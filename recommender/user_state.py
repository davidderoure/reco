"""User state store: event ingestion, preference weight accumulation, persistence."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from recommender.catalogue import StoryCatalogue
from recommender.models import Story, UserProfile

logger = logging.getLogger(__name__)

# Weight deltas applied when processing events
_WEIGHT_VIEW = 1.0
_WEIGHT_COMPLETE_BONUS = 2.0  # additive with view weight
_WEIGHT_SCORE_FACTOR = 0.5   # (score - 3) * factor

# Maximum number of mood entries to keep in memory and persist.
MOOD_HISTORY_LIMIT = 50

# Mood attribution: when mood changes, a fraction of the weight accumulated
# since the previous mood event is re-applied as a boost (positive delta) or
# dampening (negative delta).  0.5 means a full 4-point improvement adds at
# most half a "viewed" event worth of weight per theme/tag.
MOOD_ATTRIBUTION_FACTOR = 0.5

# A read-progress event whose read_percent meets or exceeds this threshold is
# treated as a "viewed" event: the story receives the standard view weight delta
# and is added to viewed_story_ids.  Repeated events above the threshold are
# idempotent — the weight is only applied once.
READ_VIEWED_THRESHOLD_PERCENT = 50


class UserStateStore:
    """Thread-safe in-memory store for all user profiles.

    Handles event ingestion, preference weight accumulation, and
    coordination with the C# server for persistence.

    The store is the single authoritative source of user state at runtime.
    The C# server holds a durable copy of the compiled user model (weights,
    IDs, scores). On startup, call :meth:`load_all_from_server` to restore
    profiles directly from the stored model — no event replay is needed.

    Args:
        stub: A ``StoryServiceStub`` gRPC client stub (or compatible mock).
        catalogue: The :class:`~recommender.catalogue.StoryCatalogue` used to
            look up story metadata when computing weight deltas.
    """

    def __init__(self, stub: Any, catalogue: StoryCatalogue) -> None:
        self._stub = stub
        self._catalogue = catalogue
        self._lock = threading.RLock()
        self._profiles: dict[str, UserProfile] = {}
        self._persist_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_all_from_server(self) -> None:
        """Restore all user profiles from the compiled model stored on the C# server.

        Calls ``StoryService.LoadUserModel`` with an empty user-ID list (meaning
        "all users") and deserializes each ``UserModelMessage`` directly into a
        :class:`~recommender.models.UserProfile` — no event replay required.
        """
        try:
            from generated import recommender_pb2

            response = self._stub.LoadUserModel(
                recommender_pb2.LoadUserModelRequest(user_ids=[])
            )
            loaded = 0
            for model_msg in response.user_models:
                profile = UserProfile(
                    user_id=model_msg.user_id,
                    viewed_story_ids=set(model_msg.viewed_story_ids),
                    completed_story_ids=set(model_msg.completed_story_ids),
                    story_scores=dict(model_msg.story_scores),
                    mood_scores=[
                        (_timestamp_to_datetime(e.timestamp), e.score)
                        for e in model_msg.mood_scores
                    ],
                    theme_weights=dict(model_msg.theme_weights),
                    tag_weights=dict(model_msg.tag_weights),
                    last_recommendations=list(model_msg.last_recommendations),
                    recommended_story_ids=set(model_msg.recommended_story_ids),
                )
                with self._lock:
                    self._profiles[model_msg.user_id] = profile
                loaded += 1
            logger.info("Loaded model for %d users from server.", loaded)
        except Exception:
            logger.exception("Failed to load user model from server.")

    def persist_all_to_server(self) -> None:
        """Serialise all user profiles as compact models and save to the C# server.

        Calls ``StoryService.SaveUserModel`` with the current derived state for
        every user in the store. Payload size is bounded by catalogue vocabulary
        size, not by interaction history length.
        """
        try:
            from generated import recommender_pb2

            with self._lock:
                profiles_snapshot = list(self._profiles.values())

            user_models = []
            for profile in profiles_snapshot:
                mood_entries = [
                    recommender_pb2.MoodEntry(
                        timestamp=_datetime_to_timestamp(ts),
                        score=score,
                    )
                    for ts, score in profile.mood_scores
                ]
                user_models.append(
                    recommender_pb2.UserModelMessage(
                        user_id=profile.user_id,
                        viewed_story_ids=list(profile.viewed_story_ids),
                        completed_story_ids=list(profile.completed_story_ids),
                        story_scores=profile.story_scores,
                        mood_scores=mood_entries,
                        theme_weights=profile.theme_weights,
                        tag_weights=profile.tag_weights,
                        last_recommendations=profile.last_recommendations,
                        recommended_story_ids=list(profile.recommended_story_ids),
                    )
                )

            self._stub.SaveUserModel(
                recommender_pb2.SaveUserModelRequest(user_models=user_models)
            )
            logger.info("Persisted model for %d users to server.", len(user_models))
        except Exception:
            logger.exception("Failed to persist user model to server.")

    def start_persist_loop(self, interval_seconds: int = 60) -> None:
        """Start a background daemon thread that periodically persists all state.

        Safe to call multiple times — only one thread is started.

        Args:
            interval_seconds: Seconds between persist calls.
        """
        if self._persist_thread is not None and self._persist_thread.is_alive():
            return
        self._persist_thread = threading.Thread(
            target=self._persist_loop,
            args=(interval_seconds,),
            name="state-persist",
            daemon=True,
        )
        self._persist_thread.start()
        logger.debug("State persist loop started (interval=%ds).", interval_seconds)

    # ------------------------------------------------------------------
    # Profile access
    # ------------------------------------------------------------------

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Return the existing profile for *user_id*, or create a new empty one.

        Args:
            user_id: The user's unique identifier.

        Returns:
            The :class:`~recommender.models.UserProfile` for *user_id*.
        """
        with self._lock:
            if user_id not in self._profiles:
                self._profiles[user_id] = UserProfile(user_id=user_id)
            return self._profiles[user_id]

    def get_all_profiles(self) -> list[UserProfile]:
        """Return a snapshot list of all user profiles.

        Returns:
            List of all :class:`~recommender.models.UserProfile` objects.
        """
        with self._lock:
            return list(self._profiles.values())

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_viewed(self, user_id: str, story_id: str, timestamp: datetime) -> None:
        """Record that *user_id* viewed *story_id* and update preference weights.

        Adds ``+1.0`` to each theme/tag weight for the story.

        Args:
            user_id: The viewing user.
            story_id: The story that was viewed.
            timestamp: When the event occurred.
        """
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.viewed_story_ids.add(story_id)
            story = self._catalogue.get_story(story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_VIEW)
                self._accumulate_mood_window(profile, story, _WEIGHT_VIEW)

    def record_completed(self, user_id: str, story_id: str, timestamp: datetime) -> None:
        """Record that *user_id* completed *story_id* and update preference weights.

        Adds a ``+2.0`` bonus to each theme/tag weight (additive with the
        view weight, so a completed story contributes +3.0 total if both
        events are received).

        Also adds *story_id* to :attr:`~recommender.models.UserProfile.completed_story_ids`.

        Args:
            user_id: The user who completed the story.
            story_id: The story that was completed.
            timestamp: When the event occurred.
        """
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.completed_story_ids.add(story_id)
            story = self._catalogue.get_story(story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_COMPLETE_BONUS)
                self._accumulate_mood_window(profile, story, _WEIGHT_COMPLETE_BONUS)

    def record_scored(
        self, user_id: str, story_id: str, score: int, timestamp: datetime
    ) -> None:
        """Record the user's end-of-story rating and update preference weights.

        Applies a weight delta of ``(score - 3) × 0.5`` per theme/tag.
        Score 3 is neutral (no change), 4–5 boost weights, 1–2 penalise.

        Args:
            user_id: The rating user.
            story_id: The rated story.
            score: Integer in [1, 5].
            timestamp: When the event occurred.

        Raises:
            ValueError: If *score* is outside [1, 5].
        """
        if not 1 <= score <= 5:
            raise ValueError(f"Score must be between 1 and 5, got {score!r}")
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.story_scores[story_id] = score
            story = self._catalogue.get_story(story_id)
            if story:
                delta = (score - 3) * _WEIGHT_SCORE_FACTOR
                self._apply_weight_delta(profile, story, delta)

    def record_read_progress(
        self, user_id: str, story_id: str, read_percent: int, timestamp: datetime
    ) -> None:
        """Record how far through a story the user has read.

        If *read_percent* is at or above :data:`READ_VIEWED_THRESHOLD_PERCENT`
        and the story has not already been recorded as viewed, the standard
        view weight delta is applied and the story is added to
        :attr:`~recommender.models.UserProfile.viewed_story_ids`.  Repeated
        calls above the threshold are idempotent — weights are only applied
        once.  Events below the threshold are a no-op for the recommendation
        engine.

        Args:
            user_id: The user.
            story_id: The story being read.
            read_percent: Integer in [0, 100] — percentage of the story read.
            timestamp: When the event occurred.

        Raises:
            ValueError: If *read_percent* is outside [0, 100].
        """
        if not 0 <= read_percent <= 100:
            raise ValueError(
                f"read_percent must be between 0 and 100, got {read_percent!r}"
            )
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            if (
                read_percent >= READ_VIEWED_THRESHOLD_PERCENT
                and story_id not in profile.viewed_story_ids
            ):
                profile.viewed_story_ids.add(story_id)
                story = self._catalogue.get_story(story_id)
                if story:
                    self._apply_weight_delta(profile, story, _WEIGHT_VIEW)

    def record_mood(self, user_id: str, mood_score: int, timestamp: datetime) -> None:
        """Record the user's current mood score and apply attribution feedback.

        Mood is stored in :attr:`~recommender.models.UserProfile.mood_scores`
        (capped at :data:`MOOD_HISTORY_LIMIT` recent entries).

        **Attribution**: the change in mood relative to the previous recorded
        mood is used to adjust theme/tag weights for content engaged with since
        that prior event.  A mood improvement boosts those weights by
        ``MOOD_ATTRIBUTION_FACTOR × (delta / 4)`` of the accumulated delta; a
        mood decline dampens them by the same fraction (capped at 0).  The
        transient accumulators are reset after each mood event.

        Args:
            user_id: The user.
            mood_score: Integer in [1, 5].
            timestamp: When the event occurred.

        Raises:
            ValueError: If *mood_score* is outside [1, 5].
        """
        if not 1 <= mood_score <= 5:
            raise ValueError(f"Mood score must be between 1 and 5, got {mood_score!r}")
        with self._lock:
            profile = self.get_or_create_profile(user_id)

            # Apply attribution: boost/dampen themes & tags engaged since last mood
            if profile.last_mood_score is not None:
                mood_delta = mood_score - profile.last_mood_score
                if mood_delta != 0:
                    attribution = (mood_delta / 4) * MOOD_ATTRIBUTION_FACTOR
                    for theme, w in profile.themes_since_last_mood.items():
                        profile.theme_weights[theme] = max(
                            0.0,
                            profile.theme_weights.get(theme, 0.0) + attribution * w,
                        )
                    for tag, w in profile.tags_since_last_mood.items():
                        profile.tag_weights[tag] = max(
                            0.0,
                            profile.tag_weights.get(tag, 0.0) + attribution * w,
                        )

            # Reset transient accumulators and record new mood
            profile.last_mood_score = mood_score
            profile.themes_since_last_mood = {}
            profile.tags_since_last_mood = {}
            profile.mood_scores.append((timestamp, mood_score))
            if len(profile.mood_scores) > MOOD_HISTORY_LIMIT:
                profile.mood_scores = profile.mood_scores[-MOOD_HISTORY_LIMIT:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_weight_delta(profile: UserProfile, story: Story, delta: float) -> None:
        """Add *delta* to all theme and tag weights associated with *story*.

        Args:
            profile: The profile to mutate in-place.
            story: The story whose themes/tags receive the delta.
            delta: The amount to add (may be negative).
        """
        for theme in story.themes:
            profile.theme_weights[theme] = profile.theme_weights.get(theme, 0.0) + delta
        for tag in story.tags:
            profile.tag_weights[tag] = profile.tag_weights.get(tag, 0.0) + delta

    @staticmethod
    def _accumulate_mood_window(
        profile: UserProfile, story: Story, delta: float
    ) -> None:
        """Accumulate *delta* into the transient mood-attribution window.

        Tracks how much weight has been applied to each theme/tag since the
        last mood event so that :meth:`record_mood` can apply proportional
        attribution when mood changes.

        Args:
            profile: The profile to mutate in-place.
            story: The story whose themes/tags are accumulated.
            delta: The weight delta already applied to the main weights.
        """
        for theme in story.themes:
            profile.themes_since_last_mood[theme] = (
                profile.themes_since_last_mood.get(theme, 0.0) + delta
            )
        for tag in story.tags:
            profile.tags_since_last_mood[tag] = (
                profile.tags_since_last_mood.get(tag, 0.0) + delta
            )

    def _persist_loop(self, interval_seconds: int) -> None:
        """Periodically persist all user state. Runs in a daemon thread."""
        while True:
            time.sleep(interval_seconds)
            self.persist_all_to_server()


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _timestamp_to_datetime(ts: Any) -> datetime:
    """Convert a ``google.protobuf.Timestamp`` to a UTC-aware ``datetime``.

    Args:
        ts: A protobuf Timestamp object with ``seconds`` and ``nanos`` fields.

    Returns:
        A UTC-aware :class:`datetime`.
    """
    return datetime.fromtimestamp(ts.seconds + ts.nanos / 1e9, tz=timezone.utc)


def _datetime_to_timestamp(dt: datetime) -> Any:
    """Convert a ``datetime`` to a ``google.protobuf.Timestamp``.

    Args:
        dt: Any :class:`datetime`; naive datetimes are assumed UTC.

    Returns:
        A ``google.protobuf.Timestamp`` instance.
    """
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.FromDatetime(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
    return ts
