"""User state store: event ingestion, preference weight accumulation, persistence."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from recommender.catalogue import StoryCatalogue
from recommender.models import EventType, Story, UserEvent, UserProfile

logger = logging.getLogger(__name__)

# Weight deltas applied when processing events
_WEIGHT_VIEW = 1.0
_WEIGHT_COMPLETE_BONUS = 2.0  # additive with view weight
_WEIGHT_SCORE_FACTOR = 0.5   # (score - 3) * factor


class UserStateStore:
    """Thread-safe in-memory store for all user profiles.

    Handles event ingestion, preference weight accumulation, and
    coordination with the C# server for persistence.

    The store is the single authoritative source of user state at runtime.
    The C# server holds the durable copy as a raw event log.  On startup,
    call :meth:`load_all_from_server` to replay events and rebuild profiles.

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
        """Reconstruct all user profiles by replaying events from the C# server.

        Calls ``StoryService.LoadUserState`` with an empty user-ID list (meaning
        "all users"), then replays each event log to rebuild in-memory profiles.
        """
        try:
            from generated import recommender_pb2

            response = self._stub.LoadUserState(
                recommender_pb2.LoadUserStateRequest(user_ids=[])
            )
            loaded = 0
            for state_msg in response.user_states:
                profile = UserProfile(user_id=state_msg.user_id)
                for event_msg in state_msg.events:
                    ts = _timestamp_to_datetime(event_msg.timestamp)
                    event_type = EventType(event_msg.event_type)
                    story_id = event_msg.story_id or None
                    score = event_msg.score if event_msg.score != 0 else None

                    event = UserEvent(
                        event_type=event_type,
                        timestamp=ts,
                        story_id=story_id,
                        score=score,
                    )
                    self._apply_event_to_profile(profile, event)
                with self._lock:
                    self._profiles[state_msg.user_id] = profile
                loaded += 1
            logger.info("Loaded state for %d users from server.", loaded)
        except Exception:
            logger.exception("Failed to load user state from server.")

    def persist_all_to_server(self) -> None:
        """Serialise all user profiles as event logs and save to the C# server.

        Calls ``StoryService.SaveUserState`` with the full event log for every
        user currently in the store.
        """
        try:
            from generated import recommender_pb2

            with self._lock:
                profiles_snapshot = list(self._profiles.values())

            user_states = []
            for profile in profiles_snapshot:
                event_msgs = []
                for event in profile.events:
                    ts = _datetime_to_timestamp(event.timestamp)
                    msg = recommender_pb2.UserEventMessage(
                        event_type=event.event_type.value,
                        story_id=event.story_id or "",
                        score=event.score or 0,
                        timestamp=ts,
                    )
                    event_msgs.append(msg)
                user_states.append(
                    recommender_pb2.UserStateMessage(
                        user_id=profile.user_id,
                        events=event_msgs,
                    )
                )

            self._stub.SaveUserState(
                recommender_pb2.SaveUserStateRequest(user_states=user_states)
            )
            logger.info("Persisted state for %d users to server.", len(user_states))
        except Exception:
            logger.exception("Failed to persist user state to server.")

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
        event = UserEvent(
            event_type=EventType.VIEWED,
            timestamp=timestamp,
            story_id=story_id,
        )
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.events.append(event)
            profile.viewed_story_ids.add(story_id)
            story = self._catalogue.get_story(story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_VIEW)

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
        event = UserEvent(
            event_type=EventType.COMPLETED,
            timestamp=timestamp,
            story_id=story_id,
        )
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.events.append(event)
            profile.completed_story_ids.add(story_id)
            story = self._catalogue.get_story(story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_COMPLETE_BONUS)

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
        event = UserEvent(
            event_type=EventType.SCORED,
            timestamp=timestamp,
            story_id=story_id,
            score=score,
        )
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.events.append(event)
            profile.story_scores[story_id] = score
            story = self._catalogue.get_story(story_id)
            if story:
                delta = (score - 3) * _WEIGHT_SCORE_FACTOR
                self._apply_weight_delta(profile, story, delta)

    def record_mood(self, user_id: str, mood_score: int, timestamp: datetime) -> None:
        """Record the user's current mood score.

        Mood is stored in the event log and
        :attr:`~recommender.models.UserProfile.mood_scores` but does not
        directly affect theme/tag preference weights in this prototype.

        Args:
            user_id: The user.
            mood_score: Integer in [1, 5].
            timestamp: When the event occurred.

        Raises:
            ValueError: If *mood_score* is outside [1, 5].
        """
        if not 1 <= mood_score <= 5:
            raise ValueError(f"Mood score must be between 1 and 5, got {mood_score!r}")
        event = UserEvent(
            event_type=EventType.MOOD,
            timestamp=timestamp,
            score=mood_score,
        )
        with self._lock:
            profile = self.get_or_create_profile(user_id)
            profile.events.append(event)
            profile.mood_scores.append((timestamp, mood_score))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_event_to_profile(self, profile: UserProfile, event: UserEvent) -> None:
        """Replay a single event onto *profile*, updating all derived state.

        Used during :meth:`load_all_from_server` to reconstruct profiles from
        their persisted event logs without going through the public ``record_*``
        methods (which also append to the event log).

        Args:
            profile: The profile to mutate in-place.
            event: The event to apply.
        """
        profile.events.append(event)

        if event.event_type == EventType.VIEWED and event.story_id:
            profile.viewed_story_ids.add(event.story_id)
            story = self._catalogue.get_story(event.story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_VIEW)

        elif event.event_type == EventType.COMPLETED and event.story_id:
            profile.completed_story_ids.add(event.story_id)
            story = self._catalogue.get_story(event.story_id)
            if story:
                self._apply_weight_delta(profile, story, _WEIGHT_COMPLETE_BONUS)

        elif event.event_type == EventType.SCORED and event.story_id and event.score:
            profile.story_scores[event.story_id] = event.score
            story = self._catalogue.get_story(event.story_id)
            if story:
                delta = (event.score - 3) * _WEIGHT_SCORE_FACTOR
                self._apply_weight_delta(profile, story, delta)

        elif event.event_type == EventType.MOOD and event.score:
            profile.mood_scores.append((event.timestamp, event.score))

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
