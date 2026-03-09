"""Recommendation engine: orchestrates the four strategies into 6 final picks."""

from __future__ import annotations

import logging
import random

from recommender.catalogue import StoryCatalogue
from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy
from recommender.user_state import UserStateStore

logger = logging.getLogger(__name__)

# Default slot allocation: (strategy_attr_name, n_slots)
_SLOT_ALLOCATION = [
    ("_content_strategy", 2),
    ("_collaborative_strategy", 2),
    ("_topical_strategy", 1),
    ("_wildcard_strategy", 1),
]

_TOTAL_SLOTS = 6

# Mood-responsive slot allocation thresholds and recency window
_MOOD_LOW_THRESHOLD = 2.5   # average ≤ this → comfort-zone allocation
_MOOD_HIGH_THRESHOLD = 3.5  # average ≥ this → exploratory allocation
_MOOD_RECENCY_N = 5         # number of most-recent mood entries to average


def _recent_mood_level(profile: UserProfile) -> float | None:
    """Return the average of the *N* most recent mood scores, or ``None``.

    Args:
        profile: The user profile.

    Returns:
        Float in [1.0, 5.0] or ``None`` if no mood data exists.
    """
    if not profile.mood_scores:
        return None
    recent = sorted(profile.mood_scores, key=lambda x: x[0], reverse=True)
    scores = [s for _, s in recent[:_MOOD_RECENCY_N]]
    return sum(scores) / len(scores)


def _mood_slot_allocation(
    mood_level: float | None,
) -> list[tuple[str, int]]:
    """Return a strategy slot allocation tuned to *mood_level*.

    Low mood → more familiar content-based picks, no wildcard surprises.
    High mood → more wildcard exploration, fewer content-based picks.
    Neutral or no data → default allocation.

    Args:
        mood_level: Recent average mood (1–5) or ``None``.

    Returns:
        List of ``(strategy_attr_name, n_slots)`` pairs summing to 6.
    """
    if mood_level is not None and mood_level <= _MOOD_LOW_THRESHOLD:
        return [
            ("_content_strategy", 3),
            ("_collaborative_strategy", 2),
            ("_topical_strategy", 1),
            ("_wildcard_strategy", 0),
        ]
    if mood_level is not None and mood_level >= _MOOD_HIGH_THRESHOLD:
        return [
            ("_content_strategy", 1),
            ("_collaborative_strategy", 2),
            ("_topical_strategy", 1),
            ("_wildcard_strategy", 2),
        ]
    return _SLOT_ALLOCATION


class RecommendationEngine:
    """Orchestrates all recommendation strategies to produce the final 6 picks.

    Default slot allocation:

    ========================  =======
    Strategy                  Slots
    ========================  =======
    Content-based             2
    Collaborative filtering   2
    Topical (tag-based)       1
    Wildcard                  1
    ========================  =======

    **Mood-responsive allocation**: the slot counts above are adjusted based on
    the user's recent average mood score.  Low mood (≤ 2.5) shifts to
    Content×3 / Wildcard×0 (familiar comfort content).  High mood (≥ 3.5)
    shifts to Content×1 / Wildcard×2 (more exploration).  Collaborative and
    Topical slots remain fixed regardless of mood.

    **Fresh calculation**: all 6 slots are recalculated on every call.  The
    result is never returned from cache, so the user can always call again to
    receive a different set ("try again").

    **Slot stability**: if a story from the previous recommendation set appears
    in the freshly-calculated picks (and the user has not acted on it), it is
    placed at its original slot index.  This prevents jarring reordering when
    the catalogue is nearly exhausted and the engine must repeat stories.

    **Progressive coverage**: the engine prefers stories the user has never been
    recommended before.  Because the previous recommendation set is added to
    ``recommended_story_ids`` after every call, the next call's first pass
    naturally excludes those stories, ensuring consecutive calls without
    interactions return different results.  A user who consistently acts on
    every recommendation will eventually be offered every story in the catalogue.

    Args:
        catalogue: The :class:`~recommender.catalogue.StoryCatalogue`.
        user_state_store: The :class:`~recommender.user_state.UserStateStore`.
        content_strategy: Content-based filtering strategy.
        collaborative_strategy: Collaborative filtering strategy.
        topical_strategy: Topical tag-based strategy.
        wildcard_strategy: Wildcard / discovery strategy.
    """

    def __init__(
        self,
        catalogue: StoryCatalogue,
        user_state_store: UserStateStore,
        content_strategy: RecommendationStrategy,
        collaborative_strategy: RecommendationStrategy,
        topical_strategy: RecommendationStrategy,
        wildcard_strategy: RecommendationStrategy,
    ) -> None:
        self._catalogue = catalogue
        self._user_state_store = user_state_store
        self._content_strategy = content_strategy
        self._collaborative_strategy = collaborative_strategy
        self._topical_strategy = topical_strategy
        self._wildcard_strategy = wildcard_strategy

    def get_recommendations(self, user_id: str) -> list[str]:
        """Return exactly 6 story IDs for *user_id*.

        All 6 slots are recalculated on every call — the result is never
        returned from cache.  Consecutive calls without any interactions return
        different stories because progressive coverage excludes the previous
        set on the next call's first pass.

        If a freshly-calculated story was in the previous recommendation set
        (and the user has not acted on it), it is placed at its original slot
        index to avoid jarring reordering.

        Args:
            user_id: The user requesting recommendations. Must be non-empty.

        Returns:
            Ordered list of up to 6 story IDs.

        Raises:
            ValueError: If *user_id* is empty.
        """
        if not user_id:
            raise ValueError("user_id must be non-empty")

        profile = self._user_state_store.get_or_create_profile(user_id)
        catalogue = self._catalogue.get_all_stories()
        all_profiles = self._user_state_store.get_all_profiles()

        acted = profile.viewed_story_ids | profile.completed_story_ids

        # --- Derive mood-responsive slot allocation ---
        mood_level = _recent_mood_level(profile)
        allocation = _mood_slot_allocation(mood_level)

        # --- Always compute a completely fresh set of 6 picks ---
        fresh_picks = self._pick_fresh(
            profile, catalogue, all_profiles,
            n=_TOTAL_SLOTS,
            exclude_ids=acted,
            allocation=allocation,
        )

        # --- Apply slot stability as post-processing reorder ---
        # If a fresh pick was in last_recommendations at position P (and wasn't
        # acted on), assign it to position P.  This preserves slot positions
        # when the catalogue is nearly exhausted and the engine must repeat
        # previously-recommended stories.
        prev_pos: dict[str, int] = {
            sid: pos
            for pos, sid in enumerate(profile.last_recommendations)
            if sid not in acted
        }
        result: list[str | None] = [None] * _TOTAL_SLOTS
        unplaced: list[str] = []
        for sid in fresh_picks:
            if sid in prev_pos and result[prev_pos[sid]] is None:
                result[prev_pos[sid]] = sid
            else:
                unplaced.append(sid)
        empty_positions = [i for i, s in enumerate(result) if s is None]
        for pos, sid in zip(empty_positions, unplaced):
            result[pos] = sid

        final = [s for s in result if s is not None]

        # --- Update profile tracking ---
        profile.last_recommendations = final
        profile.recommended_story_ids.update(final)

        logger.debug("Recommendations for user %r: %s", user_id, final)
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_fresh(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
        allocation: list[tuple[str, int]] | None = None,
    ) -> list[str]:
        """Pick up to *n* stories using three priority tiers.

        The catalogue is pre-filtered into tiers before being passed to
        :meth:`_run_strategy_pipeline`.  Pre-filtering (rather than relying on
        ``exclude_ids``) prevents :meth:`_fill_remaining`'s last-resort path
        from reaching into a higher-priority tier's stories.

        **Tier 1 — novel:** stories never recommended to this user.  Used first
        so that a user who consistently acts on recommendations will eventually
        see every story in the catalogue (progressive coverage).

        **Tier 2 — previous:** recommended before, but *outside* the most-recent
        batch.  Ensures that consecutive calls after the catalogue is exhausted
        cycle through different stories rather than repeating the same set.

        **Tier 3 — last batch:** stories from the most-recent recommendation
        set.  Used as a final fallback when the catalogue is too small to avoid
        repeating the previous batch entirely.

        Args:
            profile: Target user profile.
            catalogue: Full story catalogue.
            all_profiles: All user profiles (for collaborative strategy).
            n: Number of stories needed.
            exclude_ids: Story IDs to hard-exclude from all tiers (acted-on
                stories that must not be recommended again).
            allocation: Strategy slot allocation; defaults to
                :data:`_SLOT_ALLOCATION` if ``None``.

        Returns:
            Up to *n* unique story IDs.
        """
        prev_batch = set(profile.last_recommendations) - exclude_ids

        # Pre-filter catalogue into three tiers. exclude_ids is applied to every
        # tier; the boundary between tiers is determined by recommended_story_ids
        # and prev_batch rather than by exclude_ids passed to the pipeline.
        tier1 = [
            s for s in catalogue
            if s.story_id not in exclude_ids
            and s.story_id not in profile.recommended_story_ids
        ]
        tier2 = [
            s for s in catalogue
            if s.story_id not in exclude_ids
            and s.story_id in profile.recommended_story_ids
            and s.story_id not in prev_batch
        ]
        tier3 = [
            s for s in catalogue
            if s.story_id not in exclude_ids
            and s.story_id in prev_batch
        ]

        picks: list[str] = []
        picked_set: set[str] = set()

        for tier_cat in (tier1, tier2, tier3):
            if len(picks) >= n or not tier_cat:
                continue
            # Cap the request to what this tier can supply.  Without this cap,
            # _fill_remaining's absolute last-resort path would repeat stories
            # already picked from the same tier to satisfy an over-large request,
            # producing duplicates in the final output.
            tier_request = min(n - len(picks), len(tier_cat))
            more = self._run_strategy_pipeline(
                profile, tier_cat, all_profiles, tier_request,
                picked_set,  # dedup across tiers; tier_cat enforces the tier boundary
                allocation=allocation,
            )
            picks.extend(more)
            picked_set.update(more)

        if len(picks) < n:
            # Whole catalogue is smaller than n slots (or all tiers combined cannot
            # fill n).  Fall back to the full catalogue and let _fill_remaining's
            # last-resort path repeat stories as needed.
            more = self._run_strategy_pipeline(
                profile, catalogue, all_profiles, n - len(picks),
                picked_set,
                allocation=allocation,
            )
            picks.extend(more)

        return picks

    def _run_strategy_pipeline(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
        allocation: list[tuple[str, int]] | None = None,
    ) -> list[str]:
        """Run all strategies in allocation order to collect up to *n* unique IDs.

        Args:
            profile: Target user profile.
            catalogue: Full story catalogue.
            all_profiles: All user profiles.
            n: Maximum number of stories to collect.
            exclude_ids: Story IDs to exclude (copied internally to avoid mutation).
            allocation: Strategy slot allocation; defaults to :data:`_SLOT_ALLOCATION`.

        Returns:
            Up to *n* unique story IDs.
        """
        allocation = allocation or _SLOT_ALLOCATION
        exclude_ids = set(exclude_ids)  # local copy — don't mutate caller's set
        picks: list[str] = []

        for strategy_attr, n_slots in allocation:
            strategy: RecommendationStrategy = getattr(self, strategy_attr)
            results = strategy.recommend(
                profile=profile,
                catalogue=catalogue,
                all_profiles=all_profiles,
                n=n_slots,
                exclude_ids=exclude_ids,
            )
            for sid in results:
                if sid not in exclude_ids:
                    picks.append(sid)
                    exclude_ids.add(sid)
                    if len(picks) >= n:
                        return picks

        if len(picks) < n:
            picks = self._fill_remaining(
                picks, exclude_ids, catalogue, profile, all_profiles, target=n
            )

        return picks[:n]

    def _fill_remaining(
        self,
        picks: list[str],
        exclude_ids: set[str],
        catalogue: list[Story],
        profile: UserProfile,
        all_profiles: list[UserProfile],
        target: int = _TOTAL_SLOTS,
    ) -> list[str]:
        """Fill remaining slots up to *target* using fallback strategies.

        Tries content-based first, then collaborative, then random from the
        remaining catalogue, finally repeating catalogue entries as a last resort.

        Args:
            picks: Current picks list (modified in-place).
            exclude_ids: Already-selected story IDs.
            catalogue: Full story catalogue.
            profile: Target user profile.
            all_profiles: All user profiles.
            target: Desired total number of picks.

        Returns:
            The *picks* list padded to at most *target* entries.
        """
        needed = target - len(picks)

        # Try content-based then collaborative fallback
        for strategy in (self._content_strategy, self._collaborative_strategy):
            if needed <= 0:
                break
            results = strategy.recommend(
                profile=profile,
                catalogue=catalogue,
                all_profiles=all_profiles,
                n=needed,
                exclude_ids=exclude_ids,
            )
            for sid in results:
                if sid not in exclude_ids and needed > 0:
                    picks.append(sid)
                    exclude_ids.add(sid)
                    needed -= 1

        # Random fallback from anything left
        if needed > 0:
            remaining = [s for s in catalogue if s.story_id not in exclude_ids]
            random.shuffle(remaining)
            for story in remaining[:needed]:
                picks.append(story.story_id)
                exclude_ids.add(story.story_id)
                needed -= 1

        # Absolute last resort: allow duplicates from catalogue
        if needed > 0 and catalogue:
            all_ids = [s.story_id for s in catalogue]
            for sid in all_ids:
                if needed <= 0:
                    break
                picks.append(sid)
                needed -= 1

        return picks
