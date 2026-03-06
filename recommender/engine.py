"""Recommendation engine: orchestrates the four strategies into 6 final picks."""

from __future__ import annotations

import logging
import random

from recommender.catalogue import StoryCatalogue
from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy
from recommender.user_state import UserStateStore

logger = logging.getLogger(__name__)

# Slot allocation: (strategy_attr_name, n_slots)
_SLOT_ALLOCATION = [
    ("_content_strategy", 2),
    ("_collaborative_strategy", 2),
    ("_topical_strategy", 1),
    ("_wildcard_strategy", 1),
]

_TOTAL_SLOTS = 6


class RecommendationEngine:
    """Orchestrates all recommendation strategies to produce the final 6 picks.

    Slot allocation:

    ========================  =======
    Strategy                  Slots
    ========================  =======
    Content-based             2
    Collaborative filtering   2
    Topical (tag-based)       1
    Wildcard                  1
    ========================  =======

    **Slot stability**: stories from the previous recommendation set that the
    user has not yet acted on (viewed or completed) are kept in their original
    slot positions.  Only the slots freed by user actions are replaced with
    fresh picks.

    **Progressive coverage**: when filling open slots the engine prefers stories
    the user has never been recommended before.  This guarantees that a user who
    consistently acts on every recommendation will eventually be offered every
    story in the catalogue.

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
        """Return exactly 6 story IDs for *user_id*, preserving slot stability.

        Stories from the previous recommendation set that the user has not yet
        acted on are kept in their original positions.  Open slots (stories the
        user viewed or completed, or first-time requests) are filled with fresh
        picks, preferring stories the user has never been recommended before.

        Args:
            user_id: The user requesting recommendations. Must be non-empty.

        Returns:
            Ordered list of up to 6 story IDs.  Sticky stories appear at their
            original index; new stories are placed at open indices in random
            order.

        Raises:
            ValueError: If *user_id* is empty.
        """
        if not user_id:
            raise ValueError("user_id must be non-empty")

        profile = self._user_state_store.get_or_create_profile(user_id)
        catalogue = self._catalogue.get_all_stories()
        all_profiles = self._user_state_store.get_all_profiles()

        # --- Identify sticky slots ---
        # A slot is sticky when its story hasn't been acted on (viewed or completed)
        acted = profile.viewed_story_ids | profile.completed_story_ids
        prev = list(profile.last_recommendations)
        prev_padded: list[str | None] = (prev + [None] * _TOTAL_SLOTS)[:_TOTAL_SLOTS]

        sticky_slots: list[str | None] = [
            sid if (sid and sid not in acted) else None
            for sid in prev_padded
        ]
        sticky_ids = {sid for sid in sticky_slots if sid}
        open_positions = [i for i, sid in enumerate(sticky_slots) if sid is None]

        # --- Fill open slots with fresh picks ---
        fresh_picks: list[str] = []
        if open_positions:
            fresh_picks = self._pick_fresh(
                profile, catalogue, all_profiles,
                n=len(open_positions),
                exclude_ids=sticky_ids,
            )
            # Shuffle so open slots get a random ordering among fresh picks
            random.shuffle(fresh_picks)

        # --- Assemble ordered result ---
        result: list[str | None] = list(sticky_slots)
        fresh_iter = iter(fresh_picks)
        for pos in open_positions:
            result[pos] = next(fresh_iter, None)

        final = [sid for sid in result if sid]

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
    ) -> list[str]:
        """Pick up to *n* stories for newly opened slots.

        First pass: exclude stories already recommended to this user (prefer
        novel content for progressive coverage).  If fewer than *n* novel
        stories are available, a second pass allows previously recommended
        stories to fill the remainder.

        Args:
            profile: Target user profile.
            catalogue: Full story catalogue.
            all_profiles: All user profiles (for collaborative strategy).
            n: Number of stories needed.
            exclude_ids: Story IDs to exclude in both passes (sticky stories).

        Returns:
            Up to *n* unique story IDs.
        """
        # First pass: strongly prefer stories never recommended to this user
        exclude_with_prev = exclude_ids | profile.recommended_story_ids
        picks = self._run_strategy_pipeline(
            profile, catalogue, all_profiles, n, exclude_with_prev
        )

        if len(picks) < n:
            # Relax: allow previously recommended stories
            needed = n - len(picks)
            relaxed_exclude = exclude_ids | set(picks)
            more = self._run_strategy_pipeline(
                profile, catalogue, all_profiles, needed, relaxed_exclude
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
    ) -> list[str]:
        """Run all strategies in allocation order to collect up to *n* unique IDs.

        Args:
            profile: Target user profile.
            catalogue: Full story catalogue.
            all_profiles: All user profiles.
            n: Maximum number of stories to collect.
            exclude_ids: Story IDs to exclude (copied internally to avoid mutation).

        Returns:
            Up to *n* unique story IDs.
        """
        exclude_ids = set(exclude_ids)  # local copy — don't mutate caller's set
        picks: list[str] = []

        for strategy_attr, n_slots in _SLOT_ALLOCATION:
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
