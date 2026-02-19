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

    Results from all strategies are deduplicated (``exclude_ids`` accumulates
    as each strategy is called) then shuffled so the C# client cannot infer
    which slot each recommendation came from.

    **Fallback**: if any strategy returns fewer than its allocation, the
    engine fills remaining slots by calling the content-based strategy first,
    then collaborative, then random from the catalogue.

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
        """Return exactly 6 story IDs in random order for *user_id*.

        Runs each strategy for its allocated slots in sequence, accumulating
        ``exclude_ids`` to prevent duplicates.  If the total is fewer than 6
        (small catalogue or new user edge cases), fills remaining slots with
        fallback picks.

        Args:
            user_id: The user requesting recommendations. Must be non-empty.

        Returns:
            List of exactly 6 story IDs, shuffled.

        Raises:
            ValueError: If *user_id* is empty.
        """
        if not user_id:
            raise ValueError("user_id must be non-empty")

        profile = self._user_state_store.get_or_create_profile(user_id)
        catalogue = self._catalogue.get_all_stories()
        all_profiles = self._user_state_store.get_all_profiles()

        exclude_ids: set[str] = set()
        picks: list[str] = []

        # Run each strategy for its allocated slots
        for strategy_attr, n_slots in _SLOT_ALLOCATION:
            strategy: RecommendationStrategy = getattr(self, strategy_attr)
            results = strategy.recommend(
                profile=profile,
                catalogue=catalogue,
                all_profiles=all_profiles,
                n=n_slots,
                exclude_ids=exclude_ids,
            )
            # Add non-duplicate results
            for sid in results:
                if sid not in exclude_ids:
                    picks.append(sid)
                    exclude_ids.add(sid)
                    if len(picks) >= _TOTAL_SLOTS:
                        break
            if len(picks) >= _TOTAL_SLOTS:
                break

        # Fill any remaining slots
        if len(picks) < _TOTAL_SLOTS:
            picks = self._fill_remaining(
                picks, exclude_ids, catalogue, profile, all_profiles
            )

        random.shuffle(picks)
        logger.debug(
            "Recommendations for user %r: %s", user_id, picks
        )
        return picks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_remaining(
        self,
        picks: list[str],
        exclude_ids: set[str],
        catalogue: list[Story],
        profile: UserProfile,
        all_profiles: list[UserProfile],
    ) -> list[str]:
        """Fill remaining slots up to ``_TOTAL_SLOTS`` using fallback strategies.

        Tries content-based first, then collaborative, then random from the
        remaining catalogue.

        Args:
            picks: Current picks list (modified in-place).
            exclude_ids: Already-selected story IDs.
            catalogue: Full story catalogue.
            profile: Target user profile.
            all_profiles: All user profiles.

        Returns:
            The *picks* list padded to at most ``_TOTAL_SLOTS`` entries.
        """
        needed = _TOTAL_SLOTS - len(picks)

        # Try content-based fallback
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
