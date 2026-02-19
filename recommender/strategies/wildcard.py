"""Wildcard recommendation strategy for serendipitous discovery."""

from __future__ import annotations

import logging
import random

from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy

logger = logging.getLogger(__name__)


class WildcardStrategy(RecommendationStrategy):
    """Returns a random story to encourage serendipitous discovery.

    Specifically prefers stories in themes the user has *not* explored yet
    (zero weight in their profile), so the wildcard genuinely expands the
    user's horizons rather than repeating familiar territory.

    **Priority order:**

    1. Unviewed stories in unexplored themes (theme_weight == 0), excluding *exclude_ids*
    2. Any unviewed story, excluding *exclude_ids*
    3. Any story in unexplored themes (relaxing "unviewed" constraint)
    4. Any story at all (last resort for tiny catalogues)
    """

    def recommend(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return up to *n* random stories, preferring unexplored themes.

        Args:
            profile: Target user.
            catalogue: All available stories.
            all_profiles: All user profiles (unused; present for interface consistency).
            n: Number of recommendations to return.
            exclude_ids: Story IDs to avoid if possible.

        Returns:
            List of up to *n* story IDs, chosen randomly from the best
            available candidate pool.
        """
        if not catalogue:
            return []

        explored_themes = set(profile.theme_weights.keys())

        def is_unexplored(story: Story) -> bool:
            return all(t not in explored_themes for t in story.themes)

        # Priority 1: unviewed + unexplored + not excluded
        candidates = [
            s for s in catalogue
            if s.story_id not in profile.viewed_story_ids
            and s.story_id not in exclude_ids
            and is_unexplored(s)
        ]

        if not candidates:
            # Priority 2: any unviewed + not excluded
            candidates = [
                s for s in catalogue
                if s.story_id not in profile.viewed_story_ids
                and s.story_id not in exclude_ids
            ]

        if not candidates:
            # Priority 3: unexplored (relax "unviewed")
            candidates = [
                s for s in catalogue
                if s.story_id not in exclude_ids and is_unexplored(s)
            ]

        if not candidates:
            # Priority 4: anything not excluded
            candidates = [s for s in catalogue if s.story_id not in exclude_ids]

        if not candidates:
            # Absolute last resort
            candidates = catalogue

        chosen = random.sample(candidates, min(n, len(candidates)))
        return [s.story_id for s in chosen]
