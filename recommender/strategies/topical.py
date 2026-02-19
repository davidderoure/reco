"""Topical recommendation strategy based on the user's highest-weight tag."""

from __future__ import annotations

import logging

from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy

logger = logging.getLogger(__name__)


class TopicalStrategy(RecommendationStrategy):
    """Recommends stories based on the user's single strongest tag preference.

    Identifies the tag with the highest accumulated weight in the user's
    profile, then returns unviewed stories that carry that tag, ranked by
    how many of the user's top tags they contain.

    **Cold-start behaviour**: if the user has no tag history, falls back to
    stories containing the most globally used tag across the entire catalogue.
    """

    def recommend(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return up to *n* stories strongly aligned with the user's top tag.

        Args:
            profile: Target user.
            catalogue: All available stories.
            all_profiles: All user profiles (used for cold-start fallback).
            n: Number of recommendations to return.
            exclude_ids: Story IDs to avoid if possible.

        Returns:
            List of up to *n* story IDs.
        """
        if not catalogue:
            return []

        # Determine the target tag
        if profile.tag_weights:
            top_tag = max(profile.tag_weights, key=lambda t: profile.tag_weights[t])
        else:
            top_tag = self._most_popular_tag(catalogue)

        if top_tag is None:
            return []

        # Rank unviewed stories by number of user's top tags they contain
        user_top_tags = self._get_top_tags(profile, n_tags=10)

        candidates = [
            s for s in catalogue
            if top_tag in s.tags
            and s.story_id not in profile.viewed_story_ids
            and s.story_id not in exclude_ids
        ]

        if not candidates:
            # Relax: allow viewed stories
            candidates = [
                s for s in catalogue
                if top_tag in s.tags and s.story_id not in exclude_ids
            ]

        if not candidates:
            candidates = [s for s in catalogue if top_tag in s.tags]

        def relevance_score(story: Story) -> int:
            return sum(1 for tag in story.tags if tag in user_top_tags)

        candidates_sorted = sorted(candidates, key=relevance_score, reverse=True)
        return [s.story_id for s in candidates_sorted[:n]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _most_popular_tag(catalogue: list[Story]) -> str | None:
        """Return the tag that appears most frequently across all stories.

        Args:
            catalogue: All available stories.

        Returns:
            The most common tag string, or ``None`` if all stories have no tags.
        """
        tag_counts: dict[str, int] = {}
        for story in catalogue:
            for tag in story.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return max(tag_counts, key=lambda t: tag_counts[t]) if tag_counts else None

    @staticmethod
    def _get_top_tags(profile: UserProfile, n_tags: int) -> set[str]:
        """Return the *n_tags* highest-weight tags from the user's profile.

        Args:
            profile: The user profile to inspect.
            n_tags: Maximum number of tags to return.

        Returns:
            Set of tag strings.
        """
        if not profile.tag_weights:
            return set()
        sorted_tags = sorted(
            profile.tag_weights.items(), key=lambda x: x[1], reverse=True
        )
        return {tag for tag, _ in sorted_tags[:n_tags]}
