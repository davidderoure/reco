"""Content-based filtering strategy using cosine similarity."""

from __future__ import annotations

import logging

import numpy as np

from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy

logger = logging.getLogger(__name__)


class ContentBasedStrategy(RecommendationStrategy):
    """Recommends stories whose themes/tags best match the user's profile.

    Uses cosine similarity between the user's accumulated theme+tag weight
    vector and each story's binary theme+tag indicator vector.

    The combined feature vector is ``[theme_0, ..., theme_N, tag_0, ..., tag_M]``.
    Themes and tags each contribute equally per dimension.

    **Cold-start behaviour**: if the user has no preference weights (all
    zeros), falls back to stories with the highest average score across
    *all_profiles*, then random if no scores exist.

    Args:
        all_themes: Ordered list of all theme labels.  Defines the theme
            portion of the feature vector index.
        all_tags: Ordered list of all tag labels.  Defines the tag portion
            of the feature vector index.
    """

    def __init__(self, all_themes: list[str], all_tags: list[str]) -> None:
        self._themes = all_themes
        self._tags = all_tags
        self._theme_index = {t: i for i, t in enumerate(all_themes)}
        self._tag_index = {t: i + len(all_themes) for i, t in enumerate(all_tags)}
        self._n_features = len(all_themes) + len(all_tags)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend(
        self,
        profile: UserProfile,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return up to *n* stories most similar to the user's preference vector.

        Args:
            profile: Target user.
            catalogue: All available stories.
            all_profiles: All user profiles (used for cold-start fallback).
            n: Number of recommendations to return.
            exclude_ids: Story IDs to avoid if possible.

        Returns:
            List of up to *n* story IDs, best-first.
        """
        if self._n_features == 0 or not catalogue:
            return []

        user_vec = self._build_user_vector(profile)
        is_cold_start = np.linalg.norm(user_vec) == 0.0

        if is_cold_start:
            return self._cold_start_recommendations(
                catalogue, all_profiles, n, exclude_ids
            )

        # Prefer unviewed stories, but don't hard-exclude them if there
        # aren't enough candidates
        candidates = [
            s for s in catalogue
            if s.story_id not in profile.viewed_story_ids
            and s.story_id not in exclude_ids
        ]
        if len(candidates) < n:
            # Relax: allow viewed stories (but still respect exclude_ids)
            candidates = [s for s in catalogue if s.story_id not in exclude_ids]

        if not candidates:
            candidates = catalogue  # last resort: ignore all filters

        scores = [
            (story, self._cosine_similarity(user_vec, self._build_story_vector(story)))
            for story in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [story.story_id for story, _ in scores[:n]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_vector(self, profile: UserProfile) -> np.ndarray:
        """Convert the profile's theme/tag weights into a dense float vector."""
        vec = np.zeros(self._n_features, dtype=np.float32)
        for theme, weight in profile.theme_weights.items():
            if theme in self._theme_index:
                vec[self._theme_index[theme]] = weight
        for tag, weight in profile.tag_weights.items():
            if tag in self._tag_index:
                vec[self._tag_index[tag]] = weight
        return vec

    def _build_story_vector(self, story: Story) -> np.ndarray:
        """Convert a story's themes/tags into a binary indicator vector."""
        vec = np.zeros(self._n_features, dtype=np.float32)
        for theme in story.themes:
            if theme in self._theme_index:
                vec[self._theme_index[theme]] = 1.0
        for tag in story.tags:
            if tag in self._tag_index:
                vec[self._tag_index[tag]] = 1.0
        return vec

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity in [0, 1]; returns 0.0 for zero vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _cold_start_recommendations(
        self,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return stories with the highest average score across all users.

        Falls back to random order if no user has scored any story yet.
        """
        # Compute mean score per story across all users
        score_sums: dict[str, float] = {}
        score_counts: dict[str, int] = {}
        for p in all_profiles:
            for sid, score in p.story_scores.items():
                score_sums[sid] = score_sums.get(sid, 0.0) + score
                score_counts[sid] = score_counts.get(sid, 0) + 1

        def avg_score(story: Story) -> float:
            count = score_counts.get(story.story_id, 0)
            return score_sums.get(story.story_id, 0.0) / count if count > 0 else 0.0

        candidates = [s for s in catalogue if s.story_id not in exclude_ids]
        if not candidates:
            candidates = catalogue

        candidates_sorted = sorted(candidates, key=avg_score, reverse=True)
        return [s.story_id for s in candidates_sorted[:n]]
