"""Collaborative filtering strategy using user-user cosine similarity."""

from __future__ import annotations

import logging

import numpy as np

from recommender.models import Story, UserProfile
from recommender.strategies.base import RecommendationStrategy

logger = logging.getLogger(__name__)

_TOP_K_SIMILAR_USERS = 20


class CollaborativeFilteringStrategy(RecommendationStrategy):
    """Recommends stories liked by users with similar preference profiles.

    Computes pairwise cosine similarity between the target user's
    theme+tag weight vector and all other users' vectors.  Stories
    completed or highly scored by the most similar users are ranked by a
    weighted sum (similarity score × story engagement score) and returned.

    **Engagement score per story per similar user:**

    - Completed: 2.0
    - Scored (1–5): ``score / 5``  (on top of completion if both present)
    - Viewed only: 0.5

    **Cold-start behaviour**: if the target user is new (zero vector) or
    there is only one user in the system, falls back to globally
    most-completed stories.

    Args:
        all_themes: Ordered list of all theme labels.
        all_tags: Ordered list of all tag labels.
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
        """Return up to *n* stories favoured by similar users.

        Args:
            profile: Target user.
            catalogue: All available stories.
            all_profiles: All user profiles (including the target user).
            n: Number of recommendations to return.
            exclude_ids: Story IDs to avoid if possible.

        Returns:
            List of up to *n* story IDs, best-first.
        """
        if self._n_features == 0 or not catalogue:
            return []

        other_profiles = [p for p in all_profiles if p.user_id != profile.user_id]
        if not other_profiles:
            return self._cold_start_recommendations(catalogue, all_profiles, n, exclude_ids)

        target_vec = self._build_user_vector(profile)
        if np.linalg.norm(target_vec) == 0.0:
            return self._cold_start_recommendations(catalogue, all_profiles, n, exclude_ids)

        # Build matrix of other users
        user_matrix, user_ids = self._build_user_matrix(other_profiles)

        # Find most similar users
        similar_users = self._find_similar_users(
            target_vec, user_matrix, user_ids, top_k=_TOP_K_SIMILAR_USERS
        )

        # Build a map of user_id → profile for quick lookup
        profile_map = {p.user_id: p for p in other_profiles}

        # Aggregate candidate stories
        story_scores = self._aggregate_candidate_stories(
            similar_users,
            profile_map,
            exclude_ids=exclude_ids,
            target_viewed=profile.viewed_story_ids,
        )

        # Sort by aggregated score (descending) and return top-n
        catalogue_ids = {s.story_id for s in catalogue}
        ranked = sorted(
            [(sid, score) for sid, score in story_scores.items() if sid in catalogue_ids],
            key=lambda x: x[1],
            reverse=True,
        )
        return [sid for sid, _ in ranked[:n]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_vector(self, profile: UserProfile) -> np.ndarray:
        """Convert a profile's theme/tag weights into a dense float vector."""
        vec = np.zeros(self._n_features, dtype=np.float32)
        for theme, weight in profile.theme_weights.items():
            if theme in self._theme_index:
                vec[self._theme_index[theme]] = weight
        for tag, weight in profile.tag_weights.items():
            if tag in self._tag_index:
                vec[self._tag_index[tag]] = weight
        return vec

    def _build_user_matrix(
        self, profiles: list[UserProfile]
    ) -> tuple[np.ndarray, list[str]]:
        """Build an ``(n_users × n_features)`` matrix and matching user ID list.

        Args:
            profiles: The user profiles to include.

        Returns:
            Tuple of ``(matrix, user_ids)`` where ``matrix[i]`` is the
            feature vector for ``user_ids[i]``.
        """
        user_ids = [p.user_id for p in profiles]
        matrix = np.stack(
            [self._build_user_vector(p) for p in profiles], axis=0
        )  # shape: (n_users, n_features)
        return matrix, user_ids

    def _find_similar_users(
        self,
        target_vec: np.ndarray,
        user_matrix: np.ndarray,
        user_ids: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Return the top-*k* ``(user_id, similarity)`` pairs.

        Uses vectorised dot products for efficiency.

        Args:
            target_vec: Feature vector for the target user.
            user_matrix: ``(n_users × n_features)`` matrix of other users.
            user_ids: User IDs corresponding to each row of *user_matrix*.
            top_k: Maximum number of similar users to return.

        Returns:
            List of ``(user_id, cosine_similarity)`` tuples, sorted by
            similarity descending.  Only users with similarity > 0 are included.
        """
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0.0:
            return []

        row_norms = np.linalg.norm(user_matrix, axis=1)  # (n_users,)
        # Avoid division by zero for users with no history
        valid_mask = row_norms > 0.0
        similarities = np.zeros(len(user_ids), dtype=np.float32)
        if valid_mask.any():
            similarities[valid_mask] = (
                user_matrix[valid_mask] @ target_vec
            ) / (row_norms[valid_mask] * target_norm)

        # Select top-k (using argpartition for efficiency)
        actual_k = min(top_k, len(user_ids))
        if actual_k == 0:
            return []

        # Get indices of top-k by similarity
        if actual_k < len(user_ids):
            top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
        else:
            top_indices = np.arange(len(user_ids))

        results = [
            (user_ids[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.0
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _aggregate_candidate_stories(
        self,
        similar_users: list[tuple[str, float]],
        profile_map: dict[str, UserProfile],
        exclude_ids: set[str],
        target_viewed: set[str],
    ) -> dict[str, float]:
        """Score candidate stories by weighted sum over similar users' engagement.

        For each similar user, we score stories they engaged with and
        multiply that engagement score by their similarity to the target user.
        Stories the target has already viewed are given a small penalty
        (0.5×) so truly new stories are preferred, but not hard-excluded.

        Engagement scoring per story per user:
        - Completed: 2.0
        - Scored (1–5): ``score / 5``
        - Viewed only: 0.5

        Args:
            similar_users: ``(user_id, similarity)`` pairs.
            profile_map: Map from user_id to UserProfile.
            exclude_ids: Stories already picked by prior strategies.
            target_viewed: Stories the target user has already seen.

        Returns:
            Map from story_id to aggregated weighted score.
        """
        story_scores: dict[str, float] = {}

        for user_id, similarity in similar_users:
            p = profile_map.get(user_id)
            if p is None:
                continue

            # Stories this user has interacted with
            all_interacted = p.viewed_story_ids | p.completed_story_ids

            for sid in all_interacted:
                if sid in exclude_ids:
                    continue

                # Compute engagement score
                engagement = 0.0
                if sid in p.viewed_story_ids:
                    engagement += 0.5
                if sid in p.completed_story_ids:
                    engagement += 2.0
                if sid in p.story_scores:
                    engagement += p.story_scores[sid] / 5.0

                # Penalty for already-viewed stories (prefer novel content)
                if sid in target_viewed:
                    engagement *= 0.5

                story_scores[sid] = story_scores.get(sid, 0.0) + similarity * engagement

        return story_scores

    def _cold_start_recommendations(
        self,
        catalogue: list[Story],
        all_profiles: list[UserProfile],
        n: int,
        exclude_ids: set[str],
    ) -> list[str]:
        """Return globally most-completed stories as a cold-start fallback."""
        completion_counts: dict[str, int] = {}
        for p in all_profiles:
            for sid in p.completed_story_ids:
                completion_counts[sid] = completion_counts.get(sid, 0) + 1

        candidates = [s for s in catalogue if s.story_id not in exclude_ids]
        if not candidates:
            candidates = catalogue

        candidates_sorted = sorted(
            candidates,
            key=lambda s: completion_counts.get(s.story_id, 0),
            reverse=True,
        )
        return [s.story_id for s in candidates_sorted[:n]]
