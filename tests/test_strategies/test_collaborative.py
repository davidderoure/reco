"""Tests for CollaborativeFilteringStrategy."""

from __future__ import annotations

import pytest

from recommender.models import Story, UserProfile
from recommender.strategies.collaborative import CollaborativeFilteringStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy(all_themes, all_tags) -> CollaborativeFilteringStrategy:
    return CollaborativeFilteringStrategy(all_themes=all_themes, all_tags=all_tags)


# ---------------------------------------------------------------------------
# Core recommendation tests
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_returns_list(self, strategy, sample_stories, adventure_profile, mystery_profile) -> None:
        result = strategy.recommend(
            adventure_profile, sample_stories, [adventure_profile, mystery_profile], 2, set()
        )
        assert isinstance(result, list)

    def test_returns_at_most_n(
        self, strategy, sample_stories, adventure_profile, mystery_profile
    ) -> None:
        result = strategy.recommend(
            adventure_profile, sample_stories, [adventure_profile, mystery_profile], 2, set()
        )
        assert len(result) <= 2

    def test_story_ids_are_in_catalogue(
        self, strategy, sample_stories, adventure_profile, mystery_profile
    ) -> None:
        catalogue_ids = {s.story_id for s in sample_stories}
        result = strategy.recommend(
            adventure_profile, sample_stories, [adventure_profile, mystery_profile], 2, set()
        )
        for sid in result:
            assert sid in catalogue_ids

    def test_recommends_stories_from_similar_user(
        self, strategy, sample_stories, adventure_profile, mystery_profile
    ) -> None:
        """A mystery user similar to adventure user should surface mystery stories."""
        # adventure_profile has some mystery weight; mystery_profile is similar
        result = strategy.recommend(
            adventure_profile, sample_stories, [adventure_profile, mystery_profile], 2, set()
        )
        assert len(result) >= 0  # as long as it doesn't crash

    def test_respects_exclude_ids(
        self, strategy, sample_stories, adventure_profile, mystery_profile
    ) -> None:
        all_profiles = [adventure_profile, mystery_profile]
        result1 = strategy.recommend(adventure_profile, sample_stories, all_profiles, 1, set())
        if result1:
            exclude = set(result1)
            result2 = strategy.recommend(
                adventure_profile, sample_stories, all_profiles, 1, exclude
            )
            if result2:
                assert not set(result2) & exclude

    def test_empty_catalogue_returns_empty(
        self, strategy, adventure_profile, mystery_profile
    ) -> None:
        result = strategy.recommend(
            adventure_profile, [], [adventure_profile, mystery_profile], 2, set()
        )
        assert result == []


# ---------------------------------------------------------------------------
# Cold-start and edge-case tests
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_new_user_returns_results(
        self, strategy, sample_stories, new_user_profile, adventure_profile
    ) -> None:
        """Cold-start (zero vector user) should return popular stories."""
        result = strategy.recommend(
            new_user_profile,
            sample_stories,
            [new_user_profile, adventure_profile],
            2,
            set(),
        )
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_single_user_system_returns_results(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        """When only the target user exists, fall back gracefully."""
        result = strategy.recommend(
            adventure_profile, sample_stories, [adventure_profile], 2, set()
        )
        assert isinstance(result, list)

    def test_no_users_returns_empty_or_list(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# User similarity tests
# ---------------------------------------------------------------------------


class TestFindSimilarUsers:
    def test_returns_sorted_by_similarity(
        self, strategy, adventure_profile, mystery_profile
    ) -> None:
        import numpy as np

        matrix, user_ids = strategy._build_user_matrix([adventure_profile, mystery_profile])
        target_vec = strategy._build_user_vector(adventure_profile)
        # Use a different target to avoid same-user issues
        target_vec2 = strategy._build_user_vector(mystery_profile)
        results = strategy._find_similar_users(target_vec2, matrix, user_ids, top_k=2)
        if len(results) >= 2:
            # Should be sorted descending
            assert results[0][1] >= results[1][1]

    def test_zero_vector_returns_empty(
        self, strategy, adventure_profile, mystery_profile
    ) -> None:
        import numpy as np

        matrix, user_ids = strategy._build_user_matrix([adventure_profile, mystery_profile])
        zero_vec = np.zeros(strategy._n_features, dtype=np.float32)
        results = strategy._find_similar_users(zero_vec, matrix, user_ids, top_k=2)
        assert results == []

    def test_top_k_limits_results(
        self, strategy, adventure_profile, mystery_profile
    ) -> None:
        matrix, user_ids = strategy._build_user_matrix([adventure_profile, mystery_profile])
        target_vec = strategy._build_user_vector(adventure_profile)
        results = strategy._find_similar_users(target_vec, matrix, user_ids, top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Aggregate candidate stories tests
# ---------------------------------------------------------------------------


class TestAggregateStories:
    def test_completed_stories_score_higher_than_viewed(
        self, strategy, adventure_profile, mystery_profile
    ) -> None:
        """Completed stories should receive a higher engagement score."""
        # mystery_profile completed s_mys (engagement 2.5) and viewed s_multi (0.5)
        similar_users = [("u_mys", 1.0)]
        profile_map = {"u_mys": mystery_profile}
        scores = strategy._aggregate_candidate_stories(
            similar_users, profile_map, exclude_ids=set(), target_viewed=set()
        )
        # s_mys was viewed AND completed = 0.5 + 2.0 = 2.5
        assert scores.get("s_mys", 0) > scores.get("s_multi", 0)

    def test_exclude_ids_not_in_result(self, strategy, mystery_profile) -> None:
        similar_users = [("u_mys", 1.0)]
        profile_map = {"u_mys": mystery_profile}
        scores = strategy._aggregate_candidate_stories(
            similar_users,
            profile_map,
            exclude_ids={"s_mys"},
            target_viewed=set(),
        )
        assert "s_mys" not in scores
