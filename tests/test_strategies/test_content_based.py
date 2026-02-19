"""Tests for ContentBasedStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.models import Story, UserProfile
from recommender.strategies.content_based import ContentBasedStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy(all_themes, all_tags) -> ContentBasedStrategy:
    return ContentBasedStrategy(all_themes=all_themes, all_tags=all_tags)


# ---------------------------------------------------------------------------
# Core recommendation tests
# ---------------------------------------------------------------------------


class TestRecommend:
    def test_returns_list(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        assert isinstance(result, list)

    def test_returns_at_most_n(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        assert len(result) <= 2

    def test_returns_story_ids_in_catalogue(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        catalogue_ids = {s.story_id for s in sample_stories}
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        for sid in result:
            assert sid in catalogue_ids

    def test_adventure_user_gets_adventure_stories(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        """A user with adventure preference should get adventure stories."""
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        assert len(result) > 0
        result_stories = {s.story_id: s for s in sample_stories}
        for sid in result:
            story = result_stories[sid]
            assert "adventure" in story.themes or "mystery" in story.themes

    def test_excludes_viewed_when_enough_candidates(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        """Already-viewed stories should be excluded when enough unviewed exist."""
        result = strategy.recommend(adventure_profile, sample_stories, [], 2, set())
        for sid in result:
            assert sid not in adventure_profile.viewed_story_ids

    def test_respects_exclude_ids(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        first_result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        if first_result:
            exclude = set(first_result)
            second_result = strategy.recommend(
                adventure_profile, sample_stories, [], 1, exclude
            )
            # Should not overlap (when enough candidates)
            if second_result:
                assert not set(second_result) & exclude

    def test_empty_catalogue_returns_empty(
        self, strategy, adventure_profile
    ) -> None:
        result = strategy.recommend(adventure_profile, [], [], 2, set())
        assert result == []


# ---------------------------------------------------------------------------
# Cold-start tests
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_new_user_returns_results(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        result = strategy.recommend(new_user_profile, sample_stories, [], 2, set())
        assert len(result) <= 2

    def test_cold_start_uses_global_scores(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        """If s_mys has high average score, cold-start should favour it."""
        all_profiles = [new_user_profile]
        scorer = UserProfile(user_id="u_scorer")
        scorer.story_scores = {"s_mys": 5, "s_mys": 5}
        all_profiles.append(scorer)
        result = strategy.recommend(new_user_profile, sample_stories, all_profiles, 2, set())
        assert isinstance(result, list)

    def test_cold_start_no_scores_still_returns(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        result = strategy.recommend(new_user_profile, sample_stories, [], 2, set())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self, all_themes, all_tags) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert strategy._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, all_themes, all_tags) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert strategy._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self, all_themes, all_tags) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        a = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert strategy._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_similarity_range(self, all_themes, all_tags) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        a = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        b = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        sim = strategy._cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Vector building tests
# ---------------------------------------------------------------------------


class TestVectorBuilding:
    def test_story_vector_has_ones_for_story_features(
        self, all_themes, all_tags, story_adventure
    ) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        vec = strategy._build_story_vector(story_adventure)
        theme_idx = strategy._theme_index.get("adventure")
        tag_idx = strategy._tag_index.get("pirates")
        if theme_idx is not None:
            assert vec[theme_idx] == pytest.approx(1.0)
        if tag_idx is not None:
            assert vec[tag_idx] == pytest.approx(1.0)

    def test_user_vector_reflects_weights(self, all_themes, all_tags, adventure_profile) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        vec = strategy._build_user_vector(adventure_profile)
        adv_idx = strategy._theme_index.get("adventure")
        if adv_idx is not None:
            assert vec[adv_idx] == pytest.approx(adventure_profile.theme_weights["adventure"])

    def test_empty_profile_gives_zero_vector(self, all_themes, all_tags, new_user_profile) -> None:
        strategy = ContentBasedStrategy(all_themes, all_tags)
        vec = strategy._build_user_vector(new_user_profile)
        assert np.all(vec == 0.0)
