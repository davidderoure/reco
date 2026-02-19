"""Tests for WildcardStrategy."""

from __future__ import annotations

import pytest

from recommender.models import Story, UserProfile
from recommender.strategies.wildcard import WildcardStrategy


@pytest.fixture
def strategy() -> WildcardStrategy:
    return WildcardStrategy()


class TestRecommend:
    def test_returns_list(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        assert isinstance(result, list)

    def test_returns_at_most_n(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        assert len(result) <= 1

    def test_returns_valid_story_ids(self, strategy, sample_stories, adventure_profile) -> None:
        catalogue_ids = {s.story_id for s in sample_stories}
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        for sid in result:
            assert sid in catalogue_ids

    def test_empty_catalogue_returns_empty(self, strategy, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, [], [], 1, set())
        assert result == []

    def test_n_zero_returns_empty(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 0, set())
        assert result == []


class TestNoveltyPreference:
    def test_prefers_unexplored_themes(self, strategy, sample_stories, adventure_profile) -> None:
        """The wildcard should prefer stories in themes the user hasn't explored."""
        explored_themes = set(adventure_profile.theme_weights.keys())  # {"adventure", "mystery"}
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        if result:
            story_map = {s.story_id: s for s in sample_stories}
            story = story_map[result[0]]
            # Should prefer horror or calm (unexplored) over adventure/mystery
            has_unexplored = any(t not in explored_themes for t in story.themes)
            # Not a hard guarantee due to randomness, but unexplored should be available
            unexplored_available = any(
                all(t not in explored_themes for t in s.themes)
                and s.story_id not in adventure_profile.viewed_story_ids
                for s in sample_stories
            )
            if unexplored_available:
                assert has_unexplored

    def test_new_user_any_story_returned(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        """For a new user with no viewed stories, any story can be returned."""
        result = strategy.recommend(new_user_profile, sample_stories, [], 1, set())
        assert len(result) == 1

    def test_all_viewed_still_returns(self, strategy) -> None:
        """Even when the user has viewed everything, we return something."""
        stories = [
            Story("s1", "A", ["adventure"], ["x"]),
            Story("s2", "B", ["mystery"], ["y"]),
        ]
        profile = UserProfile(user_id="u1")
        profile.viewed_story_ids = {"s1", "s2"}
        profile.theme_weights = {"adventure": 1.0, "mystery": 1.0}
        result = strategy.recommend(profile, stories, [], 1, set())
        assert len(result) == 1


class TestExcludeIds:
    def test_does_not_return_excluded_when_alternatives_exist(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        first = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        if first:
            exclude = set(first)
            second = strategy.recommend(adventure_profile, sample_stories, [], 1, exclude)
            if second and len(sample_stories) > 1:
                assert not set(second) & exclude

    def test_returns_excluded_as_last_resort(self, strategy) -> None:
        """When only one story exists and it's excluded, return it anyway."""
        stories = [Story("s1", "A", ["adventure"], ["x"])]
        profile = UserProfile(user_id="u1")
        result = strategy.recommend(profile, stories, [], 1, exclude_ids={"s1"})
        assert result == ["s1"]


class TestRandomness:
    def test_results_are_not_always_identical(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        """Run many times; results should vary (probabilistic test)."""
        results = [
            tuple(strategy.recommend(new_user_profile, sample_stories, [], 1, set()))
            for _ in range(50)
        ]
        unique = set(results)
        # With 10 stories there should be variation across 50 runs
        assert len(unique) > 1
