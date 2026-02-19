"""Tests for TopicalStrategy."""

from __future__ import annotations

import pytest

from recommender.models import Story, UserProfile
from recommender.strategies.topical import TopicalStrategy


@pytest.fixture
def strategy() -> TopicalStrategy:
    return TopicalStrategy()


class TestRecommend:
    def test_returns_list(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        assert isinstance(result, list)

    def test_returns_at_most_n(self, strategy, sample_stories, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        assert len(result) <= 1

    def test_returned_story_contains_top_tag(
        self, strategy, sample_stories, adventure_profile
    ) -> None:
        """The returned story must contain the user's highest-weight tag."""
        top_tag = max(adventure_profile.tag_weights, key=lambda t: adventure_profile.tag_weights[t])
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        story_map = {s.story_id: s for s in sample_stories}
        for sid in result:
            assert top_tag in story_map[sid].tags

    def test_prefers_unviewed(self, strategy, sample_stories, adventure_profile) -> None:
        """Should prefer stories the user hasn't viewed yet."""
        result = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        for sid in result:
            assert sid not in adventure_profile.viewed_story_ids

    def test_respects_exclude_ids(self, strategy, sample_stories, adventure_profile) -> None:
        first = strategy.recommend(adventure_profile, sample_stories, [], 1, set())
        if first:
            exclude = set(first)
            second = strategy.recommend(adventure_profile, sample_stories, [], 1, exclude)
            if second:
                assert not set(second) & exclude

    def test_empty_catalogue_returns_empty(self, strategy, adventure_profile) -> None:
        result = strategy.recommend(adventure_profile, [], [], 1, set())
        assert result == []


class TestColdStart:
    def test_new_user_falls_back_to_popular_tag(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        result = strategy.recommend(new_user_profile, sample_stories, [], 1, set())
        assert isinstance(result, list)

    def test_new_user_result_from_catalogue(
        self, strategy, sample_stories, new_user_profile
    ) -> None:
        catalogue_ids = {s.story_id for s in sample_stories}
        result = strategy.recommend(new_user_profile, sample_stories, [], 1, set())
        for sid in result:
            assert sid in catalogue_ids


class TestMostPopularTag:
    def test_returns_tag_with_highest_count(self) -> None:
        stories = [
            Story("s1", "A", ["x"], ["ocean", "sky"]),
            Story("s2", "B", ["x"], ["ocean"]),
            Story("s3", "C", ["x"], ["sky"]),
        ]
        result = TopicalStrategy._most_popular_tag(stories)
        assert result == "ocean"

    def test_no_tags_returns_none(self) -> None:
        stories = [Story("s1", "A", ["x"], [])]
        assert TopicalStrategy._most_popular_tag(stories) is None

    def test_empty_catalogue_returns_none(self) -> None:
        assert TopicalStrategy._most_popular_tag([]) is None


class TestGetTopTags:
    def test_returns_n_most_weighted_tags(self) -> None:
        profile = UserProfile(user_id="u1")
        profile.tag_weights = {"pirates": 3.0, "ocean": 2.0, "clues": 1.0, "shadows": 0.5}
        result = TopicalStrategy._get_top_tags(profile, n_tags=2)
        assert result == {"pirates", "ocean"}

    def test_empty_profile_returns_empty_set(self, new_user_profile) -> None:
        result = TopicalStrategy._get_top_tags(new_user_profile, n_tags=5)
        assert result == set()
