"""Tests for RecommendationEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from recommender.catalogue import StoryCatalogue
from recommender.engine import RecommendationEngine
from recommender.models import Story, UserProfile
from recommender.strategies.content_based import ContentBasedStrategy
from recommender.strategies.collaborative import CollaborativeFilteringStrategy
from recommender.strategies.topical import TopicalStrategy
from recommender.strategies.wildcard import WildcardStrategy
from recommender.user_state import UserStateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_catalogue_from_stories(stories: list[Story]) -> StoryCatalogue:
    stub = MagicMock()
    cat = StoryCatalogue(stub=stub)
    with cat._lock:
        cat._stories = {s.story_id: s for s in stories}
    return cat


def _make_store(catalogue: StoryCatalogue, *profiles: UserProfile) -> UserStateStore:
    stub = MagicMock()
    stub.LoadUserState.return_value.user_states = []
    store = UserStateStore(stub=stub, catalogue=catalogue)
    for p in profiles:
        with store._lock:
            store._profiles[p.user_id] = p
    return store


def _make_engine(
    stories: list[Story],
    profiles: list[UserProfile],
) -> RecommendationEngine:
    catalogue = _make_catalogue_from_stories(stories)
    store = _make_store(catalogue, *profiles)

    all_themes = catalogue.get_all_themes()
    all_tags = catalogue.get_all_tags()

    return RecommendationEngine(
        catalogue=catalogue,
        user_state_store=store,
        content_strategy=ContentBasedStrategy(all_themes, all_tags),
        collaborative_strategy=CollaborativeFilteringStrategy(all_themes, all_tags),
        topical_strategy=TopicalStrategy(),
        wildcard_strategy=WildcardStrategy(),
    )


@pytest.fixture
def engine(sample_stories, adventure_profile, mystery_profile) -> RecommendationEngine:
    return _make_engine(sample_stories, [adventure_profile, mystery_profile])


@pytest.fixture
def engine_new_user(sample_stories) -> tuple[RecommendationEngine, str]:
    new_profile = UserProfile(user_id="u_brand_new")
    eng = _make_engine(sample_stories, [new_profile])
    return eng, "u_brand_new"


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    def test_returns_exactly_six(self, engine, adventure_profile) -> None:
        result = engine.get_recommendations(adventure_profile.user_id)
        assert len(result) == 6

    def test_no_duplicates(self, engine, adventure_profile) -> None:
        result = engine.get_recommendations(adventure_profile.user_id)
        assert len(result) == len(set(result))

    def test_all_ids_in_catalogue(self, engine, adventure_profile, sample_stories) -> None:
        catalogue_ids = {s.story_id for s in sample_stories}
        result = engine.get_recommendations(adventure_profile.user_id)
        for sid in result:
            assert sid in catalogue_ids

    def test_raises_on_empty_user_id(self, engine) -> None:
        with pytest.raises(ValueError):
            engine.get_recommendations("")

    def test_creates_profile_for_unknown_user(self, engine) -> None:
        """Engine should not raise for a user it has never seen."""
        result = engine.get_recommendations("u_completely_new")
        assert len(result) == 6


class TestColdStart:
    def test_new_user_returns_six(self, engine_new_user) -> None:
        engine, user_id = engine_new_user
        result = engine.get_recommendations(user_id)
        assert len(result) == 6

    def test_new_user_no_duplicates(self, engine_new_user) -> None:
        engine, user_id = engine_new_user
        result = engine.get_recommendations(user_id)
        assert len(result) == len(set(result))


class TestShuffling:
    def test_results_are_shuffled(self, engine, adventure_profile) -> None:
        """Run 20 times; the ordering should vary (probabilistic test)."""
        results = [
            tuple(engine.get_recommendations(adventure_profile.user_id))
            for _ in range(20)
        ]
        unique_orderings = set(results)
        # With 6 items the probability of all 20 runs being identical is negligible
        assert len(unique_orderings) > 1


class TestSmallCatalogue:
    def test_fewer_than_six_stories_does_not_crash(self) -> None:
        """With only 3 stories, engine should return 6 (repeating if needed)."""
        stories = [
            Story("s1", "A", ["adventure"], ["x"]),
            Story("s2", "B", ["mystery"], ["y"]),
            Story("s3", "C", ["calm"], ["z"]),
        ]
        profile = UserProfile(user_id="u1")
        eng = _make_engine(stories, [profile])
        result = eng.get_recommendations("u1")
        # Engine fills to 6 even if that requires repeating story IDs
        assert len(result) == 6
        # All returned IDs must be from the catalogue
        catalogue_ids = {s.story_id for s in stories}
        for sid in result:
            assert sid in catalogue_ids

    def test_single_story_catalogue(self) -> None:
        stories = [Story("s1", "A", ["adventure"], ["x"])]
        profile = UserProfile(user_id="u1")
        eng = _make_engine(stories, [profile])
        result = eng.get_recommendations("u1")
        # Engine will repeat s1 to fill 6 slots as absolute last resort
        assert len(result) <= 6


class TestFillRemaining:
    def test_slots_filled_when_strategies_return_less(self) -> None:
        """Use a catalogue barely large enough to satisfy 6 slots."""
        stories = [Story(f"s{i}", f"Story {i}", ["adventure"], [f"tag{i}"]) for i in range(6)]
        profile = UserProfile(user_id="u1")
        eng = _make_engine(stories, [profile])
        result = eng.get_recommendations("u1")
        assert len(result) == 6
        assert len(set(result)) == 6
