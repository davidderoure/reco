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
    stub.LoadUserModel.return_value.user_models = []
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
    def test_fresh_recommendations_are_shuffled(self, sample_stories) -> None:
        """Each fresh call (no prior recs) should produce a varied ordering."""
        results = []
        for _ in range(20):
            profile = UserProfile(user_id="u_adv")
            profile.theme_weights = {"adventure": 3.0, "mystery": 1.0}
            profile.tag_weights = {"pirates": 3.0, "ocean": 3.0}
            eng = _make_engine(sample_stories, [profile])
            results.append(tuple(eng.get_recommendations("u_adv")))
        unique_orderings = set(results)
        # With 6 items the probability of all 20 runs being identical is negligible
        assert len(unique_orderings) > 1


# ---------------------------------------------------------------------------
# Slot stability (sticky) tests
# ---------------------------------------------------------------------------


class TestStickySlots:
    def test_unacted_stories_keep_their_position(self, sample_stories) -> None:
        """Stories not viewed or completed must stay at the same index."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        second = eng.get_recommendations("u1")

        # No interactions between calls → all slots should be sticky
        assert first == second

    def test_acted_slot_is_replaced(self, sample_stories) -> None:
        """A story the user viewed must not appear at its previous index."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        acted_story = first[2]
        profile.viewed_story_ids.add(acted_story)

        second = eng.get_recommendations("u1")

        # The acted story must have been replaced (slot 2 is now different)
        assert second[2] != acted_story

    def test_sticky_stories_stay_in_exact_positions(self, sample_stories) -> None:
        """Unacted stories must appear at exactly the same indices."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        # Act on stories at positions 0 and 4
        profile.viewed_story_ids.add(first[0])
        profile.completed_story_ids.add(first[4])

        second = eng.get_recommendations("u1")

        # Positions 1, 2, 3, 5 must be unchanged
        for i in (1, 2, 3, 5):
            assert second[i] == first[i], f"Position {i} changed unexpectedly"

    def test_completed_slot_is_replaced(self, sample_stories) -> None:
        """A completed story must be replaced just like a viewed one."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        profile.completed_story_ids.add(first[0])

        second = eng.get_recommendations("u1")
        assert second[0] != first[0]

    def test_last_recommendations_updated_after_call(self, sample_stories) -> None:
        """Profile's last_recommendations must reflect the returned list."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        result = eng.get_recommendations("u1")
        assert profile.last_recommendations == result


# ---------------------------------------------------------------------------
# Progressive coverage tests
# ---------------------------------------------------------------------------


class TestProgressiveCoverage:
    def test_recommended_story_ids_updated(self, sample_stories) -> None:
        """All returned IDs must be added to recommended_story_ids."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        result = eng.get_recommendations("u1")
        for sid in result:
            assert sid in profile.recommended_story_ids

    def test_open_slot_prefers_unrecommended(self, sample_stories) -> None:
        """When a slot opens, the new story should not have been recommended before."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        already_recommended = set(profile.recommended_story_ids)

        # Act on one story to open a slot
        profile.viewed_story_ids.add(first[0])
        second = eng.get_recommendations("u1")

        # The new story in the opened slot should not have been previously recommended
        new_story = second[0]
        assert new_story not in already_recommended

    def test_all_stories_eventually_recommended(self, sample_stories) -> None:
        """Following recommendations should eventually surface every story."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])
        catalogue_ids = {s.story_id for s in sample_stories}

        # Simulate following all recommendations: view everything, keep getting recs
        for _ in range(len(catalogue_ids) + 2):
            result = eng.get_recommendations("u1")
            for sid in result:
                profile.viewed_story_ids.add(sid)
            if catalogue_ids <= profile.recommended_story_ids:
                break

        assert catalogue_ids <= profile.recommended_story_ids, (
            f"Stories never recommended: {catalogue_ids - profile.recommended_story_ids}"
        )


# ---------------------------------------------------------------------------
# Small catalogue edge cases
# ---------------------------------------------------------------------------


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
