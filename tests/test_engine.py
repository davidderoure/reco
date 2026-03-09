"""Tests for RecommendationEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from recommender.catalogue import StoryCatalogue
from recommender.engine import (
    RecommendationEngine,
    _mood_slot_allocation,
    _recent_mood_level,
    _SLOT_ALLOCATION,
    _MOOD_LOW_THRESHOLD,
    _MOOD_HIGH_THRESHOLD,
    _SKIP_DEPRIORITISE_THRESHOLD,
)
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
    def test_try_again_returns_fresh_stories(self, sample_stories) -> None:
        """Consecutive calls without interactions must return different stories.

        Progressive coverage adds the first set to recommended_story_ids;
        the next call's first pass excludes those, surfacing novel stories.
        """
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        first = eng.get_recommendations("u1")
        second = eng.get_recommendations("u1")

        # 10-story catalogue, 6 slots → 4 novel stories remain after first call.
        # Second call picks those 4 + 2 from relaxed pass; the sets must differ.
        assert set(first) != set(second)

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

    def test_slot_stability_preserves_positions_for_overlapping_stories(
        self,
    ) -> None:
        """A story reappearing in a fresh call must keep its original slot index.

        Uses a 7-story catalogue so the second call is forced to reuse some
        stories from the first (progressive coverage finds only 1 novel story,
        then relaxes to fill the remaining 5 slots from the previous set).
        """
        stories = [
            Story(f"s{i}", f"Title {i}", ["adventure"], [f"tag{i}"])
            for i in range(7)
        ]
        profile = UserProfile(user_id="u1")
        eng = _make_engine(stories, [profile])

        first = eng.get_recommendations("u1")  # uses 6 of 7 stories
        first_pos = {sid: pos for pos, sid in enumerate(first)}

        # No interactions — second call has only 1 novel story; 5 repeat from first.
        second = eng.get_recommendations("u1")

        for pos, sid in enumerate(second):
            if sid in first_pos:
                assert pos == first_pos[sid], (
                    f"Story {sid} was at index {first_pos[sid]} in the first call "
                    f"but appeared at index {pos} in the second call"
                )

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

    def test_cycling_continues_after_catalogue_exhausted(self, sample_stories) -> None:
        """After all stories are recommended, consecutive calls still rotate stories.

        Once recommended_story_ids covers the whole catalogue, Pass 1 yields nothing.
        The new intermediate pass excludes the last batch so the user sees different
        stories on each call rather than the same frozen set.
        """
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])  # 10-story catalogue

        # Exhaust the catalogue (ceil(10/6) = 2 calls gets all 10, but use 4 to be safe)
        catalogue_ids = {s.story_id for s in sample_stories}
        for _ in range(4):
            eng.get_recommendations("u1")

        assert profile.recommended_story_ids == catalogue_ids, (
            "Catalogue should be fully exhausted before testing post-exhaustion cycling"
        )

        # Two consecutive calls with no interactions must surface different stories
        call_a = eng.get_recommendations("u1")
        call_b = eng.get_recommendations("u1")

        assert set(call_a) != set(call_b), (
            "Post-exhaustion calls should cycle through different stories, not repeat the same set"
        )

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

    def test_skip_count_increments_when_not_viewed(self, sample_stories) -> None:
        """Stories returned but not yet viewed each have their skip count incremented."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        result = eng.get_recommendations("u1")

        for sid in result:
            assert profile.skip_counts.get(sid, 0) == 1, (
                f"Expected skip_counts[{sid!r}] == 1, got {profile.skip_counts.get(sid, 0)}"
            )

    def test_skip_count_resets_on_view(self, sample_stories) -> None:
        """Viewing a story resets its skip count to zero (story returns to Tier 1a)."""
        profile = UserProfile(user_id="u1")
        eng = _make_engine(sample_stories, [profile])

        result = eng.get_recommendations("u1")
        viewed_sid = result[0]
        assert profile.skip_counts.get(viewed_sid, 0) == 1  # count was incremented

        # Record view through the store — this should clear the skip count
        from datetime import datetime, timezone as tz
        eng._user_state_store.record_viewed("u1", viewed_sid, datetime.now(tz.utc))

        assert viewed_sid not in profile.skip_counts, (
            f"Expected skip_counts to not contain {viewed_sid!r} after viewing"
        )

    def test_selecting_away_deprioritises_story(self, sample_stories) -> None:
        """A story recommended > K times without a view is placed in Tier 2 and skipped
        when Tier 1a has enough alternatives to fill all 6 slots.
        """
        profile = UserProfile(user_id="u1")
        # Mark one story as frequently skipped (exceeds the deprioritise threshold → Tier 2)
        deprioritised_id = sample_stories[0].story_id
        profile.skip_counts[deprioritised_id] = _SKIP_DEPRIORITISE_THRESHOLD + 1

        eng = _make_engine(sample_stories, [profile])

        # 10-story catalogue: 1 story in Tier 2, 9 stories in Tier 1a (skip_count == 0).
        # All 6 slots can be filled from Tier 1a — the Tier-2 story must not appear.
        result = eng.get_recommendations("u1")

        assert deprioritised_id not in result, (
            f"Expected deprioritised story {deprioritised_id!r} to be absent "
            f"when Tier 1a has enough alternatives"
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


# ---------------------------------------------------------------------------
# Mood-responsive slot allocation tests
# ---------------------------------------------------------------------------

from datetime import datetime, timezone  # noqa: E402 (after fixtures for readability)

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestMoodSlotAllocation:
    """_mood_slot_allocation() returns the right allocation for each mood band."""

    def test_no_mood_returns_default(self) -> None:
        assert _mood_slot_allocation(None) is _SLOT_ALLOCATION

    def test_neutral_mood_returns_default(self) -> None:
        neutral = (_MOOD_LOW_THRESHOLD + _MOOD_HIGH_THRESHOLD) / 2
        assert _mood_slot_allocation(neutral) is _SLOT_ALLOCATION

    def test_low_mood_has_no_wildcard(self) -> None:
        allocation = dict(_mood_slot_allocation(_MOOD_LOW_THRESHOLD))
        assert allocation["_wildcard_strategy"] == 0
        assert allocation["_content_strategy"] == 3

    def test_low_mood_boundary(self) -> None:
        """Exactly at the low threshold should trigger the comfort allocation."""
        allocation = dict(_mood_slot_allocation(_MOOD_LOW_THRESHOLD))
        assert allocation["_wildcard_strategy"] == 0

    def test_high_mood_has_two_wildcard_slots(self) -> None:
        allocation = dict(_mood_slot_allocation(_MOOD_HIGH_THRESHOLD))
        assert allocation["_wildcard_strategy"] == 2
        assert allocation["_content_strategy"] == 1

    def test_high_mood_boundary(self) -> None:
        """Exactly at the high threshold should trigger the exploratory allocation."""
        allocation = dict(_mood_slot_allocation(_MOOD_HIGH_THRESHOLD))
        assert allocation["_wildcard_strategy"] == 2

    def test_all_allocations_sum_to_six(self) -> None:
        for mood_level in (None, 1.0, _MOOD_LOW_THRESHOLD, 3.0, _MOOD_HIGH_THRESHOLD, 5.0):
            alloc = _mood_slot_allocation(mood_level)
            assert sum(n for _, n in alloc) == 6, f"Allocation for mood={mood_level} does not sum to 6"


class TestRecentMoodLevel:
    """_recent_mood_level() returns the correct average of recent mood entries."""

    def test_no_moods_returns_none(self) -> None:
        profile = UserProfile(user_id="u1")
        assert _recent_mood_level(profile) is None

    def test_single_mood(self) -> None:
        profile = UserProfile(user_id="u1", mood_scores=[(_TS, 4)])
        assert _recent_mood_level(profile) == pytest.approx(4.0)

    def test_averages_up_to_five_recent(self) -> None:
        # 6 entries; only the 5 most recent should be averaged
        moods = [(datetime(2024, 1, i + 1, tzinfo=timezone.utc), i + 1) for i in range(6)]
        # Sorted ascending: scores 1,2,3,4,5,6. Five most recent = 2,3,4,5,6 → avg 4.0
        profile = UserProfile(user_id="u1", mood_scores=moods)
        assert _recent_mood_level(profile) == pytest.approx(4.0)


class TestMoodResponsiveRecommendations:
    """get_recommendations() uses mood-aware slot allocation end-to-end."""

    def _sample_stories(self) -> list[Story]:
        return [
            Story(f"s{i}", f"Story {i}", ["adventure"], [f"tag{i}"])
            for i in range(10)
        ]

    def test_low_mood_produces_recommendations(self) -> None:
        """Low-mood user still gets 6 recommendations (wildcard slot = 0 is fine)."""
        stories = self._sample_stories()
        profile = UserProfile(user_id="u1", mood_scores=[(_TS, 1)])
        eng = _make_engine(stories, [profile])
        result = eng.get_recommendations("u1")
        assert len(result) == 6

    def test_high_mood_produces_recommendations(self) -> None:
        """High-mood user gets 6 recommendations with broader exploration."""
        stories = self._sample_stories()
        profile = UserProfile(user_id="u1", mood_scores=[(_TS, 5)])
        eng = _make_engine(stories, [profile])
        result = eng.get_recommendations("u1")
        assert len(result) == 6
