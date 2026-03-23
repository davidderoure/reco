"""Tests for recommender.user_state.UserStateStore.

These are the most important tests in the suite — they verify the weight
accumulation maths that drive all recommendation strategies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from recommender.catalogue import StoryCatalogue
from recommender.models import Story, UserProfile
from recommender.user_state import (
    MOOD_ATTRIBUTION_FACTOR,
    MOOD_HISTORY_LIMIT,
    READ_VIEWED_THRESHOLD_PERCENT,
    READ_COMPLETED_THRESHOLD_PERCENT,
    UserStateStore,
    _datetime_to_timestamp,
    _timestamp_to_datetime,
    _WEIGHT_VIEW,
    _WEIGHT_COMPLETE_BONUS,
    _WEIGHT_SCORE_FACTOR,
    _SCORE_NEUTRAL,
    _MOOD_MAX_DELTA,
    _QUESTION_WEIGHT_FACTORS,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_catalogue(*stories: Story) -> StoryCatalogue:
    """Return a StoryCatalogue pre-loaded with the given stories (no RPC)."""
    stub = MagicMock()
    cat = StoryCatalogue(stub=stub)
    with cat._lock:
        cat._stories = {s.story_id: s for s in stories}
    return cat


def _make_store(*stories: Story) -> UserStateStore:
    stub = MagicMock()
    # LoadUserModel returns an empty response (no prior users)
    stub.LoadUserModel.return_value.user_models = []
    cat = _make_catalogue(*stories)
    return UserStateStore(stub=stub, catalogue=cat)


STORY_ADV = Story("s1", "Sea Quest", ["adventure"], ["pirates", "ocean"])
STORY_MYS = Story("s2", "Hidden Room", ["mystery"], ["clues", "shadows"])
STORY_MULTI = Story("s3", "Dual Tale", ["adventure", "mystery"], ["treasure"])


# ---------------------------------------------------------------------------
# Scored event tests
# ---------------------------------------------------------------------------


class TestRecordScored:
    @pytest.mark.parametrize("score,expected_delta", [
        (1,  (1  - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR),   # -1.0
        (5,  (5  - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR),   #  0.0 (neutral)
        (9,  (9  - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR),   # +1.0
        (10, (10 - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR),   # +1.25
    ])
    def test_score_weight_delta(self, score: int, expected_delta: float) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", score, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights.get("adventure", 0.0) == pytest.approx(expected_delta)

    def test_stores_score_in_story_scores(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 8, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 8

    def test_latest_score_overwrites(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 8, TS)
        store.record_scored("u1", "s1", 3, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 3

    def test_invalid_score_raises(self) -> None:
        store = _make_store(STORY_ADV)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 0, TS)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 11, TS)

    def test_read_progress_and_score_accumulates(self) -> None:
        """Read ≥50% (+1.0) then score 9 (+1.0) = total +2.0."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        store.record_scored("u1", "s1", 9, TS)
        profile = store.get_or_create_profile("u1")
        expected = _WEIGHT_VIEW + (9 - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR
        assert profile.theme_weights["adventure"] == pytest.approx(expected)

    def test_optional_question_applies_no_weight_delta(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 9, TS, question_number=2)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights.get("adventure", 0.0) == pytest.approx(0.0)

    def test_optional_question_not_stored_in_story_scores(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 8, TS, question_number=3)
        profile = store.get_or_create_profile("u1")
        assert "s1" not in profile.story_scores

    def test_question_1_explicit_identical_to_default(self) -> None:
        store_default = _make_store(STORY_ADV)
        store_explicit = _make_store(STORY_ADV)
        store_default.record_scored("u1", "s1", 7, TS)
        store_explicit.record_scored("u1", "s1", 7, TS, question_number=1)
        p_default = store_default.get_or_create_profile("u1")
        p_explicit = store_explicit.get_or_create_profile("u1")
        assert p_default.theme_weights == p_explicit.theme_weights
        assert p_default.story_scores == p_explicit.story_scores

    def test_invalid_score_raises_for_optional_question(self) -> None:
        store = _make_store(STORY_ADV)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 0, TS, question_number=2)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 11, TS, question_number=4)


# ---------------------------------------------------------------------------
# Mood event tests
# ---------------------------------------------------------------------------


class TestRecordMood:
    def test_stores_mood_score(self) -> None:
        store = _make_store()
        store.record_mood("u1", 4, TS)
        profile = store.get_or_create_profile("u1")
        assert len(profile.mood_scores) == 1
        assert profile.mood_scores[0] == (TS, 4)

    def test_multiple_mood_scores_accumulated(self) -> None:
        store = _make_store()
        ts2 = datetime(2024, 6, 2, tzinfo=timezone.utc)
        store.record_mood("u1", 3, TS)
        store.record_mood("u1", 8, ts2)
        profile = store.get_or_create_profile("u1")
        assert len(profile.mood_scores) == 2

    def test_mood_does_not_affect_theme_weights(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_mood("u1", 5, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights == {}

    def test_invalid_mood_raises(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError):
            store.record_mood("u1", 0, TS)
        with pytest.raises(ValueError):
            store.record_mood("u1", 11, TS)

    def test_mood_history_capped_at_limit(self) -> None:
        store = _make_store()
        for i in range(MOOD_HISTORY_LIMIT + 10):
            store.record_mood("u1", (i % 10) + 1, TS)
        profile = store.get_or_create_profile("u1")
        assert len(profile.mood_scores) == MOOD_HISTORY_LIMIT

    def test_story_id_resets_skip_count(self) -> None:
        """When story_id is provided, the story's skip count is reset."""
        store = _make_store(STORY_ADV)
        profile = store.get_or_create_profile("u1")
        profile.skip_counts["s1"] = 3  # simulate previous skips

        store.record_mood("u1", 7, TS, story_id="s1")

        assert "s1" not in profile.skip_counts


# ---------------------------------------------------------------------------
# Read progress tests
# ---------------------------------------------------------------------------


class TestRecordReadProgress:
    def test_below_threshold_does_not_mark_viewed(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", READ_VIEWED_THRESHOLD_PERCENT - 1, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" not in profile.viewed_story_ids
        assert profile.theme_weights == {}

    def test_at_threshold_marks_viewed(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", READ_VIEWED_THRESHOLD_PERCENT, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_above_threshold_but_below_100_marks_viewed_only(self) -> None:
        """Read at 75% marks as viewed but not completed."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 75, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids
        assert "s1" not in profile.completed_story_ids
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_idempotent_above_threshold(self) -> None:
        """Second call above threshold does not double-apply the view weight."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 75, TS)
        store.record_read_progress("u1", "s1", 90, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_viewed_not_double_counted_on_subsequent_above_threshold(self) -> None:
        """Second above-threshold event does not re-apply view weight."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        weight_after_first = store.get_or_create_profile("u1").theme_weights["adventure"]
        store.record_read_progress("u1", "s1", 75, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(weight_after_first)

    def test_invalid_read_percent_raises(self) -> None:
        store = _make_store()
        with pytest.raises(ValueError):
            store.record_read_progress("u1", "s1", -1, TS)
        with pytest.raises(ValueError):
            store.record_read_progress("u1", "s1", 101, TS)

    def test_unknown_story_above_threshold_does_not_crash(self) -> None:
        """Unknown story is marked viewed but no weight delta can be applied."""
        store = _make_store()
        store.record_read_progress("u1", "unknown_story", 75, TS)
        profile = store.get_or_create_profile("u1")
        assert "unknown_story" in profile.viewed_story_ids
        assert profile.theme_weights == {}

    def test_tags_receive_weight_at_threshold(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", READ_VIEWED_THRESHOLD_PERCENT, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.tag_weights["pirates"] == pytest.approx(_WEIGHT_VIEW)
        assert profile.tag_weights["ocean"] == pytest.approx(_WEIGHT_VIEW)

    # --- Completion inference (read_percent == 100) ---

    def test_read_100_marks_completed(self) -> None:
        """read_percent=100 adds the story to completed_story_ids."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.completed_story_ids

    def test_read_100_also_marks_viewed(self) -> None:
        """read_percent=100 marks the story as viewed as well as completed."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids

    def test_read_100_applies_view_and_complete_bonus(self) -> None:
        """read_percent=100 applies both the view weight and the completion bonus."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        expected = _WEIGHT_VIEW + _WEIGHT_COMPLETE_BONUS
        assert profile.theme_weights["adventure"] == pytest.approx(expected)

    def test_read_100_idempotent(self) -> None:
        """A second read_percent=100 does not re-apply the completion bonus."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 100, TS)
        weight_after_first = store.get_or_create_profile("u1").theme_weights["adventure"]
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(weight_after_first)

    def test_read_100_only_applies_complete_bonus_if_already_viewed(self) -> None:
        """If the story was already viewed at 50%, read_percent=100 adds only the completion bonus."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)  # view weight applied here
        weight_after_view = store.get_or_create_profile("u1").theme_weights["adventure"]

        store.record_read_progress("u1", "s1", 100, TS)  # only complete bonus added
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(
            weight_after_view + _WEIGHT_COMPLETE_BONUS
        )

    def test_read_100_accumulates_mood_window(self) -> None:
        """read_percent=100 accumulates the complete bonus in the mood attribution window."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)   # view (resets window after mood)
        store.record_mood("u1", 3, TS)                   # start fresh mood window
        store.record_read_progress("u1", "s1", 100, TS)  # complete (view idempotent)

        profile = store.get_or_create_profile("u1")
        assert profile.themes_since_last_mood.get("adventure", 0.0) == pytest.approx(
            _WEIGHT_COMPLETE_BONUS
        )


# ---------------------------------------------------------------------------
# Profile management tests
# ---------------------------------------------------------------------------


class TestGetOrCreateProfile:
    def test_creates_new_profile(self) -> None:
        store = _make_store()
        profile = store.get_or_create_profile("u_new")
        assert profile.user_id == "u_new"

    def test_returns_same_profile_on_second_call(self) -> None:
        store = _make_store()
        p1 = store.get_or_create_profile("u1")
        p2 = store.get_or_create_profile("u1")
        assert p1 is p2

    def test_multiple_users_are_independent(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        p1 = store.get_or_create_profile("u1")
        p2 = store.get_or_create_profile("u2")
        assert "s1" in p1.viewed_story_ids
        assert "s1" not in p2.viewed_story_ids


class TestGetAllProfiles:
    def test_returns_all_profiles(self) -> None:
        store = _make_store()
        store.get_or_create_profile("u1")
        store.get_or_create_profile("u2")
        profiles = store.get_all_profiles()
        ids = {p.user_id for p in profiles}
        assert ids == {"u1", "u2"}

    def test_returns_copy_of_list(self) -> None:
        store = _make_store()
        store.get_or_create_profile("u1")
        profiles = store.get_all_profiles()
        profiles.clear()
        assert len(store.get_all_profiles()) == 1


# ---------------------------------------------------------------------------
# Persistence round-trip tests
# ---------------------------------------------------------------------------


class TestPersistAndLoad:
    def test_persist_calls_save_user_model(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        store.persist_all_to_server()
        store._stub.SaveUserModel.assert_called_once()

    def test_persist_serialises_model_fields(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        store.record_scored("u1", "s1", 8, TS)
        store.persist_all_to_server()
        call_args = store._stub.SaveUserModel.call_args[0][0]
        user_models = list(call_args.user_models)
        assert len(user_models) == 1
        model = user_models[0]
        assert model.user_id == "u1"
        assert "s1" in list(model.viewed_story_ids)
        assert model.story_scores["s1"] == 8
        assert model.theme_weights["adventure"] == pytest.approx(
            _WEIGHT_VIEW + (8 - _SCORE_NEUTRAL) * _WEIGHT_SCORE_FACTOR
        )

    def test_persist_does_not_raise_on_stub_error(self) -> None:
        store = _make_store()
        store._stub.SaveUserModel.side_effect = RuntimeError("network error")
        store.persist_all_to_server()  # should not raise


def _make_model_msg(
    user_id: str,
    viewed: list[str] = (),
    completed: list[str] = (),
    story_scores: dict = None,
    theme_weights: dict = None,
    tag_weights: dict = None,
    mood_scores: list = (),
) -> MagicMock:
    """Build a mock UserModelMessage for use in load tests."""
    msg = MagicMock()
    msg.user_id = user_id
    msg.viewed_story_ids = list(viewed)
    msg.completed_story_ids = list(completed)
    msg.story_scores = story_scores or {}
    msg.theme_weights = theme_weights or {}
    msg.tag_weights = tag_weights or {}
    msg.mood_scores = list(mood_scores)
    return msg


class TestLoadAllFromServer:
    def test_restores_viewed_story_ids(self) -> None:
        model_msg = _make_model_msg("u1", viewed=["s1"], theme_weights={"adventure": 1.0})

        stub = MagicMock()
        stub.LoadUserModel.return_value.user_models = [model_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids

    def test_restores_theme_weights_directly(self) -> None:
        model_msg = _make_model_msg("u1", theme_weights={"adventure": 1.0, "mystery": 0.5})

        stub = MagicMock()
        stub.LoadUserModel.return_value.user_models = [model_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(1.0)
        assert profile.theme_weights["mystery"] == pytest.approx(0.5)

    def test_restores_story_scores(self) -> None:
        model_msg = _make_model_msg("u1", story_scores={"s1": 8})

        stub = MagicMock()
        stub.LoadUserModel.return_value.user_models = [model_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 8

    def test_load_does_not_raise_on_stub_error(self) -> None:
        stub = MagicMock()
        stub.LoadUserModel.side_effect = RuntimeError("server down")
        cat = _make_catalogue()
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()  # should not raise


# ---------------------------------------------------------------------------
# Timestamp helper tests
# ---------------------------------------------------------------------------


class TestTimestampHelpers:
    def test_datetime_to_timestamp_and_back(self) -> None:
        original = datetime(2024, 6, 1, 12, 30, 0, tzinfo=timezone.utc)
        ts = _datetime_to_timestamp(original)
        recovered = _timestamp_to_datetime(ts)
        assert recovered == original

    def test_naive_datetime_treated_as_utc(self) -> None:
        naive = datetime(2024, 6, 1, 12, 0, 0)
        ts = _datetime_to_timestamp(naive)
        recovered = _timestamp_to_datetime(ts)
        assert recovered.tzinfo is not None


# ---------------------------------------------------------------------------
# Mood attribution tests
# ---------------------------------------------------------------------------


class TestMoodAttribution:
    """Mood events should adjust weights for recently engaged themes/tags."""

    def test_mood_improvement_boosts_engaged_themes(self) -> None:
        """When mood rises, themes engaged since the last mood event are boosted."""
        store = _make_store(STORY_ADV)

        # First mood event — establishes baseline score, no attribution yet
        store.record_mood("u1", 2, TS)

        # View a story between mood events (enters the attribution window)
        store.record_read_progress("u1", "s1", 50, TS)
        profile = store.get_or_create_profile("u1")
        baseline = profile.theme_weights.get("adventure", 0.0)

        # Second mood event: mood rises 2 → 5 (delta = +3)
        store.record_mood("u1", 5, TS)

        after = profile.theme_weights.get("adventure", 0.0)
        # Expected boost: (delta / _MOOD_MAX_DELTA) * MOOD_ATTRIBUTION_FACTOR * window_weight
        expected_boost = (3 / _MOOD_MAX_DELTA) * MOOD_ATTRIBUTION_FACTOR * _WEIGHT_VIEW
        assert after == pytest.approx(baseline + expected_boost)

    def test_mood_decline_dampens_engaged_themes(self) -> None:
        """When mood falls, weights for engaged themes are reduced (but not below 0)."""
        store = _make_store(STORY_ADV)
        store.record_mood("u1", 5, TS)   # first mood: 5
        store.record_read_progress("u1", "s1", 50, TS)

        profile = store.get_or_create_profile("u1")
        before = profile.theme_weights.get("adventure", 0.0)

        store.record_mood("u1", 2, TS)   # second mood: 2 (delta = −3)

        after = profile.theme_weights.get("adventure", 0.0)
        expected_reduction = (3 / _MOOD_MAX_DELTA) * MOOD_ATTRIBUTION_FACTOR * _WEIGHT_VIEW
        assert after == pytest.approx(before - expected_reduction)

    def test_mood_weight_never_goes_below_zero(self) -> None:
        """Dampening is capped so weights cannot become negative."""
        store = _make_store(STORY_ADV)
        store.record_mood("u1", 8, TS)
        store.record_read_progress("u1", "s1", 50, TS)

        profile = store.get_or_create_profile("u1")
        # Override to create a scenario where attribution would push below zero
        profile.theme_weights["adventure"] = 0.01
        profile.themes_since_last_mood["adventure"] = 100.0  # large window weight

        # Big mood drop — attribution would push weight strongly negative without floor
        store.record_mood("u1", 1, TS)
        assert profile.theme_weights.get("adventure", 0.0) >= 0.0

    def test_accumulators_reset_after_mood_event(self) -> None:
        """Transient accumulators are cleared when a mood event is recorded."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)
        store.record_mood("u1", 3, TS)

        profile = store.get_or_create_profile("u1")
        assert profile.themes_since_last_mood == {}
        assert profile.tags_since_last_mood == {}

    def test_first_mood_event_no_attribution(self) -> None:
        """The very first mood event records the baseline; no weights are changed."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)

        profile = store.get_or_create_profile("u1")
        weight_before = profile.theme_weights.get("adventure", 0.0)

        store.record_mood("u1", 4, TS)

        weight_after = profile.theme_weights.get("adventure", 0.0)
        assert weight_before == weight_after

    def test_equal_mood_no_attribution(self) -> None:
        """If mood does not change, no attribution is applied."""
        store = _make_store(STORY_ADV)
        store.record_mood("u1", 3, TS)
        store.record_read_progress("u1", "s1", 50, TS)

        profile = store.get_or_create_profile("u1")
        before = profile.theme_weights.get("adventure", 0.0)

        store.record_mood("u1", 3, TS)   # same score

        after = profile.theme_weights.get("adventure", 0.0)
        assert before == after

    def test_accumulator_tracks_viewed_stories(self) -> None:
        """Viewing a story adds its themes/tags to the mood window accumulator."""
        store = _make_store(STORY_ADV)
        store.record_mood("u1", 3, TS)  # start the window

        store.record_read_progress("u1", "s1", 50, TS)

        profile = store.get_or_create_profile("u1")
        assert "adventure" in profile.themes_since_last_mood
        assert profile.themes_since_last_mood["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_accumulator_tracks_completed_stories(self) -> None:
        """Completing a story adds the completion bonus to the mood window accumulator."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 50, TS)   # mark as viewed
        store.record_mood("u1", 3, TS)                   # start fresh window

        # Now complete: view path is idempotent, only complete bonus goes into window
        store.record_read_progress("u1", "s1", 100, TS)

        profile = store.get_or_create_profile("u1")
        assert "adventure" in profile.themes_since_last_mood
        assert profile.themes_since_last_mood["adventure"] == pytest.approx(
            _WEIGHT_COMPLETE_BONUS
        )
