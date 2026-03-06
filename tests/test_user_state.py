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
    MOOD_HISTORY_LIMIT,
    READ_VIEWED_THRESHOLD_PERCENT,
    UserStateStore,
    _datetime_to_timestamp,
    _timestamp_to_datetime,
    _WEIGHT_VIEW,
    _WEIGHT_COMPLETE_BONUS,
    _WEIGHT_SCORE_FACTOR,
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
# Weight accumulation tests
# ---------------------------------------------------------------------------


class TestRecordViewed:
    def test_adds_view_weight_to_themes(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_adds_view_weight_to_tags(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert profile.tag_weights["pirates"] == pytest.approx(_WEIGHT_VIEW)
        assert profile.tag_weights["ocean"] == pytest.approx(_WEIGHT_VIEW)

    def test_marks_story_as_viewed(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids

    def test_multiple_views_accumulate(self) -> None:
        store = _make_store(STORY_ADV, STORY_MYS)
        store.record_viewed("u1", "s1", TS)
        store.record_viewed("u1", "s2", TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)
        assert profile.theme_weights["mystery"] == pytest.approx(_WEIGHT_VIEW)

    def test_unknown_story_does_not_crash(self) -> None:
        store = _make_store()
        store.record_viewed("u1", "nonexistent", TS)
        profile = store.get_or_create_profile("u1")
        assert "nonexistent" in profile.viewed_story_ids
        assert profile.theme_weights == {}

    def test_multi_theme_story(self) -> None:
        store = _make_store(STORY_MULTI)
        store.record_viewed("u1", "s3", TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)
        assert profile.theme_weights["mystery"] == pytest.approx(_WEIGHT_VIEW)
        assert profile.tag_weights["treasure"] == pytest.approx(_WEIGHT_VIEW)


class TestRecordCompleted:
    def test_adds_complete_bonus_to_themes(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_completed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_COMPLETE_BONUS)

    def test_marks_story_as_completed(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_completed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.completed_story_ids

    def test_view_plus_complete_accumulates(self) -> None:
        """Viewing then completing a story gives WEIGHT_VIEW + WEIGHT_COMPLETE_BONUS."""
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        store.record_completed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        expected = _WEIGHT_VIEW + _WEIGHT_COMPLETE_BONUS
        assert profile.theme_weights["adventure"] == pytest.approx(expected)


class TestRecordScored:
    @pytest.mark.parametrize("score,expected_delta", [
        (5, (5 - 3) * _WEIGHT_SCORE_FACTOR),   # +1.0
        (4, (4 - 3) * _WEIGHT_SCORE_FACTOR),   # +0.5
        (3, (3 - 3) * _WEIGHT_SCORE_FACTOR),   # 0.0
        (2, (2 - 3) * _WEIGHT_SCORE_FACTOR),   # -0.5
        (1, (1 - 3) * _WEIGHT_SCORE_FACTOR),   # -1.0
    ])
    def test_score_weight_delta(self, score: int, expected_delta: float) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", score, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights.get("adventure", 0.0) == pytest.approx(expected_delta)

    def test_stores_score_in_story_scores(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 4, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 4

    def test_latest_score_overwrites(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 4, TS)
        store.record_scored("u1", "s1", 2, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 2

    def test_invalid_score_raises(self) -> None:
        store = _make_store(STORY_ADV)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 0, TS)
        with pytest.raises(ValueError):
            store.record_scored("u1", "s1", 6, TS)

    def test_view_and_score_5_accumulates(self) -> None:
        """View (+1.0) then score 5 (+1.0) = total +2.0."""
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        store.record_scored("u1", "s1", 5, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(2.0)


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
        store.record_mood("u1", 5, ts2)
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
            store.record_mood("u1", 6, TS)

    def test_mood_history_capped_at_limit(self) -> None:
        store = _make_store()
        for i in range(MOOD_HISTORY_LIMIT + 10):
            store.record_mood("u1", (i % 5) + 1, TS)
        profile = store.get_or_create_profile("u1")
        assert len(profile.mood_scores) == MOOD_HISTORY_LIMIT


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

    def test_above_threshold_marks_viewed(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_idempotent_above_threshold(self) -> None:
        """Second call above threshold does not double-apply the view weight."""
        store = _make_store(STORY_ADV)
        store.record_read_progress("u1", "s1", 75, TS)
        store.record_read_progress("u1", "s1", 90, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_already_viewed_not_double_counted(self) -> None:
        """Story already in viewed_story_ids gets no extra weight delta."""
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        weight_before = store.get_or_create_profile("u1").theme_weights["adventure"]
        store.record_read_progress("u1", "s1", 100, TS)
        profile = store.get_or_create_profile("u1")
        assert profile.theme_weights["adventure"] == pytest.approx(weight_before)

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
        store.record_viewed("u1", "s1", TS)
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
        store.record_viewed("u1", "s1", TS)
        store.persist_all_to_server()
        store._stub.SaveUserModel.assert_called_once()

    def test_persist_serialises_model_fields(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        store.record_scored("u1", "s1", 5, TS)
        store.persist_all_to_server()
        call_args = store._stub.SaveUserModel.call_args[0][0]
        user_models = list(call_args.user_models)
        assert len(user_models) == 1
        model = user_models[0]
        assert model.user_id == "u1"
        assert "s1" in list(model.viewed_story_ids)
        assert model.story_scores["s1"] == 5
        assert model.theme_weights["adventure"] == pytest.approx(
            _WEIGHT_VIEW + (5 - 3) * _WEIGHT_SCORE_FACTOR
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
        model_msg = _make_model_msg("u1", story_scores={"s1": 5})

        stub = MagicMock()
        stub.LoadUserModel.return_value.user_models = [model_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 5

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
