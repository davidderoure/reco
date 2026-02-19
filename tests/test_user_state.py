"""Tests for recommender.user_state.UserStateStore.

These are the most important tests in the suite — they verify the weight
accumulation maths that drive all recommendation strategies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from recommender.catalogue import StoryCatalogue
from recommender.models import EventType, Story, UserEvent, UserProfile
from recommender.user_state import (
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
    # LoadUserState returns an empty response (no prior users)
    stub.LoadUserState.return_value.user_states = []
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

    def test_appends_event_to_log(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert len(profile.events) == 1
        assert profile.events[0].event_type == EventType.VIEWED

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

    def test_complete_event_appended(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_completed("u1", "s1", TS)
        profile = store.get_or_create_profile("u1")
        assert any(e.event_type == EventType.COMPLETED for e in profile.events)


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

    def test_score_event_appended(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_scored("u1", "s1", 5, TS)
        profile = store.get_or_create_profile("u1")
        assert any(e.event_type == EventType.SCORED for e in profile.events)


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
    def test_persist_calls_save_user_state(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        store.persist_all_to_server()
        store._stub.SaveUserState.assert_called_once()

    def test_persist_serialises_events(self) -> None:
        store = _make_store(STORY_ADV)
        store.record_viewed("u1", "s1", TS)
        store.persist_all_to_server()
        call_args = store._stub.SaveUserState.call_args[0][0]
        user_states = list(call_args.user_states)
        assert len(user_states) == 1
        assert user_states[0].user_id == "u1"
        events = list(user_states[0].events)
        assert len(events) == 1
        assert events[0].event_type == "viewed"
        assert events[0].story_id == "s1"

    def test_persist_does_not_raise_on_stub_error(self) -> None:
        store = _make_store()
        store._stub.SaveUserState.side_effect = RuntimeError("network error")
        store.persist_all_to_server()  # should not raise


class TestLoadAllFromServer:
    def _make_event_msg(
        self,
        event_type: str,
        story_id: str,
        score: int,
        ts_seconds: int = 1717243200,
    ):
        msg = MagicMock()
        msg.event_type = event_type
        msg.story_id = story_id
        msg.score = score
        msg.timestamp.seconds = ts_seconds
        msg.timestamp.nanos = 0
        return msg

    def test_replays_viewed_event(self) -> None:
        event_msg = self._make_event_msg("viewed", "s1", 0)
        state_msg = MagicMock()
        state_msg.user_id = "u1"
        state_msg.events = [event_msg]

        stub = MagicMock()
        stub.LoadUserState.return_value.user_states = [state_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert "s1" in profile.viewed_story_ids
        assert profile.theme_weights["adventure"] == pytest.approx(_WEIGHT_VIEW)

    def test_replays_scored_event(self) -> None:
        event_msg = self._make_event_msg("scored", "s1", 5)
        state_msg = MagicMock()
        state_msg.user_id = "u1"
        state_msg.events = [event_msg]

        stub = MagicMock()
        stub.LoadUserState.return_value.user_states = [state_msg]
        cat = _make_catalogue(STORY_ADV)
        store = UserStateStore(stub=stub, catalogue=cat)
        store.load_all_from_server()

        profile = store.get_or_create_profile("u1")
        assert profile.story_scores["s1"] == 5

    def test_load_does_not_raise_on_stub_error(self) -> None:
        stub = MagicMock()
        stub.LoadUserState.side_effect = RuntimeError("server down")
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
