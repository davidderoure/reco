"""Tests for RecommenderServicer (gRPC service layer)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import grpc
import pytest

from recommender.service import RecommenderServicer, _proto_ts_to_datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ts(dt: datetime = None):
    """Return a mock proto Timestamp."""
    if dt is None:
        dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    ts = MagicMock()
    ts.seconds = int(dt.timestamp())
    ts.nanos = 0
    return ts


def _make_context() -> MagicMock:
    """Return a mock gRPC context."""
    ctx = MagicMock()
    ctx.set_code = MagicMock()
    ctx.set_details = MagicMock()
    return ctx


def _make_servicer(
    recommendations: list[str] | None = None,
    engine_raises: Exception | None = None,
) -> RecommenderServicer:
    engine = MagicMock()
    if engine_raises:
        engine.get_recommendations.side_effect = engine_raises
    else:
        engine.get_recommendations.return_value = recommendations or [
            "s1", "s2", "s3", "s4", "s5", "s6"
        ]

    store = MagicMock()
    return RecommenderServicer(engine=engine, user_state_store=store)


# ---------------------------------------------------------------------------
# UserViewedStory tests
# ---------------------------------------------------------------------------


class TestUserViewedStory:
    def test_calls_record_viewed(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="u1", story_id="s1", timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserViewedStory(request, ctx)
        servicer._store.record_viewed.assert_called_once()
        call_args = servicer._store.record_viewed.call_args[0]
        assert call_args[0] == "u1"
        assert call_args[1] == "s1"
        assert isinstance(call_args[2], datetime)

    def test_returns_empty(self) -> None:
        from google.protobuf.empty_pb2 import Empty
        servicer = _make_servicer()
        request = MagicMock(user_id="u1", story_id="s1", timestamp=_make_ts())
        result = servicer.UserViewedStory(request, _make_context())
        assert isinstance(result, Empty)

    def test_store_error_sets_internal_status(self) -> None:
        servicer = _make_servicer()
        servicer._store.record_viewed.side_effect = RuntimeError("db error")
        request = MagicMock(user_id="u1", story_id="s1", timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserViewedStory(request, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


# ---------------------------------------------------------------------------
# UserCompletedStory tests
# ---------------------------------------------------------------------------


class TestUserCompletedStory:
    def test_calls_record_completed(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="u1", story_id="s1", timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserCompletedStory(request, ctx)
        servicer._store.record_completed.assert_called_once()

    def test_store_error_sets_internal_status(self) -> None:
        servicer = _make_servicer()
        servicer._store.record_completed.side_effect = RuntimeError("error")
        request = MagicMock(user_id="u1", story_id="s1", timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserCompletedStory(request, ctx)
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)


# ---------------------------------------------------------------------------
# UserAnsweredQuestion tests
# ---------------------------------------------------------------------------


class TestUserAnsweredQuestion:
    def test_calls_record_scored(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="u1", story_id="s1", score=4, timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserAnsweredQuestion(request, ctx)
        servicer._store.record_scored.assert_called_once()
        call_args = servicer._store.record_scored.call_args[0]
        assert call_args[2] == 4

    def test_invalid_score_sets_invalid_argument(self) -> None:
        servicer = _make_servicer()
        servicer._store.record_scored.side_effect = ValueError("Score out of range")
        request = MagicMock(user_id="u1", story_id="s1", score=6, timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserAnsweredQuestion(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_internal_error_sets_internal_status(self) -> None:
        servicer = _make_servicer()
        servicer._store.record_scored.side_effect = RuntimeError("crash")
        request = MagicMock(user_id="u1", story_id="s1", score=4, timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserAnsweredQuestion(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INTERNAL)


# ---------------------------------------------------------------------------
# UserProvidedMood tests
# ---------------------------------------------------------------------------


class TestUserProvidedMood:
    def test_calls_record_mood(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="u1", mood_score=3, timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserProvidedMood(request, ctx)
        servicer._store.record_mood.assert_called_once()
        call_args = servicer._store.record_mood.call_args[0]
        assert call_args[1] == 3

    def test_invalid_mood_sets_invalid_argument(self) -> None:
        servicer = _make_servicer()
        servicer._store.record_mood.side_effect = ValueError("Mood out of range")
        request = MagicMock(user_id="u1", mood_score=0, timestamp=_make_ts())
        ctx = _make_context()
        servicer.UserProvidedMood(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)


# ---------------------------------------------------------------------------
# GetRecommendations tests
# ---------------------------------------------------------------------------


class TestGetRecommendations:
    def test_returns_six_story_ids(self) -> None:
        servicer = _make_servicer(recommendations=["s1", "s2", "s3", "s4", "s5", "s6"])
        request = MagicMock(user_id="u1")
        response = servicer.GetRecommendations(request, _make_context())
        assert list(response.story_ids) == ["s1", "s2", "s3", "s4", "s5", "s6"]

    def test_empty_user_id_sets_invalid_argument(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="")
        ctx = _make_context()
        servicer.GetRecommendations(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_engine_value_error_sets_invalid_argument(self) -> None:
        servicer = _make_servicer(engine_raises=ValueError("bad user"))
        request = MagicMock(user_id="u1")
        ctx = _make_context()
        servicer.GetRecommendations(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    def test_engine_runtime_error_sets_internal(self) -> None:
        servicer = _make_servicer(engine_raises=RuntimeError("crash"))
        request = MagicMock(user_id="u1")
        ctx = _make_context()
        servicer.GetRecommendations(request, ctx)
        ctx.set_code.assert_called_with(grpc.StatusCode.INTERNAL)

    def test_slow_response_logs_warning(self) -> None:
        """Simulate a slow engine call and verify the warning is logged."""
        import time

        servicer = _make_servicer()

        def slow_recommend(user_id):
            time.sleep(0.001)  # tiny sleep, just to test the timing path
            return ["s1", "s2", "s3", "s4", "s5", "s6"]

        servicer._engine.get_recommendations = slow_recommend

        with patch("recommender.service._RECOMMENDATION_WARN_THRESHOLD_MS", 0):
            with patch("recommender.service.logger") as mock_logger:
                request = MagicMock(user_id="u1")
                servicer.GetRecommendations(request, _make_context())
                mock_logger.warning.assert_called_once()

    def test_calls_engine_with_user_id(self) -> None:
        servicer = _make_servicer()
        request = MagicMock(user_id="u42")
        servicer.GetRecommendations(request, _make_context())
        servicer._engine.get_recommendations.assert_called_once_with("u42")


# ---------------------------------------------------------------------------
# Timestamp helper tests
# ---------------------------------------------------------------------------


class TestProtoTsToDatetime:
    def test_converts_correctly(self) -> None:
        ts = MagicMock()
        ts.seconds = 1717243200  # 2024-06-01 12:00:00 UTC
        ts.nanos = 0
        result = _proto_ts_to_datetime(ts)
        assert result.tzinfo is not None
        assert result.year == 2024
        assert result.month == 6

    def test_result_is_utc_aware(self) -> None:
        ts = MagicMock()
        ts.seconds = 0
        ts.nanos = 0
        result = _proto_ts_to_datetime(ts)
        assert result.tzinfo == timezone.utc
