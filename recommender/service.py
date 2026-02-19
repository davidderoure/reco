"""gRPC servicer: the entry point for all inbound calls from the C# client."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import grpc

from recommender.engine import RecommendationEngine
from recommender.user_state import UserStateStore

logger = logging.getLogger(__name__)

_RECOMMENDATION_WARN_THRESHOLD_MS = 450  # warn if within 50ms of SLA


class RecommenderServicer:
    """Implements the ``RecommenderService`` gRPC service defined in the proto.

    This class is registered with the gRPC server via the generated
    ``add_RecommenderServiceServicer_to_server`` helper in :mod:`main`.

    Args:
        engine: The :class:`~recommender.engine.RecommendationEngine`.
        user_state_store: The :class:`~recommender.user_state.UserStateStore`.
    """

    def __init__(
        self,
        engine: RecommendationEngine,
        user_state_store: UserStateStore,
    ) -> None:
        self._engine = engine
        self._store = user_state_store

    # ------------------------------------------------------------------
    # Fire-and-forget event methods
    # ------------------------------------------------------------------

    def UserViewedStory(self, request: Any, context: Any) -> Any:
        """Record that a user viewed a story.

        Args:
            request: ``UserViewedStoryRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        try:
            self._store.record_viewed(request.user_id, request.story_id, ts)
        except Exception:
            logger.exception(
                "Error recording viewed event for user=%r story=%r",
                request.user_id,
                request.story_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error recording viewed event.")
        return Empty()

    def UserCompletedStory(self, request: Any, context: Any) -> Any:
        """Record that a user completed a story.

        Args:
            request: ``UserCompletedStoryRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        try:
            self._store.record_completed(request.user_id, request.story_id, ts)
        except Exception:
            logger.exception(
                "Error recording completed event for user=%r story=%r",
                request.user_id,
                request.story_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error recording completed event.")
        return Empty()

    def UserAnsweredQuestion(self, request: Any, context: Any) -> Any:
        """Record a user's end-of-story score (1–5).

        Args:
            request: ``UserAnsweredQuestionRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        try:
            self._store.record_scored(request.user_id, request.story_id, request.score, ts)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
        except Exception:
            logger.exception(
                "Error recording scored event for user=%r story=%r score=%r",
                request.user_id,
                request.story_id,
                request.score,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error recording score event.")
        return Empty()

    def UserProvidedMood(self, request: Any, context: Any) -> Any:
        """Record a user's current mood score (1–5).

        Args:
            request: ``UserProvidedMoodRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        try:
            self._store.record_mood(request.user_id, request.mood_score, ts)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
        except Exception:
            logger.exception(
                "Error recording mood event for user=%r score=%r",
                request.user_id,
                request.mood_score,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error recording mood event.")
        return Empty()

    # ------------------------------------------------------------------
    # Recommendation request
    # ------------------------------------------------------------------

    def GetRecommendations(self, request: Any, context: Any) -> Any:
        """Return 6 story IDs within the 500ms SLA.

        Args:
            request: ``GetRecommendationsRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``GetRecommendationsResponse`` with exactly 6 story IDs.
        """
        from generated import recommender_pb2

        if not request.user_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("user_id must be non-empty")
            return recommender_pb2.GetRecommendationsResponse()

        start_ms = time.monotonic() * 1000
        try:
            story_ids = self._engine.get_recommendations(request.user_id)
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return recommender_pb2.GetRecommendationsResponse()
        except Exception:
            logger.exception(
                "Unexpected error generating recommendations for user=%r",
                request.user_id,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error generating recommendations.")
            return recommender_pb2.GetRecommendationsResponse()
        finally:
            elapsed_ms = time.monotonic() * 1000 - start_ms
            if elapsed_ms > _RECOMMENDATION_WARN_THRESHOLD_MS:
                logger.warning(
                    "GetRecommendations for user=%r took %.1fms (SLA: 500ms)",
                    request.user_id,
                    elapsed_ms,
                )
            else:
                logger.debug(
                    "GetRecommendations for user=%r took %.1fms",
                    request.user_id,
                    elapsed_ms,
                )

        return recommender_pb2.GetRecommendationsResponse(story_ids=story_ids)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def _proto_ts_to_datetime(ts: Any) -> datetime:
    """Convert a ``google.protobuf.Timestamp`` to a UTC-aware ``datetime``.

    Args:
        ts: A protobuf Timestamp object (``seconds``, ``nanos``).

    Returns:
        UTC-aware :class:`datetime`.
    """
    return datetime.fromtimestamp(
        ts.seconds + ts.nanos / 1e9, tz=timezone.utc
    )
