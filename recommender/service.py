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

    def UserAnsweredQuestion(self, request: Any, context: Any) -> Any:
        """Record a user's answer to an end-of-story question (score 1–10).

        Args:
            request: ``UserAnsweredQuestionRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        question_number = request.question_number or 1  # 0 (unset proto3 default) → 1
        try:
            self._store.record_scored(
                request.user_id, request.story_id, request.score, ts,
                question_number=question_number,
            )
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
        """Record a user's current mood score (1–10).

        An optional ``story_id`` may be provided when the mood prompt is shown
        at the end of a story, associating the mood signal with that content.

        Args:
            request: ``UserProvidedMoodRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        story_id = request.story_id or None  # empty proto string → None
        try:
            self._store.record_mood(request.user_id, request.mood_score, ts, story_id=story_id)
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

    def UserReadStory(self, request: Any, context: Any) -> Any:
        """Record how far through a story the user has read.

        Two events are inferred from ``read_percent``:

        * ``≥ 50%`` → treated as *viewed*: themes/tags receive the view weight
          delta and the story is added to the user's viewed set (idempotent).
        * ``== 100%`` → treated as *completed*: the completion bonus weight
          delta is also applied and the story is added to completed_story_ids
          (idempotent).

        Events below 50% do not affect recommendations.

        Args:
            request: ``UserReadStoryRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        try:
            self._store.record_read_progress(
                request.user_id, request.story_id, request.read_percent, ts
            )
        except ValueError as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
        except Exception:
            logger.exception(
                "Error recording read progress for user=%r story=%r percent=%r",
                request.user_id,
                request.story_id,
                request.read_percent,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error recording read progress.")
        return Empty()

    def UserBookmarkedStory(self, request: Any, context: Any) -> Any:
        """Record that a user bookmarked a story.

        Bookmarks are captured as analytics events.  They currently have no
        effect on recommendation weights or the user profile — this may change
        in a future release (e.g. to populate a reading list or bias picks).

        Args:
            request: ``UserBookmarkedStoryRequest`` proto message.
            context: gRPC service context.

        Returns:
            ``google.protobuf.Empty``.
        """
        from google.protobuf.empty_pb2 import Empty

        ts = _proto_ts_to_datetime(request.timestamp)
        logger.info(
            "UserBookmarkedStory user=%r story=%r at=%s",
            request.user_id,
            request.story_id,
            ts.isoformat(),
        )
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

        ts = _proto_ts_to_datetime(request.timestamp)
        logger.debug(
            "GetRecommendations user=%r requested_at=%s",
            request.user_id,
            ts.isoformat(),
        )

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
