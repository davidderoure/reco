"""Entry point: wires all components and starts the gRPC server."""

from __future__ import annotations

import logging
import signal
import sys
from concurrent import futures

import grpc

import config
from generated import recommender_pb2_grpc
from recommender.catalogue import StoryCatalogue
from recommender.engine import RecommendationEngine
from recommender.service import RecommenderServicer
from recommender.strategies.collaborative import CollaborativeFilteringStrategy
from recommender.strategies.content_based import ContentBasedStrategy
from recommender.strategies.topical import TopicalStrategy
from recommender.strategies.wildcard import WildcardStrategy
from recommender.user_state import UserStateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_server(
    catalogue: StoryCatalogue,
    user_state_store: UserStateStore,
) -> grpc.Server:
    """Construct and configure the gRPC server with all dependencies wired.

    Args:
        catalogue: The loaded :class:`~recommender.catalogue.StoryCatalogue`.
        user_state_store: The loaded :class:`~recommender.user_state.UserStateStore`.

    Returns:
        A configured but not-yet-started :class:`grpc.Server`.
    """
    all_themes = catalogue.get_all_themes()
    all_tags = catalogue.get_all_tags()

    engine = RecommendationEngine(
        catalogue=catalogue,
        user_state_store=user_state_store,
        content_strategy=ContentBasedStrategy(all_themes, all_tags),
        collaborative_strategy=CollaborativeFilteringStrategy(all_themes, all_tags),
        topical_strategy=TopicalStrategy(),
        wildcard_strategy=WildcardStrategy(),
    )

    servicer = RecommenderServicer(engine=engine, user_state_store=user_state_store)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.GRPC_MAX_WORKERS)
    )
    recommender_pb2_grpc.add_RecommenderServiceServicer_to_server(servicer, server)
    server.add_insecure_port(
        f"{config.GRPC_SERVER_HOST}:{config.GRPC_SERVER_PORT}"
    )
    return server


def main() -> None:
    """Initialise all components and start the gRPC server.

    Startup sequence:
    1. Connect to the C# server as a gRPC client.
    2. Load the story catalogue (blocks until first successful fetch).
    3. Load all user state from the C# server (replay event logs).
    4. Start background threads (catalogue refresh, state persistence).
    5. Register ``SIGTERM``/``SIGINT`` shutdown handlers.
    6. Build and start the Python gRPC server.
    """
    logger.info("Connecting to C# server at %s", config.CSHARP_SERVER_ADDRESS)
    csharp_channel = grpc.insecure_channel(config.CSHARP_SERVER_ADDRESS)

    # Import generated stubs
    from generated import recommender_pb2_grpc as grpc_stubs

    story_stub = grpc_stubs.StoryServiceStub(csharp_channel)

    # Step 2: Load catalogue (synchronous — blocks until success)
    logger.info("Loading story catalogue from C# server…")
    catalogue = StoryCatalogue(
        stub=story_stub,
        refresh_interval_seconds=config.CATALOGUE_REFRESH_INTERVAL_SECONDS,
    )
    catalogue.refresh()
    logger.info("Catalogue loaded: %d stories.", len(catalogue.get_all_stories()))

    # Step 3: Load user state
    logger.info("Loading user state from C# server…")
    user_state_store = UserStateStore(stub=story_stub, catalogue=catalogue)
    user_state_store.load_all_from_server()

    # Step 4: Start background threads
    catalogue.start_refresh_loop()
    user_state_store.start_persist_loop(config.STATE_PERSIST_INTERVAL_SECONDS)

    # Step 5: Build gRPC server
    server = build_server(catalogue, user_state_store)

    # Step 6: Register shutdown handlers
    def handle_shutdown(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — persisting user state and shutting down…", sig_name)
        user_state_store.persist_all_to_server()
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    # Step 7: Start serving
    server.start()
    logger.info(
        "Recommender gRPC server listening on %s:%d",
        config.GRPC_SERVER_HOST,
        config.GRPC_SERVER_PORT,
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
