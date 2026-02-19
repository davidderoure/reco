"""Story catalogue: fetches and caches stories from the C# server."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from recommender.models import Story

logger = logging.getLogger(__name__)


class StoryCatalogue:
    """Fetches and caches the story catalogue from the C# server.

    The catalogue is loaded synchronously on first call to :meth:`refresh`,
    then kept fresh by a background daemon thread that calls
    :meth:`refresh` every *refresh_interval_seconds*.

    All public methods are thread-safe.

    Args:
        stub: A ``StoryServiceStub`` instance (generated gRPC client stub).
            In tests this can be any object with a ``GetStoryCatalogue``
            callable attribute.
        refresh_interval_seconds: How often the background thread refreshes
            the catalogue. Defaults to 300 (5 minutes).
    """

    def __init__(self, stub: Any, refresh_interval_seconds: int = 300) -> None:
        self._stub = stub
        self._refresh_interval = refresh_interval_seconds
        self._lock = threading.RLock()
        self._stories: dict[str, Story] = {}
        self._refresh_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Fetch the full catalogue from the C# server and update the cache.

        Blocks until the RPC completes. On failure, logs an error and
        preserves the existing cache so the service can continue running.
        """
        try:
            from generated import recommender_pb2

            response = self._stub.GetStoryCatalogue(
                recommender_pb2.GetStoryCatalogueRequest()
            )
            new_stories: dict[str, Story] = {}
            for msg in response.stories:
                new_stories[msg.story_id] = Story(
                    story_id=msg.story_id,
                    title=msg.title,
                    themes=list(msg.themes),
                    tags=list(msg.tags),
                )
            with self._lock:
                self._stories = new_stories
            logger.info("Story catalogue refreshed: %d stories loaded.", len(new_stories))
        except Exception:
            logger.exception(
                "Failed to refresh story catalogue; keeping existing %d stories.",
                len(self._stories),
            )

    def start_refresh_loop(self) -> None:
        """Start a background daemon thread that periodically calls :meth:`refresh`.

        Safe to call multiple times — only one refresh thread is started.
        """
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="catalogue-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        logger.debug("Catalogue refresh loop started (interval=%ds).", self._refresh_interval)

    def get_all_stories(self) -> list[Story]:
        """Return a snapshot list of all currently cached stories.

        Returns:
            List of :class:`~recommender.models.Story` objects.
            Empty list if the catalogue has never been loaded.
        """
        with self._lock:
            return list(self._stories.values())

    def get_story(self, story_id: str) -> Story | None:
        """Return a single story by ID, or ``None`` if not found.

        Args:
            story_id: The story's unique identifier.

        Returns:
            A :class:`~recommender.models.Story` or ``None``.
        """
        with self._lock:
            return self._stories.get(story_id)

    def get_all_themes(self) -> list[str]:
        """Return a sorted list of all unique theme labels across all stories.

        Returns:
            Sorted list of theme strings. Stable ordering is important as
            it defines the index positions in preference weight vectors.
        """
        with self._lock:
            themes: set[str] = set()
            for story in self._stories.values():
                themes.update(story.themes)
            return sorted(themes)

    def get_all_tags(self) -> list[str]:
        """Return a sorted list of all unique tag labels across all stories.

        Returns:
            Sorted list of tag strings. Stable ordering is important as
            it defines the index positions in preference weight vectors.
        """
        with self._lock:
            tags: set[str] = set()
            for story in self._stories.values():
                tags.update(story.tags)
            return sorted(tags)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        """Periodically refresh the catalogue. Runs in a daemon thread."""
        while True:
            time.sleep(self._refresh_interval)
            self.refresh()
