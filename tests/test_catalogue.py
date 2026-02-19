"""Tests for recommender.catalogue.StoryCatalogue."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from recommender.catalogue import StoryCatalogue
from recommender.models import Story


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_story_msg(story_id: str, title: str, themes: list[str], tags: list[str]):
    """Build a minimal proto-like object (plain namespace) for testing."""
    msg = MagicMock()
    msg.story_id = story_id
    msg.title = title
    msg.themes = themes
    msg.tags = tags
    return msg


def _make_stub(stories_msgs: list) -> MagicMock:
    """Return a stub whose GetStoryCatalogue returns the given story messages."""
    stub = MagicMock()
    response = MagicMock()
    response.stories = stories_msgs
    stub.GetStoryCatalogue.return_value = response
    return stub


@pytest.fixture
def sample_msgs():
    return [
        _make_story_msg("s1", "The Lost Isles", ["adventure"], ["pirates", "treasure"]),
        _make_story_msg("s2", "Dark Forest", ["mystery", "horror"], ["trees", "shadows"]),
        _make_story_msg("s3", "Calm Waters", ["calm"], ["ocean"]),
    ]


@pytest.fixture
def catalogue(sample_msgs):
    stub = _make_stub(sample_msgs)
    cat = StoryCatalogue(stub=stub, refresh_interval_seconds=9999)
    cat.refresh()
    return cat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_loads_all_stories(self, catalogue: StoryCatalogue) -> None:
        stories = catalogue.get_all_stories()
        assert len(stories) == 3

    def test_story_fields_parsed_correctly(self, catalogue: StoryCatalogue) -> None:
        story = catalogue.get_story("s1")
        assert story is not None
        assert story.story_id == "s1"
        assert story.title == "The Lost Isles"
        assert story.themes == ["adventure"]
        assert story.tags == ["pirates", "treasure"]

    def test_refresh_replaces_previous_catalogue(self) -> None:
        msgs1 = [_make_story_msg("s1", "A", ["x"], ["y"])]
        msgs2 = [
            _make_story_msg("s2", "B", ["x"], ["y"]),
            _make_story_msg("s3", "C", ["x"], ["y"]),
        ]
        stub = _make_stub(msgs1)
        cat = StoryCatalogue(stub=stub, refresh_interval_seconds=9999)
        cat.refresh()
        assert len(cat.get_all_stories()) == 1

        # Swap to new response and refresh
        stub.GetStoryCatalogue.return_value.stories = msgs2
        cat.refresh()
        assert len(cat.get_all_stories()) == 2
        assert cat.get_story("s1") is None
        assert cat.get_story("s2") is not None

    def test_refresh_preserves_cache_on_failure(self, sample_msgs) -> None:
        """If the RPC raises, the old cache should be retained."""
        stub = _make_stub(sample_msgs)
        cat = StoryCatalogue(stub=stub)
        cat.refresh()
        assert len(cat.get_all_stories()) == 3

        stub.GetStoryCatalogue.side_effect = RuntimeError("network error")
        cat.refresh()  # should not raise
        assert len(cat.get_all_stories()) == 3  # old cache intact

    def test_empty_catalogue(self) -> None:
        stub = _make_stub([])
        cat = StoryCatalogue(stub=stub)
        cat.refresh()
        assert cat.get_all_stories() == []


class TestGetStory:
    def test_returns_story_for_known_id(self, catalogue: StoryCatalogue) -> None:
        story = catalogue.get_story("s2")
        assert story is not None
        assert story.title == "Dark Forest"

    def test_returns_none_for_unknown_id(self, catalogue: StoryCatalogue) -> None:
        assert catalogue.get_story("nonexistent") is None

    def test_before_first_refresh(self) -> None:
        stub = MagicMock()
        cat = StoryCatalogue(stub=stub)
        assert cat.get_story("s1") is None


class TestGetAllStories:
    def test_returns_list_of_story_objects(self, catalogue: StoryCatalogue) -> None:
        stories = catalogue.get_all_stories()
        assert all(isinstance(s, Story) for s in stories)

    def test_returns_copy(self, catalogue: StoryCatalogue) -> None:
        """Modifying the returned list should not affect the catalogue."""
        stories = catalogue.get_all_stories()
        stories.clear()
        assert len(catalogue.get_all_stories()) == 3


class TestGetAllThemes:
    def test_returns_sorted_unique_themes(self, catalogue: StoryCatalogue) -> None:
        themes = catalogue.get_all_themes()
        assert themes == sorted({"adventure", "mystery", "horror", "calm"})

    def test_empty_catalogue_returns_empty(self) -> None:
        stub = _make_stub([])
        cat = StoryCatalogue(stub=stub)
        cat.refresh()
        assert cat.get_all_themes() == []

    def test_stable_ordering(self, catalogue: StoryCatalogue) -> None:
        assert catalogue.get_all_themes() == catalogue.get_all_themes()


class TestGetAllTags:
    def test_returns_sorted_unique_tags(self, catalogue: StoryCatalogue) -> None:
        tags = catalogue.get_all_tags()
        assert tags == sorted({"pirates", "treasure", "trees", "shadows", "ocean"})

    def test_empty_catalogue_returns_empty(self) -> None:
        stub = _make_stub([])
        cat = StoryCatalogue(stub=stub)
        cat.refresh()
        assert cat.get_all_tags() == []

    def test_stable_ordering(self, catalogue: StoryCatalogue) -> None:
        assert catalogue.get_all_tags() == catalogue.get_all_tags()


class TestStartRefreshLoop:
    def test_starts_daemon_thread(self, sample_msgs) -> None:
        stub = _make_stub(sample_msgs)
        cat = StoryCatalogue(stub=stub, refresh_interval_seconds=9999)
        cat.refresh()
        cat.start_refresh_loop()
        assert cat._refresh_thread is not None
        assert cat._refresh_thread.is_alive()
        assert cat._refresh_thread.daemon is True

    def test_idempotent_multiple_calls(self, sample_msgs) -> None:
        stub = _make_stub(sample_msgs)
        cat = StoryCatalogue(stub=stub, refresh_interval_seconds=9999)
        cat.refresh()
        cat.start_refresh_loop()
        thread1 = cat._refresh_thread
        cat.start_refresh_loop()
        assert cat._refresh_thread is thread1  # same thread, not a new one
