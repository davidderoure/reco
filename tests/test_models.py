"""Tests for recommender.models dataclasses."""

from datetime import datetime, timezone

import pytest

from recommender.models import EventType, Story, UserEvent, UserProfile


class TestStory:
    def test_basic_creation(self) -> None:
        story = Story(
            story_id="s1",
            title="The Lost Isles",
            themes=["adventure"],
            tags=["pirates", "treasure"],
        )
        assert story.story_id == "s1"
        assert story.title == "The Lost Isles"
        assert story.themes == ["adventure"]
        assert story.tags == ["pirates", "treasure"]

    def test_multiple_themes(self) -> None:
        story = Story(
            story_id="s2",
            title="Dark Maze",
            themes=["mystery", "horror"],
            tags=["labyrinth"],
        )
        assert len(story.themes) == 2
        assert "mystery" in story.themes
        assert "horror" in story.themes

    def test_empty_tags(self) -> None:
        story = Story(story_id="s3", title="Quiet Story", themes=["calm"], tags=[])
        assert story.tags == []

    def test_equality(self) -> None:
        s1 = Story(story_id="s1", title="A", themes=["x"], tags=["y"])
        s2 = Story(story_id="s1", title="A", themes=["x"], tags=["y"])
        assert s1 == s2

    def test_inequality(self) -> None:
        s1 = Story(story_id="s1", title="A", themes=["x"], tags=["y"])
        s2 = Story(story_id="s2", title="A", themes=["x"], tags=["y"])
        assert s1 != s2


class TestEventType:
    def test_values(self) -> None:
        assert EventType.VIEWED == "viewed"
        assert EventType.COMPLETED == "completed"
        assert EventType.SCORED == "scored"
        assert EventType.MOOD == "mood"

    def test_is_string(self) -> None:
        assert isinstance(EventType.VIEWED, str)

    def test_all_members(self) -> None:
        members = {e.value for e in EventType}
        assert members == {"viewed", "completed", "scored", "mood"}


class TestUserEvent:
    def test_viewed_event(self) -> None:
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        event = UserEvent(
            event_type=EventType.VIEWED,
            timestamp=ts,
            story_id="s1",
        )
        assert event.event_type == EventType.VIEWED
        assert event.story_id == "s1"
        assert event.score is None
        assert event.timestamp == ts

    def test_scored_event(self) -> None:
        ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
        event = UserEvent(
            event_type=EventType.SCORED,
            timestamp=ts,
            story_id="s1",
            score=4,
        )
        assert event.score == 4
        assert event.story_id == "s1"

    def test_mood_event_no_story(self) -> None:
        ts = datetime(2024, 1, 3, tzinfo=timezone.utc)
        event = UserEvent(
            event_type=EventType.MOOD,
            timestamp=ts,
            score=3,
        )
        assert event.story_id is None
        assert event.score == 3

    def test_defaults(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        event = UserEvent(event_type=EventType.VIEWED, timestamp=ts)
        assert event.story_id is None
        assert event.score is None


class TestUserProfile:
    def test_default_construction(self) -> None:
        profile = UserProfile(user_id="u1")
        assert profile.user_id == "u1"
        assert profile.events == []
        assert profile.viewed_story_ids == set()
        assert profile.completed_story_ids == set()
        assert profile.story_scores == {}
        assert profile.mood_scores == []
        assert profile.theme_weights == {}
        assert profile.tag_weights == {}

    def test_mutable_defaults_are_independent(self) -> None:
        """Each instance must get its own independent mutable collections."""
        p1 = UserProfile(user_id="u1")
        p2 = UserProfile(user_id="u2")
        p1.viewed_story_ids.add("s1")
        p1.theme_weights["adventure"] = 2.0
        assert "s1" not in p2.viewed_story_ids
        assert "adventure" not in p2.theme_weights

    def test_equality(self) -> None:
        p1 = UserProfile(user_id="u1")
        p2 = UserProfile(user_id="u1")
        assert p1 == p2

    def test_inequality_by_user_id(self) -> None:
        p1 = UserProfile(user_id="u1")
        p2 = UserProfile(user_id="u2")
        assert p1 != p2

    def test_accumulate_events(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        profile = UserProfile(user_id="u1")
        event = UserEvent(event_type=EventType.VIEWED, timestamp=ts, story_id="s1")
        profile.events.append(event)
        assert len(profile.events) == 1
        assert profile.events[0].story_id == "s1"
