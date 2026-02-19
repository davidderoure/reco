"""Shared pytest fixtures for all recommender tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recommender.models import Story, UserProfile


TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Story fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def story_adventure() -> Story:
    return Story("s_adv", "Sea Quest", ["adventure"], ["pirates", "ocean"])


@pytest.fixture
def story_mystery() -> Story:
    return Story("s_mys", "Hidden Room", ["mystery"], ["clues", "shadows"])


@pytest.fixture
def story_horror() -> Story:
    return Story("s_hor", "Dark Depths", ["horror"], ["monsters", "ocean"])


@pytest.fixture
def story_calm() -> Story:
    return Story("s_calm", "Quiet Lake", ["calm"], ["water", "peace"])


@pytest.fixture
def story_multi() -> Story:
    return Story("s_multi", "Dual Tale", ["adventure", "mystery"], ["treasure", "clues"])


@pytest.fixture
def sample_stories(
    story_adventure, story_mystery, story_horror, story_calm, story_multi
) -> list[Story]:
    """10-story catalogue spanning multiple themes and tags."""
    extra = [
        Story("s_extra1", "Night Voyage", ["adventure"], ["stars", "ocean"]),
        Story("s_extra2", "Whisper Wood", ["mystery"], ["trees", "clues"]),
        Story("s_extra3", "The Abyss", ["horror"], ["darkness", "monsters"]),
        Story("s_extra4", "Meadow Song", ["calm"], ["flowers", "peace"]),
        Story("s_extra5", "Treasure Isle", ["adventure", "calm"], ["pirates", "water"]),
    ]
    return [story_adventure, story_mystery, story_horror, story_calm, story_multi] + extra


# ---------------------------------------------------------------------------
# User profile fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def new_user_profile() -> UserProfile:
    """A brand-new user with no history (cold-start case)."""
    return UserProfile(user_id="u_new")


@pytest.fixture
def adventure_profile() -> UserProfile:
    """A user with a strong preference for adventure stories."""
    profile = UserProfile(user_id="u_adv")
    profile.viewed_story_ids = {"s_adv", "s_multi"}
    profile.completed_story_ids = {"s_adv"}
    profile.theme_weights = {"adventure": 3.0, "mystery": 1.0}
    profile.tag_weights = {"pirates": 3.0, "ocean": 3.0, "treasure": 1.0, "clues": 1.0}
    profile.story_scores = {"s_adv": 5, "s_multi": 4}
    return profile


@pytest.fixture
def mystery_profile() -> UserProfile:
    """A user with a strong preference for mystery stories."""
    profile = UserProfile(user_id="u_mys")
    profile.viewed_story_ids = {"s_mys", "s_multi"}
    profile.completed_story_ids = {"s_mys"}
    profile.theme_weights = {"mystery": 3.0, "adventure": 1.0}
    profile.tag_weights = {"clues": 3.0, "shadows": 3.0, "treasure": 1.0}
    profile.story_scores = {"s_mys": 5}
    return profile


@pytest.fixture
def all_themes(sample_stories) -> list[str]:
    themes: set[str] = set()
    for s in sample_stories:
        themes.update(s.themes)
    return sorted(themes)


@pytest.fixture
def all_tags(sample_stories) -> list[str]:
    tags: set[str] = set()
    for s in sample_stories:
        tags.update(s.tags)
    return sorted(tags)
