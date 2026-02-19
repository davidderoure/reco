"""Core domain dataclasses shared across all recommender modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Categories of user interaction events tracked by the system."""

    VIEWED = "viewed"
    COMPLETED = "completed"
    SCORED = "scored"
    MOOD = "mood"


@dataclass
class Story:
    """A single story in the catalogue.

    Attributes:
        story_id: Unique identifier for the story.
        title: Human-readable story title.
        themes: Broad category labels (e.g. ``"adventure"``, ``"mystery"``).
            A story belongs to one or more of ~10 themes.
        tags: Fine-grained content tags (e.g. ``"pirates"``, ``"treasure"``).
            Used for topical recommendations.
    """

    story_id: str
    title: str
    themes: list[str]
    tags: list[str]


@dataclass
class UserEvent:
    """A single recorded user interaction.

    Attributes:
        event_type: The category of interaction.
        timestamp: When the event occurred (UTC).
        story_id: The story involved. ``None`` for mood-only events.
        score: Numeric score (1–5) for :attr:`EventType.SCORED` and
            :attr:`EventType.MOOD` events; ``None`` otherwise.
    """

    event_type: EventType
    timestamp: datetime
    story_id: str | None = None
    score: int | None = None


@dataclass
class UserProfile:
    """Accumulated state and preference model for a single user.

    This is the primary in-memory record for a user. It is built
    incrementally from :class:`UserEvent` objects and stores pre-computed
    preference weight dictionaries for fast recommendation lookups.

    Weight accumulation rules (applied by :class:`~recommender.user_state.UserStateStore`):

    ========  ===========================================
    Event     Theme / tag weight delta
    ========  ===========================================
    Viewed    +1.0 per theme/tag
    Completed +2.0 bonus (additive with Viewed)
    Scored    ``(score - 3) × 0.5`` per theme/tag
    Mood      Stored only; no weight impact
    ========  ===========================================

    Attributes:
        user_id: Unique identifier for the user.
        events: Chronological list of all raw events (used for persistence).
        viewed_story_ids: Stories the user has viewed.
        completed_story_ids: Stories the user has completed.
        story_scores: Map from ``story_id`` to the user's end-of-story score (1–5).
        mood_scores: Chronological ``(timestamp, mood_score)`` pairs.
        theme_weights: Accumulated preference weight per theme label.
        tag_weights: Accumulated preference weight per tag label.
    """

    user_id: str
    events: list[UserEvent] = field(default_factory=list)
    viewed_story_ids: set[str] = field(default_factory=set)
    completed_story_ids: set[str] = field(default_factory=set)
    story_scores: dict[str, int] = field(default_factory=dict)
    mood_scores: list[tuple[datetime, int]] = field(default_factory=list)
    theme_weights: dict[str, float] = field(default_factory=dict)
    tag_weights: dict[str, float] = field(default_factory=dict)
