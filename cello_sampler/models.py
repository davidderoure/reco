"""Core domain dataclasses for the cello sampler pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ArticulationType(str, Enum):
    """Cello articulation categories used to label output samples."""

    STACCATO = "staccato"
    LEGATO = "legato"
    VIBRATO = "vibrato"
    PIZZICATO = "pizzicato"


class RejectionReason(str, Enum):
    """Reason a note candidate was rejected from the accepted sample set."""

    POLYPHONIC = "polyphonic"
    INTONATION = "intonation"
    LOW_CONFIDENCE = "low_confidence"
    TOO_SHORT = "too_short"


# ---------------------------------------------------------------------------
# Pipeline data objects
# ---------------------------------------------------------------------------


@dataclass
class NoteCandidate:
    """A raw note segment extracted from the recording.

    Attributes:
        audio: Multi-channel audio array of shape ``(frames, channels)``
            at the source sample rate (typically 96 000 Hz).
        sample_rate: Sample rate of :attr:`audio` in Hz.
        onset_sample: Sample index of the onset within the *source file*
            (absolute, not relative to any chunk).
        source_file: Path to the original recording file.
        candidate_index: Zero-based index of this candidate within the
            processing run (used to generate output filenames).
    """

    audio: np.ndarray          # shape (frames, channels), float32
    sample_rate: int
    onset_sample: int
    source_file: Path
    candidate_index: int


@dataclass
class PitchEstimate:
    """Per-note pitch analysis result from CREPE.

    Attributes:
        hz: Median fundamental frequency over the stable region, in Hz.
        midi_note: Nearest MIDI note number (0–127).
        note_name: Scientific pitch notation string, e.g. ``"A3"``.
        confidence: Median CREPE confidence over the stable region (0–1).
        deviation_cents: Signed deviation from the nearest equal-temperament
            pitch in cents.  Positive = sharp, negative = flat.
        pitch_contour_hz: Per-frame pitch estimates (Hz) from CREPE.  Used
            downstream by the articulation classifier to detect vibrato.
        pitch_contour_times: Timestamps (seconds) corresponding to each frame
            in :attr:`pitch_contour_hz`.
        is_stable: ``True`` when the pitch variance over the stable region is
            low enough to suggest a clean, non-multiphonic tone.
    """

    hz: float
    midi_note: int
    note_name: str
    confidence: float
    deviation_cents: float
    pitch_contour_hz: np.ndarray   # shape (n_frames,)
    pitch_contour_times: np.ndarray
    is_stable: bool


@dataclass
class ArticulationFeatures:
    """Intermediate features computed before the articulation decision.

    Attributes:
        attack_duration_ms: Time from onset to -6 dB below peak amplitude.
        decay_rate_db_per_ms: Amplitude decay rate over the first 100 ms
            after peak, derived from a linear regression on the dB envelope.
        total_duration_ms: Total sounding duration above the noise floor.
        pitch_modulation_depth_cents: Standard deviation of pitch in cents
            over the stable region, used as a proxy for vibrato depth.
        pitch_modulation_rate_hz: Dominant frequency of pitch oscillation
            (Hz).  A value of 0.0 means no clear oscillation was detected.
    """

    attack_duration_ms: float
    decay_rate_db_per_ms: float
    total_duration_ms: float
    pitch_modulation_depth_cents: float
    pitch_modulation_rate_hz: float


@dataclass
class ClassifiedNote:
    """A fully analysed and accepted note sample ready for output.

    Attributes:
        candidate: The original :class:`NoteCandidate`.
        pitch: Pitch analysis result.
        articulation: Articulation type assigned by the classifier.
        features: The intermediate features that drove the classification
            (retained for debugging and CSV export).
        take_number: 1-based index among all accepted notes with the same
            pitch and articulation (used in the output filename).
    """

    candidate: NoteCandidate
    pitch: PitchEstimate
    articulation: ArticulationType
    features: ArticulationFeatures
    take_number: int = 1


@dataclass
class RejectedNote:
    """A note candidate that failed quality gating.

    Attributes:
        candidate: The original :class:`NoteCandidate`.
        reason: Why the note was rejected.
        detail: Human-readable explanation for logging and JSON sidecar.
    """

    candidate: NoteCandidate
    reason: RejectionReason
    detail: str = ""


@dataclass
class ProcessingResult:
    """Aggregate result returned by the full pipeline run.

    Attributes:
        accepted: All notes that passed quality gating and were classified.
        rejected: All notes that were discarded, with reasons.
        source_file: Path to the processed input file.
    """

    accepted: list[ClassifiedNote] = field(default_factory=list)
    rejected: list[RejectedNote] = field(default_factory=list)
    source_file: Path = field(default_factory=lambda: Path("."))

    @property
    def n_accepted(self) -> int:
        """Number of accepted notes."""
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        """Number of rejected notes."""
        return len(self.rejected)
