"""Tests for cello_sampler.polyphony — polyphony detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cello_sampler.models import NoteCandidate
from cello_sampler.polyphony import is_polyphonic


SR = 48_000


def _make_candidate(audio: np.ndarray) -> NoteCandidate:
    """Wrap a mono audio array in a NoteCandidate with minimal metadata."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return NoteCandidate(
        audio=audio.astype(np.float32),
        sample_rate=SR,
        onset_sample=0,
        source_file=Path("test.wav"),
        candidate_index=0,
    )


def _pure_sine(freq: float, duration_s: float = 0.3, sr: int = SR) -> np.ndarray:
    """Generate a steady-state pure sine wave (no envelope — simulates middle of note)."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _two_sines(
    freq1: float,
    freq2: float,
    amp1: float = 0.5,
    amp2: float = 0.5,
    duration_s: float = 0.3,
    sr: int = SR,
) -> np.ndarray:
    """Superposition of two sine waves — simulates a double-stop or chord."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    s1 = amp1 * np.sin(2 * np.pi * freq1 * t)
    s2 = amp2 * np.sin(2 * np.pi * freq2 * t)
    return (s1 + s2).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-pitch notes — should NOT be flagged as polyphonic
# ---------------------------------------------------------------------------


class TestMonophonicSignals:
    def test_pure_sine_is_not_polyphonic(self) -> None:
        audio = _pure_sine(220.0)  # A3
        candidate = _make_candidate(audio)
        polyphonic, _ = is_polyphonic(candidate)
        assert not polyphonic, "Pure A3 sine should not be flagged as polyphonic"

    def test_cello_d_string(self) -> None:
        audio = _pure_sine(293.66)  # D4
        candidate = _make_candidate(audio)
        polyphonic, _ = is_polyphonic(candidate)
        assert not polyphonic, "Pure D4 sine should not be flagged as polyphonic"

    def test_low_c_string(self) -> None:
        audio = _pure_sine(65.41)  # C2 — lowest cello string
        candidate = _make_candidate(audio)
        polyphonic, _ = is_polyphonic(candidate)
        assert not polyphonic, "Pure C2 sine should not be flagged as polyphonic"

    def test_harmonic_series_is_not_polyphonic(self) -> None:
        """A realistic bowed tone with multiple harmonics is monophonic."""
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        f0 = 220.0
        # Add harmonics with decreasing amplitude — typical cello timbre.
        tone = sum(
            (1.0 / k) * np.sin(2 * np.pi * k * f0 * t)
            for k in range(1, 8)
        )
        tone = (tone / np.abs(tone).max() * 0.8).astype(np.float32)
        candidate = _make_candidate(tone)
        polyphonic, _ = is_polyphonic(candidate)
        assert not polyphonic, "Rich harmonic tone should not be flagged as polyphonic"


# ---------------------------------------------------------------------------
# Two-pitch signals — SHOULD be flagged as polyphonic
# ---------------------------------------------------------------------------


class TestPolyphonicSignals:
    def test_perfect_fifth_double_stop(self) -> None:
        """A3 + E4 (perfect fifth) — a common cello double stop."""
        audio = _two_sines(220.0, 329.63)  # A3 + E4
        candidate = _make_candidate(audio)
        polyphonic, detail = is_polyphonic(candidate)
        assert polyphonic, f"A3+E4 double stop should be polyphonic; detail: {detail}"

    def test_major_third_double_stop(self) -> None:
        """A3 + C#4 (major third)."""
        audio = _two_sines(220.0, 277.18)  # A3 + C#4
        candidate = _make_candidate(audio)
        polyphonic, detail = is_polyphonic(candidate)
        assert polyphonic, f"A3+C#4 major third should be polyphonic; detail: {detail}"

    def test_minor_seventh_interval(self) -> None:
        """G2 + F3 (minor seventh) — a wide interval between cello strings."""
        audio = _two_sines(98.0, 174.61)  # G2 + F3
        candidate = _make_candidate(audio)
        polyphonic, detail = is_polyphonic(candidate)
        assert polyphonic, f"G2+F3 should be polyphonic; detail: {detail}"

    def test_rejection_detail_is_nonempty(self) -> None:
        audio = _two_sines(220.0, 329.63)
        candidate = _make_candidate(audio)
        polyphonic, detail = is_polyphonic(candidate)
        if polyphonic:
            assert len(detail) > 0, "Rejection detail should explain the reason"

    def test_equal_amplitude_chord(self) -> None:
        """Two tones at equal amplitude — very clear polyphonic case."""
        audio = _two_sines(196.0, 293.66, amp1=0.5, amp2=0.5)  # G3 + D4
        candidate = _make_candidate(audio)
        polyphonic, _ = is_polyphonic(candidate)
        assert polyphonic


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_very_short_audio_does_not_crash(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        candidate = _make_candidate(audio)
        polyphonic, _ = is_polyphonic(candidate)
        assert isinstance(polyphonic, bool)

    def test_multichannel_audio_analysed_correctly(self) -> None:
        """4-channel audio should be mixed to mono before analysis."""
        mono = _pure_sine(220.0)
        audio_4ch = np.stack([mono, mono, mono, mono], axis=1)
        candidate = _make_candidate(audio_4ch)
        polyphonic, _ = is_polyphonic(candidate)
        assert not polyphonic
