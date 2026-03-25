"""Tests for cello_sampler.onset — onset detection and note segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cello_sampler import config
from cello_sampler.onset import (
    detect_onsets,
    onset_strength,
    process_chunk,
    segment_notes,
    to_mono,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 48_000  # sample rate used in all tests


def _sine_burst(
    freq: float,
    duration_s: float,
    sr: int = SR,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a mono sine-wave burst with a sharp onset.

    Uses a 5 ms linear ramp-in so the spectral flux peaks immediately at the
    start of the burst (within one STFT hop at 48 kHz).  A full Hanning
    envelope would peak at the burst midpoint, causing the onset detector to
    report the wrong time.
    """
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    attack_n = min(int(0.005 * sr), n)       # 5 ms sharp ramp
    release_n = min(int(0.050 * sr), n // 4) # 50 ms Hanning release

    envelope = np.ones(n, dtype=np.float32)
    envelope[:attack_n] = np.linspace(0, 1, attack_n)
    if n > release_n:
        envelope[-release_n:] = np.hanning(release_n * 2)[release_n:]

    return (amplitude * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    """Return a block of silence."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _multi_burst_signal(
    onset_times_s: list[float],
    note_duration_s: float = 0.3,
    gap_s: float = 0.1,
    total_duration_s: float = 3.0,
    freq: float = 220.0,
    sr: int = SR,
) -> np.ndarray:
    """Build a mono signal containing sine bursts at specified onset times."""
    total_samples = int(total_duration_s * sr)
    signal = np.zeros(total_samples, dtype=np.float32)
    burst = _sine_burst(freq, note_duration_s, sr=sr)
    for onset_s in onset_times_s:
        start = int(onset_s * sr)
        end = min(total_samples, start + len(burst))
        signal[start:end] += burst[: end - start]
    return signal


# ---------------------------------------------------------------------------
# to_mono
# ---------------------------------------------------------------------------


class TestToMono:
    def test_mono_passthrough(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_mono(audio)
        np.testing.assert_array_almost_equal(result, audio)

    def test_multichannel_mean(self) -> None:
        audio = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        result = to_mono(audio)
        expected = np.array([2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_dtype_is_float32(self) -> None:
        audio = np.ones((100, 4), dtype=np.float64)
        result = to_mono(audio)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# onset_strength
# ---------------------------------------------------------------------------


class TestOnsetStrength:
    def test_burst_produces_high_strength(self) -> None:
        """A sudden sine burst should produce a clear onset strength spike."""
        silence = _silence(0.1)
        burst = _sine_burst(220.0, 0.2)
        signal = np.concatenate([silence, burst])

        strength = onset_strength(signal, SR)

        # The strength spike should be in the region corresponding to 0.1 s.
        hop = 512
        onset_hop = int(0.1 * SR / hop)
        window = strength[max(0, onset_hop - 5): onset_hop + 10]
        assert window.max() > strength[:onset_hop - 10].mean() * 3, (
            "Expected a clear onset strength spike at the burst start"
        )

    def test_silence_produces_near_zero_strength(self) -> None:
        signal = _silence(1.0)
        strength = onset_strength(signal, SR)
        assert strength.max() < 1e-6

    def test_output_shape_consistent_with_hop(self) -> None:
        n_samples = SR  # 1 second
        signal = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32)
        hop = 512
        strength = onset_strength(signal, SR, hop=hop)
        # STFT produces ceil(n / hop) frames approximately.
        expected_frames = n_samples // hop + 2
        assert len(strength) <= expected_frames + 5  # allow small rounding


# ---------------------------------------------------------------------------
# detect_onsets
# ---------------------------------------------------------------------------


class TestDetectOnsets:
    def test_detects_known_onset_times(self) -> None:
        """Each of three bursts at known positions must be detected within ±15 ms.

        The detector may also find additional peaks inside the steady-state
        region of a burst (spectral-flux ripple from non-integer STFT bin
        alignment).  The test checks the *closest* detection to each expected
        onset, which is the meaningful semantic property: every true note onset
        is captured, regardless of spurious extras within the same burst.
        """
        known_onsets_s = [0.2, 0.7, 1.3]
        signal = _multi_burst_signal(known_onsets_s, note_duration_s=0.25, total_duration_s=2.0)
        strength = onset_strength(signal, SR)
        onsets = detect_onsets(strength, hop=config.STFT_HOP, sample_rate=SR)

        assert len(onsets) >= len(known_onsets_s), (
            f"Expected at least {len(known_onsets_s)} onsets, got {len(onsets)}"
        )

        tolerance_samples = int(0.015 * SR)  # 15 ms — well within practical requirements
        for expected_s in known_onsets_s:
            expected_sample = int(expected_s * SR)
            closest = min(onsets, key=lambda x: abs(int(x) - expected_sample))
            assert abs(int(closest) - expected_sample) <= tolerance_samples, (
                f"No onset detected within 15 ms of {expected_s:.3f} s — "
                f"closest was {int(closest) / SR:.3f} s"
            )

    def test_minimum_gap_enforced(self) -> None:
        """Two bursts closer than min_note_ms should produce at most 1 onset."""
        # Onsets 30 ms apart — below default 50 ms min.
        signal = _multi_burst_signal([0.1, 0.13], note_duration_s=0.02, total_duration_s=1.0)
        strength = onset_strength(signal, SR)
        onsets = detect_onsets(strength, hop=config.STFT_HOP, sample_rate=SR, min_note_duration_ms=50.0)
        assert len(onsets) <= 1

    def test_silence_produces_no_onsets(self) -> None:
        signal = _silence(2.0)
        strength = onset_strength(signal, SR)
        onsets = detect_onsets(strength, hop=config.STFT_HOP, sample_rate=SR)
        assert len(onsets) == 0

    def test_returns_sorted_ascending(self) -> None:
        signal = _multi_burst_signal([0.5, 1.0, 1.5], total_duration_s=2.5)
        strength = onset_strength(signal, SR)
        onsets = detect_onsets(strength, hop=config.STFT_HOP, sample_rate=SR)
        assert list(onsets) == sorted(onsets)


# ---------------------------------------------------------------------------
# segment_notes
# ---------------------------------------------------------------------------


class TestSegmentNotes:
    def test_produces_correct_number_of_candidates(self) -> None:
        onset_times = [0.2, 0.6, 1.1]
        signal = _multi_burst_signal(onset_times, total_duration_s=1.8)
        audio_2ch = np.stack([signal, signal], axis=1)
        onset_samples = np.array([int(t * SR) for t in onset_times], dtype=np.int64)

        candidates = segment_notes(
            audio=audio_2ch,
            onset_samples=onset_samples,
            sample_rate=SR,
            source_file=Path("test.wav"),
        )
        assert len(candidates) == len(onset_times)

    def test_audio_shape_is_2d(self) -> None:
        signal = _multi_burst_signal([0.1], total_duration_s=0.8)
        onset_samples = np.array([int(0.1 * SR)], dtype=np.int64)

        candidates = segment_notes(
            audio=signal,  # 1-D input
            onset_samples=onset_samples,
            sample_rate=SR,
            source_file=Path("test.wav"),
        )
        assert candidates[0].audio.ndim == 2

    def test_onset_sample_is_absolute(self) -> None:
        signal = _multi_burst_signal([0.1], total_duration_s=0.5)
        onset_samples = np.array([int(0.1 * SR)], dtype=np.int64)
        chunk_start = int(5.0 * SR)   # pretend we're deep into the file

        candidates = segment_notes(
            audio=signal,
            onset_samples=onset_samples,
            sample_rate=SR,
            source_file=Path("test.wav"),
            chunk_start_sample=chunk_start,
        )
        expected_abs = chunk_start + int(0.1 * SR)
        assert candidates[0].onset_sample == expected_abs

    def test_max_duration_cap_respected(self) -> None:
        # Create a very long last-note region; max cap should truncate it.
        signal = _multi_burst_signal([0.1], total_duration_s=15.0)
        onset_samples = np.array([int(0.1 * SR)], dtype=np.int64)

        candidates = segment_notes(
            audio=signal,
            onset_samples=onset_samples,
            sample_rate=SR,
            source_file=Path("test.wav"),
            max_note_duration_seconds=4.0,
        )
        max_samples = int(4.0 * SR)
        assert candidates[0].audio.shape[0] <= max_samples


# ---------------------------------------------------------------------------
# process_chunk integration
# ---------------------------------------------------------------------------


class TestProcessChunk:
    def test_end_to_end_returns_candidates(self) -> None:
        onset_times = [0.3, 0.8, 1.4]
        signal = _multi_burst_signal(onset_times, total_duration_s=2.0)
        audio_2ch = np.stack([signal, signal], axis=1)

        candidates = process_chunk(
            audio=audio_2ch,
            sample_rate=SR,
            source_file=Path("session.wav"),
        )
        assert len(candidates) >= 1

    def test_source_file_propagated(self) -> None:
        signal = _multi_burst_signal([0.1], total_duration_s=0.5)
        audio_2ch = np.stack([signal, signal], axis=1)
        src = Path("cello_session_01.wav")

        candidates = process_chunk(audio=audio_2ch, sample_rate=SR, source_file=src)
        for c in candidates:
            assert c.source_file == src
