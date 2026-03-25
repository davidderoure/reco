"""Tests for cello_sampler.articulation — feature extraction and classification.

All test signals are synthesised in-memory.  No audio files are required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cello_sampler.articulation import (
    _amplitude_envelope,
    _attack_duration_ms,
    _decay_rate_db_per_ms,
    _total_duration_ms,
    classify,
    extract_features,
)
from cello_sampler.models import (
    ArticulationFeatures,
    ArticulationType,
    NoteCandidate,
    PitchEstimate,
)


SR = 48_000


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


def _make_candidate(audio: np.ndarray, sr: int = SR) -> NoteCandidate:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    return NoteCandidate(
        audio=audio.astype(np.float32),
        sample_rate=sr,
        onset_sample=0,
        source_file=Path("test.wav"),
        candidate_index=0,
    )


def _make_pitch(hz: float = 220.0, n_frames: int = 100) -> PitchEstimate:
    """Create a PitchEstimate with a flat pitch contour at *hz*."""
    times = np.linspace(0, n_frames * 0.01, n_frames, dtype=np.float32)
    freqs = np.full(n_frames, hz, dtype=np.float32)
    return PitchEstimate(
        hz=hz,
        midi_note=57,
        note_name="A3",
        confidence=0.95,
        deviation_cents=0.0,
        pitch_contour_hz=freqs,
        pitch_contour_times=times,
        is_stable=True,
    )


def _make_vibrato_pitch(
    hz: float = 220.0,
    depth_cents: float = 40.0,
    rate_hz: float = 6.0,
    duration_s: float = 1.0,
    step_ms: int = 10,
) -> PitchEstimate:
    """Create a PitchEstimate with sinusoidal pitch modulation (vibrato)."""
    dt = step_ms / 1000.0
    n_frames = int(duration_s / dt)
    times = np.arange(n_frames, dtype=np.float32) * dt

    # Depth in Hz: cents → ratio → Hz.
    depth_hz = hz * (2.0 ** (depth_cents / 1200.0) - 1.0)
    freqs = (hz + depth_hz * np.sin(2 * np.pi * rate_hz * times)).astype(np.float32)

    return PitchEstimate(
        hz=hz,
        midi_note=57,
        note_name="A3",
        confidence=0.95,
        deviation_cents=0.0,
        pitch_contour_hz=freqs,
        pitch_contour_times=times,
        is_stable=True,
    )


def _pizzicato_signal(sr: int = SR) -> np.ndarray:
    """Sharply-attacked, rapidly-decaying signal (plucked string approximation).

    Decay rate 70 /s → exp(-70 * 0.1) = exp(-7) ≈ -61 dB in 100 ms
    → 0.61 dB/ms, which exceeds the PIZZICATO_MIN_DECAY_RATE_DB_PER_MS = 0.50.
    """
    n = int(0.25 * sr)  # 250 ms total window
    t = np.arange(n) / sr
    decay_rate = 70.0   # fast exponential decay — clearly pizzicato
    env = np.exp(-decay_rate * t).astype(np.float32)
    return env * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)


def _staccato_signal(sr: int = SR) -> np.ndarray:
    """Short bowed note: moderate attack, moderate decay, short duration."""
    n = int(0.15 * sr)  # 150 ms
    t = np.arange(n) / sr
    # Ramp up over first 20 ms, then decay at moderate rate.
    attack_n = int(0.020 * sr)
    envelope = np.ones(n, dtype=np.float32)
    envelope[:attack_n] = np.linspace(0, 1, attack_n)
    decay_start = attack_n
    envelope[decay_start:] *= np.exp(-8.0 * (t[decay_start:] - t[decay_start]))
    return envelope * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)


def _legato_signal(sr: int = SR) -> np.ndarray:
    """Sustained bowed note with slow attack and gentle decay."""
    n = int(1.5 * sr)  # 1.5 s
    t = np.arange(n) / sr
    attack_n = int(0.05 * sr)  # 50 ms attack
    envelope = np.ones(n, dtype=np.float32)
    envelope[:attack_n] = np.linspace(0, 1, attack_n)
    # Very slow decay.
    envelope[attack_n:] *= np.exp(-0.5 * (t[attack_n:] - t[attack_n]))
    return envelope * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Amplitude envelope helpers
# ---------------------------------------------------------------------------


class TestAmplitudeEnvelope:
    def test_pure_sine_envelope_is_smooth(self) -> None:
        n = int(0.1 * SR)
        t = np.arange(n) / SR
        signal = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        env = _amplitude_envelope(signal)
        # The Hilbert envelope of a pure sine should be approximately constant.
        assert env.std() / env.mean() < 0.05

    def test_envelope_positive(self) -> None:
        signal = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
        env = _amplitude_envelope(signal)
        assert (env >= 0).all()

    def test_envelope_length_matches_input(self) -> None:
        signal = np.ones(4096, dtype=np.float32)
        env = _amplitude_envelope(signal)
        assert len(env) == len(signal)


class TestAttackDuration:
    def test_impulse_has_near_zero_attack(self) -> None:
        """An impulse (peak at sample 0) should have essentially zero attack."""
        n = int(0.1 * SR)
        impulse = np.zeros(n, dtype=np.float32)
        impulse[0] = 1.0
        ms = _attack_duration_ms(impulse, SR)
        assert ms < 5.0, f"Impulse attack should be <5 ms, got {ms:.1f} ms"

    def test_slow_attack_gives_longer_duration(self) -> None:
        n = int(0.2 * SR)
        t = np.arange(n) / SR
        ramp = (t / t[-1]).astype(np.float32)  # linear ramp to peak
        fast_attack = np.zeros(n, dtype=np.float32)
        fast_attack[0] = 1.0

        slow_ms = _attack_duration_ms(ramp, SR)
        fast_ms = _attack_duration_ms(fast_attack, SR)
        assert slow_ms > fast_ms

    def test_empty_input_returns_zero(self) -> None:
        assert _attack_duration_ms(np.array([], dtype=np.float32), SR) == 0.0


class TestDecayRate:
    def test_fast_decay_gives_high_rate(self) -> None:
        n = int(0.3 * SR)
        t = np.arange(n) / SR
        fast = np.exp(-30.0 * t).astype(np.float32)
        slow = np.exp(-1.0 * t).astype(np.float32)
        assert _decay_rate_db_per_ms(fast, SR) > _decay_rate_db_per_ms(slow, SR)

    def test_flat_signal_gives_near_zero_rate(self) -> None:
        signal = np.ones(SR, dtype=np.float32)
        rate = _decay_rate_db_per_ms(signal, SR)
        assert rate < 0.01


class TestTotalDuration:
    def test_silence_gives_zero_duration(self) -> None:
        signal = np.zeros(SR, dtype=np.float32)
        assert _total_duration_ms(signal, SR) == 0.0

    def test_sustained_tone_gives_correct_duration(self) -> None:
        n = int(0.5 * SR)
        signal = 0.8 * np.ones(n, dtype=np.float32)
        ms = _total_duration_ms(signal, SR)
        assert abs(ms - 500.0) < 10.0  # within 10 ms


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_classify_pizzicato(self) -> None:
        features = ArticulationFeatures(
            attack_duration_ms=5.0,
            decay_rate_db_per_ms=0.8,
            total_duration_ms=150.0,
            pitch_modulation_depth_cents=5.0,
            pitch_modulation_rate_hz=0.0,
        )
        assert classify(features) == ArticulationType.PIZZICATO

    def test_classify_staccato(self) -> None:
        features = ArticulationFeatures(
            attack_duration_ms=20.0,    # too slow for pizzicato
            decay_rate_db_per_ms=0.3,
            total_duration_ms=180.0,    # short
            pitch_modulation_depth_cents=5.0,
            pitch_modulation_rate_hz=0.0,
        )
        assert classify(features) == ArticulationType.STACCATO

    def test_classify_vibrato(self) -> None:
        features = ArticulationFeatures(
            attack_duration_ms=50.0,
            decay_rate_db_per_ms=0.05,
            total_duration_ms=1200.0,
            pitch_modulation_depth_cents=35.0,  # deep vibrato
            pitch_modulation_rate_hz=6.0,       # 6 Hz rate
        )
        assert classify(features) == ArticulationType.VIBRATO

    def test_classify_legato(self) -> None:
        features = ArticulationFeatures(
            attack_duration_ms=50.0,
            decay_rate_db_per_ms=0.02,
            total_duration_ms=2000.0,   # long sustained
            pitch_modulation_depth_cents=5.0,   # no significant vibrato
            pitch_modulation_rate_hz=0.0,
        )
        assert classify(features) == ArticulationType.LEGATO

    def test_pizzicato_checked_before_staccato(self) -> None:
        """A short note with very fast decay should be pizzicato, not staccato."""
        features = ArticulationFeatures(
            attack_duration_ms=4.0,     # pizzicato attack
            decay_rate_db_per_ms=1.0,   # faster than both thresholds
            total_duration_ms=100.0,    # also qualifies as staccato
            pitch_modulation_depth_cents=0.0,
            pitch_modulation_rate_hz=0.0,
        )
        assert classify(features) == ArticulationType.PIZZICATO

    def test_vibrato_rate_out_of_range_gives_legato(self) -> None:
        """Deep pitch wobble at 2 Hz (too slow) should not be classed as vibrato."""
        features = ArticulationFeatures(
            attack_duration_ms=50.0,
            decay_rate_db_per_ms=0.02,
            total_duration_ms=1500.0,
            pitch_modulation_depth_cents=40.0,
            pitch_modulation_rate_hz=2.0,  # below 4 Hz vibrato range
        )
        assert classify(features) == ArticulationType.LEGATO


# ---------------------------------------------------------------------------
# extract_features integration (uses synthesised audio + fake pitch)
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_pizzicato_features_from_decaying_signal(self) -> None:
        audio = _pizzicato_signal()
        candidate = _make_candidate(audio)
        pitch = _make_pitch()
        features = extract_features(candidate, pitch)

        assert features.attack_duration_ms < 20.0, (
            f"Pizzicato attack expected <20 ms, got {features.attack_duration_ms:.1f}"
        )
        assert features.decay_rate_db_per_ms > 0.1, (
            f"Pizzicato decay rate expected >0.1 dB/ms, got {features.decay_rate_db_per_ms:.3f}"
        )

    def test_legato_features_from_sustained_signal(self) -> None:
        audio = _legato_signal()
        candidate = _make_candidate(audio)
        pitch = _make_pitch()
        features = extract_features(candidate, pitch)

        assert features.total_duration_ms > 500.0, (
            f"Legato duration expected >500 ms, got {features.total_duration_ms:.1f}"
        )
        assert features.decay_rate_db_per_ms < 0.3, (
            f"Legato decay rate expected <0.3 dB/ms, got {features.decay_rate_db_per_ms:.3f}"
        )

    def test_vibrato_pitch_modulation_detected(self) -> None:
        audio = _legato_signal()
        candidate = _make_candidate(audio)
        pitch = _make_vibrato_pitch(depth_cents=40.0, rate_hz=6.0)
        features = extract_features(candidate, pitch)

        assert features.pitch_modulation_depth_cents > 15.0, (
            "Vibrato depth should be significant"
        )
        assert 4.0 <= features.pitch_modulation_rate_hz <= 8.0, (
            f"Vibrato rate {features.pitch_modulation_rate_hz:.1f} Hz outside [4, 8] Hz"
        )

    def test_full_pipeline_pizzicato(self) -> None:
        """extract_features + classify round-trip for synthesised pizzicato."""
        audio = _pizzicato_signal()
        candidate = _make_candidate(audio)
        pitch = _make_pitch()
        features = extract_features(candidate, pitch)
        label = classify(features)
        assert label == ArticulationType.PIZZICATO

    def test_full_pipeline_vibrato(self) -> None:
        """extract_features + classify round-trip for synthesised vibrato."""
        audio = _legato_signal()
        candidate = _make_candidate(audio)
        pitch = _make_vibrato_pitch(depth_cents=50.0, rate_hz=6.0)
        features = extract_features(candidate, pitch)
        label = classify(features)
        assert label == ArticulationType.VIBRATO
