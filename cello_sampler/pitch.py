"""Pitch estimation and intonation assessment using CREPE.

CREPE (A Convolutional Representation for Pitch Estimation) is used because
it treats pitch detection as a 360-bin classification problem and is robust
to cello's variable harmonic balance — a property that confuses autocorrelation-
and YIN-based fundamental estimators, especially under heavy bow pressure.

CREPE operates internally at 16 kHz, so the mono mix is downsampled from
48 kHz before calling it.  The 48 kHz audio is never resampled for output.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from cello_sampler import config
from cello_sampler.models import NoteCandidate, PitchEstimate
from cello_sampler.onset import to_mono

logger = logging.getLogger(__name__)

# CREPE operates at 16 kHz.
_CREPE_SR = 16_000


def _downsample_to_crepe(mono_96k: np.ndarray, source_sr: int) -> np.ndarray:
    """Downsample a 48 kHz mono signal to 16 kHz using resampy (Kaiser-best).

    Args:
        mono_96k: 1-D float32 mono signal at *source_sr*.
        source_sr: Source sample rate (typically 48 000 Hz).

    Returns:
        1-D float32 array at 16 000 Hz.
    """
    try:
        import resampy  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "resampy is required for pitch estimation.  "
            "Install it with: pip install resampy"
        ) from exc

    return resampy.resample(
        mono_96k.astype(np.float64),
        source_sr,
        _CREPE_SR,
        filter="kaiser_best",
    ).astype(np.float32)


def _run_crepe(
    audio_16k: np.ndarray,
    step_size_ms: int = config.CREPE_STEP_SIZE_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CREPE pitch estimation on a 16 kHz mono signal.

    Args:
        audio_16k: 1-D float32 mono signal at 16 000 Hz.
        step_size_ms: Frame step size in milliseconds.

    Returns:
        Triple of ``(times, frequencies, confidences)`` — all 1-D float32
        arrays of the same length.
    """
    try:
        import crepe  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "crepe is required for pitch estimation.  "
            "Install it with: pip install crepe"
        ) from exc

    times, freqs, confidences, _ = crepe.predict(
        audio_16k,
        sr=_CREPE_SR,
        viterbi=True,
        step_size=step_size_ms,
        verbose=0,
    )
    return (
        times.astype(np.float32),
        freqs.astype(np.float32),
        confidences.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Pitch statistics helpers
# ---------------------------------------------------------------------------


def _stable_region_indices(n_frames: int, trim: float = config.PITCH_STABLE_REGION_TRIM) -> tuple[int, int]:
    """Return the ``(start, end)`` frame indices of the stable note region.

    The first and last *trim* fraction of frames are excluded to avoid attack
    and release transients contaminating the pitch statistics.

    Args:
        n_frames: Total number of CREPE frames.
        trim: Fraction to exclude from each end (default 0.20 = 20%).

    Returns:
        ``(start, end)`` pair with ``start < end``.
    """
    n_trim = max(0, int(n_frames * trim))
    start = n_trim
    end = max(start + 1, n_frames - n_trim)
    return start, end


def _hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if hz <= 0:
        return 0
    return int(round(12.0 * math.log2(hz / config.A4_HZ) + config.A4_MIDI))


def _nearest_et_hz(hz: float) -> float:
    """Return the equal-temperament frequency nearest to *hz*."""
    midi = _hz_to_midi(hz)
    return config.A4_HZ * (2.0 ** ((midi - config.A4_MIDI) / 12.0))


def _deviation_cents(hz: float, et_hz: float) -> float:
    """Return signed deviation in cents from *et_hz* to *hz*.

    Positive means sharp, negative means flat.

    Args:
        hz: Measured pitch in Hz.
        et_hz: Equal-temperament reference frequency in Hz.

    Returns:
        Deviation in cents.
    """
    if et_hz <= 0 or hz <= 0:
        return 0.0
    return 1200.0 * math.log2(hz / et_hz)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def estimate_pitch(
    candidate: NoteCandidate,
    confidence_threshold: float = config.PITCH_CONFIDENCE_THRESHOLD,
    max_deviation_cents: float = config.MAX_INTONATION_DEVIATION_CENTS,
) -> tuple[PitchEstimate | None, str]:
    """Estimate pitch for a note candidate using CREPE.

    Returns ``(None, reason)`` when the note is rejected due to low
    confidence or poor intonation; returns ``(PitchEstimate, "")`` on success.

    Args:
        candidate: The note candidate to analyse.
        confidence_threshold: Minimum median CREPE confidence to accept.
        max_deviation_cents: Maximum allowed deviation from 12-TET in cents.

    Returns:
        A ``(pitch_estimate_or_none, rejection_detail)`` pair.
    """
    mono = to_mono(candidate.audio)

    # Down-sample for CREPE.
    audio_16k = _downsample_to_crepe(mono, candidate.sample_rate)

    # Run CREPE.
    times, freqs, confidences = _run_crepe(audio_16k)

    n_frames = len(freqs)
    if n_frames == 0:
        return None, "CREPE returned no frames"

    stable_start, stable_end = _stable_region_indices(n_frames)
    stable_freqs = freqs[stable_start:stable_end]
    stable_confs = confidences[stable_start:stable_end]

    # Guard against degenerate short notes.
    if len(stable_freqs) == 0:
        stable_freqs = freqs
        stable_confs = confidences

    median_conf = float(np.median(stable_confs))
    if median_conf < confidence_threshold:
        return None, (
            f"CREPE confidence {median_conf:.3f} < threshold {confidence_threshold:.3f}"
        )

    # Filter out frames where CREPE is uncertain before computing median pitch.
    reliable_mask = stable_confs >= confidence_threshold
    if reliable_mask.sum() == 0:
        reliable_mask = np.ones(len(stable_freqs), dtype=bool)

    median_hz = float(np.median(stable_freqs[reliable_mask]))
    if median_hz <= 0:
        return None, "CREPE median frequency is zero or negative"

    midi_note = _hz_to_midi(median_hz)
    note_name = _hz_to_note_name(midi_note)
    et_hz = _nearest_et_hz(median_hz)
    dev_cents = _deviation_cents(median_hz, et_hz)

    if abs(dev_cents) > max_deviation_cents:
        return None, (
            f"Intonation deviation {dev_cents:+.1f} ¢ exceeds ±{max_deviation_cents:.0f} ¢"
        )

    # Pitch stability: variance of the reliable stable frames (in cents).
    cents_contour = np.array([
        _deviation_cents(float(f), et_hz) for f in stable_freqs[reliable_mask]
    ], dtype=np.float32)
    is_stable = bool(np.std(cents_contour) < max_deviation_cents)

    pitch = PitchEstimate(
        hz=median_hz,
        midi_note=midi_note,
        note_name=note_name,
        confidence=median_conf,
        deviation_cents=dev_cents,
        pitch_contour_hz=freqs,
        pitch_contour_times=times,
        is_stable=is_stable,
    )

    logger.debug(
        "Pitch: %s (%.1f Hz, %+.1f ¢, conf %.2f)",
        note_name, median_hz, dev_cents, median_conf,
    )
    return pitch, ""


def _hz_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to scientific pitch notation.

    This is a local copy that avoids a circular import with ``io``.
    """
    _NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    return f"{_NAMES[midi_note % 12]}{octave}"
