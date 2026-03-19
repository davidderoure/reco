"""Articulation feature extraction and rule-based classification.

Features are computed from the amplitude envelope (via the Hilbert analytic
signal) and from the CREPE pitch contour.  Classification uses an ordered
decision tree with interpretable, musically-grounded thresholds.

Decision order:
    1. Pizzicato — sharp attack, fast exponential decay, short total duration.
    2. Staccato  — short total duration, moderate decay rate.
    3. Vibrato   — sustained, with periodic pitch modulation at 4–8 Hz.
    4. Legato    — default for any remaining sustained bowed note.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import hilbert

from cello_sampler import config
from cello_sampler.models import (
    ArticulationFeatures,
    ArticulationType,
    NoteCandidate,
    PitchEstimate,
)
from cello_sampler.onset import to_mono

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Amplitude envelope helpers
# ---------------------------------------------------------------------------


def _amplitude_envelope(mono: np.ndarray) -> np.ndarray:
    """Compute the amplitude envelope using the Hilbert analytic signal.

    Args:
        mono: 1-D float32 mono audio.

    Returns:
        1-D float32 envelope array, same length as *mono*.
    """
    analytic = hilbert(mono.astype(np.float64))
    return np.abs(analytic).astype(np.float32)


def _db(amplitude: np.ndarray, reference: float) -> np.ndarray:
    """Convert amplitude to dB relative to *reference*.

    Args:
        amplitude: Linear amplitude values (must be > 0 for meaningful output).
        reference: Reference amplitude (peak value).

    Returns:
        Float64 array in dB.
    """
    eps = 1e-10
    return 20.0 * np.log10(np.maximum(amplitude, eps) / max(reference, eps))


def _attack_duration_ms(
    envelope: np.ndarray,
    sample_rate: int,
    attack_level_db: float = config.ATTACK_LEVEL_DB,
) -> float:
    """Measure time from the start to the first -6 dB point above peak.

    We scan forward from the beginning of the envelope until we find the
    amplitude peak, then measure how long it takes to rise from the start
    level to within *attack_level_db* of that peak.  For a strict impulse
    (pizzicato) the peak is at or near sample 0 and the "attack" is very short.

    Args:
        envelope: Amplitude envelope array.
        sample_rate: Sample rate in Hz.
        attack_level_db: Threshold level relative to peak (negative dB).

    Returns:
        Attack duration in milliseconds.
    """
    if len(envelope) == 0:
        return 0.0

    peak_idx = int(np.argmax(envelope))
    peak_amp = float(envelope[peak_idx])
    threshold_amp = peak_amp * (10.0 ** (attack_level_db / 20.0))

    # Scan from start to peak to find where we cross the threshold.
    onset_idx = 0
    for i in range(peak_idx):
        if envelope[i] >= threshold_amp:
            onset_idx = i
            break

    duration_samples = max(1, peak_idx - onset_idx)
    return 1000.0 * duration_samples / sample_rate


def _decay_rate_db_per_ms(
    envelope: np.ndarray,
    sample_rate: int,
    window_ms: float = config.DECAY_MEASUREMENT_WINDOW_MS,
) -> float:
    """Estimate the post-peak amplitude decay rate via linear regression.

    A higher value means the note decays faster (more staccato/pizzicato-like).

    Args:
        envelope: Amplitude envelope array.
        sample_rate: Sample rate in Hz.
        window_ms: Duration of the post-peak window to measure (ms).

    Returns:
        Decay rate in dB/ms (positive = decaying).
    """
    if len(envelope) < 4:
        return 0.0

    peak_idx = int(np.argmax(envelope))
    peak_amp = float(envelope[peak_idx])
    if peak_amp == 0.0:
        return 0.0

    window_samples = int(window_ms / 1000.0 * sample_rate)
    end_idx = min(len(envelope), peak_idx + window_samples + 1)

    if end_idx <= peak_idx + 2:
        return 0.0

    segment = envelope[peak_idx:end_idx]
    db_segment = _db(segment, peak_amp)
    t_ms = np.arange(len(db_segment), dtype=np.float64) * 1000.0 / sample_rate

    # Linear regression: dB = slope * t + intercept.  Slope is negative for
    # a decaying signal; we return the absolute rate.
    if len(t_ms) < 2:
        return 0.0
    coeffs = np.polyfit(t_ms, db_segment, 1)
    slope = float(coeffs[0])     # dB/ms — negative for decay
    return max(0.0, -slope)      # return positive decay rate


def _total_duration_ms(
    envelope: np.ndarray,
    sample_rate: int,
    noise_floor_db: float = config.NOISE_FLOOR_DB,
) -> float:
    """Measure the sounding duration above the noise floor.

    Args:
        envelope: Amplitude envelope array.
        sample_rate: Sample rate in Hz.
        noise_floor_db: Level in dB relative to peak below which the note
            is considered silent.

    Returns:
        Total sounding duration in milliseconds.
    """
    if len(envelope) == 0:
        return 0.0

    peak_amp = float(envelope.max())
    if peak_amp == 0.0:
        return 0.0

    threshold_amp = peak_amp * (10.0 ** (noise_floor_db / 20.0))
    above = envelope >= threshold_amp
    n_sounding = int(above.sum())
    return 1000.0 * n_sounding / sample_rate


# ---------------------------------------------------------------------------
# Vibrato analysis helpers
# ---------------------------------------------------------------------------


def _stable_pitch_contour_cents(
    pitch: PitchEstimate,
    reference_hz: float,
) -> np.ndarray:
    """Extract the stable-region pitch contour in cents relative to *reference_hz*.

    Args:
        pitch: PitchEstimate with ``pitch_contour_hz`` and ``pitch_contour_times``.
        reference_hz: Reference frequency (median pitch, Hz).

    Returns:
        1-D float32 array of pitch values in cents, stable region only.
    """
    n = len(pitch.pitch_contour_hz)
    if n == 0:
        return np.array([], dtype=np.float32)

    trim = config.PITCH_STABLE_REGION_TRIM
    n_trim = max(1, int(n * trim))
    start = n_trim
    end = max(start + 1, n - n_trim)
    contour_hz = pitch.pitch_contour_hz[start:end]

    cents = np.array(
        [
            1200.0 * math.log2(max(f, 1e-6) / max(reference_hz, 1e-6))
            for f in contour_hz
        ],
        dtype=np.float32,
    )
    return cents


def _pitch_modulation_features(
    pitch: PitchEstimate,
) -> tuple[float, float]:
    """Compute vibrato depth (cents std dev) and rate (Hz) from the pitch contour.

    Args:
        pitch: PitchEstimate containing the CREPE pitch contour.

    Returns:
        ``(depth_cents, rate_hz)`` pair.  ``rate_hz`` is 0.0 if no clear
        oscillation is found in the vibrato range.
    """
    cents = _stable_pitch_contour_cents(pitch, pitch.hz)
    if len(cents) < 8:
        return 0.0, 0.0

    depth_cents = float(np.std(cents))

    # Estimate the vibrato rate via FFT of the pitch contour.
    # The CREPE step size determines the frame rate.
    frame_rate_hz = 1000.0 / config.CREPE_STEP_SIZE_MS

    n = len(cents)
    # Zero-mean the contour before FFT to remove DC offset.
    fft_mag = np.abs(np.fft.rfft(cents - cents.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate_hz)

    # Look for the dominant peak within the vibrato frequency range.
    vibrato_mask = (freqs >= config.VIBRATO_MIN_RATE_HZ) & (freqs <= config.VIBRATO_MAX_RATE_HZ)
    if not vibrato_mask.any():
        return depth_cents, 0.0

    vibrato_magnitudes = fft_mag[vibrato_mask]
    vibrato_freqs = freqs[vibrato_mask]

    # Only call it a vibrato rate if the in-band peak is dominant.
    in_band_peak = float(vibrato_magnitudes.max())
    total_energy = float(fft_mag[1:].max())  # exclude DC

    if total_energy == 0.0 or in_band_peak / total_energy < 0.30:
        return depth_cents, 0.0

    rate_hz = float(vibrato_freqs[np.argmax(vibrato_magnitudes)])
    return depth_cents, rate_hz


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    candidate: NoteCandidate,
    pitch: PitchEstimate,
) -> ArticulationFeatures:
    """Compute all articulation-related features for a note candidate.

    Args:
        candidate: The note candidate (provides audio and sample rate).
        pitch: Pitch estimate (provides the CREPE pitch contour for vibrato
            analysis).

    Returns:
        :class:`~cello_sampler.models.ArticulationFeatures` instance.
    """
    mono = to_mono(candidate.audio)
    envelope = _amplitude_envelope(mono)

    attack_ms = _attack_duration_ms(envelope, candidate.sample_rate)
    decay_rate = _decay_rate_db_per_ms(envelope, candidate.sample_rate)
    duration_ms = _total_duration_ms(envelope, candidate.sample_rate)
    depth_cents, rate_hz = _pitch_modulation_features(pitch)

    return ArticulationFeatures(
        attack_duration_ms=attack_ms,
        decay_rate_db_per_ms=decay_rate,
        total_duration_ms=duration_ms,
        pitch_modulation_depth_cents=depth_cents,
        pitch_modulation_rate_hz=rate_hz,
    )


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------


def classify(features: ArticulationFeatures) -> ArticulationType:
    """Apply the ordered rule-based decision tree to return an articulation type.

    Rules are applied in the order below.  The first matching rule wins.

    **Pizzicato** (plucked):
        Sharp attack (<15 ms), fast post-peak decay (>0.5 dB/ms),
        short total duration (<400 ms).

    **Staccato** (short bowed):
        Short total duration (<250 ms), moderate decay rate (>0.2 dB/ms).
        Checked *after* pizzicato to avoid misclassifying very short plucks.

    **Vibrato** (sustained with pitch oscillation):
        Pitch modulation depth >20 ¢ *and* oscillation rate in [4, 8] Hz.

    **Legato** (default):
        All remaining sustained bowed notes.

    Args:
        features: Pre-computed :class:`~cello_sampler.models.ArticulationFeatures`.

    Returns:
        :class:`~cello_sampler.models.ArticulationType` label.
    """
    f = features

    # 1. Pizzicato
    if (
        f.attack_duration_ms < config.PIZZICATO_MAX_ATTACK_MS
        and f.decay_rate_db_per_ms > config.PIZZICATO_MIN_DECAY_RATE_DB_PER_MS
        and f.total_duration_ms < config.PIZZICATO_MAX_DURATION_MS
    ):
        logger.debug(
            "Classified as PIZZICATO (attack=%.1f ms, decay=%.3f dB/ms, dur=%.1f ms)",
            f.attack_duration_ms, f.decay_rate_db_per_ms, f.total_duration_ms,
        )
        return ArticulationType.PIZZICATO

    # 2. Staccato
    if (
        f.total_duration_ms < config.STACCATO_MAX_DURATION_MS
        and f.decay_rate_db_per_ms > config.STACCATO_MIN_DECAY_RATE_DB_PER_MS
    ):
        logger.debug(
            "Classified as STACCATO (dur=%.1f ms, decay=%.3f dB/ms)",
            f.total_duration_ms, f.decay_rate_db_per_ms,
        )
        return ArticulationType.STACCATO

    # 3. Vibrato
    if (
        f.pitch_modulation_depth_cents > config.VIBRATO_MIN_DEPTH_CENTS
        and config.VIBRATO_MIN_RATE_HZ
        <= f.pitch_modulation_rate_hz
        <= config.VIBRATO_MAX_RATE_HZ
    ):
        logger.debug(
            "Classified as VIBRATO (depth=%.1f ¢, rate=%.1f Hz)",
            f.pitch_modulation_depth_cents, f.pitch_modulation_rate_hz,
        )
        return ArticulationType.VIBRATO

    # 4. Legato (default)
    logger.debug(
        "Classified as LEGATO (dur=%.1f ms, depth=%.1f ¢)",
        f.total_duration_ms, f.pitch_modulation_depth_cents,
    )
    return ArticulationType.LEGATO
