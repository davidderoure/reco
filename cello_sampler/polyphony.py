"""Polyphony detection for note candidates.

Two complementary methods are combined:

1. **Harmonic Product Spectrum (HPS)**: downsamples the magnitude spectrum
   and multiplies successive octave-spaced versions.  Positions where a
   harmonic series sums constructively produce a strong peak.  Multiple
   strong, well-separated HPS peaks indicate multiple simultaneous voices.

2. **Multi-pitch salience map**: for each candidate MIDI pitch, sums spectral
   energy at its fundamental and upper harmonics.  If two distinct pitches
   each have high relative salience, the note is polyphonic.

Both checks must pass (neither flags polyphony) for a candidate to be
accepted.  This keeps the false-positive rate low on notes with strong
sympathetic resonance from open strings.
"""

from __future__ import annotations

import logging

import numpy as np

from cello_sampler import config
from cello_sampler.models import NoteCandidate
from cello_sampler.onset import to_mono

logger = logging.getLogger(__name__)

# Cello pitch range in MIDI notes: C2 (36) to A6 (93).
_CELLO_MIDI_MIN = 36
_CELLO_MIDI_MAX = 93

_A4_HZ = config.A4_HZ
_A4_MIDI = config.A4_MIDI


def _midi_to_hz(midi: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return _A4_HZ * (2.0 ** ((midi - _A4_MIDI) / 12.0))


def _analysis_frame(
    mono: np.ndarray,
    sample_rate: int,
    max_duration_ms: float = config.POLYPHONY_ANALYSIS_DURATION_MS,
) -> np.ndarray:
    """Return a Hanning-windowed analysis frame for polyphony detection.

    Uses as much of the available audio as possible up to *max_duration_ms*.
    Longer windows give finer FFT frequency resolution, reducing the Hanning
    spectral-leakage contribution of the dominant pitch into adjacent pitch
    bins, which would otherwise cause false-positive polyphony detections on
    monophonic notes.

    Short notes (fewer than 16 samples) are padded with zeros.

    Args:
        mono: 1-D mono audio array.
        sample_rate: Sample rate in Hz.
        max_duration_ms: Maximum analysis window length in milliseconds.

    Returns:
        Hanning-windowed 1-D float32 frame.
    """
    n_max = int(max_duration_ms / 1000.0 * sample_rate)
    n = max(16, min(len(mono), n_max))
    frame = np.zeros(n, dtype=np.float32)
    frame[: min(len(mono), n)] = mono[: min(len(mono), n)]
    return frame * np.hanning(n)


def _magnitude_spectrum(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freqs, magnitude)`` for the given windowed frame."""
    n = len(frame)
    magnitude = np.abs(np.fft.rfft(frame, n=n)).astype(np.float64)
    freqs = np.fft.rfftfreq(n)  # normalised 0–0.5; multiply by sr to get Hz
    return freqs, magnitude


# ---------------------------------------------------------------------------
# HPS check
# ---------------------------------------------------------------------------


def _hps_peaks(
    magnitude: np.ndarray,
    order: int = config.HPS_ORDER,
    n_peaks: int = 3,
) -> list[int]:
    """Compute the Harmonic Product Spectrum and return the top-*n_peaks* bin indices.

    Returns an empty list when the input has insufficient harmonic content for
    a reliable analysis (e.g. pure sine waves, silence).

    Args:
        magnitude: Magnitude spectrum (one-sided, linear scale).
        order: Number of harmonic downsampling stages.
        n_peaks: Maximum number of peaks to return.

    Returns:
        List of bin indices corresponding to HPS peaks, strongest first.
    """
    hps = magnitude.copy()
    for k in range(2, order + 1):
        # Downsample by k: take every k-th bin, truncated to same length.
        downsampled = magnitude[::k][: len(hps)]
        hps[: len(downsampled)] *= downsampled

    # For a pure sine (no harmonics) the HPS is effectively zero everywhere
    # because the product includes near-zero factors.  Guard against this by
    # requiring the HPS peak to be a meaningful fraction of what a fully
    # harmonic signal would produce (magnitude_max^order).
    magnitude_max = magnitude.max()
    if magnitude_max == 0.0:
        return []
    minimum_meaningful_hps = (magnitude_max * 0.05) ** order
    if hps.max() < minimum_meaningful_hps:
        return []  # insufficient harmonic content for HPS analysis

    # Simple peak picking: local maxima above 1% of global max.
    peak_threshold = hps.max() * 0.01
    peaks: list[tuple[float, int]] = []
    for i in range(1, len(hps) - 1):
        if hps[i] > hps[i - 1] and hps[i] > hps[i + 1] and hps[i] > peak_threshold:
            peaks.append((hps[i], i))

    peaks.sort(reverse=True)
    return [idx for _, idx in peaks[:n_peaks]]


def _is_polyphonic_hps(
    magnitude: np.ndarray,
    freqs_normalised: np.ndarray,
    sample_rate: int,
    order: int = config.HPS_ORDER,
    unison_ratio_tol: float = config.POLYPHONY_UNISON_RATIO_TOLERANCE,
) -> bool:
    """Return ``True`` if the HPS shows two or more distinct strong pitches.

    Args:
        magnitude: One-sided magnitude spectrum.
        freqs_normalised: Normalised frequencies (0–0.5) matching magnitude.
        sample_rate: Sample rate in Hz (used to convert bins to Hz).
        order: HPS order.
        unison_ratio_tol: Peaks with a frequency ratio within this tolerance
            of 1.0 are considered the same pitch.

    Returns:
        ``True`` if polyphony is detected.
    """
    peak_bins = _hps_peaks(magnitude, order=order, n_peaks=3)
    if len(peak_bins) < 2:
        return False

    # Convert to Hz.
    bin_hz = freqs_normalised * sample_rate
    peak_hz = [bin_hz[b] for b in peak_bins if b < len(bin_hz) and bin_hz[b] > 30.0]

    if len(peak_hz) < 2:
        return False

    dominant = peak_hz[0]
    for candidate_hz in peak_hz[1:]:
        if dominant == 0.0:
            continue
        ratio = candidate_hz / dominant
        # Check that the candidate is not just an octave / partial harmonic
        # (ratio close to a small integer) and not within unison tolerance.
        is_unison = abs(ratio - 1.0) < unison_ratio_tol
        is_octave_harmonic = any(
            abs(ratio - k) < unison_ratio_tol for k in [2.0, 3.0, 4.0, 0.5, 0.333]
        )
        if not is_unison and not is_octave_harmonic:
            logger.debug(
                "HPS: second peak at %.1f Hz (ratio %.3f vs dominant %.1f Hz) → polyphonic",
                candidate_hz, ratio, dominant,
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Multi-pitch salience check
# ---------------------------------------------------------------------------


def _pitch_salience_map(
    magnitude: np.ndarray,
    freqs_normalised: np.ndarray,
    sample_rate: int,
    n_harmonics: int = config.POLYPHONY_N_HARMONICS,
) -> dict[int, float]:
    """Build a salience value for each MIDI pitch in the cello range.

    Salience is the weighted sum of spectral energy at the pitch's harmonics:
    ``salience(p) = Σ_k  magnitude[bin(k * f_p)] / k``  for k = 1 … n_harmonics.

    Args:
        magnitude: One-sided magnitude spectrum (linear scale).
        freqs_normalised: Normalised frequencies corresponding to ``magnitude``.
        sample_rate: Sample rate in Hz.
        n_harmonics: Number of harmonics to accumulate.

    Returns:
        Dict mapping MIDI note number → salience value (un-normalised).
    """
    bin_hz = freqs_normalised * sample_rate
    bin_spacing = bin_hz[1] - bin_hz[0] if len(bin_hz) > 1 else 1.0

    salience: dict[int, float] = {}
    for midi in range(_CELLO_MIDI_MIN, _CELLO_MIDI_MAX + 1):
        f0 = _midi_to_hz(midi)
        s = 0.0
        for k in range(1, n_harmonics + 1):
            target_hz = k * f0
            if target_hz > bin_hz[-1]:
                break
            # Nearest-bin lookup with ±1 bin interpolation.
            bin_idx = int(round(target_hz / bin_spacing))
            bin_idx = max(0, min(bin_idx, len(magnitude) - 1))
            s += magnitude[bin_idx] / k
        salience[midi] = s

    return salience


def _is_polyphonic_salience(
    magnitude: np.ndarray,
    freqs_normalised: np.ndarray,
    sample_rate: int,
    threshold: float = config.POLYPHONY_SALIENCE_THRESHOLD,
) -> bool:
    """Return ``True`` if two pitches have high relative salience.

    Args:
        magnitude: One-sided magnitude spectrum.
        freqs_normalised: Normalised frequencies.
        sample_rate: Sample rate in Hz.
        threshold: A secondary pitch must exceed this fraction of the maximum
            salience to trigger polyphony rejection.

    Returns:
        ``True`` if polyphony is detected.
    """
    salience = _pitch_salience_map(magnitude, freqs_normalised, sample_rate)
    if not salience:
        return False

    values = sorted(salience.values(), reverse=True)
    if len(values) < 2:
        return False

    max_sal = values[0]
    if max_sal == 0.0:
        return False

    # The second-strongest pitch must be spatially separated from the first
    # (at least 2 semitones) to avoid counting the fundamental twice.
    ranked_midis = sorted(salience, key=lambda m: salience[m], reverse=True)
    dominant_midi = ranked_midis[0]

    bin_hz = freqs_normalised * sample_rate
    bin_spacing = bin_hz[1] - bin_hz[0] if len(bin_hz) > 1 else 1.0
    spec_peak = magnitude.max()

    for midi in ranked_midis[1:]:
        interval = abs(midi - dominant_midi)
        if interval < 2:
            continue   # Too close — same note with slight detuning
        if interval % 12 == 0:
            continue   # Octave/double-octave: sub-harmonic accumulation artefact

        relative_sal = salience[midi] / max_sal
        if relative_sal <= threshold:
            continue

        # Guard: the candidate second voice must have real energy at its OWN
        # fundamental frequency.  Without this check, a note P2 whose k-th
        # harmonic coincides with the dominant P1's fundamental accumulates
        # salience from P1's energy (e.g. G2's 3rd harmonic ≈ D4), producing
        # a false positive on a monophonic note.  If P2 is actually sounding,
        # its fundamental bin will have significant direct energy.
        f2 = _midi_to_hz(midi)
        bin_idx = int(round(f2 / bin_spacing))
        bin_idx = max(0, min(bin_idx, len(magnitude) - 1))
        fundamental_fraction = magnitude[bin_idx] / spec_peak if spec_peak > 0 else 0.0
        if fundamental_fraction < 0.05:
            logger.debug(
                "Salience: MIDI %d (%.1f Hz) rel=%.3f but fundamental %.4f "
                "< 0.05 — sub-harmonic artefact, not polyphonic",
                midi, f2, relative_sal, fundamental_fraction,
            )
            continue

        logger.debug(
            "Salience: MIDI %d (%.1f Hz) salience %.3f > threshold %.3f "
            "alongside dominant MIDI %d → polyphonic",
            midi, f2, relative_sal, threshold, dominant_midi,
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def is_polyphonic(candidate: NoteCandidate) -> tuple[bool, str]:
    """Test whether a note candidate contains simultaneous pitches.

    Runs both the HPS and the multi-pitch salience check.  If either flags
    polyphony the note is rejected.

    Args:
        candidate: The note candidate to analyse.

    Returns:
        A ``(polyphonic, detail)`` pair.  ``polyphonic`` is ``True`` when
        polyphony is detected; ``detail`` is a human-readable explanation.
    """
    mono = to_mono(candidate.audio)
    frame = _analysis_frame(mono, candidate.sample_rate)
    freqs_norm, magnitude = _magnitude_spectrum(frame)

    if _is_polyphonic_hps(magnitude, freqs_norm, candidate.sample_rate):
        return True, "HPS detected multiple harmonic series"

    if _is_polyphonic_salience(magnitude, freqs_norm, candidate.sample_rate):
        return True, "Multi-pitch salience map detected two simultaneous pitches"

    return False, ""
