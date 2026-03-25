"""Onset detection and note segmentation.

Uses a spectral-flux based onset strength envelope computed via STFT, with
an adaptive local-mean threshold and non-maximum suppression.  All analysis
is performed on a mono mix at the native 48 kHz sample rate so no resampling
is introduced before writing output files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.signal import stft

from cello_sampler import config
from cello_sampler.models import NoteCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mono mix helper
# ---------------------------------------------------------------------------


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return the mean across channels as a 1-D float32 array.

    Args:
        audio: Shape ``(frames,)`` or ``(frames, channels)``.

    Returns:
        1-D array of shape ``(frames,)``.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Onset strength envelope
# ---------------------------------------------------------------------------


def onset_strength(
    mono: np.ndarray,
    sample_rate: int,
    nperseg: int = config.STFT_NPERSEG,
    hop: int = config.STFT_HOP,
    freq_min: float = config.ANALYSIS_FREQ_MIN_HZ,
    freq_max: float = config.ANALYSIS_FREQ_MAX_HZ,
) -> np.ndarray:
    """Compute a superflux-style spectral-flux onset strength envelope.

    The envelope is band-limited to ``[freq_min, freq_max]`` to suppress
    sub-bass rumble and high-frequency bow noise that would otherwise produce
    spurious onsets.

    Args:
        mono: 1-D float32 mono audio array.
        sample_rate: Sample rate in Hz.
        nperseg: STFT window length in samples.
        hop: STFT hop size in samples.
        freq_min: Lower frequency bound for the analysis band (Hz).
        freq_max: Upper frequency bound for the analysis band (Hz).

    Returns:
        1-D float32 array of onset strength values, one per STFT hop.
    """
    _, _, Zxx = stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
    )
    magnitude = np.abs(Zxx)  # shape: (freq_bins, time_frames)

    # Restrict to the cello analysis band.
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    magnitude = magnitude[band_mask, :]

    # Superflux: positive spectral flux summed across all band frequencies.
    flux = np.diff(magnitude, axis=1, prepend=magnitude[:, :1])
    flux = np.maximum(flux, 0.0)
    strength = flux.sum(axis=0).astype(np.float32)

    return strength


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------


def detect_onsets(
    strength: np.ndarray,
    hop: int,
    sample_rate: int,
    threshold_multiplier: float = config.ONSET_THRESHOLD_MULTIPLIER,
    local_mean_half_width: int = config.ONSET_LOCAL_MEAN_HALF_WIDTH,
    min_note_duration_ms: float = config.MIN_NOTE_DURATION_MS,
) -> np.ndarray:
    """Pick onset sample indices from an onset strength envelope.

    An onset is accepted when the strength exceeds a local adaptive threshold
    and is the maximum in a minimum-inter-onset window.

    Args:
        strength: 1-D onset strength array (one value per STFT hop).
        hop: STFT hop size in samples, used to convert hop indices to sample
            indices.
        sample_rate: Sample rate in Hz.
        threshold_multiplier: Onset must exceed this multiple of the local mean.
        local_mean_half_width: Half-width in hops for the local mean window.
        min_note_duration_ms: Minimum gap between successive onsets in ms.

    Returns:
        Sorted 1-D int64 array of onset sample positions (absolute, relative
        to the start of the provided ``strength`` array's source signal).
    """
    n = len(strength)
    min_gap_hops = max(1, int((min_note_duration_ms / 1000.0) * sample_rate / hop))

    # Adaptive threshold: local mean over a sliding window.
    threshold = np.empty(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - local_mean_half_width)
        hi = min(n, i + local_mean_half_width + 1)
        threshold[i] = strength[lo:hi].mean() * threshold_multiplier

    # Find candidate peaks above threshold.
    above = strength > threshold
    candidates: list[int] = []
    for i in range(1, n - 1):
        if above[i] and strength[i] >= strength[i - 1] and strength[i] >= strength[i + 1]:
            candidates.append(i)

    # Non-maximum suppression: enforce minimum inter-onset gap.
    onsets: list[int] = []
    last_hop: int = -min_gap_hops - 1
    for hop_idx in candidates:
        if hop_idx - last_hop >= min_gap_hops:
            onsets.append(hop_idx)
            last_hop = hop_idx
        elif strength[hop_idx] > strength[onsets[-1]]:
            # Replace the previous onset if this one is stronger and within
            # the minimum gap — catches cases where the STFT slightly smears
            # the true attack peak.
            onsets[-1] = hop_idx
            last_hop = hop_idx

    sample_indices = np.array([h * hop for h in onsets], dtype=np.int64)
    logger.debug("Detected %d onsets", len(sample_indices))
    return sample_indices


# ---------------------------------------------------------------------------
# Note segmentation
# ---------------------------------------------------------------------------


def segment_notes(
    audio: np.ndarray,
    onset_samples: np.ndarray,
    sample_rate: int,
    source_file: Path,
    chunk_start_sample: int = 0,
    pre_onset_samples: int = config.PRE_ONSET_SAMPLES,
    post_onset_gap_samples: int = config.POST_ONSET_GAP_SAMPLES,
    max_note_duration_seconds: float = config.MAX_NOTE_DURATION_SECONDS,
    start_candidate_index: int = 0,
) -> list[NoteCandidate]:
    """Slice note windows from a chunk of audio around detected onsets.

    Each note window starts ``pre_onset_samples`` before the onset and ends
    either ``post_onset_gap_samples`` before the next onset, or at the
    maximum note duration cap, whichever is shorter.

    Args:
        audio: Multi-channel audio of shape ``(frames, channels)``.
        onset_samples: 1-D array of onset positions relative to the start of
            ``audio``.
        sample_rate: Sample rate in Hz.
        source_file: Path to the originating file (stored in each candidate).
        chunk_start_sample: Absolute sample position of the first frame of
            ``audio`` within the source file.  Added to onset positions so
            that ``NoteCandidate.onset_sample`` is an absolute file offset.
        pre_onset_samples: Samples prepended before the onset.
        post_onset_gap_samples: Samples trimmed from the end before the next
            onset.
        max_note_duration_seconds: Hard cap on note window length.
        start_candidate_index: First candidate index to assign in this batch
            (allows globally unique indices across multiple chunks).

    Returns:
        List of :class:`~cello_sampler.models.NoteCandidate` objects.
    """
    max_samples = int(max_note_duration_seconds * sample_rate)
    n_frames = audio.shape[0]
    candidates: list[NoteCandidate] = []

    for i, onset in enumerate(onset_samples):
        onset = int(onset)
        start = max(0, onset - pre_onset_samples)

        if i + 1 < len(onset_samples):
            next_onset = int(onset_samples[i + 1])
            end = min(n_frames, next_onset - post_onset_gap_samples)
        else:
            end = min(n_frames, onset + max_samples)

        # Enforce max duration cap regardless of next onset position.
        end = min(end, start + max_samples)

        if end <= start:
            logger.debug("Skipping zero-length note at onset %d", onset)
            continue

        window = audio[start:end]
        # Ensure 2-D even for single-channel files.
        if window.ndim == 1:
            window = window[:, np.newaxis]

        candidate = NoteCandidate(
            audio=window.copy(),
            sample_rate=sample_rate,
            onset_sample=chunk_start_sample + onset,
            source_file=source_file,
            candidate_index=start_candidate_index + i,
        )
        candidates.append(candidate)

    logger.debug("Segmented %d note candidates from chunk", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Combined single-chunk entry point
# ---------------------------------------------------------------------------


def process_chunk(
    audio: np.ndarray,
    sample_rate: int,
    source_file: Path,
    chunk_start_sample: int = 0,
    start_candidate_index: int = 0,
) -> list[NoteCandidate]:
    """Run onset detection and segmentation on one audio chunk.

    This is the primary entry point called by the pipeline for each streamed
    chunk.

    Args:
        audio: Multi-channel chunk of shape ``(frames, channels)``.
        sample_rate: Sample rate in Hz.
        source_file: Originating file path.
        chunk_start_sample: Absolute sample offset of this chunk.
        start_candidate_index: First candidate index for this batch.

    Returns:
        List of :class:`~cello_sampler.models.NoteCandidate` objects.
    """
    mono = to_mono(audio)
    strength = onset_strength(mono, sample_rate)
    onset_samples = detect_onsets(strength, hop=config.STFT_HOP, sample_rate=sample_rate)

    return segment_notes(
        audio=audio,
        onset_samples=onset_samples,
        sample_rate=sample_rate,
        source_file=source_file,
        chunk_start_sample=chunk_start_sample,
        start_candidate_index=start_candidate_index,
    )
