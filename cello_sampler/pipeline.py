"""Orchestration: runs all pipeline stages and coordinates I/O.

The pipeline:
    1. Streams the input file in overlapping chunks.
    2. Detects onsets and segments note candidates per chunk.
    3. For each candidate (in parallel across workers):
        a. Polyphony check  → reject if polyphonic.
        b. Pitch estimation  → reject if low confidence or poor intonation.
        c. Articulation feature extraction + classification.
    4. Writes accepted notes and rejected sidecars via SampleWriter.
    5. Returns a ProcessingResult summary.

Duplicate-onset handling: onsets detected within the carry-buffer region of a
chunk (i.e. already processed in the previous chunk) are discarded by only
considering candidates whose ``onset_sample`` >= ``chunk_start + overlap_frames``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cello_sampler import config
from cello_sampler import articulation as art_module
from cello_sampler import onset as onset_module
from cello_sampler import polyphony as poly_module
from cello_sampler import pitch as pitch_module
from cello_sampler.io import SampleWriter, detect_subtype, stream_chunks
from cello_sampler.models import (
    ArticulationType,
    ClassifiedNote,
    NoteCandidate,
    ProcessingResult,
    RejectedNote,
    RejectionReason,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-note analysis (runs inside worker processes)
# ---------------------------------------------------------------------------


def _analyse_candidate(
    candidate: NoteCandidate,
) -> ClassifiedNote | RejectedNote:
    """Run polyphony, pitch and articulation analysis on a single candidate.

    Designed to be called inside a worker process via ProcessPoolExecutor.
    Returns either a :class:`~cello_sampler.models.ClassifiedNote` or a
    :class:`~cello_sampler.models.RejectedNote`.

    Args:
        candidate: The note candidate to analyse.

    Returns:
        A :class:`~cello_sampler.models.ClassifiedNote` on success, or a
        :class:`~cello_sampler.models.RejectedNote` on rejection.
    """
    # 1. Polyphony check.
    polyphonic, poly_detail = poly_module.is_polyphonic(candidate)
    if polyphonic:
        return RejectedNote(
            candidate=candidate,
            reason=RejectionReason.POLYPHONIC,
            detail=poly_detail,
        )

    # 2. Pitch estimation + intonation gate.
    pitch, rejection_detail = pitch_module.estimate_pitch(candidate)
    if pitch is None:
        reason = (
            RejectionReason.LOW_CONFIDENCE
            if "confidence" in rejection_detail.lower()
            else RejectionReason.INTONATION
        )
        return RejectedNote(
            candidate=candidate,
            reason=reason,
            detail=rejection_detail,
        )

    # 3. Articulation features + classification.
    features = art_module.extract_features(candidate, pitch)
    articulation = art_module.classify(features)

    # 4. Minimum-duration gate for sustaining articulations.
    #    Very short legato/vibrato takes are not useful for looped playback.
    min_ms: float | None = None
    if articulation == ArticulationType.LEGATO:
        min_ms = config.MIN_LEGATO_DURATION_MS
    elif articulation == ArticulationType.VIBRATO:
        min_ms = config.MIN_VIBRATO_DURATION_MS

    if min_ms is not None and features.total_duration_ms < min_ms:
        return RejectedNote(
            candidate=candidate,
            reason=RejectionReason.TOO_SHORT,
            detail=(
                f"{articulation.value} note is {features.total_duration_ms:.0f} ms "
                f"(minimum {min_ms:.0f} ms)"
            ),
        )

    return ClassifiedNote(
        candidate=candidate,
        pitch=pitch,
        articulation=articulation,
        features=features,
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run(
    input_path: Path,
    output_dir: Path,
    n_workers: int = config.N_WORKERS,
    chunk_size: int = config.CHUNK_SIZE_FRAMES,
    overlap_seconds: float = config.OVERLAP_SECONDS,
) -> ProcessingResult:
    """Process a single recording file end-to-end.

    Streams the file in overlapping chunks, detects and segments note
    candidates, analyses each candidate (in parallel), then writes labelled
    output files.

    Args:
        input_path: Path to the source 48 kHz multi-channel audio file.
        output_dir: Root directory for labelled output files.
        n_workers: Number of parallel worker processes for per-note analysis.
        chunk_size: Streaming chunk size in frames.
        overlap_seconds: Carry-buffer overlap between chunks in seconds.

    Returns:
        :class:`~cello_sampler.models.ProcessingResult` with accepted and
        rejected note lists.
    """
    logger.info("Processing: %s", input_path)

    subtype = detect_subtype(input_path)
    result = ProcessingResult(source_file=input_path)
    candidate_count = 0

    # Collect all candidates first, then analyse in parallel.
    all_candidates: list[NoteCandidate] = []

    for chunk, sample_rate, chunk_start in stream_chunks(
        input_path, chunk_size=chunk_size, overlap_seconds=overlap_seconds
    ):
        overlap_frames = int(overlap_seconds * sample_rate)

        candidates = onset_module.process_chunk(
            audio=chunk,
            sample_rate=sample_rate,
            source_file=input_path,
            chunk_start_sample=chunk_start,
            start_candidate_index=candidate_count,
        )

        # Discard candidates whose onset falls inside the carry-buffer region.
        # Those were already yielded by the previous chunk's segmentation.
        if chunk_start > 0:
            candidates = [
                c for c in candidates
                if c.onset_sample >= chunk_start + overlap_frames
            ]

        all_candidates.extend(candidates)
        candidate_count += len(candidates)

    logger.info("Total candidates from onset detection: %d", len(all_candidates))

    # Analyse candidates in parallel.
    analysed: list[ClassifiedNote | RejectedNote] = []

    if n_workers > 1 and len(all_candidates) > 0:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_analyse_candidate, c): c
                for c in all_candidates
            }
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]
                progress = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Analysing notes",
                    unit="note",
                )
            except ImportError:
                progress = as_completed(futures)  # type: ignore[assignment]

            for future in progress:
                analysed.append(future.result())
    else:
        # Single-process fallback (simpler for debugging / short files).
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            iterable = tqdm(all_candidates, desc="Analysing notes", unit="note")
        except ImportError:
            iterable = all_candidates  # type: ignore[assignment]

        for candidate in iterable:
            analysed.append(_analyse_candidate(candidate))

    # Assign per-(pitch, articulation) take numbers and write output.
    take_counters: dict[tuple[str, str], int] = {}

    with SampleWriter(output_dir, source_bit_depth=subtype, sample_rate=sample_rate) as writer:
        for item in analysed:
            if isinstance(item, RejectedNote):
                result.rejected.append(item)
                writer.write_rejected(item)
            else:
                key = (item.pitch.note_name, item.articulation.value)
                take = take_counters.get(key, 0) + 1
                take_counters[key] = take
                item.take_number = take
                result.accepted.append(item)
                writer.write_accepted(item)

    logger.info(
        "Done. Accepted: %d, Rejected: %d "
        "(polyphonic: %d, intonation: %d, low_conf: %d, too_short: %d)",
        result.n_accepted,
        result.n_rejected,
        sum(1 for r in result.rejected if r.reason == RejectionReason.POLYPHONIC),
        sum(1 for r in result.rejected if r.reason == RejectionReason.INTONATION),
        sum(1 for r in result.rejected if r.reason == RejectionReason.LOW_CONFIDENCE),
        sum(1 for r in result.rejected if r.reason == RejectionReason.TOO_SHORT),
    )

    return result
