"""add_loop_points.py — embed sustain loop markers into legato and vibrato WAV files.

Logic Pro's Sampler (and most professional samplers) reads loop start/end points
from the ``smpl`` chunk embedded in a WAV file and uses them to sustain a note
for as long as the player holds a key.

Algorithm
---------
For each WAV file in the ``legato/`` or ``vibrato/`` subdirectory of an output
directory:

1. Read the mono mix of the audio and the source sample rate.
2. Locate the *sustain region*: the portion of the note after the attack and
   before the release, identified by the Hilbert amplitude envelope.
3. Estimate the fundamental period from the median pitch (via autocorrelation
   on the sustain region).
4. Search within the sustain region for a loop of at least ``MIN_LOOP_CYCLES``
   complete fundamental periods where the waveform at the end of the loop is
   maximally similar to the waveform at the start, using normalised cross-
   correlation.  This minimises the click at the loop splice point.
5. Refine the loop end to the nearest zero crossing.
6. Write the loop markers into a ``smpl`` chunk and save the modified WAV in
   place (original backed up as ``*.wav.bak``).

Usage
-----
::

    python add_loop_points.py OUTPUT_DIR [options]

    # Process a specific articulation only
    python add_loop_points.py samples/mono_1/ --articulation legato

    # Require a longer loop and accept higher splice error
    python add_loop_points.py samples/mono_1/ --min-cycles 6 --max-splice-error 0.05
"""

from __future__ import annotations

import argparse
import logging
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, hilbert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fraction of note duration trimmed from each end to find the sustain region.
SUSTAIN_TRIM: float = 0.20

#: Noise floor in dB relative to peak — frames below this are treated as silence.
NOISE_FLOOR_DB: float = -40.0

#: Minimum number of complete fundamental-period cycles the loop must span.
MIN_LOOP_CYCLES: int = 4

#: Maximum normalised splice error (0 = perfect, 1 = maximally different).
#: Loops with a higher error are skipped and the file is left unmodified.
MAX_SPLICE_ERROR: float = 0.02

#: Width of the zero-crossing search window around the candidate loop end
#: (in samples).
ZERO_CROSSING_WINDOW: int = 256

#: Articulations that benefit from looping.
LOOP_ARTICULATIONS: tuple[str, ...] = ("legato", "vibrato")

# ---------------------------------------------------------------------------
# WAV smpl chunk construction
# ---------------------------------------------------------------------------

_SMPL_HEADER_FMT = "<IIIIIIIIiI"   # 10 × 4-byte fields before loop blocks
_SMPL_LOOP_FMT   = "<IIIIII"       # 6 × 4-byte fields per loop


def _build_smpl_chunk(loop_start: int, loop_end: int, sample_rate: int) -> bytes:
    """Serialise a WAV ``smpl`` chunk containing one forward sustain loop.

    Args:
        loop_start: Loop start in samples (absolute from file start).
        loop_end:   Loop end in samples (inclusive, absolute from file start).
        sample_rate: File sample rate in Hz (used for MIDI unity note period).

    Returns:
        Raw bytes of the complete ``smpl`` chunk including the 8-byte RIFF
        chunk header (``b"smpl"`` + 4-byte size).
    """
    manufacturer      = 0
    product           = 0
    sample_period_ns  = int(1_000_000_000 / sample_rate)
    midi_unity_note   = 60        # middle C — Sampler overrides this from zone mapping
    midi_pitch_frac   = 0
    smpte_format      = 0
    smpte_offset      = 0
    num_sample_loops  = 1
    sampler_data_len  = 0

    header = struct.pack(
        _SMPL_HEADER_FMT,
        manufacturer,
        product,
        sample_period_ns,
        midi_unity_note,
        midi_pitch_frac,
        smpte_format,
        smpte_offset,
        num_sample_loops,
        sampler_data_len,
        0,                        # padding / extra sampler data length
    )

    loop_block = struct.pack(
        _SMPL_LOOP_FMT,
        0,              # cue point ID
        0,              # loop type: 0 = forward
        loop_start,
        loop_end,
        0,              # fractional loop end (unused)
        0,              # play count: 0 = infinite
    )

    payload = header + loop_block
    chunk   = b"smpl" + struct.pack("<I", len(payload)) + payload
    return chunk


def _embed_smpl_chunk(wav_path: Path, smpl_chunk: bytes) -> None:
    """Re-write *wav_path* with *smpl_chunk* appended to its RIFF container.

    Backs up the original to ``<name>.wav.bak`` before writing.

    Args:
        wav_path:   Path to the WAV file to modify (modified in place).
        smpl_chunk: Fully serialised ``smpl`` chunk bytes.
    """
    raw = wav_path.read_bytes()

    # Parse the outer RIFF header.
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{wav_path} is not a valid RIFF/WAVE file")

    # Strip any existing smpl chunk.
    pos = 12
    out_chunks: list[bytes] = []
    while pos < len(raw):
        if pos + 8 > len(raw):
            break
        chunk_id   = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk_end  = pos + 8 + chunk_size
        if chunk_id != b"smpl":
            out_chunks.append(raw[pos:chunk_end])
        pos = chunk_end

    # Back up original.
    shutil.copy2(wav_path, wav_path.with_suffix(".wav.bak"))

    # Reassemble: RIFF header + existing chunks + new smpl chunk.
    body       = b"".join(out_chunks) + smpl_chunk
    riff_size  = 4 + len(body)   # 4 = len(b"WAVE")
    new_raw    = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + body
    wav_path.write_bytes(new_raw)


# ---------------------------------------------------------------------------
# Loop-point detection
# ---------------------------------------------------------------------------


def _amplitude_envelope(mono: np.ndarray) -> np.ndarray:
    """Hilbert analytic-signal magnitude envelope."""
    return np.abs(hilbert(mono.astype(np.float64))).astype(np.float32)


def _sustain_region(mono: np.ndarray, envelope: np.ndarray) -> tuple[int, int]:
    """Return (start, end) sample indices of the sustain region.

    The sustain region excludes the attack and release, defined as the inner
    ``1 - 2 * SUSTAIN_TRIM`` fraction of the sounding portion of the note.

    Args:
        mono:     1-D mono audio array.
        envelope: Amplitude envelope (same length as *mono*).

    Returns:
        ``(start, end)`` sample indices, both inclusive.
    """
    peak_db    = 20 * np.log10(envelope.max() + 1e-12)
    floor_db   = peak_db + NOISE_FLOOR_DB
    floor_lin  = 10 ** (floor_db / 20)

    above = np.where(envelope > floor_lin)[0]
    if len(above) < 16:
        return 0, len(mono) - 1

    sound_start = int(above[0])
    sound_end   = int(above[-1])
    sound_len   = sound_end - sound_start

    trim = int(sound_len * SUSTAIN_TRIM)
    return sound_start + trim, sound_end - trim


def _estimate_period(mono: np.ndarray, sr: int, start: int, end: int) -> int | None:
    """Estimate the fundamental period (in samples) via autocorrelation.

    Analyses a 100 ms window from the middle of the sustain region.

    Args:
        mono:  1-D mono audio.
        sr:    Sample rate in Hz.
        start: Sustain region start sample.
        end:   Sustain region end sample.

    Returns:
        Fundamental period in samples, or ``None`` if estimation fails.
    """
    window_n = min(int(0.1 * sr), end - start)
    mid      = (start + end) // 2
    segment  = mono[mid - window_n // 2: mid + window_n // 2].astype(np.float64)
    segment -= segment.mean()

    acf = correlate(segment, segment, mode="full")
    acf = acf[len(acf) // 2:]   # keep positive lags only

    # Search for first peak in the range corresponding to 60–1000 Hz.
    min_period = max(1, int(sr / 1000))
    max_period = int(sr / 60)

    if max_period >= len(acf):
        return None

    search = acf[min_period:max_period]
    if len(search) == 0:
        return None

    peak_lag = int(np.argmax(search)) + min_period
    return peak_lag


def _find_loop_points(
    mono: np.ndarray,
    sr: int,
    min_cycles: int = MIN_LOOP_CYCLES,
    max_splice_error: float = MAX_SPLICE_ERROR,
) -> tuple[int, int] | None:
    """Find the best sustain loop start and end in *mono*.

    Searches the sustain region for a loop spanning at least *min_cycles*
    fundamental periods where the normalised cross-correlation between the
    waveform neighbourhood at loop-start and loop-end is maximised (i.e. the
    splice is as seamless as possible).

    Args:
        mono:             1-D float32 mono audio.
        sr:               Sample rate in Hz.
        min_cycles:       Minimum number of fundamental cycles the loop spans.
        max_splice_error: Maximum allowed normalised splice discontinuity.

    Returns:
        ``(loop_start, loop_end)`` in samples, or ``None`` if no suitable loop
        is found within *max_splice_error*.
    """
    envelope = _amplitude_envelope(mono)
    s_start, s_end = _sustain_region(mono, envelope)

    if s_end - s_start < 512:
        logger.warning("Sustain region too short to find loop point.")
        return None

    period = _estimate_period(mono, sr, s_start, s_end)
    if period is None or period < 2:
        logger.warning("Fundamental period estimation failed.")
        return None

    min_loop_len = min_cycles * period
    if s_end - s_start < min_loop_len + period:
        logger.warning(
            "Sustain region (%d samples) shorter than minimum loop "
            "length (%d samples for %d cycles at period %d).",
            s_end - s_start, min_loop_len, min_cycles, period,
        )
        return None

    # Compare neighbourhood windows at loop start vs candidate loop ends.
    compare_n = min(period * 2, 1024)
    loop_start = s_start
    start_window = mono[loop_start: loop_start + compare_n].astype(np.float64)
    start_norm   = np.linalg.norm(start_window)
    if start_norm < 1e-9:
        return None

    best_error  = float("inf")
    best_end    = -1

    # Slide the candidate loop-end across the sustain region.
    candidate_start = loop_start + min_loop_len
    candidate_stop  = s_end - compare_n

    for loop_end_candidate in range(candidate_start, candidate_stop, period // 4):
        end_window = mono[loop_end_candidate: loop_end_candidate + compare_n].astype(np.float64)
        end_norm   = np.linalg.norm(end_window)
        if end_norm < 1e-9:
            continue

        # Normalised difference energy.
        diff  = start_window - end_window
        error = float(np.linalg.norm(diff)) / (start_norm + end_norm)

        if error < best_error:
            best_error = error
            best_end   = loop_end_candidate

    if best_end < 0 or best_error > max_splice_error:
        logger.warning(
            "No loop point found within error threshold %.4f (best %.4f).",
            max_splice_error, best_error,
        )
        return None

    # Refine loop_end to nearest zero crossing.
    search_start = max(0, best_end - ZERO_CROSSING_WINDOW)
    search_end   = min(len(mono) - 1, best_end + ZERO_CROSSING_WINDOW)
    region       = mono[search_start:search_end]
    zero_crossings = np.where(np.diff(np.sign(region)))[0] + search_start
    if len(zero_crossings) > 0:
        best_end = int(zero_crossings[np.argmin(np.abs(zero_crossings - best_end))])

    logger.debug(
        "Loop: start=%d end=%d length=%d samples splice_error=%.4f",
        loop_start, best_end, best_end - loop_start, best_error,
    )
    return loop_start, best_end


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_file(
    wav_path: Path,
    min_cycles: int = MIN_LOOP_CYCLES,
    max_splice_error: float = MAX_SPLICE_ERROR,
    dry_run: bool = False,
) -> bool:
    """Detect and embed loop points in a single WAV file.

    Args:
        wav_path:         Path to the WAV file.
        min_cycles:       Minimum fundamental cycles for the loop.
        max_splice_error: Maximum normalised splice discontinuity.
        dry_run:          If ``True``, detect but do not write anything.

    Returns:
        ``True`` if loop points were found (and written, unless *dry_run*).
    """
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    except Exception as exc:
        logger.error("Cannot read %s: %s", wav_path, exc)
        return False

    mono = audio.mean(axis=1)
    result = _find_loop_points(mono, sr, min_cycles=min_cycles,
                               max_splice_error=max_splice_error)

    if result is None:
        logger.info("  %-50s  no suitable loop found", wav_path.name)
        return False

    loop_start, loop_end = result
    logger.info(
        "  %-50s  loop %d–%d  (%.2f s)",
        wav_path.name, loop_start, loop_end, (loop_end - loop_start) / sr,
    )

    if not dry_run:
        smpl = _build_smpl_chunk(loop_start, loop_end, sr)
        _embed_smpl_chunk(wav_path, smpl)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog="add_loop_points",
        description="Embed sustain loop markers into legato/vibrato WAV samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Root output directory produced by process_recording.py.",
    )
    p.add_argument(
        "--articulation",
        choices=list(LOOP_ARTICULATIONS),
        default=None,
        help="Process only this articulation (default: both legato and vibrato).",
    )
    p.add_argument(
        "--min-cycles",
        type=int,
        default=MIN_LOOP_CYCLES,
        metavar="N",
        help="Minimum number of fundamental-period cycles the loop must span.",
    )
    p.add_argument(
        "--max-splice-error",
        type=float,
        default=MAX_SPLICE_ERROR,
        metavar="ERR",
        help=(
            "Maximum normalised waveform discontinuity at the splice point "
            "(0 = perfect, 1 = maximally different).  Raise if few files "
            "get loop points; lower for cleaner splices."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Detect loop points but do not modify any files.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    articulations = (
        (args.articulation,) if args.articulation else LOOP_ARTICULATIONS
    )

    total = found = 0

    for art in articulations:
        art_dir = args.output_dir / art
        if not art_dir.is_dir():
            logger.warning("Directory not found, skipping: %s", art_dir)
            continue

        wav_files = sorted(art_dir.glob("*.wav"))
        logger.info("\n%s/  (%d files)", art, len(wav_files))

        for wav_path in wav_files:
            total += 1
            if process_file(
                wav_path,
                min_cycles=args.min_cycles,
                max_splice_error=args.max_splice_error,
                dry_run=args.dry_run,
            ):
                found += 1

    action = "would embed" if args.dry_run else "embedded"
    logger.info(
        "\nDone. Loop points %s in %d / %d files.", action, found, total
    )
    if found < total:
        logger.info(
            "%d files had no suitable loop — try raising --max-splice-error "
            "(e.g. 0.05) or lowering --min-cycles.",
            total - found,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
