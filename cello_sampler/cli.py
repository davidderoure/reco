"""Command-line interface for the cello sampler pipeline.

Usage::

    python process_recording.py INPUT_FILE OUTPUT_DIR [options]

All pipeline thresholds can be overridden from the command line so a musician
can tune the tool without editing source code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cello_sampler import config


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="process_recording",
        description=(
            "Extract, classify and label individual note samples from a "
            "multi-channel solo cello studio recording."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required positional arguments ---
    p.add_argument(
        "input_file",
        type=Path,
        help="Path to the source audio file (WAV, AIFF, FLAC; 48 kHz recommended).",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Root directory for labelled output files.  Created if absent.",
    )

    # --- I/O options ---
    io_group = p.add_argument_group("I/O options")
    io_group.add_argument(
        "--workers",
        type=int,
        default=config.N_WORKERS,
        metavar="N",
        help="Number of parallel worker processes for per-note analysis.",
    )
    io_group.add_argument(
        "--chunk-seconds",
        type=float,
        default=config.CHUNK_SIZE_FRAMES / 48_000,
        metavar="SECS",
        help="Streaming chunk size in seconds.",
    )
    io_group.add_argument(
        "--overlap-seconds",
        type=float,
        default=config.OVERLAP_SECONDS,
        metavar="SECS",
        help="Carry-buffer overlap between consecutive chunks.",
    )

    # --- Onset detection options ---
    onset_group = p.add_argument_group("Onset detection")
    onset_group.add_argument(
        "--onset-threshold",
        type=float,
        default=config.ONSET_THRESHOLD_MULTIPLIER,
        metavar="MULT",
        help="Onset strength threshold as a multiple of the local mean.",
    )
    onset_group.add_argument(
        "--min-note-ms",
        type=float,
        default=config.MIN_NOTE_DURATION_MS,
        metavar="MS",
        help="Minimum inter-onset gap in milliseconds.",
    )

    # --- Quality gating options ---
    quality_group = p.add_argument_group("Quality gating")
    quality_group.add_argument(
        "--pitch-confidence",
        type=float,
        default=config.PITCH_CONFIDENCE_THRESHOLD,
        metavar="CONF",
        help="Minimum CREPE confidence (0–1) to accept a note.",
    )
    quality_group.add_argument(
        "--max-intonation-cents",
        type=float,
        default=config.MAX_INTONATION_DEVIATION_CENTS,
        metavar="CENTS",
        help="Maximum allowed intonation deviation from 12-TET in cents.",
    )
    quality_group.add_argument(
        "--polyphony-threshold",
        type=float,
        default=config.POLYPHONY_SALIENCE_THRESHOLD,
        metavar="FRAC",
        help=(
            "Fraction of max salience at which a secondary pitch triggers "
            "polyphony rejection (0–1)."
        ),
    )

    # --- Articulation thresholds ---
    art_group = p.add_argument_group("Articulation classification")
    art_group.add_argument(
        "--pizz-max-attack-ms",
        type=float,
        default=config.PIZZICATO_MAX_ATTACK_MS,
        metavar="MS",
        help="Maximum attack duration (ms) for pizzicato classification.",
    )
    art_group.add_argument(
        "--stacc-max-duration-ms",
        type=float,
        default=config.STACCATO_MAX_DURATION_MS,
        metavar="MS",
        help="Maximum sounding duration (ms) for staccato classification.",
    )
    art_group.add_argument(
        "--vibrato-min-depth-cents",
        type=float,
        default=config.VIBRATO_MIN_DEPTH_CENTS,
        metavar="CENTS",
        help="Minimum pitch modulation depth (cents) to classify as vibrato.",
    )

    # --- Logging ---
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return p


def apply_overrides(args: argparse.Namespace) -> None:
    """Patch the config module constants with any CLI overrides.

    This mutates :mod:`cello_sampler.config` in-place so that all downstream
    modules pick up the overridden values without needing to pass arguments
    through every function.

    Args:
        args: Parsed CLI arguments.
    """
    config.ONSET_THRESHOLD_MULTIPLIER = args.onset_threshold
    config.MIN_NOTE_DURATION_MS = args.min_note_ms
    config.PITCH_CONFIDENCE_THRESHOLD = args.pitch_confidence
    config.MAX_INTONATION_DEVIATION_CENTS = args.max_intonation_cents
    config.POLYPHONY_SALIENCE_THRESHOLD = args.polyphony_threshold
    config.PIZZICATO_MAX_ATTACK_MS = args.pizz_max_attack_ms
    config.STACCATO_MAX_DURATION_MS = args.stacc_max_duration_ms
    config.VIBRATO_MIN_DEPTH_CENTS = args.vibrato_min_depth_cents
    config.N_WORKERS = args.workers
    config.CHUNK_SIZE_FRAMES = int(args.chunk_seconds * 48_000)
    config.OVERLAP_SECONDS = args.overlap_seconds


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input_file.is_file():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        return 1

    apply_overrides(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import pipeline here (after config overrides are applied).
    from cello_sampler.pipeline import run  # noqa: PLC0415

    result = run(
        input_path=args.input_file,
        output_dir=args.output_dir,
        n_workers=args.workers,
        chunk_size=config.CHUNK_SIZE_FRAMES,
        overlap_seconds=config.OVERLAP_SECONDS,
    )

    print(
        f"\nProcessing complete.\n"
        f"  Accepted : {result.n_accepted}\n"
        f"  Rejected : {result.n_rejected}\n"
        f"  Output   : {args.output_dir}\n"
    )

    return 0
