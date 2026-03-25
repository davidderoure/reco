"""Audio I/O: streaming reader and labelled sample writer.

The reader yields overlapping chunks of multi-channel audio from a large file
without loading the whole file into memory.  The writer saves classified notes
as labelled WAV files and produces a CSV index for DAW import.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf

from cello_sampler import config
from cello_sampler.models import (
    ArticulationType,
    ClassifiedNote,
    RejectedNote,
    RejectionReason,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------


def stream_chunks(
    path: Path,
    chunk_size: int = config.CHUNK_SIZE_FRAMES,
    overlap_seconds: float = config.OVERLAP_SECONDS,
) -> Generator[tuple[np.ndarray, int, int], None, None]:
    """Yield overlapping audio chunks from a large multi-channel file.

    Each chunk is returned as a ``(audio, sample_rate, chunk_start_sample)``
    triple.  ``audio`` has shape ``(frames, channels)`` and dtype ``float32``.
    The carry-buffer overlap ensures that notes straddling chunk boundaries
    are fully captured in at least one chunk.

    Args:
        path: Path to the source audio file (WAV, AIFF, FLAC, etc.).
        chunk_size: Number of frames per chunk (excluding overlap).
        overlap_seconds: Seconds of audio carried over from the previous chunk.

    Yields:
        Tuples of ``(audio_chunk, sample_rate, chunk_start_sample)`` where
        ``chunk_start_sample`` is the absolute sample position of the first
        frame in ``audio_chunk`` within the source file.
    """
    with sf.SoundFile(path) as f:
        sample_rate: int = f.samplerate
        overlap_frames = int(overlap_seconds * sample_rate)

        logger.info(
            "Opened %s: %d ch, %d Hz, %d frames (%.1f s)",
            path.name, f.channels, sample_rate,
            f.frames, f.frames / sample_rate,
        )

        carry: np.ndarray | None = None
        file_pos = 0                    # read cursor in the source file

        while True:
            raw = f.read(chunk_size, dtype="float32", always_2d=True)
            if raw.shape[0] == 0:
                break

            if carry is not None:
                chunk = np.concatenate([carry, raw], axis=0)
                chunk_start = file_pos - carry.shape[0]
            else:
                chunk = raw
                chunk_start = 0

            yield chunk, sample_rate, chunk_start

            file_pos += raw.shape[0]

            # Carry the last `overlap_frames` samples into the next iteration.
            carry = chunk[-overlap_frames:] if overlap_frames > 0 else None

            if raw.shape[0] < chunk_size:
                break   # reached end of file


def read_file(path: Path) -> tuple[np.ndarray, int]:
    """Read an entire audio file into memory.

    Intended for short files (test fixtures, individual note segments).  For
    production use on large recordings, prefer :func:`stream_chunks`.

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of ``(audio, sample_rate)`` where ``audio`` has shape
        ``(frames, channels)``, dtype ``float32``.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio, sr


# ---------------------------------------------------------------------------
# MIDI / note-name helpers
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_note: int) -> str:
    """Convert a MIDI note number to scientific pitch notation.

    Args:
        midi_note: MIDI note number (0–127).  60 = middle C (C4).

    Returns:
        Note name string such as ``"A3"``, ``"C4"``, ``"F#5"``.
    """
    octave = (midi_note // 12) - 1
    name = _NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


class SampleWriter:
    """Writes labelled note samples to an organised directory tree.

    Directory layout::

        output_dir/
            legato/
                A3_legato_001.wav
                ...
            staccato/
                ...
            vibrato/
                ...
            pizzicato/
                ...
            rejected/
                polyphonic/
                    A3_rejected_polyphonic_001.json
                intonation/
                    ...
                low_confidence/
                    ...
            _index.csv

    The JSON sidecar files for rejected notes capture diagnostic metadata
    to help tune rejection thresholds.

    Args:
        output_dir: Root directory for all output.  Created if absent.
        source_bit_depth: Bit depth / subtype string for output WAV files
            (e.g. ``"PCM_24"``, ``"FLOAT"``).  Defaults to ``"FLOAT"``
            (32-bit float), matching typical studio session formats.
        sample_rate: Output sample rate.  Should match the source recording.
    """

    def __init__(
        self,
        output_dir: Path,
        source_bit_depth: str = "FLOAT",
        sample_rate: int = 48_000,
    ) -> None:
        self._root = output_dir
        self._subtype = source_bit_depth
        self._sr = sample_rate

        # Create output directories up front.
        for art in ArticulationType:
            (self._root / art.value).mkdir(parents=True, exist_ok=True)
        for reason in RejectionReason:
            (self._root / config.REJECTED_DIR / reason.value).mkdir(
                parents=True, exist_ok=True
            )

        # CSV index: open in append mode so the writer can be called
        # incrementally as notes are processed.
        index_path = self._root / config.INDEX_FILENAME
        self._csv_file = index_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "filename", "note", "midi_note", "articulation",
            "pitch_hz", "deviation_cents", "duration_ms",
            "attack_ms", "decay_db_per_ms",
            "vibrato_depth_cents", "vibrato_rate_hz",
            "source_file", "onset_sample",
        ])

        # Track per-(note, articulation) take counters.
        self._take_counters: dict[tuple[str, str], int] = {}

    def write_accepted(self, note: ClassifiedNote) -> Path:
        """Write a classified note to disk and append a CSV index row.

        Args:
            note: A fully classified note from the pipeline.

        Returns:
            Path to the written WAV file.
        """
        art = note.articulation.value
        note_name = note.pitch.note_name
        key = (note_name, art)
        take = self._take_counters.get(key, 0) + 1
        self._take_counters[key] = take

        filename = f"{note_name}_{art}_{take:03d}.wav"
        out_path = self._root / art / filename

        sf.write(
            out_path,
            note.candidate.audio,
            self._sr,
            subtype=self._subtype,
        )

        f = note.features
        self._csv_writer.writerow([
            filename,
            note_name,
            note.pitch.midi_note,
            art,
            f"{note.pitch.hz:.3f}",
            f"{note.pitch.deviation_cents:.2f}",
            f"{f.total_duration_ms:.1f}",
            f"{f.attack_duration_ms:.2f}",
            f"{f.decay_rate_db_per_ms:.4f}",
            f"{f.pitch_modulation_depth_cents:.2f}",
            f"{f.pitch_modulation_rate_hz:.2f}",
            note.candidate.source_file.name,
            note.candidate.onset_sample,
        ])
        self._csv_file.flush()

        logger.debug("Wrote accepted note: %s", out_path)
        return out_path

    def write_rejected(self, note: RejectedNote) -> Path:
        """Write a JSON sidecar for a rejected note candidate.

        Args:
            note: A rejected note from the pipeline.

        Returns:
            Path to the written JSON file.
        """
        reason = note.reason.value
        cand = note.candidate
        key = (f"unknown_{cand.candidate_index}", reason)
        take = self._take_counters.get(key, 0) + 1
        self._take_counters[key] = take

        filename = f"candidate_{cand.candidate_index:05d}_rejected_{reason}.json"
        out_path = self._root / config.REJECTED_DIR / reason / filename

        metadata = {
            "reason": reason,
            "detail": note.detail,
            "source_file": str(cand.source_file),
            "onset_sample": cand.onset_sample,
            "candidate_index": cand.candidate_index,
            "audio_frames": cand.audio.shape[0],
            "channels": cand.audio.shape[1] if cand.audio.ndim == 2 else 1,
        }
        out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.debug("Wrote rejected note sidecar: %s", out_path)
        return out_path

    def close(self) -> None:
        """Flush and close the CSV index file."""
        self._csv_file.close()

    def __enter__(self) -> "SampleWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def detect_subtype(path: Path) -> str:
    """Return the soundfile subtype string of an existing audio file.

    Used to preserve the original bit depth when writing output files.

    Args:
        path: Path to an existing audio file.

    Returns:
        Subtype string (e.g. ``"PCM_24"``, ``"FLOAT"``).
    """
    info = sf.info(path)
    return info.subtype
