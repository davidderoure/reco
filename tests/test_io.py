"""Tests for cello_sampler.io — MIDI helpers and SampleWriter."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cello_sampler.io import SampleWriter, midi_to_note_name
from cello_sampler.models import (
    ArticulationFeatures,
    ArticulationType,
    ClassifiedNote,
    NoteCandidate,
    PitchEstimate,
    RejectedNote,
    RejectionReason,
)


SR = 48_000


# ---------------------------------------------------------------------------
# midi_to_note_name
# ---------------------------------------------------------------------------


class TestMidiToNoteName:
    @pytest.mark.parametrize("midi, expected", [
        (60, "C4"),
        (69, "A4"),
        (57, "A3"),
        (36, "C2"),
        (61, "C#4"),
        (70, "A#4"),
        (48, "C3"),
    ])
    def test_known_notes(self, midi: int, expected: str) -> None:
        assert midi_to_note_name(midi) == expected


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _candidate(index: int = 0) -> NoteCandidate:
    n = int(0.3 * SR)
    audio = np.random.default_rng(index).standard_normal((n, 2)).astype(np.float32)
    return NoteCandidate(
        audio=audio,
        sample_rate=SR,
        onset_sample=index * SR,
        source_file=Path("test_session.wav"),
        candidate_index=index,
    )


def _pitch(note: str = "A3", midi: int = 57, hz: float = 220.0) -> PitchEstimate:
    times = np.linspace(0, 1.0, 100, dtype=np.float32)
    freqs = np.full(100, hz, dtype=np.float32)
    return PitchEstimate(
        hz=hz,
        midi_note=midi,
        note_name=note,
        confidence=0.95,
        deviation_cents=2.0,
        pitch_contour_hz=freqs,
        pitch_contour_times=times,
        is_stable=True,
    )


def _features() -> ArticulationFeatures:
    return ArticulationFeatures(
        attack_duration_ms=30.0,
        decay_rate_db_per_ms=0.05,
        total_duration_ms=1200.0,
        pitch_modulation_depth_cents=5.0,
        pitch_modulation_rate_hz=0.0,
    )


def _classified_note(
    index: int = 0,
    articulation: ArticulationType = ArticulationType.LEGATO,
    note: str = "A3",
) -> ClassifiedNote:
    return ClassifiedNote(
        candidate=_candidate(index),
        pitch=_pitch(note=note),
        articulation=articulation,
        features=_features(),
        take_number=1,
    )


def _rejected_note(index: int = 0, reason: RejectionReason = RejectionReason.POLYPHONIC) -> RejectedNote:
    return RejectedNote(
        candidate=_candidate(index),
        reason=reason,
        detail="HPS detected multiple harmonic series",
    )


# ---------------------------------------------------------------------------
# SampleWriter tests
# ---------------------------------------------------------------------------


class TestSampleWriter:
    def test_accepted_note_creates_wav_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            note = _classified_note(articulation=ArticulationType.LEGATO, note="A3")

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                path = writer.write_accepted(note)

            assert path.exists()
            assert path.suffix == ".wav"
            assert "legato" in str(path)
            assert "A3" in path.name

    def test_wav_file_is_readable_with_correct_sr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            note = _classified_note(articulation=ArticulationType.PIZZICATO, note="D4")

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                path = writer.write_accepted(note)

            audio, sr = sf.read(path)
            assert sr == SR
            assert audio.shape[1] == 2    # 2 channels preserved

    def test_take_counter_increments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            note1 = _classified_note(index=0, articulation=ArticulationType.LEGATO, note="A3")
            note2 = _classified_note(index=1, articulation=ArticulationType.LEGATO, note="A3")

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                p1 = writer.write_accepted(note1)
                p2 = writer.write_accepted(note2)

            assert "001" in p1.name
            assert "002" in p2.name

    def test_different_notes_independent_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            a3 = _classified_note(index=0, note="A3", articulation=ArticulationType.LEGATO)
            d4 = _classified_note(index=1, note="D4", articulation=ArticulationType.LEGATO)
            a3_2 = _classified_note(index=2, note="A3", articulation=ArticulationType.LEGATO)

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                writer.write_accepted(a3)
                writer.write_accepted(d4)
                p = writer.write_accepted(a3_2)

            assert "A3_legato_002" in p.name

    def test_rejected_note_creates_json_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            note = _rejected_note(reason=RejectionReason.POLYPHONIC)

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                path = writer.write_rejected(note)

            assert path.exists()
            assert path.suffix == ".json"
            data = json.loads(path.read_text())
            assert data["reason"] == "polyphonic"
            assert "source_file" in data

    def test_csv_index_created_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            note = _classified_note()

            with SampleWriter(output_dir, sample_rate=SR) as writer:
                writer.write_accepted(note)

            index_path = output_dir / "_index.csv"
            assert index_path.exists()
            with index_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert "note" in rows[0]
            assert "articulation" in rows[0]
            assert "pitch_hz" in rows[0]

    def test_articulation_subdirs_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with SampleWriter(output_dir, sample_rate=SR):
                pass
            for art in ArticulationType:
                assert (output_dir / art.value).is_dir()

    def test_rejected_subdirs_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with SampleWriter(output_dir, sample_rate=SR):
                pass
            for reason in RejectionReason:
                assert (output_dir / "rejected" / reason.value).is_dir()
