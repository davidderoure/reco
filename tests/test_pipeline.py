"""End-to-end pipeline tests using synthesised audio.

CREPE is mocked so the test suite runs without TensorFlow or network access.
The mock returns plausible pitch/confidence values based on the dominant
frequency of each candidate's audio.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from cello_sampler.pipeline import run


SR = 48_000


# ---------------------------------------------------------------------------
# Synthesised multi-note recording fixture
# ---------------------------------------------------------------------------


def _make_recording(
    output_path: Path,
    n_channels: int = 2,
) -> list[float]:
    """Write a short multi-channel WAV file with 4 clear note onsets.

    Returns the expected onset times in seconds.
    """
    onset_times_s = [0.2, 0.7, 1.3, 1.9]
    note_freqs_hz = [220.0, 293.66, 329.63, 220.0]   # A3, D4, E4, A3
    total_s = 2.5
    total_n = int(total_s * SR)

    signal = np.zeros(total_n, dtype=np.float32)

    for onset_s, freq in zip(onset_times_s, note_freqs_hz):
        start = int(onset_s * SR)
        note_n = int(0.35 * SR)
        end = min(total_n, start + note_n)
        t = np.arange(end - start) / SR
        env = np.hanning(end - start)
        note = (0.7 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        signal[start:end] += note

    # Stack to multi-channel.
    audio = np.stack([signal] * n_channels, axis=1)
    sf.write(output_path, audio, SR, subtype="FLOAT")

    return onset_times_s


def _make_crepe_mock(audio_16k: np.ndarray, sr: int, **kwargs) -> tuple:
    """Fake CREPE predict: returns a constant 220 Hz pitch with high confidence."""
    n_frames = max(1, len(audio_16k) // 160)
    times = np.arange(n_frames, dtype=np.float32) * 0.01
    freqs = np.full(n_frames, 220.0, dtype=np.float32)
    confidences = np.full(n_frames, 0.92, dtype=np.float32)
    activation = np.zeros((n_frames, 360), dtype=np.float32)
    return times, freqs, confidences, activation


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    @pytest.fixture
    def tmp_dir(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_accepted_wav_files_created(self, tmp_dir: Path) -> None:
        """Pipeline should write at least one accepted WAV per distinct onset."""
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"

        _make_recording(recording_path)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        assert result.n_accepted >= 1, (
            f"Expected at least 1 accepted note, got {result.n_accepted}"
        )

        # Check WAV files actually exist on disk.
        wav_files = list(output_dir.rglob("*.wav"))
        assert len(wav_files) == result.n_accepted

    def test_wav_files_have_correct_sample_rate(self, tmp_dir: Path) -> None:
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        for wav in output_dir.rglob("*.wav"):
            _, sr = sf.read(wav)
            assert sr == SR, f"Expected {SR} Hz, got {sr} for {wav}"

    def test_wav_files_are_multichannel(self, tmp_dir: Path) -> None:
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path, n_channels=2)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        for wav in output_dir.rglob("*.wav"):
            audio, _ = sf.read(wav)
            assert audio.ndim == 2, f"Expected 2-D multichannel audio in {wav}"
            assert audio.shape[1] == 2

    def test_csv_index_populated(self, tmp_dir: Path) -> None:
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        import csv
        index_path = output_dir / "_index.csv"
        assert index_path.exists()
        with index_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == result.n_accepted

    def test_output_filenames_follow_convention(self, tmp_dir: Path) -> None:
        """Filenames must match {note}{octave}_{articulation}_{take:03d}.wav."""
        import re
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        pattern = re.compile(r"^[A-G]#?\d_(legato|staccato|vibrato|pizzicato)_\d{3}\.wav$")
        for wav in output_dir.rglob("*.wav"):
            assert pattern.match(wav.name), (
                f"Filename '{wav.name}' does not match expected pattern"
            )

    def test_rejected_notes_have_json_sidecars(self, tmp_dir: Path) -> None:
        """Any rejected note should produce a JSON sidecar."""
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path)

        # Make CREPE return low confidence so some notes are rejected.
        def _low_confidence_crepe(audio_16k, sr, **kwargs):
            n_frames = max(1, len(audio_16k) // 160)
            times = np.arange(n_frames, dtype=np.float32) * 0.01
            freqs = np.full(n_frames, 220.0, dtype=np.float32)
            confidences = np.full(n_frames, 0.50, dtype=np.float32)  # below threshold
            activation = np.zeros((n_frames, 360), dtype=np.float32)
            return times, freqs, confidences, activation

        with patch("crepe.predict", side_effect=_low_confidence_crepe):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        json_files = list((output_dir / "rejected").rglob("*.json"))
        assert len(json_files) == result.n_rejected

    def test_processing_result_counts_consistent(self, tmp_dir: Path) -> None:
        recording_path = tmp_dir / "session.wav"
        output_dir = tmp_dir / "output"
        _make_recording(recording_path)

        with patch("crepe.predict", side_effect=_make_crepe_mock):
            result = run(
                input_path=recording_path,
                output_dir=output_dir,
                n_workers=1,
            )

        assert result.n_accepted + result.n_rejected == len(result.accepted) + len(result.rejected)
        assert result.source_file == recording_path
