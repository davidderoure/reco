"""Tests for cello_sampler.pitch — pitch helpers (not full CREPE integration).

The CREPE model requires a TensorFlow installation and network access on first
use, so we do not call ``estimate_pitch`` in the unit tests.  Instead we test
all the pure-Python helper functions that wrap CREPE's output.

CREPE integration is exercised in ``test_pipeline.py`` (marked as slow/optional).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cello_sampler.pitch import (
    _deviation_cents,
    _hz_to_midi,
    _hz_to_note_name,
    _nearest_et_hz,
    _stable_region_indices,
)


class TestHzToMidi:
    def test_a4(self) -> None:
        assert _hz_to_midi(440.0) == 69

    def test_a3(self) -> None:
        assert _hz_to_midi(220.0) == 57

    def test_c4_middle_c(self) -> None:
        assert _hz_to_midi(261.63) == 60

    def test_rounds_to_nearest_semitone(self) -> None:
        # 450 Hz is between A4 (440) and A#4/Bb4 (466.16).
        # 450 is 38.9 cents sharp of A4 and 61.1 cents flat of Bb4.
        # Nearest semitone is A4.
        assert _hz_to_midi(450.0) == 69

    def test_low_c2(self) -> None:
        assert _hz_to_midi(65.41) == 36


class TestNearestEtHz:
    def test_a4_returns_440(self) -> None:
        assert abs(_nearest_et_hz(440.0) - 440.0) < 0.01

    def test_slightly_sharp_returns_a4(self) -> None:
        """450 Hz is nearest to A4 (440 Hz) not Bb4 (466 Hz)."""
        assert abs(_nearest_et_hz(450.0) - 440.0) < 0.01

    def test_d4(self) -> None:
        # D4 = 293.665 Hz in 12-TET.
        et = _nearest_et_hz(295.0)
        assert abs(et - 293.665) < 0.5


class TestDeviationCents:
    def test_zero_deviation_on_exact_pitch(self) -> None:
        dev = _deviation_cents(440.0, 440.0)
        assert abs(dev) < 0.001

    def test_positive_deviation_sharp(self) -> None:
        # 450 Hz vs 440 Hz reference.
        dev = _deviation_cents(450.0, 440.0)
        expected = 1200.0 * math.log2(450.0 / 440.0)
        assert abs(dev - expected) < 0.01
        assert dev > 0

    def test_negative_deviation_flat(self) -> None:
        dev = _deviation_cents(430.0, 440.0)
        assert dev < 0

    def test_one_semitone_is_100_cents(self) -> None:
        # A4 to A#4.
        a4 = 440.0
        bb4 = 440.0 * (2.0 ** (1.0 / 12.0))
        dev = _deviation_cents(bb4, a4)
        assert abs(dev - 100.0) < 0.01

    def test_known_deviation(self) -> None:
        """450 Hz is approximately +38.9 cents sharp of A4."""
        dev = _deviation_cents(450.0, 440.0)
        assert abs(dev - 38.9) < 1.0


class TestHzToNoteName:
    @pytest.mark.parametrize("midi, expected", [
        (60, "C4"),    # middle C
        (69, "A4"),    # concert A
        (57, "A3"),
        (36, "C2"),    # lowest cello string
        (93, "A6"),    # near highest cello note
        (61, "C#4"),   # sharp
        (70, "A#4"),   # Bb4
    ])
    def test_known_notes(self, midi: int, expected: str) -> None:
        assert _hz_to_note_name(midi) == expected


class TestStableRegionIndices:
    def test_trims_20_percent_each_end(self) -> None:
        start, end = _stable_region_indices(100, trim=0.20)
        assert start == 20
        assert end == 80

    def test_start_less_than_end(self) -> None:
        for n in [5, 10, 50, 100, 1000]:
            start, end = _stable_region_indices(n)
            assert start < end

    def test_minimum_one_frame(self) -> None:
        start, end = _stable_region_indices(2)
        assert end > start

    def test_full_coverage_with_zero_trim(self) -> None:
        start, end = _stable_region_indices(100, trim=0.0)
        assert start == 0
        assert end == 100
