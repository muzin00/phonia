from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from scripts.run_julius_alignment import (
    parse_alignment_log,
    resample_wave,
    to_julius_phonemes,
    validate_intervals,
    write_alignment_inputs,
)


class PhonemeConversionTest(unittest.TestCase):
    def test_maps_only_pause_label(self) -> None:
        self.assertEqual(
            to_julius_phonemes(["m", "a", "pau", "o", "o"]),
            ["m", "a", "sp", "o", "o"],
        )


class InputGenerationTest(unittest.TestCase):
    def test_writes_linear_grammar_and_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stem = Path(temporary_directory) / "sample"
            dfa, dictionary = write_alignment_inputs(stem, ["m", "a"])

            self.assertEqual(
                dfa.read_text(encoding="utf-8"),
                "0 2 1 0 1\n1 1 2 0 0\n2 0 3 0 0\n3 -1 -1 1 0\n",
            )
            self.assertEqual(
                dictionary.read_text(encoding="utf-8"),
                "0 [w_0] silB\n1 [w_1] m a\n2 [w_2] silE\n",
            )


class WaveConversionTest(unittest.TestCase):
    def test_resamples_to_mono_16khz_pcm(self) -> None:
        source_samples = array(
            "h",
            [
                round(10_000 * math.sin(2 * math.pi * index / 40))
                for index in range(24_000)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "source.wav"
            output_path = temporary_path / "output.wav"
            with wave.open(str(source_path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(24_000)
                destination.writeframes(source_samples.tobytes())

            metadata = resample_wave(source_path, output_path)

            with wave.open(str(output_path), "rb") as converted:
                self.assertEqual(converted.getframerate(), 16_000)
                self.assertEqual(converted.getnchannels(), 1)
                self.assertEqual(converted.getsampwidth(), 2)
                self.assertEqual(converted.getnframes(), 16_000)
            self.assertEqual(metadata["source_sample_rate_hz"], 24_000)
            self.assertEqual(metadata["derived_sample_rate_hz"], 16_000)


class AlignmentParsingTest(unittest.TestCase):
    LOG = """\
=== begin forced alignment ===
-- phoneme alignment --
[   0   10]  -20.0 silB
[  11   20]  -21.5 a
[  21   30]  -22.0 silE
=== end forced alignment ===
"""

    def test_parses_frames_scores_and_offset_times(self) -> None:
        intervals = parse_alignment_log(self.LOG)

        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0]["start_sec"], 0.0)
        self.assertEqual(intervals[0]["end_sec"], 0.1225)
        self.assertEqual(intervals[1]["start_sec"], 0.1225)
        self.assertEqual(intervals[1]["score"], -21.5)

    def test_rejects_incomplete_log(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "completed forced alignment"):
            parse_alignment_log("=== begin forced alignment ===")

    def test_validates_expected_labels_and_time_order(self) -> None:
        intervals = parse_alignment_log(self.LOG)

        validate_intervals(intervals, ["a"], 0.4)

        with self.assertRaisesRegex(RuntimeError, "differ from input"):
            validate_intervals(intervals, ["i"], 0.4)


if __name__ == "__main__":
    unittest.main()
