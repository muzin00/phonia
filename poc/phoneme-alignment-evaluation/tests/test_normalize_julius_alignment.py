from __future__ import annotations

import unittest

from scripts.normalize_julius_alignment import (
    classify_julius_vowel,
    extract_vowel_intervals,
    validate_expected,
)


def make_record(duration_sec: float = 2.0) -> dict[str, object]:
    return {
        "utterance_id": "sample_001",
        "speaker_id": "speaker_001",
        "session_id": "session_001",
        "source_file": "sample.wav",
        "transcript": "sample",
        "duration_sec": duration_sec,
    }


def entry(phone: str, start: float, end: float) -> dict[str, object]:
    return {
        "phoneme": phone,
        "start_sec": start,
        "end_sec": end,
        "begin_frame": round(start * 100),
        "end_frame": round(end * 100) - 1,
        "score": -20.0,
    }


class ClassifyJuliusVowelTest(unittest.TestCase):
    def test_normalizes_only_five_vowels(self) -> None:
        self.assertEqual(classify_julius_vowel("o"), ("o", 1, False))
        self.assertIsNone(classify_julius_vowel("o:"))
        self.assertIsNone(classify_julius_vowel("m"))


class ExtractVowelIntervalsTest(unittest.TestCase):
    def test_keeps_contiguous_identical_vowels_separate(self) -> None:
        entries = [
            entry("my", 0.0, 0.1),
            entry("o", 0.1, 0.2),
            entry("o", 0.2, 0.3),
            entry("o", 0.3, 0.4),
        ]

        intervals = extract_vowel_intervals(make_record(), entries)

        self.assertEqual(len(intervals), 3)
        self.assertEqual([item["expected_units"] for item in intervals], [1, 1, 1])
        self.assertEqual(
            [item["vowel_interval_id"] for item in intervals],
            [
                "sample_001-julius-vowel-000",
                "sample_001-julius-vowel-001",
                "sample_001-julius-vowel-002",
            ],
        )
        self.assertEqual(intervals[0]["score_kind"], "julius_n_score")

    def test_rejects_invalid_intervals(self) -> None:
        entries = [entry("a", 0.2, 0.3), entry("i", 0.1, 0.2)]

        with self.assertRaisesRegex(ValueError, "Invalid Julius interval"):
            extract_vowel_intervals(make_record(), entries)


class ValidateExpectedTest(unittest.TestCase):
    def test_accepts_matching_sequence(self) -> None:
        intervals = [
            {"normalized_phoneme": "a"},
            {"normalized_phoneme": "o"},
        ]

        validate_expected("sample_001", intervals, {"sample_001": ["a", "o"]})

    def test_rejects_different_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ from expected"):
            validate_expected(
                "sample_001",
                [{"normalized_phoneme": "a"}],
                {"sample_001": ["i"]},
            )


if __name__ == "__main__":
    unittest.main()
