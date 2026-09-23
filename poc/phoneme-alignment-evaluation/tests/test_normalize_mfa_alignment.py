from __future__ import annotations

import unittest

from scripts.normalize_mfa_alignment import (
    classify_mfa_vowel,
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


class ClassifyMfaVowelTest(unittest.TestCase):
    def test_normalizes_mfa_vowels(self) -> None:
        self.assertEqual(classify_mfa_vowel("oː"), ("o", 2, False))
        self.assertEqual(classify_mfa_vowel("ɨ"), ("u", 1, False))
        self.assertEqual(classify_mfa_vowel("ɯ̥"), ("u", 1, True))
        self.assertIsNone(classify_mfa_vowel("m"))


class ExtractVowelIntervalsTest(unittest.TestCase):
    def test_keeps_contiguous_identical_vowels_separate(self) -> None:
        entries = [
            [0.0, 0.1, "mʲ"],
            [0.1, 0.3, "oː"],
            [0.3, 0.4, "oː"],
            [0.4, 0.5, "t"],
            [0.5, 0.6, "o"],
        ]

        intervals = extract_vowel_intervals(make_record(), entries)

        self.assertEqual(len(intervals), 3)
        self.assertEqual(intervals[0]["raw_phonemes"], ["oː"])
        self.assertEqual(intervals[1]["raw_phonemes"], ["oː"])
        self.assertEqual([item["expected_units"] for item in intervals], [2, 2, 1])
        self.assertEqual(intervals[0]["start_sec"], 0.1)
        self.assertEqual(intervals[0]["end_sec"], 0.3)
        self.assertEqual(intervals[0]["duration_sec"], 0.2)
        self.assertEqual(
            [item["vowel_interval_id"] for item in intervals],
            [
                "sample_001-mfa-vowel-000",
                "sample_001-mfa-vowel-001",
                "sample_001-mfa-vowel-002",
            ],
        )

    def test_does_not_merge_across_unlabelled_silence(self) -> None:
        entries = [[0.0, 0.1, "o"], [0.3, 0.4, "o"]]

        intervals = extract_vowel_intervals(make_record(), entries)

        self.assertEqual(len(intervals), 2)

    def test_rejects_invalid_intervals(self) -> None:
        entries = [[0.2, 0.3, "a"], [0.1, 0.2, "i"]]

        with self.assertRaisesRegex(ValueError, "Invalid MFA interval"):
            extract_vowel_intervals(make_record(), entries)

    def test_preserves_devoicing(self) -> None:
        entries = [[0.0, 0.1, "ɨ̥"]]

        run = extract_vowel_intervals(make_record(), entries)[0]

        self.assertTrue(run["is_devoiced"])
        self.assertTrue(run["contains_devoiced"])


class ValidateExpectedTest(unittest.TestCase):
    def test_accepts_matching_expanded_sequence(self) -> None:
        intervals = [
            {"normalized_phoneme": "a", "expected_units": 1},
            {"normalized_phoneme": "o", "expected_units": 2},
        ]
        expected = {"sample_001": [("a", 1), ("o", 1), ("o", 1)]}

        validate_expected("sample_001", intervals, expected)

    def test_rejects_different_sequence(self) -> None:
        intervals = [{"normalized_phoneme": "a", "expected_units": 1}]
        expected = {"sample_001": [("i", 1)]}

        with self.assertRaisesRegex(ValueError, "differ from expected"):
            validate_expected("sample_001", intervals, expected)


if __name__ == "__main__":
    unittest.main()
