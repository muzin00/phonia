from __future__ import annotations

import unittest

from scripts.normalize_wav2vec2_alignment import extract_vowel_intervals


class ExtractVowelIntervalsTest(unittest.TestCase):
    def test_normalizes_devoiced_vowels_and_preserves_flag(self) -> None:
        record = {
            "utterance_id": "sample",
            "speaker_id": "speaker",
            "session_id": "session",
            "source_file": "sample.wav",
            "transcript": "test",
            "duration_sec": 1.0,
        }
        entries = [
            {
                "phoneme": "I",
                "start_sec": 0.1,
                "end_sec": 0.2,
                "mean_log_probability": -1.0,
            },
            {
                "phoneme": "a",
                "start_sec": 0.2,
                "end_sec": 0.3,
                "mean_log_probability": -2.0,
            },
        ]

        intervals = extract_vowel_intervals(record, entries, "1.30.0")

        self.assertEqual(
            [interval["normalized_phoneme"] for interval in intervals], ["i", "a"]
        )
        self.assertTrue(intervals[0]["is_devoiced"])
        self.assertFalse(intervals[1]["is_devoiced"])
