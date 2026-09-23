from __future__ import annotations

import unittest

import numpy as np

from scripts.run_wav2vec2_alignment import (
    ctc_forced_align,
    edit_distance,
    map_phonemes_to_tokens,
    spans_to_phone_intervals,
)


class PhoneMappingTest(unittest.TestCase):
    def test_maps_japanese_phones_and_omits_pause(self) -> None:
        vocabulary = {
            "m": 1,
            "a": 2,
            "tɕ": 3,
            "ɯ": 4,
            "ɲ": 5,
            "q": 6,
            "ɴ": 7,
            "i": 8,
        }
        tokens, token_ids, source_indices = map_phonemes_to_tokens(
            ["m", "a", "pau", "ch", "u", "ny", "cl", "N", "I"], vocabulary
        )

        self.assertEqual(tokens, ["m", "a", "tɕ", "ɯ", "ɲ", "q", "ɴ", "i"])
        self.assertEqual(token_ids, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(source_indices, [0, 1, 3, 4, 5, 6, 7, 8])


class CtcForcedAlignmentTest(unittest.TestCase):
    def test_aligns_repeated_tokens_through_blank(self) -> None:
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
        spans, score = ctc_forced_align(np.log(probabilities), [1, 1])

        self.assertEqual(
            [(span["begin_frame"], span["end_frame"]) for span in spans],
            [(1, 1), (3, 3)],
        )
        self.assertTrue(np.isfinite(score))


class IntervalExpansionTest(unittest.TestCase):
    def test_splits_blank_gap_between_adjacent_phones(self) -> None:
        spans = [
            {"begin_frame": 1, "end_frame": 1, "token_id": 1},
            {"begin_frame": 4, "end_frame": 4, "token_id": 2},
        ]
        intervals = spans_to_phone_intervals(
            ["a", "i"],
            [0, 1],
            spans,
            frame_shift_sec=0.02,
            audio_duration_sec=1.0,
        )

        self.assertEqual(intervals[0]["start_sec"], 0.02)
        self.assertEqual(intervals[0]["end_sec"], 0.06)
        self.assertEqual(intervals[1]["start_sec"], 0.06)
        self.assertEqual(intervals[1]["end_sec"], 0.1)

    def test_preserves_explicit_pause_gap(self) -> None:
        spans = [
            {"begin_frame": 1, "end_frame": 1, "token_id": 1},
            {"begin_frame": 5, "end_frame": 5, "token_id": 2},
        ]
        intervals = spans_to_phone_intervals(
            ["a", "pau", "i"],
            [0, 2],
            spans,
            frame_shift_sec=0.02,
            audio_duration_sec=1.0,
        )

        self.assertEqual(intervals[0]["end_sec"], 0.04)
        self.assertEqual(intervals[1]["start_sec"], 0.04)
        self.assertEqual(intervals[1]["end_sec"], 0.1)
        self.assertEqual(intervals[2]["start_sec"], 0.1)


class EditDistanceTest(unittest.TestCase):
    def test_counts_insertions_deletions_and_substitutions(self) -> None:
        self.assertEqual(edit_distance([1, 2, 3], [1, 4, 3]), 1)
        self.assertEqual(edit_distance([1, 2], [1, 2, 3]), 1)
        self.assertEqual(edit_distance([1, 2, 3], [1, 3]), 1)


if __name__ == "__main__":
    unittest.main()
