"""Protect metadata-only selection and source-matched short-audio budgets."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from short_input_smoke import add_length_diagnostics, select_cases


class ShortInputSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sources = {
            filename: {
                "source_file": filename,
                "source_sha256": filename,
                "frame_count": 24000,
                "split": "validation",
                "speaker_id": "speaker",
                "evaluation_role": "enrollment",
            }
            for filename in ("short.wav", "long.wav")
        }
        self.records = {
            "enrollment": [
                {
                    "split": "validation",
                    "user_id": "speaker",
                    "enrollment_count": 1,
                    "segments": [
                        {
                            "segment_id": filename,
                            "source_file": filename,
                            "source_sha256": filename,
                            "start_frame": 5000,
                            "end_frame": 5000 + frames,
                        }
                        for filename, frames in (("short.wav", 720), ("long.wav", 2400))
                    ],
                }
            ],
            "queries": [],
        }

    def test_extremes_use_actual_union_budgets_and_same_source(self):
        cases = select_cases(self.records, self.sources)
        self.assertEqual(len(cases), 4)
        durations = {(row["method"], row["budget_frames"]) for row in cases}
        self.assertEqual(
            durations,
            {
                ("ecapa_time_exact", 720),
                ("ecapa_time_exact", 2400),
                ("ecapa_time_context20", 1680),
                ("ecapa_time_context20", 3360),
            },
        )
        for row in cases:
            self.assertEqual(
                row["end_frame"] - row["start_frame"], row["budget_frames"]
            )
            self.assertEqual(row["start_frame"], (24000 - row["budget_frames"]) // 2)
            self.assertEqual(row["purpose"], "actual_comparison_input")

    def test_selection_is_independent_of_input_order(self):
        shuffled = copy.deepcopy(self.records)
        shuffled["enrollment"][0]["segments"].reverse()
        self.assertEqual(
            select_cases(self.records, self.sources),
            select_cases(shuffled, dict(reversed(list(self.sources.items())))),
        )

    def test_test_split_is_rejected_before_audio_access(self):
        sources = copy.deepcopy(self.sources)
        sources["short.wav"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only validation"):
            select_cases(self.records, sources)
        records = copy.deepcopy(self.records)
        records["enrollment"][0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only validation"):
            select_cases(records, self.sources)

    def test_source_role_mismatch_and_no_budget_are_rejected(self):
        sources = copy.deepcopy(self.sources)
        sources["short.wav"]["evaluation_role"] = "verification"
        with self.assertRaisesRegex(ValueError, "speaker or role"):
            select_cases(self.records, sources)
        records = copy.deepcopy(self.records)
        records["enrollment"][0]["segments"] = []
        with self.assertRaisesRegex(ValueError, "no positive"):
            select_cases(records, self.sources)

    def test_length_grid_is_diagnostic_and_cannot_exceed_source(self):
        cases = select_cases(self.records, self.sources)
        combined = add_length_diagnostics(cases, [70, 30, 40])
        self.assertEqual(combined[:4], cases)
        self.assertEqual(
            [row["budget_frames"] for row in combined[4:]], [720, 960, 1680]
        )
        self.assertTrue(
            all("excluded_from_comparison" in row["purpose"] for row in combined[4:])
        )
        self.assertTrue(
            all(row["source"]["source_file"] == "short.wav" for row in combined[4:])
        )
        for lengths in ([30, 30], [0], [30.5]):
            with self.assertRaises(ValueError):
                add_length_diagnostics(cases, lengths)
        with self.assertRaisesRegex(ValueError, "exceeds source"):
            add_length_diagnostics(cases, [1001])


if __name__ == "__main__":
    unittest.main()
