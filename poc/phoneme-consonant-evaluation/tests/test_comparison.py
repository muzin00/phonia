"""Exact duration, availability fallback and calibration-leak guards."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study
from matched_audio import PHONEMES, VOWELS, matched_query, plan


def candidates(phones=PHONEMES, size=2400, count=10):
    return [
        {
            "segment_id": f"{p}-{i}",
            "vowel": p,
            "source_file": f"{p}.wav",
            "source_sha256": "fixture",
            "start_frame": i * 12000,
            "end_frame": i * 12000 + size,
        }
        for p in phones
        for i in range(count)
    ]


class ComparisonTest(unittest.TestCase):
    def test_trial_labels_duplicates_and_fused_scores_are_checked(self):
        selected = candidates(("a",), count=1)
        query = {
            "query_id": "q",
            "speaker_id": "s1",
            "role": "verification",
            "split": "test",
            "common7": False,
            "conditions": {c: selected for c in study.CONDITIONS},
            "budget_frames": 2400,
        }
        inputs = {"queries": [query], "speakers": ["s1", "s2"]}
        rows = [
            {
                "query_id": "q",
                "speaker_id": "s1",
                "claimed_speaker_id": speaker,
                "role": "verification",
                "split": "test",
                "common7": False,
                "condition": condition,
                "is_genuine": speaker == "s1",
                "status": "scored",
                "score": 0.5,
                "phone_scores": {"a": 0.5},
                "used_frames": 2400,
            }
            for condition in study.CONDITIONS
            for speaker in inputs["speakers"]
        ]
        study.validate_scores(inputs, rows)
        for change in (
            {"is_genuine": False},
            {"score": 0.2},
            {"used_frames": 2399},
            {"phone_scores": {"n": 0.5}},
        ):
            with self.assertRaises(ValueError):
                study.validate_scores(inputs, [{**rows[0], **change}, *rows[1:]])
        for broken in (rows[:-1], [*rows, rows[0]]):
            with self.assertRaises(ValueError):
                study.validate_scores(inputs, broken)

    def test_equal_time_and_valid_center_bounds(self):
        rows = candidates()
        a = plan(rows, VOWELS, 72000)
        b = plan(rows, PHONEMES, 72000)
        self.assertEqual(sum(s["end_frame"] - s["start_frame"] for s in a), 72000)
        self.assertEqual(sum(s["end_frame"] - s["start_frame"] for s in b), 72000)
        originals = {r["segment_id"]: r for r in rows}
        for chosen in (a, b):
            for row in chosen:
                source = originals[row["original_segment_id"]]
                self.assertGreaterEqual(row["start_frame"], source["start_frame"])
                self.assertLessEqual(row["end_frame"], source["end_frame"])
                self.assertGreaterEqual(row["end_frame"] - row["start_frame"], 720)
        self.assertEqual(a, plan(rows, VOWELS, 72000))

    def test_subminimum_residual_is_reallocated_not_padded(self):
        chosen = plan(candidates(("a",), size=1400, count=2), ("a",), 1800)
        self.assertEqual(
            [r["end_frame"] - r["start_frame"] for r in chosen], [1080, 720]
        )
        self.assertIsNone(plan(candidates(("a",), size=1000, count=2), ("a",), 1200))

    def test_missing_optional_nasals_and_short_budget_fall_back(self):
        vowels = candidates(VOWELS, size=720, count=1)
        plans, common = matched_query(vowels, [], 24000)
        self.assertFalse(common)
        self.assertEqual(plans[study.CONDITIONS[0]], plans[study.CONDITIONS[1]])
        plans, common = matched_query(
            vowels, candidates(("m", "n"), size=720, count=1), 24000
        )
        self.assertFalse(common)
        self.assertEqual(plans[study.CONDITIONS[0]], plans[study.CONDITIONS[1]])

    def test_complete_seven_and_missing_vowel_support(self):
        rows = candidates()
        plans, common = matched_query(
            [r for r in rows if r["vowel"] in VOWELS],
            [r for r in rows if r["vowel"] in ("m", "n")],
            24000,
        )
        self.assertTrue(common)
        self.assertEqual(
            {r["vowel"] for r in plans[study.CONDITIONS[1]]}, set(PHONEMES)
        )
        plans, _ = matched_query(
            [r for r in rows if r["vowel"] in VOWELS[1:]], [], 24000
        )
        self.assertTrue(all(p is None for p in plans.values()))

    def test_test_requires_unchanged_validation_thresholds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"fixed": True}
            study.write_json(
                root / "design-freeze.json",
                {
                    "config": config,
                    "runtime": study.old.previous.runtime(),
                    "files": {},
                },
            )
            study.write_json(root / "test-inputs.json", {"split": "test"})
            with self.assertRaises(FileNotFoundError):
                study.frozen(config, root, "test")
            threshold = root / "thresholds.json"
            threshold.write_text("fixed")
            study.write_json(
                root / "evaluation-freeze.json",
                {
                    "status": "thresholds_frozen_before_test_inference",
                    "files": {"thresholds.json": study.sha256_file(threshold)},
                },
            )
            self.assertEqual(study.frozen(config, root, "test")["split"], "test")
            threshold.write_text("changed")
            with self.assertRaises(ValueError):
                study.frozen(config, root, "test")

    def test_calibration_rejects_test_and_cross_text(self):
        for split, role in (
            ("test", "verification"),
            ("validation", "cross_text_verification"),
        ):
            with self.assertRaises(ValueError):
                study.old.calibrate_cell([{"split": split, "role": role}], ["s1"])


if __name__ == "__main__":
    unittest.main()
