"""Guard comparable support, time budgets, trial identities and held-out calibration."""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("pair_study", BASE / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def candidates(size=2400, count=10):
    return [
        {
            "segment_id": f"{p}-{i}",
            "vowel": p,
            "source_file": f"{p}.wav",
            "source_sha256": f"{p}-fixture",
            "start_frame": i * 12000,
            "end_frame": i * 12000 + size,
        }
        for p in study.PHONEMES
        for i in range(count)
    ]


class AblationTest(unittest.TestCase):
    def test_registration_reduces_shared_time_if_a_slice_is_quiet(self):
        def rms(_root, row):
            return -60 if row["end_frame"] - row["start_frame"] == 720 else -10

        config = {
            "minimum_consonant_rms_dbfs": -50,
            "enrollment_budget_step_frames": 120,
        }
        with patch.object(study, "pcm_rms_dbfs", side_effect=rms):
            budget, plans = study.enrollment_plans(candidates(), 72000, config)
        self.assertLess(budget, 72000)
        self.assertEqual(
            {
                sum(r["end_frame"] - r["start_frame"] for r in rows)
                for rows in plans.values()
            },
            {budget},
        )
        self.assertTrue(
            all(rms(None, r) >= -50 for rows in plans.values() for r in rows)
        )

    def test_shortened_stops_keep_the_candidate_end(self):
        rows = candidates(size=2400, count=1)
        planned = study.plan(rows, ("t", "d", "k", "g"), 2880)
        self.assertEqual(sum(r["end_frame"] - r["start_frame"] for r in planned), 2880)
        for row in planned:
            self.assertEqual(row["end_frame"], 2400)
            self.assertEqual(row["start_frame"], 1680)

    def test_same_seven_phone_count_different_consonants_and_equal_time(self):
        rows = candidates()
        plans = study.four_plans(rows, 72000)
        profile = {
            "budget_frames": 72000,
            "conditions": plans,
            "candidate_source_files": sorted({r["source_file"] for r in rows}),
        }
        self.assertEqual(
            study.validate_plans({"profiles": {"s": profile}, "queries": []}), 4
        )
        for name, extra in (
            ("vowels_mn", {"m", "n"}),
            ("vowels_tk", {"t", "k"}),
            ("vowels_dg", {"d", "g"}),
        ):
            self.assertEqual(
                {r["vowel"] for r in plans[name]}, set(study.VOWELS) | extra
            )
        self.assertEqual(len(study.PAIRS), 6)

    def test_missing_k_cannot_be_scored_as_a_complete_pair(self):
        rows = [r for r in candidates() if r["vowel"] != "k"]
        plans = study.four_plans(rows, 24000)
        self.assertIsNotNone(plans["vowels_mn"])
        self.assertIsNone(plans["vowels_tk"])
        self.assertIsNotNone(plans["vowels_dg"])
        with self.assertRaises(ValueError):
            study.validate_plans(
                {
                    "profiles": {},
                    "queries": [
                        {"common11": True, "budget_frames": 24000, "conditions": plans}
                    ],
                }
            )

    def test_budget_repeat_wrong_phone_and_content_leak_rejected(self):
        rows = candidates()
        plans = {c: study.plan(rows, p, 24000) for c, p in study.PHONE_SETS.items()}
        query = {"common11": True, "budget_frames": 24000, "conditions": plans}
        valid = {"profiles": {}, "queries": [query]}
        study.validate_plans(valid)
        for change in ("budget", "repeat", "phone", "leak"):
            broken = copy.deepcopy(valid)
            if change == "budget":
                broken["queries"][0]["budget_frames"] += 1
            elif change == "repeat":
                chosen = broken["queries"][0]["conditions"]["vowels_tk"]
                chosen[1]["original_segment_id"] = chosen[0]["original_segment_id"]
            elif change == "phone":
                broken["queries"][0]["conditions"]["vowels_tk"][-1]["vowel"] = "n"
            else:
                broken["profiles"]["s"] = {
                    "conditions": {"vowels5": [plans["vowels5"][0]]}
                }
            with self.assertRaises(ValueError):
                study.validate_plans(broken)

    def test_trial_matrix_checks_four_conditions_and_labels(self):
        query = {
            "query_id": "q",
            "speaker_id": "s1",
            "role": "verification",
            "split": "test",
            "common11": True,
            "budget_frames": 24000,
        }
        inputs = {"queries": [query], "speakers": ["s1", "s2"]}
        rows = [
            {
                **{
                    k: query[k]
                    for k in ("query_id", "speaker_id", "role", "split", "common11")
                },
                "claimed_speaker_id": speaker,
                "condition": condition,
                "is_genuine": speaker == "s1",
                "status": "scored",
                "phone_scores": dict.fromkeys(phones, 0.5),
                "score": 0.5,
                "used_frames": 24000,
            }
            for condition, phones in study.PHONE_SETS.items()
            for speaker in inputs["speakers"]
        ]
        study.validate_scores(inputs, rows)
        for broken in (
            rows[:-1],
            [*rows, rows[0]],
            [{**rows[0], "is_genuine": False}, *rows[1:]],
            [{**rows[0], "score": 0.1}, *rows[1:]],
        ):
            with self.assertRaises(ValueError):
                study.validate_scores(inputs, broken)

    def test_test_requires_frozen_unchanged_validation_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"fixed": True}
            study.write_json(
                root / "design-freeze.json",
                {
                    "config": config,
                    "files": {},
                    "runtime": study.old.previous.runtime(),
                },
            )
            study.write_json(root / "test-inputs.json", {"split": "test"})
            with self.assertRaises(FileNotFoundError):
                study.frozen(config, root, "test")
            threshold = root / "threshold.json"
            threshold.write_text("fixed")
            study.write_json(
                root / "evaluation-freeze.json",
                {
                    "status": "thresholds_frozen_before_test_inference",
                    "files": {"threshold.json": study.sha256_file(threshold)},
                },
            )
            study.frozen(config, root, "test")
            threshold.write_text("changed")
            with self.assertRaises(ValueError):
                study.frozen(config, root, "test")

    def test_calibration_rejects_test_or_cross_text(self):
        for split, role in (
            ("test", "verification"),
            ("validation", "cross_text_verification"),
        ):
            with self.assertRaises(ValueError):
                study.old.calibrate_cell([{"split": split, "role": role}], ["s1"])


if __name__ == "__main__":
    unittest.main()
