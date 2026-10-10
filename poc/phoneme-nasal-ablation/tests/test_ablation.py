"""Guard comparable support, time budgets, trial identities and held-out calibration."""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("nasal_study", BASE / "study.py")
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
    def test_four_conditions_share_time_but_isolate_each_nasal(self):
        rows = candidates()
        baseline = {
            c: study.plan(rows, study.PHONE_SETS[c], 72000) for c in study.BASELINES
        }
        plans = study.four_plans(rows, 72000, baseline)
        profile = {
            "budget_frames": 72000,
            "conditions": plans,
            "candidate_source_files": sorted({r["source_file"] for r in rows}),
        }
        self.assertEqual(
            study.validate_plans({"profiles": {"s": profile}, "queries": []}), 4
        )
        self.assertEqual(
            {r["vowel"] for r in plans["vowels_m"]}, set(study.VOWELS) | {"m"}
        )
        self.assertEqual(
            {r["vowel"] for r in plans["vowels_n"]}, set(study.VOWELS) | {"n"}
        )
        self.assertEqual(len(study.PAIRS), 6)

    def test_unfit_single_nasal_cannot_silently_change_cohort(self):
        rows = candidates(size=1000, count=1)
        # Losing m must stop preparation, rather than falling back or dropping q.
        self.assertIsNotNone(study.plan(rows, study.PHONEMES, 5040))
        broken = [r for r in rows if r["vowel"] != "m"]
        baseline = {
            c: study.plan(rows, study.PHONE_SETS[c], 5040) for c in study.BASELINES
        }
        with self.assertRaises(ValueError):
            study.four_plans(broken, 5040, baseline)

    def test_prior_baseline_crop_must_be_identical(self):
        rows = candidates()
        baseline = {
            c: study.plan(rows, study.PHONE_SETS[c], 24000) for c in study.BASELINES
        }
        baseline["vowels5"][0]["start_frame"] += 1
        with self.assertRaisesRegex(ValueError, "baseline source"):
            study.four_plans(rows, 24000, baseline)

    def test_budget_repeat_wrong_phone_and_content_leak_rejected(self):
        rows = candidates()
        plans = {c: study.plan(rows, p, 24000) for c, p in study.PHONE_SETS.items()}
        query = {"common7": True, "budget_frames": 24000, "conditions": plans}
        valid = {"profiles": {}, "queries": [query]}
        study.validate_plans(valid)
        for change in ("budget", "repeat", "phone", "leak"):
            broken = copy.deepcopy(valid)
            if change == "budget":
                broken["queries"][0]["budget_frames"] += 1
            elif change == "repeat":
                chosen = broken["queries"][0]["conditions"]["vowels_m"]
                chosen[1]["original_segment_id"] = chosen[0]["original_segment_id"]
            elif change == "phone":
                broken["queries"][0]["conditions"]["vowels_m"][-1]["vowel"] = "n"
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
            "common7": True,
            "budget_frames": 24000,
        }
        inputs = {"queries": [query], "speakers": ["s1", "s2"]}
        rows = [
            {
                **{
                    k: query[k]
                    for k in ("query_id", "speaker_id", "role", "split", "common7")
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
