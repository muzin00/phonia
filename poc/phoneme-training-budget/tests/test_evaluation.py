"""Protect the common populations, all-validation test gate and paired direction."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget_evaluation as study
import numpy as np


class EvaluationTests(unittest.TestCase):
    def test_only_preregistered_sixty_thousand_trajectory_snapshots_are_valid(self):
        self.assertEqual(study.specification("src_u45000_s60k"), ("src", 45000))
        for condition in ("src70_u30000", "src_u15000_s60k", "cv_u60000_s30k"):
            with self.assertRaises(ValueError):
                study.specification(condition)

    def test_comparison_cannot_select_checkpoints_or_tune_thresholds_on_test(self):
        config = study.read_json(study.training.CONFIG)
        evaluation = study.read_json(study.CONFIG)
        study.validate_protocol(config, evaluation)
        for change in (
            {"enrollment_count_per_vowel": 30},
            {"test_threshold_recalibration": True},
            {"no_score": "drop"},
        ):
            with self.assertRaises(ValueError):
                study.validate_protocol({**config, **change}, evaluation)
        with self.assertRaises(ValueError):
            study.validate_protocol(
                config, {**evaluation, "checkpoint_selection": "validation_best"}
            )

    def test_test_gate_requires_all_validation_thresholds_and_unchanged_files(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            pinned = run / "input.txt"
            pinned.write_text("fixed")
            (run / "design-freeze.json").write_text(
                json.dumps(
                    {
                        "status": "design_frozen_before_new_inference",
                        "runtime": study.old.previous.runtime(),
                        "files": {"input.txt": study.sha256_file(pinned)},
                    }
                )
            )
            for split in ("validation", "test"):
                (run / f"{split}-inputs.json").write_text(json.dumps({"split": split}))
            with patch.object(study.training, "ROOT", run):
                study.frozen_inputs(run, "validation")
                with self.assertRaises(FileNotFoundError):
                    study.frozen_inputs(run, "test")
                threshold = run / "threshold.json"
                threshold.write_text("fixed")
                (run / "evaluation-freeze.json").write_text(
                    json.dumps(
                        {
                            "status": "all_validation_thresholds_frozen_before_any_new_test_inference",
                            "files": {"threshold.json": study.sha256_file(threshold)},
                        }
                    )
                )
                study.frozen_inputs(run, "test")
                threshold.write_text("changed")
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "test")

    def test_paired_differences_use_later_minus_thirty_thousand_with_shared_draws(self):
        cells, replicas = {}, {}
        for role in study.ROLES:
            for corpus in ("cv", "src"):
                for update, value in ((30000, 0.3), (45000, 0.2), (60000, 0.1)):
                    key = f"{corpus}_u{update}_s60k/{role}"
                    cells[key] = {
                        "pooled_eer": value,
                        "query_coverage": 1,
                        "operating_points": {
                            p: {m: value for m in study.METRICS}
                            for p, _ in study.POINTS
                        },
                    }
                    replicas[key] = {
                        f"{p}/{m}": np.array([value, value + 0.01])
                        for p, _ in study.POINTS
                        for m in study.METRICS
                    }
                    replicas[key].update(
                        pooled_eer=np.array([value, value]), query_coverage=np.ones(2)
                    )
        differences, arrays = study.paired(
            cells, replicas, study.read_json(study.CONFIG)
        )
        self.assertEqual(len(differences), 112)
        key = (
            "src_u60000_s60k_minus_src_u30000_s60k/verification/far_1pct/all_input_frr"
        )
        self.assertTrue(differences[key]["primary"])
        self.assertAlmostEqual(differences[key]["difference_percentage_points"], -20)
        np.testing.assert_allclose(arrays[key], -20)
        self.assertFalse(differences[key.replace("60000", "45000")]["primary"])

    def test_scores_reject_wrong_model_changed_truth_and_missing_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            condition = "cv_u30000_s60k"
            output = run / "validation" / condition
            output.mkdir(parents=True)
            trial = {"trial_id": "t", "is_genuine": True, "role": "verification"}
            inputs = {"trials": [trial]}
            scores = output / "scores.jsonl.gz"
            for change in (
                {},
                {"model_id": "cv_u45000_s60k"},
                {"is_genuine": False},
                {"trial_id": "other"},
                {"scores": None},
            ):
                if scores.exists():
                    scores.unlink()
                with study.old.writer(scores) as write:
                    write(
                        {
                            **trial,
                            "model_id": condition,
                            "status": "scored",
                            "scores": {
                                "fused": 0.5,
                                **{v: 0.5 for v in study.old.VOWELS},
                            },
                            **change,
                        }
                    )
                (output / "report.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "outputs_sha256": {
                                "scores.jsonl.gz": study.sha256_file(scores)
                            },
                        }
                    )
                )
                if not change:
                    self.assertEqual(
                        len(study.new_rows(run, "validation", inputs, condition)), 1
                    )
                else:
                    with self.assertRaises(ValueError):
                        study.new_rows(run, "validation", inputs, condition)


if __name__ == "__main__":
    unittest.main()
