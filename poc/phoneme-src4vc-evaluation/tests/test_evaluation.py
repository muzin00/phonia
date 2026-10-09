"""Frozen test gate and paired comparisons with both reused references."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import src_evaluation as study
from diagnostics import paired_decisions


class EvaluationTests(unittest.TestCase):
    def test_protocol_rejects_test_tuning_or_changed_scoring_conditions(self):
        good = study.read_json(study.CONFIG)
        study.validate_protocol(good)
        for change in (
            {"test_threshold_recalibration": True},
            {"enrollment_count_per_vowel": 30},
            {"required_vowels": ["a", "i", "u"]},
            {"no_score": "drop"},
            {"retrain_encoder": True},
            {"checkpoint_selection": "test_best"},
        ):
            with self.assertRaises(ValueError):
                study.validate_protocol({**good, **change})

    def test_decision_transitions_include_equality_and_no_score_rejection(self):
        before, after = [], []
        for genuine in (True, False):
            for index, (a, b) in enumerate(
                ((0.5, 0.6), (0.6, 0.4), (0.4, 0.6), (0.4, 0.4), (None, None))
            ):
                common = {
                    "trial_id": f"{genuine}/{index}",
                    "is_genuine": genuine,
                    "speaker_id": "s1",
                    "claimed_speaker_id": "s1" if genuine else "s2",
                    "status": "scored" if a is not None else "no_score",
                }
                before.append({**common, "score": a})
                after.append({**common, "score": b})
        result = paired_decisions(before, after, 0.5, 0.5)
        expected = {
            "all_trials": 5,
            "no_score": 1,
            "accepted_both": 1,
            "accepted_before_only": 1,
            "accepted_after_only": 1,
            "rejected_both": 2,
        }
        self.assertEqual(result["counts"], {"genuine": expected, "impostor": expected})
        with self.assertRaises(ValueError):
            paired_decisions(before, after[:-1], 0.5, 0.5)
        with self.assertRaises(ValueError):
            paired_decisions(
                before, [{**after[0], "is_genuine": False}, *after[1:]], 0.5, 0.5
            )

    def test_test_requires_frozen_validation_and_rejects_changed_threshold(self):
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
            with patch.object(study, "ROOT", run):
                study.frozen_inputs(run, "validation")
                with self.assertRaises(FileNotFoundError):
                    study.frozen_inputs(run, "test")
                threshold = run / "threshold.json"
                threshold.write_text("fixed")
                (run / "evaluation-freeze.json").write_text(
                    json.dumps(
                        {
                            "status": "thresholds_frozen_before_new_test_inference",
                            "files": {"threshold.json": study.sha256_file(threshold)},
                        }
                    )
                )
                study.frozen_inputs(run, "test")
                threshold.write_text("changed")
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "test")

    def test_paired_primary_direction_and_shared_replicates(self):
        cells, replicas = {}, {}
        for role in study.ROLES:
            for condition, value in zip(study.CONDITIONS, (0.3, 0.2, 0.1)):
                key = f"{condition}/{role}"
                cells[key] = {
                    "pooled_eer": value,
                    "query_coverage": 1,
                    "operating_points": {
                        p: {m: value for m in study.METRICS} for p, _ in study.POINTS
                    },
                }
                replicas[key] = {
                    f"{p}/{m}": np.array([value, value + 0.01, value + 0.02])
                    for p, _ in study.POINTS
                    for m in study.METRICS
                }
                replicas[key].update(
                    pooled_eer=np.array([value] * 3), query_coverage=np.ones(3)
                )
        differences, arrays = study.paired(cells, replicas)
        self.assertEqual(len(differences), 56)
        key = "src70_u30000_minus_cv70_u30000/verification/far_1pct/all_input_frr"
        self.assertTrue(differences[key]["primary"])
        np.testing.assert_allclose(arrays[key], -10)
        self.assertAlmostEqual(differences[key]["difference_percentage_points"], -10)
        self.assertFalse(
            differences["src70_u30000_minus_jvs70_u15000/verification/pooled_eer"][
                "primary"
            ]
        )

    def test_new_score_wrong_label_missing_trial_and_inconsistent_fusion_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            output = run / "validation/jvs70_src4vc70"
            output.mkdir(parents=True)
            trial = {"trial_id": "t", "is_genuine": True, "role": "verification"}
            inputs = {"trials": [trial]}
            scores = output / "scores.jsonl.gz"
            for change in (
                {},
                {"is_genuine": False},
                {"trial_id": "other"},
                {"scores": {"fused": 0.9, **{v: 0.5 for v in study.old.VOWELS}}},
            ):
                if scores.exists():
                    scores.unlink()
                with study.old.writer(scores) as write:
                    write(
                        {
                            **trial,
                            "model_id": "jvs70_src4vc70",
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
                    self.assertEqual(len(study.new_rows(run, "validation", inputs)), 1)
                else:
                    with self.assertRaises(ValueError):
                        study.new_rows(run, "validation", inputs)


if __name__ == "__main__":
    unittest.main()
