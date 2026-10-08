"""Frozen thresholds, paired populations and missing-score denominators."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation_study as study
import numpy as np


def rows():
    return [
        {
            "query_id": f"{speaker}-{index}",
            "speaker_id": speaker,
            "claimed_speaker_id": claimed,
            "split": "validation",
            "role": "verification",
            "is_genuine": speaker == claimed,
            "status": "scored" if score is not None else "no_score",
            "score": score if speaker == claimed or score is None else 0.2,
        }
        for speaker, values in (("s1", (0.8, 0.5, None)), ("s2", (0.7, 0.1, None)))
        for index, score in enumerate(values)
        for claimed in ("s1", "s2")
    ]


class EvaluationTests(unittest.TestCase):
    def test_calibration_cannot_use_test_or_cross_text(self):
        for key, value in (("split", "test"), ("role", "cross_text_verification")):
            with self.assertRaises(ValueError):
                study.calibrate_cell([{**r, key: value} for r in rows()], ["s1", "s2"])

    def test_missing_scores_remain_in_denominators_and_equality_accepts(self):
        thresholds = {"operating_points": {"far_1pct": {"threshold": 0.5}}}
        cell, _ = study.evaluate_cell(rows(), thresholds, ["s1", "s2"])
        p = cell["operating_points"]["far_1pct"]
        self.assertEqual(p["frr"], 1 / 4)
        self.assertEqual(p["all_input_frr"], 3 / 6)
        self.assertEqual(p["no_score_genuine"], 2)
        self.assertEqual(p["false_rejects"], 1)
        self.assertEqual(p["false_accepts"], 0)

    def test_paired_draws_preserve_direction_and_percentage_point_units(self):
        cells, replicas = {}, {}
        for role in study.ROLES:
            for model, value in (("jvs70", 0.2), ("jvs70_cv70", 0.1)):
                key = f"{model}/{role}"
                cells[key] = {
                    "pooled_eer": value,
                    "query_coverage": 1,
                    "operating_points": {
                        point: {m: value for m in study.METRICS}
                        for point, _ in study.POINTS
                    },
                }
                replicas[key] = {
                    f"{point}/{m}": np.array([value, value + 0.01, value + 0.02])
                    for point, _ in study.POINTS
                    for m in study.METRICS
                }
                replicas[key].update(
                    pooled_eer=np.array([value] * 3), query_coverage=np.ones(3)
                )
        differences, arrays = study.paired(cells, replicas)
        self.assertEqual(len(differences), 28)
        key = "verification/far_1pct/all_input_frr"
        np.testing.assert_allclose(arrays[key], -10)
        self.assertAlmostEqual(differences[key]["difference_percentage_points"], -10)
        self.assertEqual(
            differences["verification/query_coverage"]["difference_percentage_points"],
            0,
        )

    def fixture(self, folder):
        run = Path(folder)
        pinned = run / "input.txt"
        pinned.write_text("fixed")
        (run / "design-freeze.json").write_text(
            json.dumps(
                {
                    "status": "design_frozen_before_new_inference",
                    "runtime": study.previous.runtime(),
                    "files": {"input.txt": study.sha256_file(pinned)},
                }
            )
        )
        for split in ("validation", "test"):
            (run / f"{split}-inputs.json").write_text(json.dumps({"split": split}))
        return run

    def test_new_test_inference_requires_validation_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            with patch.object(study, "ROOT", run):
                self.assertEqual(
                    study.frozen_inputs(run, "validation")["split"], "validation"
                )
                with self.assertRaises(FileNotFoundError):
                    study.frozen_inputs(run, "test")

    def test_changed_threshold_or_audio_fails_before_test(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            threshold = run / "thresholds.json"
            threshold.write_text("fixed")
            (run / "evaluation-freeze.json").write_text(
                json.dumps(
                    {
                        "status": "thresholds_frozen_before_new_test_inference",
                        "files": {"thresholds.json": study.sha256_file(threshold)},
                    }
                )
            )
            with patch.object(study, "ROOT", run):
                self.assertEqual(study.frozen_inputs(run, "test")["split"], "test")
                threshold.write_text("changed")
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "test")
                (run / "input.txt").write_text("changed")
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "validation")

    def test_swapped_model_or_label_and_missing_trials_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            trial = {
                "trial_id": "one",
                "query_id": "q",
                "role": "verification",
                "is_genuine": True,
                "enrollment_count": 10,
            }
            (run / "validation-inputs.json").write_text(json.dumps({"trials": [trial]}))
            for model in study.MODELS:
                output = run / "validation" / model
                output.mkdir(parents=True)
                scores = output / "scores.jsonl.gz"
                with study.writer(scores) as write:
                    write(
                        {
                            **trial,
                            "model_id": model,
                            "status": "scored",
                            "scores": {"fused": 0.5, **{v: 0.5 for v in study.VOWELS}},
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
            study.score_groups(run, "validation")
            output = run / "validation" / "jvs70_cv70"
            for change in (
                {"model_id": "jvs70"},
                {"is_genuine": False},
                {"trial_id": "other"},
                {"status": "no_score", "scores": None},
            ):
                scores = output / "scores.jsonl.gz"
                scores.unlink()
                with study.writer(scores) as write:
                    write(
                        {
                            **trial,
                            "model_id": "jvs70_cv70",
                            "status": "scored",
                            "scores": {"fused": 0.5, **{v: 0.5 for v in study.VOWELS}},
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
                with self.assertRaises(ValueError):
                    study.score_groups(run, "validation")


if __name__ == "__main__":
    unittest.main()
