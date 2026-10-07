"""Failure-sensitive checks for selection, frozen gates and metric denominators."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import scaling as study
from phase3_data.manifest import Segment


def segments():
    return [
        Segment(
            f"{speaker}-{vowel}-{index}",
            speaker,
            vowel,
            "validation",
            "enrollment",
            (),
            f"{speaker}/{index % 4}.wav",
            0,
            1000,
            "a" * 64,
        )
        for speaker in ("s1", "s2")
        for vowel in study.VOWELS
        for index in range(32)
    ]


def rows():
    result = []
    for speaker, scores in (("s1", (0.8, 0.5, None)), ("s2", (0.7, 0.1, None))):
        for index, score in enumerate(scores):
            for claimed in ("s1", "s2"):
                genuine = claimed == speaker
                result.append(
                    {
                        "query_id": f"{speaker}-{index}",
                        "speaker_id": speaker,
                        "claimed_speaker_id": claimed,
                        "split": "validation",
                        "role": "verification",
                        "is_genuine": genuine,
                        "status": "no_score" if score is None else "scored",
                        "score": score if genuine or score is None else 0.2,
                    }
                )
    return result


class SelectionTests(unittest.TestCase):
    def test_nested_counts_and_source_round_robin_are_order_invariant(self):
        data = segments()
        a = study.expanded_enrollment(
            data, "validation", ["s1", "s2"], [10, 20, 30], 20260930
        )
        b = study.expanded_enrollment(
            list(reversed(data)), "validation", ["s1", "s2"], [10, 20, 30], 20260930
        )
        self.assertEqual(a, b)
        for speaker in ("s1", "s2"):
            previous = set()
            for count in (10, 20, 30):
                chosen = next(
                    r
                    for r in a
                    if r["user_id"] == speaker and r["enrollment_count"] == count
                )
                ids = {r["segment_id"] for r in chosen["segments"]}
                self.assertEqual(len(ids), 5 * count)
                self.assertTrue(previous <= ids)
                previous = ids
            first = next(
                r for r in a if r["user_id"] == speaker and r["enrollment_count"] == 10
            )
            for vowel in study.VOWELS:
                self.assertEqual(
                    len(
                        {
                            r["source_file"]
                            for r in first["segments"]
                            if r["vowel"] == vowel
                        }
                    ),
                    4,
                )

    def test_baseline_prefix_matches_original_selection(self):
        data = segments()
        result = study.expanded_enrollment(
            data, "validation", ["s1", "s2"], [10, 20, 30], 20260930
        )
        old = study.make_enrollment(data, split="validation", count=10, seed=20260930)
        self.assertEqual(
            [
                r
                for item in result
                if item["enrollment_count"] == 10
                for r in item["ranks"]
            ],
            old,
        )

    def test_bad_counts_and_insufficient_segments_fail(self):
        for counts in ([], [0], [20, 10], [10, 10], [33]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                study.expanded_enrollment(
                    segments(), "validation", ["s1", "s2"], counts, 20260930
                )

    def test_other_split_query_role_and_missing_vowel_fail(self):
        for data in (
            [replace(s, split="test") for s in segments()],
            [replace(s, role="verification") for s in segments()],
            [s for s in segments() if s.vowel != "o"],
        ):
            with self.assertRaises(ValueError):
                study.expanded_enrollment(
                    data, "validation", ["s1", "s2"], [10, 20, 30], 20260930
                )

    def test_duplicate_segments_and_speaker_mismatch_fail(self):
        for data, speakers in (
            (segments() + [segments()[0]], ["s1", "s2"]),
            (segments(), ["s1"]),
            (segments(), ["s1", "s1", "s2"]),
        ):
            with self.assertRaises(ValueError):
                study.expanded_enrollment(
                    data, "validation", speakers, [10, 20, 30], 20260930
                )


class MetricTests(unittest.TestCase):
    def test_calibration_rejects_test_and_cross_text(self):
        for field, value in (("split", "test"), ("role", "cross_text_verification")):
            with self.assertRaises(ValueError):
                study.calibrate_cell(
                    [{**r, field: value} for r in rows()], ["s1", "s2"]
                )

    def test_missing_scores_reject_and_threshold_equality_accepts(self):
        thresholds = {"operating_points": {"far_1pct": {"threshold": 0.5}}}
        cell, _ = study.evaluate_cell(rows(), thresholds, ["s1", "s2"])
        p = cell["operating_points"]["far_1pct"]
        self.assertEqual(p["false_accepts"], 0)
        self.assertEqual(p["false_rejects"], 1)
        self.assertEqual(p["frr"], 1 / 4)
        self.assertEqual(p["no_score_genuine"], 2)
        self.assertEqual(p["all_input_frr"], 3 / 6)
        self.assertEqual(
            study.audit_rates(
                {(10, "verification"): rows()},
                {"n10/verification": cell},
                {"n10": thresholds},
            ),
            1,
        )
        cell["operating_points"]["far_1pct"]["all_input_frr"] = 0
        with self.assertRaises(ValueError):
            study.audit_rates(
                {(10, "verification"): rows()},
                {"n10/verification": cell},
                {"n10": thresholds},
            )

    def test_paired_ci_uses_aligned_draws_and_percentage_points(self):
        cells, replicas = {}, {}
        for role in ("verification", "cross_text_verification"):
            for count in (10, 20, 30):
                key = f"n{count}/{role}"
                cells[key] = {
                    "operating_points": {
                        point: {
                            name: 0.1 - count / 1000
                            for name in ("far", "frr", "all_input_far", "all_input_frr")
                        }
                        for point, _ in study.POINTS
                    }
                }
                replicas[key] = {
                    f"{point}/{name}": np.array([0.2, 0.3, 0.4]) - count / 1000
                    for point, _ in study.POINTS
                    for name in ("far", "frr", "all_input_far", "all_input_frr")
                }
        differences, arrays = study.paired(cells, replicas)
        self.assertEqual(len(differences), 72)
        key = "verification/n30_minus_n10/far_1pct/all_input_frr"
        np.testing.assert_allclose(arrays[key], -2)
        self.assertAlmostEqual(differences[key]["difference_percentage_points"], -2)
        self.assertAlmostEqual(differences[key]["ci95_percentage_points"]["upper"], -2)


class FreezeTests(unittest.TestCase):
    def fixture(self, folder):
        run = Path(folder)
        pinned = run / "immutable.txt"
        pinned.write_text("fixed")
        (run / "design-freeze.json").write_text(
            json.dumps(
                {
                    "runtime": study.runtime(),
                    "files": {"immutable.txt": study.sha256_file(pinned)},
                }
            )
        )
        (run / "validation-inputs.json").write_text(json.dumps({"split": "validation"}))
        (run / "test-inputs.json").write_text(json.dumps({"split": "test"}))
        return run

    def test_new_test_requires_evaluation_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            with patch.object(study, "ROOT", run):
                self.assertEqual(
                    study.frozen_inputs(run, "validation")["split"], "validation"
                )
                with self.assertRaises(FileNotFoundError):
                    study.frozen_inputs(run, "test")

    def test_modified_source_or_threshold_fails(self):
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
                (run / "immutable.txt").write_text("changed")
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "validation")

    def test_runtime_and_input_split_changes_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            run = self.fixture(folder)
            with patch.object(study, "ROOT", run):
                (run / "validation-inputs.json").write_text(
                    json.dumps({"split": "test"})
                )
                with self.assertRaises(ValueError):
                    study.frozen_inputs(run, "validation")
                with (
                    patch.object(study, "runtime", return_value={"changed": True}),
                    self.assertRaises(ValueError),
                ):
                    study.frozen_inputs(run, "validation")

    def test_trial_label_mutation_and_missing_trial_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            output = run / "validation"
            output.mkdir()
            trial = {
                "trial_id": "id",
                "query_id": "q",
                "split": "validation",
                "role": "verification",
                "speaker_id": "s1",
                "claimed_speaker_id": "s1",
                "is_genuine": True,
                "enrollment_count": 10,
            }
            (run / "validation-inputs.json").write_text(json.dumps({"trials": [trial]}))
            for saved in (
                [{**trial, "is_genuine": False, "status": "no_score", "scores": None}],
                [],
            ):
                score = output / "scores.jsonl.gz"
                if score.exists():
                    score.unlink()
                with study.writer(score) as write:
                    for row in saved:
                        write(row)
                (output / "report.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "outputs_sha256": {score.name: study.sha256_file(score)},
                        }
                    )
                )
                with self.assertRaises(ValueError):
                    study.score_groups(run, "validation")


if __name__ == "__main__":
    unittest.main()
