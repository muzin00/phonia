"""Freeze guards and independent bootstrap/stratum correctness checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "poc/phoneme-speaker-encoder/scripts")
)
import numpy as np
from weighted_bootstrap import speaker_draws, weighted_eers

from final_metrics import (
    bootstrap_cell,
    duration_diagnostics,
    interval,
    paired_differences,
)
from frozen_test import BoundedAudio, checked_file, load_inputs
from smoke import BASE, sha256_file
from validation_metrics import evaluate_cell, roc_eer


def records():
    return [
        {
            "query_id": f"q{a}",
            "speaker_id": a,
            "claimed_speaker_id": b,
            "is_genuine": a == b,
            "status": "scored",
            "score": value,
            "source_seconds": 5.0,
            "actual_input_seconds": 1.0,
        }
        for a, b, value in (
            ("a", "a", 0.8),
            ("a", "b", 0.8),
            ("b", "a", 0.2),
            ("b", "b", 0.7),
        )
    ]


class FinalTests(unittest.TestCase):
    def test_weighted_eer_matches_materialized_pairs_at_ties(self):
        rows = records()
        qi = np.array([0, 0, 1, 1])
        ci = np.array([0, 1, 0, 1])
        counts = np.array([[1, 1], [2, 1], [1, 3]])
        values = np.array([r["score"] for r in rows])
        calculated = weighted_eers(values, qi, ci, counts)
        for index, draw in enumerate(counts):
            multiplicity = np.array(
                [
                    draw[a] if a == b else draw[a] * draw[b]
                    for a, b in zip(qi, ci, strict=True)
                ]
            )
            expanded = np.repeat(values, multiplicity)
            labels = np.repeat(qi == ci, multiplicity)
            self.assertAlmostEqual(
                calculated[index], roc_eer(expanded, labels)["eer"], places=14
            )

    def test_no_score_reject_weights_and_conditional_ne(self):
        rows = records()
        for row in rows[:2]:
            row.update(status="no_score", score=None)
        t = {"operating_points": {"far_1pct": {"threshold": 0.75}}}
        cell, _ = evaluate_cell(rows, t, ["a", "b"])
        bounds, draws = bootstrap_cell(
            rows, cell, t, ["a", "b"], np.array([[2, 1], [1, 3]])
        )
        np.testing.assert_allclose(draws["far_1pct/all_input_frr"], [1, 1])
        np.testing.assert_allclose(draws["query_coverage"], [1 / 3, 3 / 4])
        self.assertNotIn("frr", bounds["operating_points"]["far_1pct"])
        self.assertIsNone(bounds["pooled_eer"])

    def test_shared_draws_are_deterministic_and_non_degenerate(self):
        a, n = speaker_draws(3, 100, 12)
        b, m = speaker_draws(3, 100, 12)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(n, m)
        self.assertTrue((n.sum(axis=1) == 3).all())
        self.assertTrue(((n > 0).sum(axis=1) >= 2).all())

    def test_interval_retains_undefined_replicates(self):
        self.assertEqual(
            interval([0.0, 1.0, np.nan]),
            {
                "lower": 0.025,
                "upper": 0.975,
                "valid_replicates": 2,
                "undefined_replicates": 1,
            },
        )
        self.assertIsNone(interval([np.nan])["lower"])

    def test_duration_edges_and_empty_strata_kept(self):
        rows = records()
        protocol = json.loads((BASE / "config/comparison-protocol.json").read_text())
        t = {"operating_points": {"far_1pct": {"threshold": 0.75}}}
        d = duration_diagnostics(rows, t, ["a", "b"], protocol)
        self.assertEqual(len(d["source_seconds"]), 4)
        self.assertEqual(len(d["actual_input_seconds"]), 5)
        self.assertEqual(d["source_seconds"]["5:10"]["queries"], 2)
        self.assertEqual(d["source_seconds"]["3:5"]["queries"], 0)
        self.assertEqual(
            d["source_seconds"]["3:5"]["conditional_status"], "not_evaluable"
        )
        self.assertEqual(d["actual_input_seconds"]["1:2"]["queries"], 2)

    def test_paired_ci_uses_aligned_replicates(self):
        keys = [
            "native/vowel_exact/n10/full/verification",
            "native/ecapa_whole/n10/full/verification",
        ]
        cells = {
            k: {"operating_points": {"far_1pct": {"all_input_frr": v}}}
            for k, v in zip(keys, [0.4, 0.1], strict=True)
        }
        replicas = {
            keys[0]: {"far_1pct/all_input_frr": np.array([0.2, 0.4, 0.6])},
            keys[1]: {"far_1pct/all_input_frr": np.array([0, 0.1, 0.2])},
        }
        differences, draws = paired_differences(cells, replicas)
        key = f"method_minus_exact/{keys[1]}/far_1pct"
        np.testing.assert_allclose(draws[key], [-20, -30, -40])
        self.assertAlmostEqual(
            differences[key]["point_difference_percentage_points"], -30
        )

    def test_missing_freeze_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "frozen_test.load_preflight", return_value={"config": {"split": "test"}}
            ),
            self.assertRaises(FileNotFoundError),
        ):
            load_inputs(Path(directory))

    def test_mutated_pinned_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("before")
            checksum = sha256_file(path)
            path.write_text("after")
            with self.assertRaises(ValueError):
                checked_file(path, checksum)

    def test_test_worker_rejects_validation_source(self):
        with self.assertRaises(ValueError):
            BoundedAudio({"x": {"split": "validation"}}, 16)

    def test_test_scoring_matches_frozen_validation_implementation(self):
        # Independent validation/pilot exercised the original worker. Only the
        # audio/freeze imports and pinned baseline split differ in the test copy.
        old = (BASE / "validation_inference.py").read_text()
        new = (BASE / "test_inference.py").read_text()
        new = new.replace(
            "Frozen test model workers; unchanged validation scoring math.",
            "Full-validation model workers; identical pilot math with streaming scores.",
        )
        new = new.replace("from frozen_test import BoundedAudio, load_inputs\n", "")
        new = new.replace(
            "from validation import scored_row, writer",
            "from validation import BoundedAudio, scored_row, writer",
        )
        new = new.replace("    mean_profile,", "    load_inputs,\n    mean_profile,")
        new = new.replace("profiles/test/", "profiles/validation/").replace(
            "scores/test.jsonl", "scores/validation.jsonl"
        )
        self.assertEqual(new, old)


if __name__ == "__main__":
    unittest.main()
