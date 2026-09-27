"""Frozen Phase 3 sanity matrix and resource budget contracts."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

from run_sanity_matrix import expand_matrix


class SanityMatrixTests(unittest.TestCase):
    def setUp(self):
        self.search = json.loads(
            (BASE / "config/search-space.json").read_text(encoding="utf-8")
        )
        self.budget = json.loads(
            (BASE / "config/execution-budget.json").read_text(encoding="utf-8")
        )

    def test_exactly_16_main_and_two_limited_with_unique_ids(self):
        rows = expand_matrix(self.search, self.budget)
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["comparison"] == "main" for row in rows), 16)
        self.assertEqual(len({row["config_id"] for row in rows}), 18)
        self.assertEqual(len({row["run_id"] for row in rows}), 18)
        self.assertEqual(
            {row["encoder"] for row in rows if row["comparison"] != "main"},
            {"framewise_cnn", "waveform_cnn_k240_context27"},
        )
        self.assertTrue(all(row["cohort"] == 10 for row in rows))
        self.assertTrue(all(row["seed"] == 20260926 for row in rows))

    def test_missing_and_duplicate_axes_are_rejected(self):
        missing = deepcopy(self.search)
        missing["families"][0]["axes"]["loss"] = ["aam_softmax"]
        with self.assertRaises(ValueError):
            expand_matrix(missing, self.budget)
        duplicate = deepcopy(self.search)
        duplicate["limited_comparisons"][0]["encoder"] = "tdnn"
        with self.assertRaises(ValueError):
            expand_matrix(duplicate, self.budget)

    def test_cpu_budget_is_fixed_before_runs(self):
        environment = self.budget["environment"]
        protocol = self.budget["protocol"]
        limits = self.budget["limits"]
        self.assertEqual(environment["device"], "cpu")
        self.assertEqual(environment["precision"], "float32")
        self.assertEqual(environment["data_loader_workers"], 0)
        self.assertFalse(protocol["test_split_allowed"])
        self.assertEqual(protocol["maximum_updates"], 2000)
        self.assertGreater(limits["maximum_run_wall_seconds"], 0)
        self.assertGreater(limits["maximum_process_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
