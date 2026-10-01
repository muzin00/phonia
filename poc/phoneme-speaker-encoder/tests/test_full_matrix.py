"""Frozen 70-speaker, three-seed comparison and resource contracts."""

import json
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

from run_full_matrix import expand_matrix


class FullMatrixTests(unittest.TestCase):
    def setUp(self):
        self.search = json.loads(
            (BASE / "config/search-space.json").read_text(encoding="utf-8")
        )
        self.budget = json.loads(
            (BASE / "config/full-execution-budget.json").read_text(encoding="utf-8")
        )

    def test_all_18_configurations_have_three_distinct_seeds(self):
        rows = expand_matrix(self.search, self.budget)
        self.assertEqual(len(rows), 54)
        self.assertEqual(len({row["run_id"] for row in rows}), 54)
        config_ids = {row["config_id"] for row in rows}
        self.assertEqual(len(config_ids), 18)
        for config_id in config_ids:
            self.assertEqual(
                {row["seed"] for row in rows if row["config_id"] == config_id},
                set(self.search["seeds"]),
            )
        self.assertEqual(sum(row["comparison"] == "main" for row in rows), 48)
        self.assertTrue(all(row["cohort"] == 70 for row in rows))

    def test_original_full_training_protocol_is_preserved(self):
        protocol = self.budget["protocol"]
        environment = self.budget["environment"]
        self.assertEqual(protocol["maximum_updates"], 30000)
        self.assertEqual(protocol["warmup_updates"], 1000)
        self.assertEqual(protocol["validation_interval"], 1000)
        self.assertEqual(environment["device"], "cpu")
        self.assertEqual(environment["maximum_parallel_runs"], 4)
        self.assertFalse(protocol["test_split_allowed"])


if __name__ == "__main__":
    unittest.main()
