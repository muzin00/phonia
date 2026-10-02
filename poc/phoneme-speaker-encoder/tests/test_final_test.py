"""Guard the one-time test evaluator against threshold fitting on test scores."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_final_test import COUNTS, ROLES, VOWELS, _summarize


class FrozenThresholdTests(unittest.TestCase):
    def test_eer_and_validation_threshold_rates_are_distinct(self) -> None:
        groups = {
            (role, count, vowel): ([0.8, 0.2], [True, False])
            for role in ROLES
            for count in COUNTS
            for vowel in VOWELS
        }
        operating = {
            name: {"threshold": 0.9}
            for name in ("far_1pct", "far_0_1pct", "eer_operating")
        }
        thresholds = {
            "per_count": {
                str(count): {key: operating for key in (*VOWELS, "pooled")}
                for count in COUNTS
            }
        }
        result = _summarize(groups, {}, thresholds)
        primary = result["roles"]["verification"]["10"]
        self.assertEqual(result["macro_eer"], 0.0)
        for vowel in VOWELS:
            rates = primary["vowels"][vowel]["fixed_threshold_rates"]["far_1pct"]
            self.assertEqual((rates["far"], rates["frr"]), (0.0, 1.0))
        self.assertEqual(
            primary["pooled"]["fixed_threshold_rates"]["far_1pct"]["frr"], 1.0
        )
        self.assertEqual(
            thresholds["per_count"]["10"]["a"]["far_1pct"]["threshold"], 0.9
        )


if __name__ == "__main__":
    unittest.main()
