import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schedule import restrict_to_cv


class ScheduleTests(unittest.TestCase):
    def test_only_cv_uses_existing_hard_choices_without_mutating_controls(self):
        original = {
            "queries": np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]]),
            "negative_claims": np.full((3, 3), 10),
        }
        hard = {
            "queries": original["queries"].copy(),
            "negative_claims": np.array([[10, 10, 10], [11, 12, 13], [14, 15, 16]]),
        }
        result = restrict_to_cv(original, hard, np.array([False, True, False]), 1)
        np.testing.assert_array_equal(result["queries"], original["queries"])
        np.testing.assert_array_equal(
            result["negative_claims"], [[10, 10, 10], [10, 10, 13], [14, 10, 10]]
        )
        np.testing.assert_array_equal(original["negative_claims"], 10)
        np.testing.assert_array_equal(
            hard["negative_claims"], [[10, 10, 10], [11, 12, 13], [14, 15, 16]]
        )

    def test_rejects_mismatched_queries_or_changed_prefix(self):
        original = {
            "queries": np.array([[0], [1]]),
            "negative_claims": np.array([[2], [3]]),
        }
        for hard in (
            {
                "queries": np.array([[1], [0]]),
                "negative_claims": original["negative_claims"],
            },
            {"queries": original["queries"], "negative_claims": np.array([[9], [3]])},
        ):
            with self.assertRaises(AssertionError):
                restrict_to_cv(original, hard, np.array([True, False]), 1)


if __name__ == "__main__":
    unittest.main()
