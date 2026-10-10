"""Known ROC examples for the independent numerical checker."""

import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import audit


class AuditTest(unittest.TestCase):
    def test_perfect_separation_and_tied_scores(self):
        labels = np.array([True, True, False, False])
        self.assertEqual(audit.eer(np.array([0.9, 0.8, 0.2, 0.1]), labels), 0.0)
        self.assertEqual(audit.eer(np.array([0.5, 0.5, 0.5, 0.5]), labels), 0.5)

    def test_weighted_tie_crossing(self):
        # At score .9: FAR=0, FRR=2/3. At .5: FAR=1/3, FRR=0.
        # Linear interpolation crosses at FAR=FRR=2/9.
        scores = np.array([0.9, 0.5, 0.5, 0.1])
        labels = np.array([True, True, False, False])
        weights = np.array([1, 2, 1, 2])
        self.assertAlmostEqual(audit.eer(scores, labels, weights), 2 / 9)


if __name__ == "__main__":
    unittest.main()
