"""Compare the optimized speaker bootstrap against expanded original trials."""

import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE / "scripts")]
from phase3_train.metrics import roc_eer
from weighted_bootstrap import speaker_draws, weighted_eers


class WeightedBootstrapTests(unittest.TestCase):
    def test_exact_weighting_ties_and_zero_weight_pairs(self):
        rng = np.random.default_rng(14)
        query = np.repeat(np.arange(3), 9)
        claimed = np.tile(np.repeat(np.arange(3), 3), 3)
        scores = rng.integers(0, 5, len(query)).astype(float) / 4
        counts = np.array([[1, 1, 1], [2, 0, 1], [0, 1, 2], [1, 2, 0]])
        actual = weighted_eers(scores, query, claimed, counts)
        for index, draw in enumerate(counts):
            weights = np.where(
                query == claimed, draw[query], draw[query] * draw[claimed]
            )
            expected = roc_eer(
                np.repeat(scores, weights), np.repeat(query == claimed, weights)
            )["eer"]
            self.assertAlmostEqual(actual[index], expected, places=14)

    def test_all_tied_scores_have_eer_one_half(self):
        query = np.array([0, 0, 1, 1])
        claimed = np.array([0, 1, 0, 1])
        result = weighted_eers(np.ones(4), query, claimed, [[1, 1], [3, 1]])
        np.testing.assert_allclose(result, 0.5, atol=1e-14)

    def test_draws_are_deterministic_and_have_two_distinct_speakers(self):
        indices, counts = speaker_draws(3, 100)
        other, other_counts = speaker_draws(3, 100)
        np.testing.assert_array_equal(indices, other)
        np.testing.assert_array_equal(counts, other_counts)
        self.assertTrue(np.all(counts.sum(axis=1) == 3))
        self.assertTrue(np.all((counts > 0).sum(axis=1) >= 2))


if __name__ == "__main__":
    unittest.main()
