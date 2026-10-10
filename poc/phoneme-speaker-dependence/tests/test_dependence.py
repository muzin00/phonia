import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dependence_math as dm
import numpy as np


class VarianceTests(unittest.TestCase):
    def test_balanced_reml_equals_independent_anova(self):
        rng = np.random.default_rng(12)
        s, n, d = 20, 12, 16
        y = rng.normal(size=(s, 1, d)) * 0.6 + rng.normal(size=(s, n, d)) * 0.8
        blocks, dimensions = dm.sufficient_statistics(
            np.ones((s * n, 1)),
            y.reshape(s * n, d),
            np.repeat(np.arange(s), n),
            np.arange(s),
        )
        result = dm.fit_components(blocks, dimensions)
        center = y.mean(axis=1)
        msb = n * np.sum((center - center.mean(axis=0)) ** 2) / (s - 1)
        msw = np.sum((y - center[:, None]) ** 2) / (s * (n - 1))
        b = max((msb - msw) / n, 0)
        self.assertAlmostEqual(result["B_trace"], b, places=5)
        self.assertAlmostEqual(result["W_trace"], msw, places=5)
        self.assertAlmostEqual(result["R"], b / (b + msw), places=6)

    def test_covariate_removes_spurious_speaker_signal(self):
        rng = np.random.default_rng(43)
        s, n, d = 30, 15, 8
        c = np.repeat(rng.normal(size=s), n) + rng.normal(size=s * n) * 0.3
        y = c[:, None] * rng.normal(size=(1, d)) * 3 + rng.normal(size=(s * n, d)) * 0.1
        labels = np.repeat(np.arange(s), n)
        raw, _ = dm.sufficient_statistics(np.ones((s * n, 1)), y, labels, np.arange(s))
        adjusted, _ = dm.sufficient_statistics(
            np.column_stack([np.ones(s * n), c]), y, labels, np.arange(s)
        )
        self.assertGreater(dm.fit_components(raw, d)["R"], 0.7)
        self.assertLess(dm.fit_components(adjusted, d)["R"], 0.03)

    def test_orthogonal_embedding_rotation_preserves_components(self):
        rng = np.random.default_rng(89)
        y = rng.normal(size=(10, 1, 8)) + rng.normal(size=(10, 10, 8))
        y = y.reshape(100, 8)
        rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        labels = np.repeat(np.arange(10), 10)
        a, _ = dm.sufficient_statistics(np.ones((100, 1)), y, labels, np.arange(10))
        b, _ = dm.sufficient_statistics(
            np.ones((100, 1)), y @ rotation, labels, np.arange(10)
        )
        self.assertAlmostEqual(
            dm.fit_components(a, 8)["R"], dm.fit_components(b, 8)["R"], places=6
        )

    def test_bootstrap_copies_have_separate_random_intercepts(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=(40, 4))
        blocks, _ = dm.sufficient_statistics(
            np.ones((40, 1)), y, np.repeat(np.arange(4), 10), np.arange(4)
        )
        fit = dm.fit_components(blocks, 4, [0, 0, 1, 2])
        self.assertEqual(fit["speakers"], 4)
        self.assertEqual(fit["tokens"], 40)


class GroupingTests(unittest.TestCase):
    def test_exact_clustering_matches_exhaustive_partitions(self):
        scores = dict(zip("abcdef", [0.01, 0.02, 0.03, 0.5, 0.6, 0.7]))
        result = dm.cluster(scores)
        losses = []
        for size in range(2, 5):
            for low in itertools.combinations(scores, size):
                a = np.array([scores[p] for p in low])
                b = np.array([scores[p] for p in scores if p not in low])
                losses.append(np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2))
        self.assertAlmostEqual(result["two_group_sse"], min(losses))
        self.assertEqual(result["low"], list("abc"))

    def test_equal_scores_do_not_claim_a_gap(self):
        result = dm.cluster(dict.fromkeys("abcdef", 0.2))
        self.assertEqual(result["relative_sse_reduction"], 0)

    def test_context_classes_keep_unvoiced_vowels_and_N_distinct(self):
        self.assertEqual(dm.context_class("I"), "unvoiced_vowel")
        self.assertEqual(dm.context_class("N"), "nasal")
        self.assertEqual(dm.context_class("i"), "i")
        self.assertEqual(dm.context_class("pau"), "other")

    def test_spearman_ties_and_inverse_order(self):
        self.assertAlmostEqual(dm.rank_correlation([1, 2, 2, 4], [4, 2, 2, 1]), -1)
        self.assertIsNone(dm.rank_correlation([1, 1], [1, 2]))


if __name__ == "__main__":
    unittest.main()
