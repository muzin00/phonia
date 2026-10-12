import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mining


class MiningTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(9)
        speakers = [
            f"{prefix}{i}" for prefix in ("jvs", "cv17_", "jtube_") for i in range(10)
        ]
        data = {"speakers": speakers, "queries": [{"speaker_id": v} for v in speakers]}
        arrays = {
            "enrollment": rng.normal(size=(30, 8, 4)),
            "queries": rng.normal(size=(30, 8, 4)),
            "emask": rng.random((30, 8)) > 0.3,
            "qmask": rng.random((30, 8)) > 0.3,
        }
        arrays["emask"][:, :5] = arrays["qmask"][:, :5] = True
        return arrays, data

    def test_scores_use_exact_shared_mask_and_exclude_own_or_other_corpus(self):
        arrays, data = self.fixture()
        table, owner = mining.score_table(arrays, data)
        for q in range(30):
            for c in range(30):
                if c == q or q // 10 != c // 10:
                    self.assertEqual(table[q, c], -np.inf)
                else:
                    mask = (
                        arrays["qmask"][q]
                        & arrays["emask"][owner[q]]
                        & arrays["emask"][c]
                    )
                    expected = np.mean(
                        [
                            np.dot(arrays["queries"][q, p], arrays["enrollment"][c, p])
                            for p in np.flatnonzero(mask)
                        ]
                    )
                    self.assertAlmostEqual(table[q, c], expected, places=12)

    def test_schedule_keeps_prefix_queries_random_half_and_best_of_distinct_candidates(
        self,
    ):
        arrays, data = self.fixture()
        table, owner = mining.score_table(arrays, data)
        queries = np.arange(32).reshape(8, 4) % 30
        old = (queries // 10) * 10 + (queries % 10 + 1) % 10
        original = {"queries": queries, "negative_claims": old}
        config = {
            "mining_seed": 47,
            "budgets": [2, 8],
            "hard_queries_per_update": 2,
            "candidates": 8,
        }
        schedule, diagnostic = mining.make_schedule(original, table, owner, config)
        again, again_diagnostic = mining.make_schedule(original, table, owner, config)
        for key in schedule:
            np.testing.assert_array_equal(schedule[key], again[key])
        for key in diagnostic:
            np.testing.assert_array_equal(diagnostic[key], again_diagnostic[key])
        np.testing.assert_array_equal(schedule["queries"], queries)
        np.testing.assert_array_equal(schedule["negative_claims"][:2], old[:2])
        hard = np.zeros(queries.size, bool)
        hard[diagnostic["positions"]] = True
        np.testing.assert_array_equal(hard.reshape(8, 4)[2:].sum(axis=1), 2)
        np.testing.assert_array_equal(
            schedule["negative_claims"].ravel()[~hard], old.ravel()[~hard]
        )
        for pos, candidates in zip(
            diagnostic["positions"], diagnostic["candidates"], strict=True
        ):
            self.assertEqual(len(set(candidates.tolist())), 8)
            self.assertEqual(candidates[0], old.ravel()[pos])
            q = queries.ravel()[pos]
            self.assertTrue(np.isfinite(table[q, candidates]).all())
            self.assertEqual(
                schedule["negative_claims"].ravel()[pos],
                candidates[table[q, candidates].argmax()],
            )
        np.testing.assert_array_equal(original["negative_claims"], old)


if __name__ == "__main__":
    unittest.main()
