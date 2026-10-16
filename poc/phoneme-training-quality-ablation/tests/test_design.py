import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_data import common_banks, matched_random_keep

# isort: split
from ablation_data import make_schedule


class DesignTests(unittest.TestCase):
    def pool(self):
        return SimpleNamespace(
            features=np.zeros((60, 128)),
            groups=[np.arange(i, i + 10) for i in range(0, 60, 10)],
            speakers=["s1", "s2"],
            phones=["a", "i", "u"],
            group_speaker=np.repeat([0, 1], 3),
            group_phone=np.tile(np.arange(3), 2),
        )

    def test_counts_match_including_empty_singleton_and_untouched_groups(self):
        pool = self.pool()
        keep = np.zeros(60, bool)
        for ids, count in zip(pool.groups, [0, 1, 4, 10, 7, 3]):
            keep[ids[:count]] = True
        result = matched_random_keep(pool, keep, 42)
        self.assertEqual(
            [int(result[g].sum()) for g in pool.groups], [0, 1, 4, 10, 7, 3]
        )
        np.testing.assert_array_equal(result, matched_random_keep(pool, keep, 42))
        self.assertFalse(np.array_equal(result, matched_random_keep(pool, keep, 43)))

    def test_unexpected_mask_size_fails(self):
        with self.assertRaises(ValueError):
            matched_random_keep(self.pool(), np.ones(59, bool), 42)

    def bank(self, order):
        return {
            "speakers": ["jvs001", "jvs002"],
            "profiles": {"jvs001": {"a": [1]}, "jvs002": {"a": [2]}},
            "queries": [
                {
                    "query_id": str(i),
                    "speaker_id": "jvs001" if i % 2 else "jvs002",
                    "groups": {"a": [i]},
                }
                for i in order
            ],
        }

    def test_common_query_order_preserves_each_variants_segments(self):
        banks = {
            "original": self.bank([1, 2, 3, 4]),
            "screened": self.bank([4, 3, 2]),
            "random": self.bank([3, 2, 1]),
        }
        banks["screened"]["queries"][1]["groups"] = {"a": [777]}
        common = common_banks(banks)
        for bank in common.values():
            self.assertEqual([q["query_id"] for q in bank["queries"]], ["2", "3"])
            self.assertEqual(bank["speakers"], ["jvs001", "jvs002"])
        self.assertEqual(common["screened"]["queries"][1]["groups"], {"a": [777]})
        a, _ = make_schedule(common["original"], 20, 4, 42)
        b, _ = make_schedule(common["screened"], 20, 4, 42)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        owners = np.array([1, 0])
        self.assertFalse(np.any(owners[a["queries"]] == a["negative_claims"]))

    def test_inconsistent_owner_and_duplicate_query_are_rejected(self):
        banks = {"a": self.bank([1, 2]), "b": self.bank([1, 2])}
        banks["b"]["queries"][0]["speaker_id"] = "jvs002"
        with self.assertRaises(ValueError):
            common_banks(banks)
        with self.assertRaises(ValueError):
            common_banks({"a": self.bank([1, 1]), "b": self.bank([1, 2])})

    def test_owner_without_shared_queries_is_removed(self):
        banks = {"a": self.bank([1, 2]), "b": self.bank([1, 3])}
        common = common_banks(banks)
        self.assertEqual(common["a"]["speakers"], ["jvs001"])


if __name__ == "__main__":
    unittest.main()
