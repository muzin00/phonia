import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare


class PairedComparisonTests(unittest.TestCase):
    def rows(self, values):
        return [
            {"corpus": "A", "speaker_id": "x", "query_id": str(i), "score": v}
            for i, v in enumerate(values)
        ]

    def test_crossings_include_equality_and_keep_unscored(self):
        before = self.rows([0.7, 0.5, None, 0.8])
        after = self.rows([None, 0.6, 0.7, 0.9])
        result = compare.paired(before, after, 0.6)
        self.assertEqual(result["accept_to_reject_or_unscored"], 1)
        self.assertEqual(result["reject_or_unscored_to_accept"], 2)
        self.assertEqual(result["scored_pairs"], 2)
        self.assertAlmostEqual(result["mean_score_difference"], 0.1)

    def test_mismatched_queries_are_rejected(self):
        before = self.rows([0.7, 0.8])
        with self.assertRaises(AssertionError):
            compare.paired(before, before[::-1], 0.6)


if __name__ == "__main__":
    unittest.main()
