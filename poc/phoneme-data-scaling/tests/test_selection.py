import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selection as s


class SelectionTests(unittest.TestCase):
    def test_new_labels_exclude_existing_and_evaluation(self):
        groups = {
            "train": list(range(100)),
            "test": list(range(100)),
            "short": [1],
            "a": [1, 2],
            "b": [1, 2, 3],
        }
        selected, eligible = s.select_labels(groups, {"train", "test"}, 2, 2, 7)
        self.assertEqual(set(selected), {"a", "b"})
        self.assertEqual(eligible, 2)
        with self.assertRaisesRegex(AssertionError, "only 2 eligible"):
            s.select_labels(groups, {"train", "test"}, 3, 2, 7)

    def test_round_robin_uses_distinct_sources_without_replacement(self):
        groups = {
            "short": [{"source_id": "s"}],
            "long": [{"source_id": f"l{i}"} for i in range(5)],
        }
        result = s.balanced_take(groups, 4, 19)
        self.assertEqual(len({v["source_id"] for v in result}), 4)
        self.assertIn("s", [v["source_id"] for v in result[:2]])
        self.assertEqual(
            result, s.balanced_take(dict(reversed(list(groups.items()))), 4, 19)
        )
        with self.assertRaisesRegex(AssertionError, "Insufficient"):
            s.balanced_take(groups, 7, 19)

    def test_frozen_output_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            s.fixed(path, {"labels": ["a"]})
            s.fixed(path, {"labels": ["a"]})
            with self.assertRaisesRegex(AssertionError, "Frozen output changed"):
                s.fixed(path, {"labels": ["b"]})
            self.assertEqual(s.read(path), {"labels": ["a"]})


if __name__ == "__main__":
    unittest.main()
