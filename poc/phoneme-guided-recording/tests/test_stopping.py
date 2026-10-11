import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as e


class StoppingTests(unittest.TestCase):
    def sequence(self):
        return [
            {
                "query_id": "first",
                "groups": {"a": [0], "i": [1], "u": [], "e": [], "o": []},
            },
            {
                "query_id": "second",
                "groups": {"a": [2], "i": [], "u": [3], "e": [4], "o": [5]},
            },
            {
                "query_id": "third",
                "groups": {"a": [6], "i": [7], "u": [8], "e": [9], "o": [10]},
            },
        ]

    def test_stop_at_first_complete_boundary_without_score(self):
        sequence = self.sequence()
        groups, used = e.select_sequence(sequence, "adaptive1")
        self.assertEqual(len(used), 2)
        self.assertEqual(groups["a"], [0, 2])
        self.assertEqual(groups["o"], [5])
        groups, used = e.select_sequence(sequence, "adaptive2")
        self.assertEqual(len(used), 3)
        self.assertTrue(all(len(groups[p]) >= 2 for p in e.s.VOWELS))
        self.assertEqual(sequence[0]["groups"]["a"], [0])

    def test_insufficient_support_is_retained_and_fixed_controls_ignore_it(self):
        sequence = self.sequence()
        for q in sequence:
            q["groups"]["o"] = []
        groups, used = e.select_sequence(sequence, "adaptive1")
        self.assertEqual(len(used), 3)
        self.assertEqual(groups["o"], [])
        for arm, count in [("single", 1), ("fixed2", 2), ("fixed3", 3)]:
            _, used = e.select_sequence(sequence, arm)
            self.assertEqual(len(used), count)

    def test_already_complete_first_clip_requires_no_retry(self):
        sequence = self.sequence()[2:] + self.sequence()[:2]
        _, used = e.select_sequence(sequence, "adaptive1")
        self.assertEqual(len(used), 1)


if __name__ == "__main__":
    unittest.main()
