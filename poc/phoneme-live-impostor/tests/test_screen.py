import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screen


class ScreeningTests(unittest.TestCase):
    def test_threshold_ties_and_inclusive_decision(self):
        values = [0.9, 0.8, 0.8, 0.7]
        cut = screen.threshold(values, 0.5)
        self.assertGreater(cut, 0.8)
        self.assertEqual(sum(v >= cut for v in values), 1)
        self.assertGreater(screen.threshold(values, 0), max(values))

    def test_adaptive_stops_only_after_all_vowels(self):
        sequence = [
            {"groups": {p: [0] for p in "aiue"}},
            {"groups": {"o": [1]}},
            {"groups": {"o": [2]}},
        ]
        groups, selected = screen.combine(sequence, "adaptive1", screen.VOWELS)
        self.assertEqual(len(selected), 2)
        self.assertEqual(groups["o"], [1])
        groups, selected = screen.combine(sequence, "single", screen.VOWELS)
        self.assertEqual(len(selected), 1)
        self.assertFalse(groups["o"])

    def test_query_mean_not_renormalized_and_equal_phone_weights(self):
        np = screen.np
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        profile = {p: np.array([1, 0], dtype=np.float32) for p in screen.VOWELS}
        groups = {p: [0] for p in screen.VOWELS}
        groups["a"] = [0, 1]
        value, parts, error = screen.score(groups, vectors, profile)
        self.assertEqual(parts["a"], 0.5)
        self.assertAlmostEqual(value, 0.9)
        self.assertLess(error, 1e-6)
        groups["i"] = []
        self.assertIsNone(screen.score(groups, vectors, profile)[0])

    def test_missing_scores_and_dependent_labels_remain_in_denominator(self):
        rows = [
            {"corpus": "A", "speaker_id": "x", "score": 0.8},
            {"corpus": "A", "speaker_id": "x", "score": 0.9},
            {"corpus": "A", "speaker_id": "y", "score": None},
        ]
        result = screen.rates(rows, 0.8)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["accepted_fraction_all"], 2 / 3)
        self.assertEqual(result["accepted_fraction_scored"], 1)
        self.assertEqual(result["speaker_labels_with_accept"], 1)
        self.assertEqual(result["unscored"], 1)


if __name__ == "__main__":
    unittest.main()
