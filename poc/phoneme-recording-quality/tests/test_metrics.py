import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report as r


class MetricTests(unittest.TestCase):
    def test_missing_inputs_and_speaker_pair_resampling(self):
        speakers = ["a", "b", "c"]
        rows = []
        for query in speakers:
            for claim in speakers:
                rows.append(
                    {
                        "speaker_id": query,
                        "claimed_speaker_id": claim,
                        "is_genuine": query == claim,
                        "status": "no_score" if query == "c" else "scored",
                        "score": None
                        if query == "c"
                        else (
                            0.9
                            if query == claim or (query, claim) == ("a", "b")
                            else 0.1
                        ),
                    }
                )
        counts = r.np.array([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
        result, draws = r.metrics(rows, 0.5, speakers, counts)
        self.assertEqual(result["scored_queries"], 2)
        self.assertEqual(result["false_rejects"], 1)
        self.assertEqual(result["false_accepts"], 1)
        self.assertEqual(result["frr"], 1 / 3)
        self.assertEqual(result["far"], 1 / 6)
        r.np.testing.assert_allclose(draws["frr"], [1 / 3, 0, 2 / 3])
        r.np.testing.assert_allclose(draws["far"], [1 / 6, 1 / 2, 0])


if __name__ == "__main__":
    unittest.main()
