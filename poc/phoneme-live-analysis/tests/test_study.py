import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose as study

np = study.np


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        self.profile = {p: self.matrix[0] for p in study.VOWELS}
        self.query = {
            "id": "q",
            "status": "complete",
            "groups": {p: [0, 1] for p in study.VOWELS},
        }

    def test_query_mean_is_not_renormalized(self):
        value = study.score(self.profile, True, self.query, self.matrix, self.profile)
        self.assertEqual(value["score"], 0.5)

    def test_insufficient_and_aborted_inputs_have_no_score(self):
        self.assertIsNone(
            study.score(self.profile, False, self.query, self.matrix, self.profile)[
                "score"
            ]
        )
        self.query["status"] = "aborted"
        self.assertIsNone(
            study.score(self.profile, True, self.query, self.matrix, self.profile)[
                "score"
            ]
        )
        self.query["status"] = "complete"
        del self.query["groups"]["i"]
        self.assertIsNone(
            study.score(self.profile, True, self.query, self.matrix, self.profile)[
                "score"
            ]
        )

    def test_fixed_phone_comparison_excludes_extra_phone(self):
        self.profile["x"] = self.matrix[0]
        self.query["groups"]["x"] = [2]
        native = study.score(self.profile, True, self.query, self.matrix, self.profile)
        common = study.score(self.profile, True, self.query, self.matrix, study.VOWELS)
        self.assertEqual(native["score"], 0.25)
        self.assertEqual(common["score"], 0.5)

    def test_registration_minimum_applies_to_each_vowel(self):
        rows = [
            {"phoneme": p, "cache_index": 0, "segment_id": p, "source_file": "one"}
            for p in study.VOWELS
        ]
        config = {"limit_per_phone": 30, "selection_seed": 1, "minimum_per_vowel": 2}
        _, _, eligible = study.make_profile(rows, self.matrix, config)
        self.assertFalse(eligible)
        rows += [
            {**row, "segment_id": row["segment_id"] + "-2", "source_file": "two"}
            for row in rows
        ]
        _, chosen, eligible = study.make_profile(rows, self.matrix, config)
        self.assertTrue(eligible)
        self.assertTrue(all(len(chosen[p]) == 2 for p in study.VOWELS))


if __name__ == "__main__":
    unittest.main()
