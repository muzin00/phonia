"""Edge cases that would otherwise inflate coverage or hide false accepts."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s
from auditing import paired_draws


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.policies = {p["name"]: p for p in s.read_json(s.CONFIG)["policies"]}

    def test_reused_transformer_accepts_missing_vowel_mask(self):
        model = s.fusion.create_model(
            {"dimension": 64, "transformer_layers": 2, "attention_heads": 4},
            "transformer",
        ).eval()
        mask = s.torch.zeros((1, 36), dtype=s.torch.bool)
        mask[0, [0, 1, 2, 5, 6]] = True
        with s.torch.inference_mode():
            score, weights = model(s.torch.zeros(1, 36, 520), s.torch.ones(1, 36), mask)
        self.assertAlmostEqual(float(score[0]), 1.0)
        self.assertTrue(
            s.torch.equal(weights[~mask], s.torch.zeros_like(weights[~mask]))
        )

    def test_three_vowels_alone_are_not_sufficient(self):
        self.assertFalse(s.accepted(["a", "i", "u"], self.policies["vowels3_phones5"]))
        self.assertFalse(
            s.accepted(["a", "i", "u", "n"], self.policies["vowels3_phones5"])
        )
        self.assertTrue(
            s.accepted(["a", "i", "u", "n", "m"], self.policies["vowels3_phones5"])
        )

    def test_many_consonants_do_not_replace_minimum_vowels(self):
        self.assertFalse(
            s.accepted(["a", "i", "n", "m", "s", "t"], self.policies["vowels3_phones5"])
        )

    def test_four_vowels_require_no_consonants(self):
        self.assertTrue(s.accepted(["a", "i", "e", "o"], self.policies["vowels4"]))
        self.assertFalse(s.accepted(["a", "i", "e", "o"], self.policies["vowels5"]))

    def test_count_distinct_phonemes_and_empty_support(self):
        self.assertFalse(
            s.accepted(["a", "i", "u", "n", "n"], self.policies["vowels3_phones5"])
        )
        for policy in self.policies.values():
            self.assertFalse(s.accepted([], policy))

    def test_rejection_preserves_input_and_never_leaves_a_score(self):
        values = [
            {
                "used_phones": ["a", "i", "u", "m", "n"],
                "score": 0.99,
                "status": "scored",
            }
        ]
        before = copy.deepcopy(values)
        rejected = s.apply_policy(values, self.policies["vowels5"])
        self.assertIsNone(rejected[0]["score"])
        self.assertEqual(rejected[0]["status"], "no_score")
        self.assertEqual(values, before)

    def test_existing_scores_identical_across_policies(self):
        row = {
            "used_phones": ["a", "i", "u", "e", "o", "n"],
            "score": 0.7123456789,
            "status": "scored",
        }
        for policy in self.policies.values():
            self.assertEqual(s.apply_policy([row], policy)[0], row)

    def test_bootstrap_counts_missing_genuine_as_rejected(self):
        # Three speakers; the third query has no score for any claim.
        values = []
        for q in range(3):
            for c in range(3):
                values.append(
                    {
                        "speaker_id": str(q),
                        "claimed_speaker_id": str(c),
                        "status": "scored" if q < 2 else "no_score",
                        "score": (0.9 if q == c else 0.1) if q < 2 else None,
                    }
                )
        counts = s.np.ones((2, 3), dtype=int)
        draws, _ = paired_draws(
            values, {"far_1pct": 0.5}, counts, {str(i): i for i in range(3)}
        )
        s.np.testing.assert_allclose(draws["far_1pct/frr"], 1 / 3)
        s.np.testing.assert_allclose(draws["far_1pct/far"], 0)
        s.np.testing.assert_allclose(draws["coverage"], 2 / 3)
        s.np.testing.assert_allclose(draws["eer"], 0)


if __name__ == "__main__":
    unittest.main()
