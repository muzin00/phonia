import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (
    Scorer,
    backend,
    eer,
    independent_eer,
    neighbors,
    np,
    partition,
    positions,
    search,
    summarize,
    torch,
)


class SelectionTests(unittest.TestCase):
    def test_partition_excludes_cross_group_claims(self):
        speakers = [f"s{i}" for i in range(7)]
        groups = partition(speakers, 42)
        self.assertEqual(groups, partition(speakers[::-1], 42))
        self.assertFalse(set(groups["discovery"]) & set(groups["confirmation"]))
        rows = [
            {
                "speaker_id": q,
                "claimed_speaker_id": c,
                "role": "verification",
                "score": 0.5,
            }
            for q in speakers
            for c in speakers
        ]
        chosen = positions(rows, groups["confirmation"])
        self.assertEqual(len(chosen), 9)
        self.assertTrue(
            all(
                rows[i]["speaker_id"] in groups["confirmation"]
                and rows[i]["claimed_speaker_id"] in groups["confirmation"]
                for i in chosen
            )
        )

    def test_search_supports_deletion_and_swap(self):
        phones = ["a", "n", "m", "s"]
        objectives = {
            (0, 1, 2, 3): 0.4,
            (0, 1, 2): 0.3,
            (0, 1): 0.2,
            (0, 2): 0.25,
            (0, 3): 0.1,
        }
        config = {
            "single_move_limit": 2,
            "swap_move_limit": 1,
            "minimum_improvement": 1e-12,
        }
        # Two deletions reach a+n; swapping n for s then improves further.
        best, history = search(
            phones, lambda items: {v: objectives.get(v, 0.5) for v in items}, config
        )
        self.assertEqual(best, (0, 3))
        self.assertEqual(
            [v["move"] for v in history], ["start", "delete", "delete", "swap"]
        )
        self.assertIn((0, 1), neighbors(best, phones, swap=True))
        self.assertIn((0, 1, 3), neighbors(best, phones))

    def test_no_tie_moves_and_no_loss_of_vowel_anchor(self):
        phones = ["a", "i", "n"]
        config = {
            "single_move_limit": 8,
            "swap_move_limit": 2,
            "minimum_improvement": 1e-12,
        }
        best, history = search(phones, lambda items: dict.fromkeys(items, 0.1), config)
        self.assertEqual(best, (0, 1, 2))
        self.assertEqual(len(history), 1)
        self.assertNotIn((2,), neighbors((0, 2), phones))
        self.assertIn((1, 2), neighbors((0, 2), phones, swap=True))

    def test_tie_aware_independent_eer(self):
        values = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.8])
        labels = np.array([True, False, True, False, True, False])
        self.assertAlmostEqual(eer(values, labels), independent_eer(values, labels))
        self.assertEqual(
            eer(np.array([1, 1, 0, 0]), np.array([True, True, False, False])), 0
        )

    def test_no_scores_count_as_rejections(self):
        result = summarize(np.array([0.9, 0.1]), np.array([True, False]), 0.5, 2, 2)
        self.assertEqual(result["eer"], 0)
        self.assertEqual(result["frr"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["far"], 0)

    def test_projection_cache_matches_attention_recomputation(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        rng = np.random.default_rng(456)
        model = backend.Fusion("transformer").eval()
        with torch.no_grad():
            model.output.weight.normal_()
        arrays = {
            k: rng.normal(size=shape).astype(np.float32)
            for k, shape in {
                "enrollment": (3, 36, 128),
                "emeta": (3, 36, 4),
                "queries": (4, 36, 128),
                "qmeta": (4, 36, 4),
            }.items()
        }
        masks = rng.random((12, 36)) > 0.4
        masks[:, :5] = True
        arrays["masks"] = masks
        arrays["components"] = rng.normal(size=(12, 36))
        scorer = Scorer(model, arrays, np.arange(12), batch_size=5)
        for subset in (tuple(range(36)), (0,), tuple(range(1, 36)), (0, 5, 20, 22, 28)):
            np.testing.assert_array_equal(
                scorer.score(subset), scorer.score(subset, direct=True)
            )
        arrays["masks"][:, 35] = False
        scorer = Scorer(model, arrays, np.arange(12))
        with self.assertRaises(ValueError):
            scorer.score((35,))


if __name__ == "__main__":
    unittest.main()
