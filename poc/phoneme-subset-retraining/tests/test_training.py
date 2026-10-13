import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import assert_nested_equal, np, prior, seed_all, torch, training_batch
from trainer import learning_rate, train_loop


def toy_data():
    rng = np.random.default_rng(55)
    arrays = {
        k: torch.from_numpy(rng.normal(size=shape).astype(np.float32))
        for k, shape in {
            "enrollment": (3, 36, 128),
            "emeta": (3, 36, 4),
            "queries": (6, 36, 128),
            "qmeta": (6, 36, 4),
        }.items()
    }
    arrays["emask"] = torch.ones(3, 36, dtype=torch.bool)
    arrays["qmask"] = torch.ones(6, 36, dtype=torch.bool)
    arrays["emask"][1, 9] = False
    owners = torch.tensor([0, 0, 1, 1, 2, 2])
    q = rng.integers(6, size=(8, 2), dtype=np.int32)
    schedule = {
        "queries": q,
        "negative_claims": ((owners.numpy()[q] + 1) % 3).astype(np.int16),
    }
    return arrays, owners, schedule


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = {
            "prefix_updates": 4,
            "learning_rate": 0.0003,
            "continuation_learning_rate": 0.00003,
            "weight_decay": 0.01,
            "uniform_kl_coefficient": 0.001,
            "gradient_clip": 5.0,
            "validation_every": 2,
            "fusion_seed": 99,
        }

    def test_schedule_matches_original_cosine_then_constant(self):
        self.assertEqual(learning_rate(self.config, 1), 0.0003)
        for u in range(1, 5):
            expected = 0.0003 * (0.1 + 0.9 * (1 + math.cos(math.pi * (u - 1) / 4)) / 2)
            self.assertEqual(learning_rate(self.config, u), expected)
        self.assertEqual(learning_rate(self.config, 5), 0.00003)
        self.assertEqual(learning_rate(self.config, 64000), 0.00003)

    def test_masked_training_preserves_pairs_features_labels(self):
        arrays, owners, schedule = toy_data()
        all36 = torch.ones(36, dtype=torch.bool)
        subset = all36.clone()
        subset[[8, 18]] = False
        full = training_batch(arrays, owners, schedule, 0, all36)
        reduced = training_batch(arrays, owners, schedule, 0, subset)
        for index in (0, 1, 3):
            torch.testing.assert_close(full[index], reduced[index], rtol=0, atol=0)
        self.assertTrue(torch.equal(reduced[2][:2], reduced[2][2:]))
        self.assertFalse(reduced[2][:, [8, 18]].any())
        self.assertTrue(reduced[2][:, :5].all())
        self.assertTrue(torch.equal(reduced[2], full[2] & subset))

    def test_excluded_features_have_zero_loss_gradient(self):
        arrays, owners, schedule = toy_data()
        allowed = torch.ones(36, dtype=torch.bool)
        allowed[[8, 18]] = False
        f, c, mask, labels = training_batch(arrays, owners, schedule, 0, allowed)
        seed_all(99)
        model = prior.backend.Fusion(
            "transformer", dimension=8, layers=1, heads=2
        ).eval()
        with torch.no_grad():
            model.output.weight.normal_()
        f.requires_grad_()
        model.loss(f, c, mask, labels).backward()
        self.assertEqual(float(f.grad[:, [8, 18]].abs().max()), 0.0)
        self.assertGreater(float(f.grad[:, :5].abs().max()), 0.0)

    def test_atomic_resume_restores_optimizer_rng_best_and_history(self):
        arrays, owners, schedule = toy_data()
        subset = tuple(i for i in range(36) if i not in (8, 18))

        def model():
            seed_all(99)
            return prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)

        def evaluate(network, _subset):
            network.eval()
            return {"jvs": 0.25, "common_voice": 0.5, "jtube": 0.75}

        with tempfile.TemporaryDirectory() as temp:
            full, interrupted = Path(temp) / "full", Path(temp) / "interrupted"
            expected = train_loop(
                self.config,
                model(),
                arrays,
                owners,
                schedule,
                subset,
                full,
                "design",
                evaluate,
                8,
            )
            train_loop(
                self.config,
                model(),
                arrays,
                owners,
                schedule,
                subset,
                interrupted,
                "design",
                evaluate,
                8,
                stop_after=4,
            )
            (interrupted / "history.json").write_text('"stale sidecar"')
            actual = train_loop(
                self.config,
                model(),
                arrays,
                owners,
                schedule,
                subset,
                interrupted,
                "design",
                evaluate,
                8,
            )
            assert_nested_equal(expected, actual)
            self.assertEqual(actual["best_bundle"]["update"], 0)
            # Completed restart also reconstructs the same immutable output.
            repeated = train_loop(
                self.config,
                model(),
                arrays,
                owners,
                schedule,
                subset,
                interrupted,
                "design",
                evaluate,
                8,
            )
            assert_nested_equal(repeated, actual)
            with self.assertRaises(AssertionError):
                train_loop(
                    self.config,
                    model(),
                    arrays,
                    owners,
                    schedule,
                    subset,
                    interrupted,
                    "changed-design",
                    evaluate,
                    8,
                )

    def test_identical_initialization_with_only_mask_intervention(self):
        arrays, owners, schedule = toy_data()
        seed_all(99)
        model = prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)
        initial = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        for allowed in (
            torch.ones(36, dtype=torch.bool),
            torch.tensor([i not in (8, 18) for i in range(36)]),
        ):
            training_batch(arrays, owners, schedule, 0, allowed)
            assert_nested_equal(initial, model.state_dict())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
