import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import assert_nested_equal, batch, np, prior, seed_all, torch
from losses import objective, soft_eer
from trainer import train_loop


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
    for key in ("enrollment", "queries"):
        arrays[key] = torch.nn.functional.normalize(arrays[key], dim=-1)
    arrays["emask"] = torch.ones(3, 36, dtype=torch.bool)
    arrays["qmask"] = torch.ones(6, 36, dtype=torch.bool)
    arrays["emask"][1, 9] = False
    owners = torch.tensor([0, 0, 1, 1, 2, 2])
    q = rng.integers(6, size=(8, 4), dtype=np.int32)
    schedule = {
        "queries": q,
        "negative_claims": ((owners.numpy()[q] + 1) % 3).astype(np.int16),
    }
    return arrays, owners, schedule


def configuration():
    return {
        "learning_rate": 0.00001,
        "weight_decay": 0.01,
        "uniform_kl_coefficient": 0.001,
        "gradient_clip": 5.0,
        "soft_eer_coefficient": 1.0,
        "soft_eer_temperature": 0.03,
        "root_iterations": 60,
        "validation_every": 2,
        "updates": 8,
    }


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_implicit_gradient_matches_finite_differences_with_unequal_classes(self):
        labels = torch.tensor([True, True, True, False, False])
        for values in (
            [0.45, 0.5, 0.67, 0.42, 0.57],
            [0.3] * 5,
            [-0.7, -0.6, 0.2, 0.7, 0.8],
        ):
            scores = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda x: soft_eer(x, labels, 0.08),
                    (scores,),
                    eps=1e-6,
                    atol=2e-6,
                    rtol=2e-4,
                )
            )

    def test_known_rates_orientation_and_temperature_limit(self):
        labels = torch.tensor([True, True, False, False])
        for values, expected in (
            ([0.8, 0.8, 0.2, 0.2], 0),
            ([0.3] * 4, 0.5),
            ([0.2, 0.2, 0.8, 0.8], 1),
        ):
            actual = float(
                soft_eer(torch.tensor(values, dtype=torch.float64), labels, 0.005)
            )
            self.assertAlmostEqual(actual, expected, places=10)
        values = torch.tensor(
            [0.9, 0.7, 0.4, 0.1, 0.8, 0.6, 0.3, 0.0], dtype=torch.float64
        )
        labels = torch.tensor([True] * 4 + [False] * 4)
        exact = prior.independent_eer(values.numpy(), labels.numpy())
        self.assertAlmostEqual(float(soft_eer(values, labels, 0.001)), exact, places=8)

    def test_translation_invariance_and_correct_gradient_directions(self):
        scores = torch.tensor(
            [0.3, 0.7, 0.4, 0.5, 0.8], dtype=torch.float64, requires_grad=True
        )
        labels = torch.tensor([True, True, False, False, False])
        value = soft_eer(scores, labels, 0.08)
        value.backward()
        self.assertAlmostEqual(float(scores.grad.sum()), 0, places=12)
        self.assertTrue((scores.grad[labels] < 0).all())
        self.assertTrue((scores.grad[~labels] > 0).all())
        self.assertAlmostEqual(
            float(value.detach()),
            float(soft_eer(scores.detach() + 17, labels, 0.08)),
            places=12,
        )
        improved = soft_eer(scores.detach() - 0.01 * scores.grad, labels, 0.08)
        self.assertLess(float(improved), float(value.detach()))

    def test_extreme_scores_and_invalid_inputs(self):
        labels = torch.tensor([True, False])
        for values in ([1e4, -1e4], [-1e4, 1e4], [1e4, 1e4]):
            scores = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            value = soft_eer(scores, labels)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(scores.grad).all())
        for y in (torch.tensor([True, True]), torch.tensor([False, False])):
            with self.assertRaises(ValueError):
                soft_eer(torch.zeros(2), y)
        with self.assertRaises(ValueError):
            soft_eer(torch.tensor([float("nan"), 1.0]), labels)
        with self.assertRaises(ValueError):
            soft_eer(torch.zeros(2), labels, 0)

    def test_zero_coefficient_matches_original_bce_value_and_all_gradients(self):
        arrays, owners, schedule = toy_data()
        f, c, mask, labels = batch(arrays, owners, schedule, 0)
        seed_all(7)
        old = prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)
        with torch.no_grad():
            old.output.weight.normal_(std=0.1)
        new = copy.deepcopy(old)
        seed_all(101)
        expected = old.loss(f, c, mask, labels, 0.001)
        expected.backward()
        end_rng = torch.get_rng_state().clone()
        seed_all(101)
        actual, _ = objective(new, f, c, mask, labels, configuration(), "bce")
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def model():
        seed_all(99)
        model = prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)
        seed_all(77)
        return model

    @staticmethod
    def evaluate(model):
        model.eval()
        return {"jvs": 0.25, "common_voice": 0.5, "jtube": 0.75}

    def test_paired_batch_mask_and_source_preservation(self):
        arrays, owners, schedule = toy_data()
        original = copy.deepcopy((arrays, owners, schedule))
        f, c, mask, labels = batch(arrays, owners, schedule, 0)
        self.assertEqual(f.shape, (8, 36, 520))
        self.assertTrue(torch.equal(mask[:4], mask[4:]))
        self.assertTrue(mask[:, :5].all())
        torch.testing.assert_close(labels, torch.tensor([1.0] * 4 + [0.0] * 4))
        q = torch.from_numpy(schedule["queries"][0].astype(np.int64))
        claims = torch.cat(
            (
                owners[q],
                torch.from_numpy(schedule["negative_claims"][0].astype(np.int64)),
            )
        )
        queries = torch.cat((q, q))
        expected = (arrays["enrollment"][claims] * arrays["queries"][queries]).sum(-1)
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
        assert_nested_equal((arrays, owners), original[:2])
        for key in schedule:
            np.testing.assert_array_equal(schedule[key], original[2][key])

    def test_atomic_resume_and_completed_restart_for_both_arms(self):
        arrays, owners, schedule = toy_data()
        config = configuration()
        with tempfile.TemporaryDirectory() as temp:
            for arm in ("bce", "soft_eer"):
                full, partial = Path(temp) / f"{arm}-full", Path(temp) / f"{arm}-resume"
                expected = train_loop(
                    config,
                    self.model(),
                    arrays,
                    owners,
                    schedule,
                    full,
                    "design",
                    arm,
                    self.evaluate,
                )
                train_loop(
                    config,
                    self.model(),
                    arrays,
                    owners,
                    schedule,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                    stop_after=4,
                )
                (partial / "history.json").write_text('"stale sidecar"')
                actual = train_loop(
                    config,
                    self.model(),
                    arrays,
                    owners,
                    schedule,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                )
                assert_nested_equal(expected, actual)
                self.assertEqual(actual["best_bundle"]["update"], 0)
                repeated = train_loop(
                    config,
                    self.model(),
                    arrays,
                    owners,
                    schedule,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                )
                assert_nested_equal(actual, repeated)
                with self.assertRaises(AssertionError):
                    train_loop(
                        config,
                        self.model(),
                        arrays,
                        owners,
                        schedule,
                        partial,
                        "changed",
                        arm,
                        self.evaluate,
                    )

    def test_zero_coefficient_reproduces_entire_training_and_nonzero_changes_weights(
        self,
    ):
        arrays, owners, schedule = toy_data()
        config = {**configuration(), "soft_eer_coefficient": 0.0}
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for arm, coefficient, name in (
                ("bce", 0.0, "base"),
                ("soft_eer", 0.0, "zero"),
                ("soft_eer", 1.0, "nonzero"),
            ):
                result = train_loop(
                    {**config, "soft_eer_coefficient": coefficient},
                    self.model(),
                    arrays,
                    owners,
                    schedule,
                    Path(temp) / name,
                    "design",
                    arm,
                    self.evaluate,
                )
                results.append(result)
            for key in ("model", "optimizer", "rng", "history", "best_bundle"):
                assert_nested_equal(results[0][key], results[1][key])
            self.assertTrue(torch.equal(results[0]["rng"], results[2]["rng"]))
            self.assertTrue(
                any(
                    not torch.equal(results[0]["model"][k], results[2]["model"][k])
                    for k in results[0]["model"]
                )
            )


if __name__ == "__main__":
    unittest.main()
