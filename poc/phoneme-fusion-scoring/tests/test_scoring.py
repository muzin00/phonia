import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mining import cv_score_table, make_schedule
from scoring import DirectFusion, FrozenModel, arm_configuration
from shared import ARMS, Bank, CachedEvaluation, configuration, fit, rt
from verification import Evaluation


def tiny_bank():
    rng = rt.np.random.default_rng(17)
    phones = ["a", "i", "u", "e", "o", "n"]
    speakers = ["jvs001", "jvs002"]
    info = {
        "speakers": speakers,
        "profiles": {
            s: {p: [i * 6 + j] for j, p in enumerate(phones)}
            for i, s in enumerate(speakers)
        },
        "queries": [
            {
                "query_id": str(i),
                "speaker_id": s,
                "groups": {p: [12 + i * 6 + j] for j, p in enumerate(phones)},
            }
            for i, s in enumerate(speakers)
        ],
    }
    bank = Bank(
        info,
        rng.normal(size=(24, 128)).astype("float32"),
        rt.np.tile([0.1, -25.0], (24, 1)),
        phones,
    )
    rows = [
        {
            "speaker_id": s,
            "claimed_speaker_id": c,
            "role": "verification",
            "is_genuine": s == c,
            "score": 0.0,
        }
        for s in speakers
        for c in speakers
    ]
    return bank, Evaluation(bank, rows)


def tiny_model(config, arm):
    model = object.__new__(FrozenModel)
    model.config, model.arm, model.seed = config, arm, 123
    rt.torch.manual_seed(12)
    model.encoder = rt.create_encoder("statistics_mlp").requires_grad_(False)
    model.reset_fusion()
    return model


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rt.torch.set_num_threads(1)
        rt.torch.use_deterministic_algorithms(True)

    def features(self):
        rt.torch.manual_seed(99)
        f = rt.torch.randn(4, 36, 520)
        c = rt.torch.rand(4, 36)
        mask = rt.torch.rand(4, 36) > 0.4
        mask[:, :5] = True
        return f, c, mask

    def test_initial_state_scores_and_parameter_count_are_matched(self):
        f, c, mask = self.features()
        models = [tiny_model(configuration(), arm) for arm in ARMS]
        self.assertEqual(len({rt.tree_hash(m.fusion.state_dict()) for m in models}), 1)
        self.assertEqual(
            len({sum(p.numel() for p in m.fusion.parameters()) for m in models}), 1
        )
        values = [m.fusion.eval()(f, c, mask)[0] for m in models]
        for value in values[1:]:
            rt.torch.testing.assert_close(value, values[0], atol=1e-7, rtol=0)

    def test_direct_ignores_padding_and_preserves_phone_permutation(self):
        f, c, mask = self.features()
        model = DirectFusion("transformer").eval()
        with rt.torch.no_grad():
            model.output.weight.normal_(0, 0.1)
        expected = model(f, c, mask)[0]
        corrupt_f, corrupt_c = f.clone(), c.clone()
        corrupt_f[~mask], corrupt_c[~mask] = float("nan"), float("nan")
        rt.torch.testing.assert_close(model(corrupt_f, corrupt_c, mask)[0], expected)
        order = rt.torch.randperm(36)
        rt.torch.testing.assert_close(
            model(f[:, order], c[:, order], mask[:, order], order)[0],
            expected,
            atol=2e-6,
            rtol=1e-5,
        )
        with self.assertRaises(ValueError):
            model(f, c, rt.torch.zeros_like(mask))

    def test_direct_can_leave_cosine_range_and_train(self):
        f, c, mask = self.features()
        model = DirectFusion("transformer")
        with rt.torch.no_grad():
            model.output.bias.fill_(2)
        self.assertTrue((model.eval()(f, c, mask)[0] > c.max(1).values).all())
        loss = model.loss(f, c, mask, rt.torch.tensor([1.0, 0.0, 1.0, 0.0]))
        self.assertTrue(rt.torch.isfinite(loss))
        loss.backward()
        self.assertTrue(rt.torch.isfinite(model.output.weight.grad).all())
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            model.loss(f, c, mask, rt.torch.ones(4), regularization=0.001)

    def test_cached_evaluation_equals_original(self):
        _, data = tiny_bank()
        for arm in ARMS:
            model = tiny_model(configuration(), arm)
            cached = CachedEvaluation(data, model.encoder)
            rt.np.testing.assert_array_equal(data.scores(model), cached.scores(model))

    def test_all_arms_resume_exactly_and_leave_encoder_fixed(self):
        bank, data = tiny_bank()
        options = {
            **configuration(),
            "final_fusion_updates": 6,
            "final_fusion_initial_updates": 2,
            "final_fusion_validation_every": 2,
        }
        schedule = {
            "queries": rt.np.tile([0, 1], (6, 1)),
            "negative_claims": rt.np.tile([1, 0], (6, 1)),
        }
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                full, resumed = tiny_model(options, arm), tiny_model(options, arm)
                initial = rt.tree_hash(full.encoder.state_dict())
                selection = {"tiny": CachedEvaluation(data, full.encoder)}
                a, b = Path(folder) / "full", Path(folder) / "resumed"
                config = arm_configuration(options, arm)
                fit(config, a, full, bank, selection, schedule_override=schedule)
                fit(
                    config,
                    b,
                    resumed,
                    bank,
                    selection,
                    schedule_override=schedule,
                    stop_after=2,
                )
                fit(config, b, resumed, bank, selection, schedule_override=schedule)
                self.assertEqual(
                    rt.tree_hash(full.fusion.state_dict()),
                    rt.tree_hash(resumed.fusion.state_dict()),
                )
                self.assertEqual(
                    rt.tree_hash(full.fusion_optimizer.state_dict()),
                    rt.tree_hash(resumed.fusion_optimizer.state_dict()),
                )
                self.assertEqual(
                    rt.read(a / "history.json"), rt.read(b / "history.json")
                )
                self.assertEqual(
                    rt.tree_hash(rt.torch.load(a / "selected.pt", weights_only=True)),
                    rt.tree_hash(rt.torch.load(b / "selected.pt", weights_only=True)),
                )
                self.assertEqual(initial, rt.tree_hash(resumed.encoder.state_dict()))
                self.assertTrue(
                    all(p.grad is None for p in resumed.encoder.parameters())
                )

    def test_mining_is_train_cv_only_deterministic_and_not_easier(self):
        rng = rt.np.random.default_rng(3)
        corpora = rt.np.array(["common_voice"] * 10 + ["jvs"] * 2)
        owners = rt.np.arange(12)
        arrays = {
            "enrollment": rt.torch.tensor(
                rng.normal(size=(12, 6, 128)).astype("float32")
            ),
            "queries": rt.torch.tensor(rng.normal(size=(12, 6, 128)).astype("float32")),
            "emask": rt.torch.ones(12, 6, dtype=rt.torch.bool),
            "qmask": rt.torch.ones(12, 6, dtype=rt.torch.bool),
        }
        table = cv_score_table(arrays, owners, corpora)
        self.assertTrue(rt.np.isneginf(rt.np.diag(table)).all())
        self.assertTrue(rt.np.isneginf(table[10:]).all())
        config = {
            **configuration(),
            "final_fusion_updates": 5,
            "final_fusion_initial_updates": 1,
            "hard_queries_per_update": 4,
        }
        original = {
            "queries": rt.np.tile([0, 1, 10, 11], (5, 1)),
            "negative_claims": rt.np.tile([2, 3, 11, 10], (5, 1)),
        }
        result, diagnostics = make_schedule(
            original, table, owners, corpora, config, 17
        )
        again, _ = make_schedule(original, table, owners, corpora, config, 17)
        rt.np.testing.assert_array_equal(
            result["negative_claims"], again["negative_claims"]
        )
        rt.np.testing.assert_array_equal(result["queries"], original["queries"])
        rt.np.testing.assert_array_equal(
            result["negative_claims"][:, 2:], original["negative_claims"][:, 2:]
        )
        rt.np.testing.assert_array_equal(
            result["negative_claims"][0], original["negative_claims"][0]
        )
        self.assertTrue(
            (diagnostics["selected_scores"] >= diagnostics["original_scores"]).all()
        )
        self.assertFalse((owners[result["queries"]] == result["negative_claims"]).any())


if __name__ == "__main__":
    unittest.main()
