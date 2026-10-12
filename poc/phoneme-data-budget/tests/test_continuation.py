import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_budget as d

s = d.acoustic.s


class ContinuationTests(unittest.TestCase):
    def test_continuation_restores_optimizer_rng_without_changing_source(self):
        s.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=d.ROOT / "artifacts") as tmp:
            root = Path(tmp)
            config = {
                "encoder": "statistics_mlp",
                "training_seed": 99,
                "maximum_updates": 12,
                "warmup_updates": 1,
                "checkpoint_interval": 2,
            }
            options = {
                "encoder_budgets": [12, 24],
                "encoder_continuation_learning_rate": 1e-5,
                "encoder_schedule_seed_stride": 1000003,
            }
            s.torch.manual_seed(100)
            phones = s.ordered(["a", "i", "u", "e", "o"])
            pool = SimpleNamespace(
                speakers=["a", "b", "c"],
                speaker_to_class={"a": 0, "b": 1, "c": 2},
                groups={
                    (speaker, phone): list(range(i * 10, i * 10 + 10))
                    for i, (speaker, phone) in enumerate(
                        (sp, ph) for sp in ("a", "b", "c") for ph in phones
                    )
                },
                features=s.torch.randn(150, 128),
                manifest_sha256="toy-manifest",
                feature_sha256="toy-features",
                statistics_sha256="toy-statistics",
            )
            d.fixed(root / "design-freeze.json", {"config": config})
            source = root / "source"
            s.train_trial(config, root, source, phones, pool)
            original = d.pin(source.iterdir())
            full, resumed = root / "full", root / "resumed"
            d.acoustic.train(
                options,
                source,
                full,
                pool,
                phones,
                99,
                lambda *args, **kwargs: None,
            )

            def interrupt(status, **details):
                if details["update"] == 18:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                d.acoustic.train(options, source, resumed, pool, phones, 99, interrupt)
            d.acoustic.train(
                options,
                source,
                resumed,
                pool,
                phones,
                99,
                lambda *args, **kwargs: None,
            )
            a = s.torch.load(full / "last.pt", weights_only=False)
            b = s.torch.load(resumed / "last.pt", weights_only=False)
            self.assertEqual(a["update"], 24)
            for part in ("model", "head"):
                for key in a[part]:
                    s.torch.testing.assert_close(
                        a[part][key], b[part][key], rtol=0, atol=0
                    )
            for key, values in a["optimizer"]["state"].items():
                for field, value in values.items():
                    s.torch.testing.assert_close(
                        value, b["optimizer"]["state"][key][field], rtol=0, atol=0
                    )
            self.assertEqual(
                a["optimizer"]["param_groups"], b["optimizer"]["param_groups"]
            )
            self.assertEqual(
                d.read(full / "training-summary.json")["final_encoder_sha256"],
                d.read(resumed / "training-summary.json")["final_encoder_sha256"],
            )
            d.check_files(original)

    def test_schedule_has_distinct_pairs_and_seed_changes(self):
        pool = SimpleNamespace(
            groups={
                (speaker, phone): list(range(i * 10, i * 10 + 10))
                for i, (speaker, phone) in enumerate(
                    (sp, ph) for sp in ("a", "b") for ph in ("a", "i")
                )
            },
            speaker_to_class={"a": 0, "b": 1},
        )
        config = {"encoder_budgets": [12, 24], "encoder_schedule_seed_stride": 1000003}
        first = d.acoustic.make_schedule(config, pool, ["a", "i"], 99)
        again = d.acoustic.make_schedule(config, pool, ["a", "i"], 99)
        other = d.acoustic.make_schedule(config, pool, ["a", "i"], 100)
        self.assertEqual(
            d.acoustic.schedule_digest(first), d.acoustic.schedule_digest(again)
        )
        self.assertNotEqual(
            d.acoustic.schedule_digest(first), d.acoustic.schedule_digest(other)
        )
        pairs = first["planned"].reshape(-1, 2)
        mask = first["valid"].reshape(-1, 2)
        self.assertTrue(s.np.array_equal(mask[:, 0], mask[:, 1]))
        self.assertTrue((pairs[mask[:, 0], 0] != pairs[mask[:, 0], 1]).all())
        with self.assertRaisesRegex(AssertionError, "phone phase"):
            d.acoustic.make_schedule(
                {**config, "encoder_budgets": [11, 24]}, pool, ["a", "i"], 99
            )


if __name__ == "__main__":
    unittest.main()
