import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replication as x

d, s, e = x.d, x.s, x.e


class ReplicationTests(unittest.TestCase):
    def test_new_initializations_keep_schedule_and_share_exact_fork(self):
        s.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=d.ROOT / "artifacts") as tmp:
            root = Path(tmp)
            source, run = root / "source", root / "run"
            source.mkdir()
            run.mkdir()
            config = {
                "budgets": [2, 4, 6],
                "validation_every": 2,
                "continuation_learning_rate": 0.00003,
            }
            original = {
                "updates": 2,
                "seeds": [100],
                "architectures": ["transformer"],
                "dimension": 8,
                "transformer_layers": 1,
                "attention_heads": 2,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "uniform_kl_coefficient": 0.001,
                "encoder_sha256": "fixed-toy",
                "validation_every": 2,
                "queries_per_update": 3,
            }
            d.fixed(run / "design-freeze.json", {"config": config})
            d.fixed(source / "design-freeze.json", {"config": original})
            data = {
                "speakers": ["jvs001", "jvs002", "jvs003"],
                "queries": [{"speaker_id": f"jvs00{i}"} for i in (1, 2, 3)],
            }
            d.fixed(source / "train-inputs.json", data)
            s.torch.manual_seed(42)
            arrays = {
                "enrollment": s.torch.randn(3, 6, 128),
                "queries": s.torch.randn(3, 6, 128),
                "emeta": s.torch.randn(3, 6, 4),
                "qmeta": s.torch.randn(3, 6, 4),
                "emask": s.torch.ones(3, 6, dtype=s.torch.bool),
                "qmask": s.torch.ones(3, 6, dtype=s.torch.bool),
            }
            plain = {k: v.numpy() for k, v in arrays.items()}
            s.np.savez(source / "train.npz", **plain)
            schedule = e.make_schedule({**original, "updates": 6}, data, 100)
            saved_schedule = source / "fixed-schedule.npz"
            s.np.savez(saved_schedule, **schedule)
            q, c = s.torch.arange(3).repeat_interleave(3), s.torch.arange(3).repeat(3)
            _, components, masks = e.batch(arrays, q, c)
            rows = [
                {"role": "verification", "is_genuine": i == j, "score": 0.0}
                for i, j in zip(q.tolist(), c.tolist())
            ]
            for corpus in s.CORPORA:
                s.np.savez(
                    source / f"{corpus}-validation.npz",
                    **plain,
                    components=components.numpy().astype("float64"),
                    masks=masks.numpy(),
                )
                d.fixed(source / f"{corpus}-validation-baseline.json", rows)
            with (
                patch.object(x, "sources", return_value={1: {"fusion": str(source)}}),
                patch.object(x, "schedule_path", return_value=saved_schedule),
            ):
                for seed in (101, 102):
                    x.prefix_worker(config, run, 1, seed)
                    x.prefix_worker(
                        config, run, 1, seed
                    )  # Completed prefix is immutable/reusable.
                    path = x.prefix_path(run, 1, seed)
                    with s.np.load(path / f"schedule-{seed}.npz") as actual:
                        for key in schedule:
                            s.np.testing.assert_array_equal(
                                actual[key], schedule[key][:2]
                            )
                for arm in ("random", "cv_only"):
                    x.arm_worker(config, run, 1, 101, arm)
                prefix = s.torch.load(
                    x.prefix_path(run, 1, 101) / "transformer-101/last.pt",
                    weights_only=False,
                )
                other = s.torch.load(
                    x.prefix_path(run, 1, 102) / "transformer-102/last.pt",
                    weights_only=False,
                )
                self.assertTrue(
                    any(
                        not s.torch.equal(v, other["model"][k])
                        for k, v in prefix["model"].items()
                    )
                )
                endings = []
                for arm in ("random", "cv_only"):
                    folder = x.arm_path(run, 1, 101, arm) / "seed-101"
                    fork = s.torch.load(folder / "budget-2/final.pt", weights_only=True)
                    self.assertTrue(
                        all(
                            s.torch.equal(v, fork["state_dict"][k])
                            for k, v in prefix["model"].items()
                        )
                    )
                    endings.append(s.torch.load(folder / "last.pt", weights_only=False))
                # With identical toy schedules both arms must train identically,
                # demonstrating that the arm name itself changes no optimization.
                self.assertTrue(
                    all(
                        s.torch.equal(v, endings[1]["model"][k])
                        for k, v in endings[0]["model"].items()
                    )
                )
                self.assertTrue(s.torch.equal(endings[0]["rng"], endings[1]["rng"]))


if __name__ == "__main__":
    unittest.main()
