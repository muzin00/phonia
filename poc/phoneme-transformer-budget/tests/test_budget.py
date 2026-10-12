import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as b

s, e = b.s, b.e


class BudgetTests(unittest.TestCase):
    def test_selection_uses_earliest_tie_and_only_available_history(self):
        history = [
            {"update": u, "macro_eer": v}
            for u, v in [(0, 0.4), (4000, 0.2), (8000, 0.2), (16000, 0.1)]
        ]
        self.assertEqual(b.select_best(history[:3])["update"], 4000)
        self.assertEqual(b.select_best(history)["update"], 16000)

    def test_schedule_preserves_original_prefix_and_budget_prefix(self):
        with tempfile.TemporaryDirectory(dir=s.ROOT / "artifacts") as tmp:
            root = Path(tmp)
            config = {
                "reference_run": str(root.relative_to(s.ROOT)),
                "budgets": [2, 4, 6],
                "schedule_block_updates": 2,
                "schedule_seed_stride": 1000003,
            }
            data = {
                "speakers": ["jvs001", "jvs002"],
                "queries": [{"speaker_id": "jvs001"}, {"speaker_id": "jvs002"}],
            }
            original = {"updates": 2, "queries_per_update": 2}
            first = e.make_schedule(original, data, 99)
            s.np.savez(root / "schedule-99.npz", **first)
            full = b.build_schedule(config, original, data, 99)
            shorter = b.build_schedule(
                {**config, "budgets": [2, 4]}, original, data, 99
            )
            for key in first:
                s.np.testing.assert_array_equal(full[key][:2], first[key])
                s.np.testing.assert_array_equal(full[key][:4], shorter[key])

    def test_resume_restores_optimizer_rng_and_best_checkpoint(self):
        s.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=s.ROOT / "artifacts") as tmp:
            root = Path(tmp)
            ref = root / "reference"
            ref.mkdir()
            config = {
                "reference_run": str(ref.relative_to(s.ROOT)),
                "budgets": [2, 4, 6],
                "continuation_learning_rate": 0.00003,
                "validation_every": 2,
            }
            original = {
                "dimension": 8,
                "transformer_layers": 1,
                "attention_heads": 2,
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "uniform_kl_coefficient": 0.001,
                "encoder_sha256": "toy-fixed-encoder",
            }
            s.fixed(ref / "design-freeze.json", {"config": original})
            s.torch.manual_seed(99)
            model = e.create_model(original, "transformer")
            initial = copy.deepcopy(model.state_dict())
            optimizer = s.torch.optim.AdamW(
                model.parameters(), lr=0.0003, weight_decay=0.01
            )
            arrays = {
                "enrollment": s.torch.randn(2, 6, 128),
                "queries": s.torch.randn(2, 6, 128),
                "emeta": s.torch.randn(2, 6, 4),
                "qmeta": s.torch.randn(2, 6, 4),
                "emask": s.torch.ones(2, 6, dtype=s.torch.bool),
                "qmask": s.torch.ones(2, 6, dtype=s.torch.bool),
            }
            data = {
                "speakers": ["jvs001", "jvs002"],
                "queries": [{"speaker_id": "jvs001"}, {"speaker_id": "jvs002"}],
            }
            schedule = e.make_schedule(
                {"updates": 6, "queries_per_update": 2}, data, 99
            )
            for u in range(2):
                optimizer.zero_grad(set_to_none=True)
                f, c, m, y = e.training_batch(arrays, data, schedule, u)
                model.loss(f, c, m, y).backward()
                optimizer.step()
            history = [{"update": 0, "macro_eer": 0.5}, {"update": 2, "macro_eer": 0.6}]
            prior = ref / "transformer-99"
            prior.mkdir()
            s.torch.save(
                {"kind": "transformer", "seed": 99, "update": 0, "state_dict": initial},
                prior / "fusion.pt",
            )
            s.torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": s.torch.get_rng_state(),
                    "history": history,
                    "best": 0.5,
                    "selected": 0,
                    "update": 2,
                    "design_sha256": s.sha(ref / "design-freeze.json"),
                },
                prior / "last.pt",
            )
            s.fixed(ref / "train-inputs.json", data)
            plain = {k: v.numpy() for k, v in arrays.items()}
            s.np.savez(ref / "train.npz", **plain)
            qids, claims = s.torch.tensor([0, 0, 1, 1]), s.torch.tensor([0, 1, 0, 1])
            _, components, masks = e.batch(arrays, qids, claims)
            rows = [
                {"role": "verification", "is_genuine": q == c, "score": 0.0}
                for q, c in zip(qids.tolist(), claims.tolist())
            ]
            for corpus in s.CORPORA:
                s.np.savez(
                    ref / f"{corpus}-validation.npz",
                    **plain,
                    components=components.numpy().astype("float64"),
                    masks=masks.numpy(),
                )
                s.fixed(ref / f"{corpus}-validation-baseline.json", rows)
            full, resumed = root / "full", root / "resumed"
            for run in (full, resumed):
                run.mkdir()
                s.fixed(run / "design-freeze.json", {"config": config})
                s.np.savez(run / "schedule-99.npz", **schedule)
            b.worker(config, full, 99)
            write = b.write_state

            def stop_after_checkpoint(path, status, **details):
                write(path, status, **details)
                if status == "training" and details.get("update") == 4:
                    raise RuntimeError("simulated interruption")

            with (
                patch.object(b, "write_state", side_effect=stop_after_checkpoint),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                b.worker(config, resumed, 99)
            # Simulate an interrupted write leaving a newer, incompatible best file.
            s.torch.save({"bad": True}, resumed / "seed-99/best.pt")
            b.worker(config, resumed, 99)
            for name in ("last.pt", "budget-6/selected.pt", "budget-6/final.pt"):
                a = s.torch.load(full / "seed-99" / name, weights_only=False)
                z = s.torch.load(resumed / "seed-99" / name, weights_only=False)
                key = "model" if name == "last.pt" else "state_dict"
                self.assertEqual(a["update"], z["update"])
                for parameter in a[key]:
                    s.torch.testing.assert_close(
                        a[key][parameter], z[key][parameter], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
