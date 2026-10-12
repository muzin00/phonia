"""Test fourth-corpus isolation and the unchanged CV-only intervention."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as x

study = x.module("src490_study_test", x.BASE / "study.py")
LEGACY_CORPUS = x.e.make_schedule.__globals__["corpus"]


class CorpusTests(unittest.TestCase):
    def setUp(self):
        x.s.torch.set_num_threads(1)
        x.e.make_schedule.__globals__["corpus"] = LEGACY_CORPUS

    def test_legacy_schedule_equivalence_and_src_isolation(self):
        legacy = x.e.make_schedule.__globals__["corpus"]
        speakers = [
            f"{prefix}{i:03d}"
            for prefix in ("jvs", "cv17_", "jtube_")
            for i in range(10)
        ]
        data = {"speakers": speakers, "queries": [{"speaker_id": v} for v in speakers]}
        config = {"updates": 10, "queries_per_update": 32}
        before = x.e.make_schedule(config, data, 123)
        try:
            x.install_corpus_adapter()
            after = x.e.make_schedule(config, data, 123)
            for key in before:
                x.s.np.testing.assert_array_equal(before[key], after[key])
            self.assertEqual(x.corpus("ext_src4vc001"), "src4vc")
            with self.assertRaises(ValueError):
                x.corpus("unknown001")
            data["speakers"] += [f"ext_src4vc{i:03d}" for i in range(10)]
            data["queries"] = [{"speaker_id": v} for v in data["speakers"]]
            mixed = x.e.make_schedule(config, data, 123)
            owners = x.s.np.array([x.corpus(v) for v in data["speakers"]])
            self.assertTrue(
                (owners[mixed["queries"]] == owners[mixed["negative_claims"]]).all()
            )
            self.assertTrue((mixed["queries"] != mixed["negative_claims"]).all())
        finally:
            x.e.make_schedule.__globals__["corpus"] = legacy

    def test_cv_only_and_real_training_step(self):
        x.install_corpus_adapter()
        np = x.s.np
        rng = np.random.default_rng(40)
        speakers = [
            f"{prefix}{i:03d}"
            for prefix in ("jvs", "cv17_", "jtube_", "ext_src4vc")
            for i in range(10)
        ]
        data = {"speakers": speakers, "queries": [{"speaker_id": v} for v in speakers]}
        arrays = {
            "enrollment": rng.normal(size=(40, 36, 128)).astype(np.float32),
            "queries": rng.normal(size=(40, 36, 128)).astype(np.float32),
            "emask": np.ones((40, 36), bool),
            "qmask": np.ones((40, 36), bool),
            "emeta": rng.normal(size=(40, 36, 4)).astype(np.float32),
            "qmeta": rng.normal(size=(40, 36, 4)).astype(np.float32),
        }
        for key in ("enrollment", "queries"):
            arrays[key] /= np.linalg.norm(arrays[key], axis=2, keepdims=True)
        config = {
            "updates": 12,
            "queries_per_update": 32,
            "budgets": [4, 8, 12],
            "mining_seed": 12,
            "candidates": 8,
            "hard_queries_per_update": 16,
        }
        original = x.e.make_schedule(config, data, 123)
        actual, table, _ = study.make_cv_schedule(config, original, arrays, data)
        for key in original:
            np.testing.assert_array_equal(original[key][:4], actual[key][:4])
        np.testing.assert_array_equal(original["queries"], actual["queries"])
        changed = actual["negative_claims"] != original["negative_claims"]
        self.assertTrue(changed.any())
        self.assertTrue(
            all(
                x.corpus(speakers[i]) == "common_voice"
                for i in actual["queries"][changed]
            )
        )
        q = actual["queries"][changed]
        self.assertTrue(
            (
                table[q, actual["negative_claims"][changed]]
                >= table[q, original["negative_claims"][changed]]
            ).all()
        )
        self.assertTrue(np.isneginf(table[0, 30]))
        tensors = {k: x.s.torch.from_numpy(v) for k, v in arrays.items()}
        f, c, mask, labels = x.e.training_batch(tensors, data, actual, 6)
        self.assertEqual(len(labels), 64)
        self.assertTrue(x.s.torch.equal(mask[:32], mask[32:]))
        model = x.e.create_model(
            {"dimension": 64, "transformer_layers": 2, "attention_heads": 4},
            "transformer",
        )
        optimizer = x.s.torch.optim.AdamW(model.parameters(), lr=0.0003)
        before = [p.detach().clone() for p in model.parameters()]
        loss = model.loss(f, c, mask, labels, 0.001)
        self.assertTrue(x.s.torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertTrue(
            any(
                not x.s.torch.equal(a, b)
                for a, b in zip(before, model.parameters(), strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
