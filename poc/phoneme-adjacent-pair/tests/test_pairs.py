import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import independent_eer
from evaluate import score, summarize
from inventory import candidate
from pair_common import CONFIG, metrics, np, read, torch, write
from pair_model import PairEncoder
from report import report
from train import optimize, schedule


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            nested_equal(x, y)
    else:
        assert a == b, (a, b)


class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = read(CONFIG)

    def test_only_real_contiguous_uncropped_cv_intervals(self):
        left = {
            "source_file": "one.wav",
            "speaker_id": "s1",
            "source_sha256": "abc",
            "alignment_file": "one.json",
            "phoneme": "k",
            "interval_index": 1,
            "raw_start_frame": 0,
            "raw_end_frame": 1200,
            "start_frame": 0,
            "end_frame": 1200,
            "quality_flags": [],
            "segment_id": "left",
            "cache_index": 0,
        }
        right = {
            **left,
            "phoneme": "a",
            "interval_index": 2,
            "raw_start_frame": 1200,
            "raw_end_frame": 3600,
            "start_frame": 1200,
            "end_frame": 3600,
            "segment_id": "right",
            "cache_index": 1,
        }
        got = candidate(left, right, self.config, {"k"})
        self.assertEqual(got["frames"], [0, 1200, 3600])
        self.assertEqual(got["pair"], "k+a")
        for change in (
            {"interval_index": 3},
            {"source_file": "other.wav"},
            {"raw_start_frame": 1201},
            {"raw_end_frame": 7000},
            {"raw_end_frame": 1300},
            {"phoneme": "pau"},
        ):
            self.assertIsNone(candidate(left, {**right, **change}, self.config, {"k"}))
        with self.assertRaises(AssertionError):
            candidate(left, {**right, "start_frame": 1201}, self.config, {"k"})

    def test_identical_inputs_have_identical_outputs_gradients_and_rng(self):
        x = torch.linspace(-2, 2, 5 * 128).reshape(5, 1, 128).expand(-1, 3, -1).clone()
        outcomes = []
        for arm in ("split", "joined"):
            torch.manual_seed(13)
            model = PairEncoder(arm).train()
            y = model(x)
            y[:, 0].sum().backward()
            outcomes.append(
                (
                    y.detach(),
                    [p.grad.clone() for p in model.parameters()],
                    torch.get_rng_state(),
                )
            )
        nested_equal(outcomes[0], outcomes[1])

    def test_unused_input_branches_have_zero_gradient(self):
        for arm, unused, used in (("split", [2], [0, 1]), ("joined", [0, 1], [2])):
            torch.manual_seed(7)
            model = PairEncoder(arm).eval()
            x = torch.randn(5, 3, 128, requires_grad=True)
            model(x)[:, 0].sum().backward()
            self.assertEqual(torch.count_nonzero(x.grad[:, unused]).item(), 0)
            self.assertGreater(torch.count_nonzero(x.grad[:, used]).item(), 0)

    def test_schedule_real_positive_pairs_unique_speakers_and_balance(self):
        units = [
            {"speaker_id": f"s{s}", "pair": p}
            for p in ("k+a", "t+e")
            for s in range(12)
            for _ in range(3)
        ]
        speakers = sorted({u["speaker_id"] for u in units})
        ids, labels, digest = schedule(units, ["k+a", "t+e"], 12, 42, speakers)
        ids2, labels2, digest2 = schedule(units, ["k+a", "t+e"], 12, 42, speakers)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_array_equal(labels, labels2)
        self.assertEqual(digest, digest2)
        for i, group in enumerate(ids.reshape(-1, 20)):
            self.assertEqual({units[j]["pair"] for j in group}, {["k+a", "t+e"][i % 2]})
            self.assertEqual(len({units[j]["speaker_id"] for j in group}), 10)
            for a, b in zip(group[::2], group[1::2], strict=True):
                self.assertNotEqual(a, b)
                self.assertEqual(units[a]["speaker_id"], units[b]["speaker_id"])

    def test_resume_exact_optimizer_model_rng_and_history(self):
        units = [
            {"speaker_id": f"s{s}", "pair": p}
            for p in ("k+a", "t+e")
            for s in range(10)
            for _ in range(2)
        ]
        ids, labels, digest = schedule(
            units, ["k+a", "t+e"], 12, 42, sorted({u["speaker_id"] for u in units})
        )
        ids, labels = torch.from_numpy(ids), torch.from_numpy(labels)
        features = torch.from_numpy(
            np.random.default_rng(2)
            .normal(size=(len(units), 3, 128))
            .astype(np.float32)
        )
        config = {**self.config, "warmup_updates": 1, "checkpoint_interval": 3}
        for arm in ("split", "joined"):
            with tempfile.TemporaryDirectory() as d:
                full, resumed = Path(d) / "full", Path(d) / "resumed"
                identity = {"schedule": digest}
                optimize(config, full, arm, 9, features, ids, labels, 10, identity)
                optimize(
                    config,
                    resumed,
                    arm,
                    9,
                    features,
                    ids,
                    labels,
                    10,
                    identity,
                    stop_after=6,
                )
                optimize(config, resumed, arm, 9, features, ids, labels, 10, identity)
                nested_equal(
                    torch.load(full / "last.pt", weights_only=False),
                    torch.load(resumed / "last.pt", weights_only=False),
                )
                before = (resumed / "last.pt").read_bytes()
                optimize(config, resumed, arm, 9, features, ids, labels, 10, identity)
                self.assertEqual(before, (resumed / "last.pt").read_bytes())
                with self.assertRaises(AssertionError):
                    optimize(
                        config,
                        resumed,
                        arm,
                        9,
                        features,
                        ids,
                        labels,
                        10,
                        {"schedule": "wrong"},
                    )

    def test_claim_independent_masks_and_no_score_rejections(self):
        vectors = np.zeros((5, 128))
        vectors[0, 0] = vectors[2, 0] = 1
        vectors[1, 1] = vectors[3, 1] = vectors[4, 1] = 1
        data = {
            "speakers": ["s0", "s1"],
            "profiles": {"s0": {"k+a": [0], "t+e": [4]}, "s1": {"k+a": [1], "t+e": []}},
            "universally_registered": ["k+a"],
            "queries": [
                {
                    "groups": {"k+a": [2], "t+e": [3]},
                    "old_eligible": True,
                    "scorable": True,
                },
                {
                    "groups": {"k+a": [], "t+e": [3]},
                    "old_eligible": True,
                    "scorable": False,
                },
                {
                    "groups": {"k+a": [2], "t+e": []},
                    "old_eligible": False,
                    "scorable": False,
                },
            ],
        }
        scores, _, mask, error = score(data, vectors, ["k+a", "t+e"])
        np.testing.assert_array_equal(scores[0], [1.0, 0.0])
        self.assertTrue(np.isnan(scores[1:]).all())
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(error, 0.0)
        result = summarize(
            scores.ravel(), np.array([1, 0, 0, 1, 1, 0], bool), 0.5, 3, 3
        )
        self.assertEqual(result["coverage"], 1 / 3)
        self.assertEqual(result["frr"], 2 / 3)
        self.assertEqual(result["far"], 0)

    def test_independent_tied_score_eer(self):
        for seed in range(8):
            rng = np.random.default_rng(seed)
            values = rng.integers(0, 7, size=50).astype(float)
            labels = np.arange(50) % 3 == 0
            self.assertAlmostEqual(
                independent_eer(values, labels),
                metrics.roc_eer(values, labels)["eer"],
                places=14,
            )

    def test_report_keeps_dataset_split_and_arm_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            base = Path(temp) / "published"
            config = {"seeds": [1, 2, 3], "corpora": ["a", "b", "c"]}
            for split in ("validation", "test"):
                rows = []
                pairs = []
                for ci, c in enumerate(config["corpora"]):
                    for seed in config["seeds"]:
                        for arm in ("split", "joined"):
                            value = 0.01 * (ci + seed) + (
                                0.005 if arm == "joined" else 0
                            )
                            rows.append(
                                {
                                    "split": split,
                                    "corpus": c,
                                    "seed": seed,
                                    "arm": arm,
                                    "role": "verification",
                                    "eer": value,
                                    "far": 0.01,
                                    "frr": 0.1,
                                    "coverage": 0.9,
                                }
                            )
                            pairs.append(
                                {
                                    "split": split,
                                    "corpus": c,
                                    "seed": seed,
                                    "arm": arm,
                                    "pair": "k+a",
                                    "eer": value,
                                    "queries": 10,
                                    "total_queries": 20,
                                }
                            )
                write(run / f"{split}-metrics.json", rows)
                write(run / f"{split}-pairs.json", pairs)
            write(
                run / "pair-selection.json",
                {
                    "selected": ["k+a"],
                    "support": {"k+a": {}},
                    "eligible_ranked": ["k+a"],
                },
            )
            write(
                run / "bootstrap.json",
                {
                    "eer_delta_ci95_pp": {
                        c: [0.0, 1.0] for c in [*config["corpora"], "macro"]
                    }
                },
            )
            write(run / "independent-audit.json", {})
            for seed in config["seeds"]:
                for arm in ("split", "joined"):
                    write(run / f"seed-{seed}/{arm}/summary.json", {})
            with patch("report.BASE", base):
                report(config, run)
            result = read(run / "results.json")
            self.assertEqual(len(result["summary"]), 8)
            for row in result["summary"]:
                self.assertIn(row["dataset_split"], ("validation", "test"))
                self.assertAlmostEqual(row["delta_pp"], 0.5)
                self.assertEqual(row["worsened"], 3)
            self.assertEqual(
                (run / "results.md").read_bytes(),
                (base / "evaluation-results.md").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
