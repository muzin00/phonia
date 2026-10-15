import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_data import audit_bank, make_schedule, subset_bank, subset_pool
from ablation_train import AvailableCandidates, train_uniform
from pool import Candidates
from shared_ablation import LEGACY, Bank, Student, fit, rt
from trainer import train_episode

sys.path.insert(0, str(LEGACY / "tests"))
from test_selection import fixture


def mixed_fixture():
    pool, bank, _, config = fixture()
    names = {
        s: f"cv17_{i:03d}" if i % 3 == 0 else s for i, s in enumerate(pool.speakers)
    }
    pool.speakers = [names[s] for s in pool.speakers]
    info = copy.deepcopy(bank.info)
    info["speakers"] = pool.speakers
    info["profiles"] = {names[s]: v for s, v in info["profiles"].items()}
    for q in info["queries"]:
        q["speaker_id"] = names[q["speaker_id"]]
    keep = [s for s in pool.speakers if rt.corpus(s) != "common_voice"]
    details, arrays, source_ids, mapping = subset_pool(pool, keep)
    filtered = SimpleNamespace(
        **arrays,
        phones=pool.phones,
        speakers=details["speakers"],
        features=pool.features[source_ids],
    )
    filtered.groups = [
        filtered.indices[a:b]
        for a, b in zip(filtered.offsets[:-1], filtered.offsets[1:])
    ]
    filtered.eligible = rt.np.array(
        [i for i, v in enumerate(filtered.groups) if len(v) >= 2]
    )
    filtered.singletons = rt.np.array(
        [v[0] for v in filtered.groups if len(v) == 1], dtype=rt.np.int64
    )
    subset = subset_bank(info, keep, mapping)
    metadata = rt.np.tile(rt.np.array([0.1, -25], rt.np.float32), (len(source_ids), 1))
    bank = Bank(subset, filtered.features, metadata, filtered.phones)
    return pool, info, filtered, subset, source_ids, mapping, bank, config


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rt.torch.set_num_threads(1)
        rt.torch.use_deterministic_algorithms(True)

    def test_filter_preserves_raw_features_and_all_references(self):
        original, info, pool, subset, source_ids, mapping, _, _ = mixed_fixture()
        audit = audit_bank(subset, pool, exclude_cv=True)
        self.assertEqual(len(pool.speakers), 6)
        self.assertEqual(audit["fusion_speakers"], 6)
        self.assertEqual(len(source_ids), 360)
        self.assertTrue((mapping[source_ids] == rt.np.arange(360)).all())
        self.assertEqual(int((mapping < 0).sum()), 240)
        rt.np.testing.assert_array_equal(pool.features, original.features[source_ids])
        for speaker in pool.speakers:
            for phone, ids in subset["profiles"][speaker].items():
                rt.np.testing.assert_array_equal(
                    source_ids[ids], info["profiles"][speaker][phone]
                )
        sampler = Candidates(pool, 123)
        for update in range(20):
            batch = sampler.draw(update)
            values = batch["pairs"][batch["mask"]].reshape(-1)
            self.assertTrue(
                all(
                    rt.corpus(original.speakers[s]) != "common_voice"
                    for s in original.segment_speaker[source_ids[values]]
                )
            )

    def test_rejects_cross_speaker_reference_and_excluded_reference(self):
        _, info, pool, subset, _, mapping, _, _ = mixed_fixture()
        bad = copy.deepcopy(subset)
        bad["profiles"][pool.speakers[0]]["a"][0] = bad["profiles"][pool.speakers[1]][
            "a"
        ][0]
        with self.assertRaisesRegex(ValueError, "speaker remapping"):
            audit_bank(bad, pool, exclude_cv=True)
        bad_source = copy.deepcopy(info)
        bad_source["profiles"][pool.speakers[0]]["a"][0] = int(
            rt.np.flatnonzero(mapping < 0)[0]
        )
        with self.assertRaisesRegex(ValueError, "excluded segment"):
            subset_bank(bad_source, pool.speakers, mapping)

    def test_schedule_is_reproducible_and_uses_only_within_corpus_impostors(self):
        _, info, _, subset, _, _, _, _ = mixed_fixture()
        for data in (info, subset):
            schedule, counts = make_schedule(data, 100, 32, 123)
            other, _ = make_schedule(data, 100, 32, 123)
            self.assertEqual(rt.tree_hash(schedule), rt.tree_hash(other))
            self.assertEqual(sum(counts.values()), 3200)
            owners = [q["speaker_id"] for q in data["queries"]]
            for q, n in zip(
                schedule["queries"].reshape(-1), schedule["negative_claims"].reshape(-1)
            ):
                self.assertNotEqual(owners[q], data["speakers"][n])
                self.assertEqual(rt.corpus(owners[q]), rt.corpus(data["speakers"][n]))
        self.assertNotIn("common_voice", counts)

    def test_encoder_and_fusion_initialization_match_across_head_sizes(self):
        _, _, pool, _, _, _, _, config = mixed_fixture()
        a, b = Student(config, 10, 123), Student(config, len(pool.speakers), 123)
        self.assertEqual(
            rt.tree_hash(a.encoder.state_dict()), rt.tree_hash(b.encoder.state_dict())
        )
        self.assertEqual(
            rt.tree_hash(a.fusion.state_dict()), rt.tree_hash(b.fusion.state_dict())
        )

    def test_available_sampler_preserves_legacy_training_when_all_phones_exist(self):
        pool, bank, _, config = fixture()
        old, new = Candidates(pool, 123), AvailableCandidates(pool, 123)
        for update in range(10):
            self.assertEqual(
                rt.tree_hash(old.draw(update)), rt.tree_hash(new.draw(update))
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, _, _ = train_episode(
                config,
                root / "old",
                pool,
                bank,
                None,
                "uniform",
                123,
                None,
                None,
                training_teacher=False,
            )
            b, _, _ = train_uniform(config, root / "new", pool, 123)
            self.assertEqual(rt.tree_hash(a.state_dict()), rt.tree_hash(b.state_dict()))

    def test_phone_with_only_singletons_still_gets_classification_updates(self):
        pool, _, _, config = fixture()
        for g in range(len(pool.groups)):
            if pool.group_phone[g] == 4:
                pool.groups[g] = pool.groups[g][:1]
        pool.eligible = rt.np.array(
            [g for g in range(len(pool.groups)) if len(pool.groups[g]) >= 2]
        )
        pool.singletons = rt.np.array([v[0] for v in pool.groups if len(v) == 1])
        sampler = AvailableCandidates(pool, 123)
        self.assertEqual(sampler.active_phones, [0, 1, 2, 3])
        config["singleton_interval"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            model, coverage, done = train_uniform(config, Path(tmp), pool, 123)
        self.assertTrue(done)
        self.assertEqual(int(coverage.singleton.sum()), 40)
        self.assertTrue((coverage.singleton[pool.singletons] > 0).all())
        self.assertTrue(
            all(rt.torch.isfinite(v).all() for v in model.encoder.state_dict().values())
        )

    def test_filtered_training_and_fusion_resume_exactly(self):
        _, _, pool, subset, _, _, bank, config = mixed_fixture()
        # Evaluation still includes the original unfiltered speakers/data.
        _, _, evaluation, _ = fixture()
        config.update(final_fusion_updates=4, final_fusion_validation_every=2)
        schedule, _ = make_schedule(subset, 4, 2, 123)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whole, cov, done = train_uniform(config, root / "whole", pool, 123)
            self.assertTrue(done)
            _, _, done = train_uniform(
                config, root / "resumed", pool, 123, stop_after_blocks=1
            )
            self.assertFalse(done)
            resumed, cov2, done = train_uniform(config, root / "resumed", pool, 123)
            self.assertTrue(done)
            self.assertEqual(
                rt.tree_hash(whole.state_dict()), rt.tree_hash(resumed.state_dict())
            )
            self.assertEqual(
                rt.tree_hash(cov.state_dict()), rt.tree_hash(cov2.state_dict())
            )
            fit(
                config,
                root / "fusion-whole",
                whole,
                bank,
                evaluation,
                schedule_override=schedule,
            )
            fit(
                config,
                root / "fusion-resumed",
                resumed,
                bank,
                evaluation,
                schedule_override=schedule,
                stop_after=2,
            )
            fit(
                config,
                root / "fusion-resumed",
                resumed,
                bank,
                evaluation,
                schedule_override=schedule,
            )
            for name in ("selected.pt", "final.pt"):
                a = rt.torch.load(root / "fusion-whole" / name, weights_only=True)
                b = rt.torch.load(root / "fusion-resumed" / name, weights_only=True)
                self.assertEqual(rt.tree_hash(a), rt.tree_hash(b))


if __name__ == "__main__":
    unittest.main()
