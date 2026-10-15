"""Behavioral checks: actual choices, probability gradients, forks, and replay."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import Teacher, sample, uniform_probabilities
from pool import Candidates, Coverage, policy_features, selected_indices
from runtime import BASE, acoustic, np, read, torch, tree_hash
from student import Student
from verification import Bank, Evaluation


def fixture():
    rng = np.random.default_rng(1701)
    phones, speakers, per_group = list("aiueo"), [f"jvs{i:03d}" for i in range(10)], 12
    groups, gs, gp, ss, sp = [], [], [], [], []
    for s in range(len(speakers)):
        for p in range(len(phones)):
            start = len(ss)
            groups.append(np.arange(start, start + per_group))
            gs.append(s)
            gp.append(p)
            ss.extend([s] * per_group)
            sp.extend([p] * per_group)
    size = len(ss)
    pool = SimpleNamespace(
        phones=phones,
        speakers=speakers,
        groups=groups,
        group_speaker=np.array(gs),
        group_phone=np.array(gp),
        segment_speaker=np.array(ss),
        segment_phone=np.array(sp),
        eligible=np.arange(len(groups)),
        singletons=np.array([], dtype=np.int64),
        features=rng.normal(size=(size, 128)).astype(np.float32),
        metadata=np.tile(np.array([-2.0, -0.5], np.float32), (size, 1)),
        recording=np.arange(size),
    )
    profiles, queries = {}, []
    for s, speaker in enumerate(speakers):
        profiles[speaker] = {
            p: groups[s * 5 + j][:3].tolist() for j, p in enumerate(phones)
        }
        queries.append(
            {
                "speaker_id": speaker,
                "query_id": str(s),
                "role": "verification",
                "groups": {
                    p: groups[s * 5 + j][3:].tolist() for j, p in enumerate(phones)
                },
            }
        )
    info = {
        "speakers": speakers,
        "profiles": profiles,
        "queries": queries,
        "universally_registered": phones,
    }
    metadata = np.tile(np.array([0.1, -25], np.float32), (size, 1))
    bank = Bank(info, pool.features, metadata, phones)
    rows = [
        {
            "speaker_id": s,
            "claimed_speaker_id": c,
            "role": "verification",
            "score": 0.0,
            "is_genuine": s == c,
        }
        for s in speakers
        for c in speakers
    ]
    evaluation = Evaluation(bank, rows)
    config = read(BASE / "config.json")
    config.update(
        updates=40,
        warmup_updates=2,
        block_updates=20,
        candidate_pairs=4,
        training_seeds=[123],
        fusion_steps_per_block=1,
        fusion_queries_per_update=2,
        long_probe_every_blocks=1,
        long_probe_updates=10,
        proxy_interval=10,
        proxy_queries_per_corpus=2,
        final_fusion_updates=4,
    )
    return pool, bank, {"jvs": evaluation}, config


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_candidates_preserve_pairs_speakers_and_resume(self):
        pool, _, _, _ = fixture()
        sampler = Candidates(pool, 123)
        coverage = Coverage(len(pool.features))
        for update in range(4):
            data = sampler.draw(update)
            self.assertTrue((data["pairs"][..., 0] != data["pairs"][..., 1]).all())
            self.assertTrue(
                (
                    pool.segment_speaker[data["pairs"]][..., 0]
                    == pool.segment_speaker[data["pairs"]][..., 1]
                ).all()
            )
            self.assertTrue(
                (
                    pool.segment_phone[data["pairs"]][..., 0]
                    == pool.segment_phone[data["pairs"]][..., 1]
                ).all()
            )
            choices = sample(uniform_probabilities(data["mask"]), np.full(50, 0.1))
            ids, mask = selected_indices(data, choices)
            labels = pool.segment_speaker[ids].reshape(5, 20)[:, ::2]
            self.assertTrue(all(len(np.unique(row)) == 10 for row in labels))
            coverage.record(data, ids)
            self.assertTrue(mask.all())
        self.assertEqual(np.count_nonzero(coverage.candidate), len(pool.features))
        restored = Candidates(pool, 123)
        restored.load_state_dict(sampler.state_dict())
        self.assertEqual(tree_hash(sampler.draw(4)), tree_hash(restored.draw(4)))
        # With the same seed, group slots exactly match the existing acoustic learner.
        mapping = {
            (pool.speakers[s], pool.phones[p]): pool.groups[g]
            for g, (s, p) in enumerate(zip(pool.group_speaker, pool.group_phone))
        }
        _, old_labels, old_valid = acoustic.schedule(
            mapping, pool.phones, 5, 123, {s: i for i, s in enumerate(pool.speakers)}
        )
        slots = restored.group_slots(4)
        np.testing.assert_array_equal(
            pool.group_speaker[slots].repeat(2), old_labels[4]
        )
        self.assertTrue(old_valid.all())

    def test_policy_gradient_matches_exact_expected_reward(self):
        torch.manual_seed(3)
        teacher = Teacher(5)
        features = torch.randn(1, 4, 269)
        phones = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.tensor([[True, True, True, False]])
        p = teacher.probabilities(features, phones, mask)
        reward = torch.tensor([-0.2, 0.1, 0.7, 0.0])
        parameters = tuple(teacher.parameters())
        exact = torch.autograd.grad(
            -(p[0] * reward).sum(), parameters, retain_graph=True
        )
        estimate = sum(-p[0, i].detach() * reward[i] * p[0, i].log() for i in range(3))
        actual = torch.autograd.grad(estimate, parameters)
        for a, b in zip(exact, actual):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
        self.assertEqual(float(p[0, 3].detach()), 0.0)
        self.assertTrue((p[0, :3] >= 0.2 / 3).all())

    def test_sparse_groups_mask_padding_and_no_duplicate_pair_actions(self):
        pool, _, _, _ = fixture()
        pool.groups[0] = pool.groups[0][:2]
        pool.groups[1] = pool.groups[1][:3]
        pool.eligible = np.array(
            [
                g
                for g in pool.eligible
                if pool.group_phone[g] != 4 or pool.group_speaker[g] < 2
            ]
        )
        data = Candidates(pool, 12).draw(0)
        for i, g in enumerate(data["groups"]):
            if g < 0:
                self.assertFalse(data["mask"][i].any())
            else:
                valid = data["pairs"][i, data["mask"][i]]
                self.assertEqual(
                    len(set(map(tuple, np.sort(valid, axis=1)))), len(valid)
                )
        choices = sample(uniform_probabilities(data["mask"]), np.full(50, 0.9))
        ids, mask = selected_indices(data, choices)
        self.assertEqual(int(mask.sum()), 84)
        pool.features[0] = 1e4
        student = Student(fixture()[3], 10, 12)
        self.assertTrue(np.isfinite(student.step(pool, ids, mask, 0)))

    def test_fork_optimizer_isolation_and_fresh_pooling(self):
        pool, bank, evaluations, config = fixture()
        student = Student(config, 10, 123)
        batch = Candidates(pool, 123).draw(0)
        ids, mask = selected_indices(batch, np.zeros(50, dtype=int))
        student.step(pool, ids, mask, 0)
        parent = tree_hash(student.state_dict())
        before = evaluations["jvs"].scores(student)
        child = student.fork()
        child.step(pool, ids, mask, 1)
        self.assertEqual(parent, tree_hash(student.state_dict()))
        after = evaluations["jvs"].scores(child)
        self.assertGreater(float(np.abs(before - after).max()), 1e-7)
        arrays = bank.materialize(child.encoder, np.array([0]), np.array([0]))
        with torch.no_grad():
            z = torch.nn.functional.normalize(
                child.encoder.projection(
                    torch.from_numpy(pool.features[pool.groups[0][:3]])
                ),
                dim=1,
            )
        expected = torch.nn.functional.normalize(z.mean(0), dim=0)
        torch.testing.assert_close(
            arrays["enrollment"][0, 0], expected, atol=1e-6, rtol=1e-6
        )

    def test_pair_features_have_expected_shape(self):
        pool, _, _, config = fixture()
        student = Student(config, 10, 123)
        values, phone = policy_features(
            pool,
            student,
            Candidates(pool, 123).draw(0),
            Coverage(len(pool.features)),
            0.1,
        )
        self.assertEqual(values.shape, (50, 4, 269))
        self.assertEqual(phone.shape, (50, 4))

    def test_uniform_teacher_and_control_use_independent_draws(self):
        from trainer import choices_for, teacher_for

        pool, _, _, config = fixture()
        student = Student(config, 10, 123)
        teacher, _ = teacher_for(config, 5)
        batch = Candidates(pool, 123).draw(0)
        a, _ = choices_for(
            config, pool, student, batch, Coverage(len(pool.features)), teacher, 0, 123
        )
        b, _ = choices_for(
            config, pool, student, batch, None, None, 0, 123, stream="control"
        )
        self.assertGreater(np.count_nonzero(a != b), 10)

    def test_inverse_cdf_never_selects_padding_at_float_boundary(self):
        p = torch.tensor([[0.3333333, 0.3333333, 0.3333333, 0.0], [1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(
            sample(p, np.array([np.nextafter(1.0, 0.0), np.nextafter(1.0, 0.0)])),
            [2, 0],
        )

    def test_final_fusion_resumes_exactly(self):
        from final_fusion import fit

        _, bank, evaluation, config = fixture()
        config.update(final_fusion_updates=4, final_fusion_validation_every=2)
        schedule = {
            "queries": np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            "negative_claims": np.array([[2, 3], [4, 5], [6, 7], [8, 9]]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = Student(config, 10, 123)
            fit(config, root / "whole", a, bank, evaluation, schedule_override=schedule)
            b = Student(config, 10, 123)
            fit(
                config,
                root / "resumed",
                b,
                bank,
                evaluation,
                schedule_override=schedule,
                stop_after=2,
            )
            c = Student(config, 10, 123)
            fit(
                config,
                root / "resumed",
                c,
                bank,
                evaluation,
                schedule_override=schedule,
            )
            self.assertEqual(
                tree_hash(a.fusion.state_dict()), tree_hash(c.fusion.state_dict())
            )
            self.assertEqual(
                tree_hash(a.fusion_optimizer.state_dict()),
                tree_hash(c.fusion_optimizer.state_dict()),
            )
            self.assertEqual(
                read(root / "whole/history.json"), read(root / "resumed/history.json")
            )

    def test_training_replay_with_delayed_reward_and_negative_commit(self):
        from trainer import teacher_for, train_episode

        pool, bank, evaluations, config = fixture()
        for arm in ("eer", "proxy"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                teacher, opt = teacher_for(config, 5)
                a, coverage_a, done = train_episode(
                    config,
                    root / "whole",
                    pool,
                    bank,
                    evaluations,
                    arm,
                    0,
                    teacher,
                    opt,
                )
                self.assertTrue(done)
                teacher_hash = tree_hash(teacher.state_dict())
                teacher2, opt2 = teacher_for(config, 5)
                _, _, done = train_episode(
                    config,
                    root / "resumed",
                    pool,
                    bank,
                    evaluations,
                    arm,
                    0,
                    teacher2,
                    opt2,
                    stop_after_blocks=1,
                )
                self.assertFalse(done)
                teacher3, opt3 = teacher_for(config, 5)
                b, coverage_b, done = train_episode(
                    config,
                    root / "resumed",
                    pool,
                    bank,
                    evaluations,
                    arm,
                    0,
                    teacher3,
                    opt3,
                )
                self.assertEqual(tree_hash(a.state_dict()), tree_hash(b.state_dict()))
                self.assertEqual(teacher_hash, tree_hash(teacher3.state_dict()))
                self.assertEqual(
                    tree_hash(coverage_a.state_dict()),
                    tree_hash(coverage_b.state_dict()),
                )
                history = read(root / "whole/history.json")
                if arm == "eer":
                    self.assertIn("delayed_reward", history[0])
                    self.assertTrue(
                        all(
                            v["committed"] == "sampled_even_if_negative"
                            for v in history
                        )
                    )
                    self.assertLess(history[0]["reward"], 0.0)
                # A final checkpoint must also recover if the completion seal
                # was never written (e.g. interruption while exporting files).
                (root / "resumed/complete.json").unlink()
                resumed, _, done = train_episode(
                    config,
                    root / "resumed",
                    pool,
                    bank,
                    evaluations,
                    arm,
                    0,
                    teacher3,
                    opt3,
                )
                self.assertTrue(done)
                self.assertEqual(
                    tree_hash(a.state_dict()), tree_hash(resumed.state_dict())
                )

    def test_partition_filters_both_query_and_claim(self):
        pool, bank, _, _ = fixture()
        rows = [
            {
                "speaker_id": s,
                "claimed_speaker_id": c,
                "role": "verification",
                "score": 0.5,
                "is_genuine": s == c,
            }
            for s in pool.speakers
            for c in pool.speakers
        ]
        data = Evaluation(bank, rows, pool.speakers[:3])
        self.assertEqual(len(data.labels), 9)
        self.assertTrue((data.claims < 3).all())
        self.assertTrue((data.query_speakers < 3).all())


if __name__ == "__main__":
    unittest.main()
