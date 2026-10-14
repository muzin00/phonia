import copy
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (
    ARMS,
    CONFIG,
    assert_nested_equal,
    batch,
    np,
    prior,
    read,
    seed_all,
    torch,
)
from losses import eer_rewards, objective, reinforce
from policies import (
    Policy,
    Scorer,
    deterministic_subset,
    phone_logits,
    sample_subsets,
    subset_kl,
    subset_log_prob,
)
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
    return {**read(CONFIG), "updates": 8, "validation_every": 2}


def model(arm):
    seed_all(99)
    base = prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)
    with torch.no_grad():
        base.output.weight.normal_(std=0.1)
    result = Policy(base, arm, sigma=1.5)
    seed_all(77)
    return result


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_conditional_density_kl_and_score_gradient_match_full_enumeration(self):
        z = torch.tensor([[0.2, -0.9, 1.1]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        actions = torch.tensor(
            [
                [[a, False, b]]
                for a, b in itertools.product([False, True], repeat=2)
                if a or b
            ]
        )
        logp = subset_log_prob(z, actions, mask).squeeze(-1)
        probabilities = logp.exp()
        self.assertAlmostEqual(float(probabilities.detach().sum()), 1.0, places=14)
        utility = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64)
        expected_gradient = torch.autograd.grad(
            (probabilities * utility).sum(), z, retain_graph=True
        )[0]
        score_gradient = torch.autograd.grad(
            (probabilities.detach() * utility * logp).sum(), z, retain_graph=True
        )[0]
        torch.testing.assert_close(
            expected_gradient, score_gradient, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(float(score_gradient[0, 1]), 0)
        ref = torch.tensor([[-0.7, 0.5, 0.4]], dtype=torch.float64)
        exact_kl = (
            probabilities * (logp - subset_log_prob(ref, actions, mask).squeeze(-1))
        ).sum()
        analytic = subset_kl(z, ref, mask).sum()
        torch.testing.assert_close(exact_kl, analytic, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            torch.autograd.grad(exact_kl, z, retain_graph=True)[0],
            torch.autograd.grad(analytic, z)[0],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_sampling_nonempty_mask_and_one_phone_zero_gradient(self):
        seed_all(123)
        mask = torch.tensor([[True, False, True], [False, True, False]])
        logits = torch.full((2, 3), -4.0, dtype=torch.float64, requires_grad=True)
        actions = sample_subsets(logits, mask, 2000)
        self.assertTrue(actions.any(-1).all())
        self.assertFalse((actions & ~mask).any())
        # With one available phone the conditioned law has exactly one action.
        logp = subset_log_prob(logits, actions, mask)
        torch.testing.assert_close(
            logp[:, 1], torch.zeros(2000, dtype=torch.float64), rtol=0, atol=1e-12
        )
        gradient = torch.autograd.grad(logp[:, 1].sum(), logits)[0]
        torch.testing.assert_close(
            gradient, torch.zeros_like(gradient), atol=1e-10, rtol=0
        )
        # Exchangeable rare Bernoulli events conditional on nonempty give symmetric marginals.
        self.assertLess(
            abs(
                float(actions[:, 0, 0].float().mean() - actions[:, 0, 2].float().mean())
            ),
            0.06,
        )
        with self.assertRaises(ValueError):
            subset_log_prob(logits, torch.zeros_like(mask), mask)

    def test_normal_score_function_and_no_pathwise_reward_gradient(self):
        arrays, owners, schedule = toy_data()
        f, c, mask, _ = batch(arrays, owners, schedule, 0)
        m = model("weight")
        seed_all(12)
        means = phone_logits(m.actor, f, mask).double()
        law = torch.distributions.Normal(means, m.sigma)
        actions = law.sample((4,))
        value = (law.log_prob(actions) * mask).sum()
        gradient = torch.autograd.grad(value, means)[0]
        torch.testing.assert_close(
            gradient,
            ((actions - means.detach()) / m.sigma.square() * mask).sum(0),
            rtol=1e-12,
            atol=1e-12,
        )
        scores, logp, kl, _ = m.rollouts(f, c, mask, 4)
        self.assertFalse(scores.requires_grad)
        self.assertTrue(logp.requires_grad)
        self.assertAlmostEqual(float(kl.detach()), 0.0, places=12)

    def test_deterministic_subset_initial_equivalence_and_attention_rescore(self):
        arrays, owners, schedule = toy_data()
        f, c, mask, _ = batch(arrays, owners, schedule, 0)
        m = model("subset")
        with torch.no_grad():
            expected = m.reference(f, c, mask)
            actual = m(f, c, mask)
            assert_nested_equal(actual, expected)
            assert_nested_equal(actual, m(f, c, mask))
            m.actor.output.weight.normal_(std=1)
            m.actor.output.bias.fill_(-1)
            chosen = m.selected_mask(f, mask)
            self.assertTrue(chosen.any(-1).all())
            self.assertFalse((chosen & ~mask).any())
            self.assertTrue((chosen != mask).any())
            actual = m(f, c, mask)
            assert_nested_equal(actual, m.reference(f, c, chosen))
        z = torch.tensor([[-3.0, -1.0, -2.0], [0.0, 0.0, 0.0]])
        available = torch.tensor([[True, False, True], [True, False, True]])
        torch.testing.assert_close(
            deterministic_subset(z, available),
            torch.tensor([[False, False, True], [True, False, True]]),
        )

    def test_reward_corpus_grouping_independent_eer_and_joint_baseline(self):
        scores = torch.tensor(
            [
                [0.9, 0.8, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4],
                [0.1, 0.2, 0.9, 0.8, 0.9, 0.8, 0.1, 0.2],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([True, True, False, False] * 2)
        corpora = torch.tensor([0] * 4 + [1] * 4)
        rewards, groups = eer_rewards(scores, labels, corpora)
        torch.testing.assert_close(
            rewards, torch.tensor([[0.0, -0.5], [-1.0, 0.0]], dtype=torch.float64)
        )
        for k in range(2):
            for g in range(2):
                keep = corpora.numpy() == g
                self.assertAlmostEqual(
                    float(-rewards[k, g]),
                    prior.independent_eer(
                        scores[k].numpy()[keep], labels.numpy()[keep]
                    ),
                    places=12,
                )
        logp = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        loss, adv = reinforce(logp, rewards, corpora, groups)
        gradient = torch.autograd.grad(loss, logp)[0]
        torch.testing.assert_close(gradient, -adv[:, corpora] / 4, rtol=0, atol=0)
        loss, _ = reinforce(logp, torch.zeros_like(rewards), corpora, groups)
        torch.testing.assert_close(
            torch.autograd.grad(loss, logp)[0], torch.zeros_like(logp), rtol=0, atol=0
        )

    def test_inference_manual_direct_and_original_scorer_agree(self):
        arrays, _, _ = toy_data()
        a = {k: v.numpy() for k, v in arrays.items()}
        n, q = len(a["enrollment"]), len(a["queries"])
        ids = np.arange(n * q)
        a["masks"] = a["emask"][ids % n] & a["qmask"][ids // n]
        a["components"] = (a["enrollment"][ids % n] * a["queries"][ids // n]).sum(-1)
        for arm in ARMS:
            m = model(arm)
            scorer = Scorer(m, a, ids)
            values = scorer.score()
            np.testing.assert_array_equal(values, scorer.score(direct=True))
            expected = prior.Scorer(m.reference, a, ids).score(tuple(range(36)))
            np.testing.assert_array_equal(values, expected)
            if arm == "subset":
                with torch.no_grad():
                    m.actor.output.weight.normal_()
                    m.actor.output.bias.fill_(-1)
                np.testing.assert_array_equal(scorer.score(), scorer.score(direct=True))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def evaluate(m):
        m.eval()
        return {"jvs": 0.25, "common_voice": 0.5, "jtube": 0.75}

    def test_bce_matches_original_and_rl_reference_is_frozen(self):
        arrays, owners, schedule = toy_data()
        f, c, mask, labels = batch(arrays, owners, schedule, 0)
        groups = torch.zeros(len(labels), dtype=torch.long)
        m = model("bce")
        expected = m.actor.loss(f, c, mask, labels, 0.001)
        actual, _ = objective(m, f, c, mask, labels, configuration(), "bce", groups)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for arm in ("weight", "subset"):
            m = model(arm)
            loss, details = objective(
                m, f, c, mask, labels, configuration(), arm, groups
            )
            loss.backward()
            self.assertTrue(all(p.grad is None for p in m.reference.parameters()))
            self.assertTrue(
                any(
                    p.grad is not None and torch.any(p.grad != 0)
                    for p in m.actor.parameters()
                )
            )
            self.assertGreater(details["nonzero_advantage_fraction"], 0)

    def test_exact_atomic_resume_all_arms_and_frozen_source(self):
        arrays, owners, schedule = toy_data()
        source = copy.deepcopy((arrays, owners, schedule))
        config = configuration()
        query_groups = torch.tensor([0, 0, 1, 1, 2, 2])
        with tempfile.TemporaryDirectory() as temp:
            for arm in ARMS:
                initial = model(arm).state_dict()
                full, partial = (
                    Path(temp) / f"{arm}-full",
                    Path(temp) / f"{arm}-partial",
                )
                args = (arrays, owners, schedule)
                expected = train_loop(
                    config,
                    model(arm),
                    *args,
                    full,
                    "design",
                    arm,
                    self.evaluate,
                    query_groups,
                )
                train_loop(
                    config,
                    model(arm),
                    *args,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                    query_groups,
                    stop_after=4,
                )
                (partial / "history.json").write_text('"stale sidecar"')
                actual = train_loop(
                    config,
                    model(arm),
                    *args,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                    query_groups,
                )
                assert_nested_equal(expected, actual)
                self.assertEqual(actual["best_bundle"]["update"], 0)
                repeated = train_loop(
                    config,
                    model(arm),
                    *args,
                    partial,
                    "design",
                    arm,
                    self.evaluate,
                    query_groups,
                )
                assert_nested_equal(actual, repeated)
                for k, v in initial.items():
                    if k.startswith("reference."):
                        torch.testing.assert_close(
                            actual["model"][k], v, rtol=0, atol=0
                        )
                self.assertTrue(
                    any(
                        not torch.equal(actual["model"][k], v)
                        for k, v in initial.items()
                        if k.startswith("actor.")
                    )
                )
                with self.assertRaises(AssertionError):
                    train_loop(
                        config,
                        model(arm),
                        *args,
                        partial,
                        "changed",
                        arm,
                        self.evaluate,
                        query_groups,
                    )
        assert_nested_equal((arrays, owners), source[:2])
        for k, v in schedule.items():
            np.testing.assert_array_equal(v, source[2][k])


if __name__ == "__main__":
    unittest.main()
