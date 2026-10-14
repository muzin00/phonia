"""Behavioral checks for paired data selection, REINFORCE and exact restart."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import assert_nested_equal, np, partitions, prior, seed_all, torch
from selection import (
    Teacher,
    actions_for,
    make_draws,
    select_indices,
    selected_batch,
    teacher_loss,
)
from trainer import (
    fork,
    optimizer_for,
    reinforce_block,
    train_block,
    train_loop,
    tree_hash,
)


def make_fixture():
    torch.set_num_threads(1)
    seed_all(31)
    config = {
        "updates": 4,
        "queries_per_update": 4,
        "block_updates": 2,
        "validation_every": 2,
        "sampling_seed": 43,
        "dropout_seed": 63,
        "phone_sampling_multiplier": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "uniform_kl_coefficient": 0.001,
        "gradient_clip": 5,
        "teacher_exploration": 0.2,
        "teacher_learning_rate": 0.03,
        "teacher_uniform_kl": 0.001,
        "teacher_rollouts": 2,
    }
    arrays = {
        "enrollment": torch.nn.functional.normalize(torch.randn(4, 36, 128), dim=-1),
        "queries": torch.nn.functional.normalize(torch.randn(8, 36, 128), dim=-1),
        "emeta": torch.randn(4, 36, 4),
        "qmeta": torch.randn(8, 36, 4),
        "emask": torch.ones(4, 36, dtype=torch.bool),
        "qmask": torch.ones(8, 36, dtype=torch.bool),
    }
    arrays["qmask"][::2, 5] = False
    arrays["qmask"][1::2, 6] = False
    arrays["qmask"][:, 7] = False
    owners = torch.arange(8) % 4
    q = np.tile(np.arange(8), (4, 1))
    pools = {
        "queries": q,
        "negative_claims": (q + 1) % 4,
        "common": arrays["qmask"].numpy()[q],
        "corpora": np.tile([0, 0, 1, 1, 0, 0, 1, 1], (4, 1)),
    }
    model = prior.backend.Fusion("transformer", dimension=8, layers=1, heads=2)
    return config, model, arrays, owners, pools, make_draws(config, pools)


def test_sampler_preserves_corpus_pairs_and_all_input_phones(fixture):
    config, _, arrays, owners, pools, draws = fixture
    assert actions_for(pools) == [-1, 5, 6]
    for phone in actions_for(pools):
        batch, ids = selected_batch(
            config, arrays, owners, pools, draws["keys"], 0, phone
        )
        assert len(np.unique(ids)) == config["queries_per_update"]
        assert np.bincount(pools["corpora"][0, ids]).tolist() == [2, 2]
        f, components, mask, labels = batch
        assert labels.tolist() == [1] * 4 + [0] * 4
        expected = arrays["qmask"][pools["queries"][0, ids]]
        assert torch.equal(mask[:4], expected) and torch.equal(mask[4:], expected)
        # Neither target-only masking nor target removal is allowed.
        assert mask[:, :5].all() and mask[:, 8:].all()
        assert f.shape[0] == components.shape[0] == 8
    uniform = select_indices(config, pools, draws["keys"], 0, -1)
    unweighted = select_indices(
        {**config, "phone_sampling_multiplier": 1}, pools, draws["keys"], 0, 5
    )
    np.testing.assert_array_equal(uniform, unweighted)


def test_emphasis_increases_target_exposure(fixture):
    config, _, _, _, pools, _ = fixture
    rng = np.random.default_rng(51)
    baseline, emphasis = [], []
    for _ in range(1000):
        keys = -np.log(rng.uniform(1e-10, 1, (4, 8)))
        for phone, result in ((-1, baseline), (5, emphasis)):
            ids = select_indices(config, pools, keys, 0, phone)
            result.append(pools["common"][0, ids, 5].sum())
    assert np.mean(emphasis) > np.mean(baseline) + 0.3


def test_reinforce_gradient_matches_exact_expected_reward():
    teacher = Teacher(3, 0.2)
    with torch.no_grad():
        teacher.theta[:, 0] = torch.tensor([-0.4, 0.1, 0.3])
    context, rewards = (
        [1, 0.4, 0.1],
        torch.tensor([-0.02, 0.04, 0.01], dtype=torch.float64),
    )
    p = teacher.probabilities(context)
    exact = torch.autograd.grad(-(p * rewards).sum(), teacher.theta)[0]
    expected = torch.zeros_like(exact)
    for i in range(3):
        p = teacher.probabilities(context)
        gradient = torch.autograd.grad(
            teacher_loss(p, [i], [rewards[i]], 0), teacher.theta
        )[0]
        expected += p[i].detach() * gradient
    torch.testing.assert_close(exact, expected, rtol=1e-12, atol=1e-12)
    assert teacher.probabilities(context).min() >= 0.2 / 3
    zero = Teacher(3, 0.2)
    teacher_loss(zero.probabilities(context), [0, 2], [0, 0], 0.001).backward()
    torch.testing.assert_close(
        zero.theta.grad, torch.zeros_like(zero.theta), rtol=0, atol=1e-15
    )


def test_optimizer_forks_are_independent_and_dropout_is_coupled(fixture):
    config, model, arrays, owners, pools, draws = fixture
    optimizer = optimizer_for(config, model)
    train_block(config, model, optimizer, arrays, owners, pools, draws["keys"], 0, -1)
    parent = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    a, oa = fork(config, model, optimizer)
    b, ob = fork(config, model, optimizer)
    train_block(config, a, oa, arrays, owners, pools, draws["keys"], 2, 5)
    assert_nested_equal((model.state_dict(), optimizer.state_dict()), parent)
    assert_nested_equal((b.state_dict(), ob.state_dict()), parent)
    torch.rand(
        100
    )  # Branch order / unrelated RNG consumption must not change training.
    train_block(config, b, ob, arrays, owners, pools, draws["keys"], 2, 5)
    assert_nested_equal(
        (a.state_dict(), oa.state_dict()), (b.state_dict(), ob.state_dict())
    )
    assert tree_hash(oa.state_dict()) != tree_hash(parent[1])


def first_rollout_is_committed_even_when_reward_is_negative(fixture):
    config, model, arrays, owners, pools, draws = fixture
    teacher = Teacher(3, 0.2)
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.03)
    draws["actions"][0] = [0.4, 0.9]

    def fake_train(config, child, optimizer, arrays, owners, pools, keys, start, phone):
        with torch.no_grad():
            child.bias.fill_({-1: 0.10, 5: 0.12, 6: 0.08}[phone])
        return np.zeros((2, 4), dtype=np.int64), {"loss": 0}

    with patch("trainer.train_block", fake_train):
        child, _, _, _, log = reinforce_block(
            config,
            model,
            optimizer_for(config, model),
            teacher,
            teacher_optimizer,
            [-1, 5, 6],
            draws,
            0,
            arrays,
            owners,
            pools,
            lambda m: {"toy": float(m.bias.detach())},
            {"toy": 0.1},
        )
    assert log["rewards"][0] < 0 < log["rewards"][1]
    assert log["committed_phone"] == 5 and abs(float(child.bias.detach()) - 0.12) < 1e-7
    assert log["teacher_after"] != log["teacher_before"]


def test_resume_is_bitwise_identical_and_selection_ties_keep_initial(
    fixture, tmp_path, arm
):
    config, model, arrays, owners, pools, draws = fixture
    initial = copy.deepcopy(model)

    def run(directory, stop=None):
        return train_loop(
            config,
            copy.deepcopy(initial),
            arrays,
            owners,
            pools,
            draws,
            [-1, 5, 6],
            directory,
            "frozen-test",
            arm,
            lambda m: {"toy": 0.1},
            lambda m: {"toy": float(m.output.weight.detach().square().mean())},
            stop,
        )

    full = run(tmp_path / "full")
    run(tmp_path / "resume", 2)
    torch.rand(100)
    resumed = run(tmp_path / "resume")
    assert_nested_equal(full, resumed)
    assert resumed["best_bundle"]["update"] == 0
    for mode in ("selected", "final"):
        assert_nested_equal(
            torch.load(tmp_path / "full" / f"{mode}.pt", weights_only=True),
            torch.load(tmp_path / "resume" / f"{mode}.pt", weights_only=True),
        )


def test_partitions_are_disjoint_exhaustive_and_order_invariant():
    speakers = [f"speaker-{i}" for i in range(15)]
    a = partitions({"partition_seed": 45}, speakers, "corpus")
    assert a == partitions({"partition_seed": 45}, speakers[::-1], "corpus")
    assert len(a["reward"]) == 7 and len(a["selection"]) == 8
    assert set(a["reward"]).isdisjoint(a["selection"])
    assert set(a["reward"]) | set(a["selection"]) == set(speakers)


class TrainingDataTests(unittest.TestCase):
    def test_sampler(self):
        test_sampler_preserves_corpus_pairs_and_all_input_phones(make_fixture())

    def test_emphasis(self):
        test_emphasis_increases_target_exposure(make_fixture())

    def test_gradient(self):
        test_reinforce_gradient_matches_exact_expected_reward()

    def test_fork_isolation(self):
        test_optimizer_forks_are_independent_and_dropout_is_coupled(make_fixture())

    def test_negative_branch_committed(self):
        first_rollout_is_committed_even_when_reward_is_negative(make_fixture())

    def test_uniform_resume(self):
        self.resume("uniform")

    def test_random_resume(self):
        self.resume("random")

    def test_rl_resume(self):
        self.resume("rl")

    def resume(self, arm):
        with tempfile.TemporaryDirectory() as directory:
            test_resume_is_bitwise_identical_and_selection_ties_keep_initial(
                make_fixture(), Path(directory), arm
            )

    def test_partitions(self):
        test_partitions_are_disjoint_exhaustive_and_order_invariant()


if __name__ == "__main__":
    unittest.main()
