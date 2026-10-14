"""Select training PAIRS; never change inference phones or label balance."""

from common import np, reused, torch


def actions_for(pools):
    common = pools["common"]
    counts = common.sum(axis=(0, 1))
    return [
        -1,
        *np.flatnonzero(
            (counts > 0) & (counts < common.shape[0] * common.shape[1])
        ).tolist(),
    ]


def make_draws(config, pools):
    rng = np.random.default_rng(config["sampling_seed"])
    return {
        "keys": -np.log(rng.uniform(np.finfo(float).eps, 1, pools["queries"].shape)),
        "actions": rng.uniform(
            size=(
                config["updates"] // config["block_updates"],
                config["teacher_rollouts"],
            )
        ),
    }


def select_indices(config, pools, keys, update, phone):
    groups = pools["corpora"][update]
    size = config["queries_per_update"]
    weights = np.ones(len(groups))
    if phone >= 0:
        weights[pools["common"][update, :, phone]] = config["phone_sampling_multiplier"]
    priority = keys[update] / weights
    chosen = []
    for group in np.unique(groups[:size]):
        quota = int(np.sum(groups[:size] == group))
        candidates = np.flatnonzero(groups == group)
        ranked = candidates[np.argsort(priority[candidates], kind="stable")]
        chosen.extend(ranked[:quota].tolist())
    ids = np.sort(chosen)
    assert len(ids) == size and len(set(ids.tolist())) == size
    for group in np.unique(groups):
        assert np.sum(groups[ids] == group) == np.sum(groups[:size] == group)
    return ids


def selected_batch(config, arrays, owners, pools, keys, update, phone):
    ids = select_indices(config, pools, keys, update, phone)
    schedule = {k: pools[k][update, ids][None] for k in ("queries", "negative_claims")}
    batch = reused.training_batch(
        arrays, owners, schedule, 0, torch.ones(36, dtype=torch.bool)
    )
    return batch, ids


class Teacher(torch.nn.Module):
    def __init__(self, count, exploration):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(count, 3, dtype=torch.float64))
        self.exploration = exploration

    def probabilities(self, context):
        logits = self.theta @ torch.as_tensor(context, dtype=torch.float64)
        return (1 - self.exploration) * logits.softmax(0) + self.exploration / len(
            logits
        )


def choose(probabilities, draw):
    assert 0 <= draw < 1
    cdf = probabilities.detach().numpy().cumsum()
    return min(int(np.searchsorted(cdf, draw, side="right")), len(cdf) - 1)


def teacher_loss(probabilities, actions, rewards, coefficient):
    r = torch.as_tensor(rewards, dtype=torch.float64).detach()
    logp = probabilities.log()
    policy = -(r * logp[actions]).mean()
    kl = (probabilities * (logp + np.log(len(probabilities)))).sum()
    return policy + coefficient * kl
