"""Actual batch EER rewards, corpus-wise leave-one-out REINFORCE."""

from common import np, prior, torch


def eer_rewards(scores, labels, corpora):
    values = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(bool)
    groups = torch.unique(corpora, sorted=True)
    rewards = torch.empty((len(values), len(groups)), dtype=torch.float64)
    for j, group in enumerate(groups):
        keep = (corpora == group).numpy()
        if not y[keep].any() or y[keep].all():
            raise ValueError(
                "Each reward corpus needs both genuine and impostor trials"
            )
        for k, row in enumerate(values):
            rewards[k, j] = -prior.eer(row[keep], y[keep])
    return rewards, groups


def reinforce(log_prob, rewards, corpora, groups):
    if len(rewards) < 2:
        raise ValueError("Leave-one-out baseline requires at least two rollouts")
    advantage = (rewards - (rewards.sum(0) - rewards) / (len(rewards) - 1)).detach()
    joint = torch.stack([log_prob[:, corpora == g].sum(-1) for g in groups], -1)
    return -(advantage * joint).mean(), advantage


def objective(model, features, components, mask, labels, config, arm, corpora):
    if arm == "bce":
        loss = model.actor.loss(
            features, components, mask, labels, config["uniform_kl_coefficient"]
        )
        return loss, {"loss": float(loss.detach())}
    scores, logp, kl, size = model.rollouts(
        features, components, mask, config["rollouts"], config["inference_batch_size"]
    )
    rewards, groups = eer_rewards(scores, labels, corpora)
    policy_loss, advantage = reinforce(logp, rewards, corpora, groups)
    loss = policy_loss + config["policy_kl_coefficient"] * kl
    details = {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "policy_kl": float(kl.detach()),
        "reward_macro_eer": float(-rewards.mean()),
        "reward_rollout_std": float(rewards.std(0, unbiased=False).mean()),
        "nonzero_advantage_fraction": float((advantage.abs() > 1e-12).double().mean()),
        "sampled_mean_phones": float(size),
    }
    assert all(np.isfinite(v) for v in details.values())
    return loss, details
