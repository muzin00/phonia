"""Train-only exploration calibration, before any learning or validation selection."""

from common import batch, np, query_corpora, read, seed_all, torch, write
from losses import objective
from policies import build, phone_logits


def calibrate(config, seed, arrays, owners, schedule, path):
    if path.exists():
        return read(path)
    f, c, mask, labels = batch(arrays, owners, schedule, 0)
    groups = query_corpora(config, seed)[schedule["queries"][0]]
    groups = torch.cat((groups, groups))
    base = build(config, seed, "bce")
    with torch.no_grad():
        logits = phone_logits(base.actor, f, mask).masked_fill(~mask, -torch.inf)
        top = logits.topk(2, dim=-1).values
        gaps = (top[:, 0] - top[:, 1]).double().numpy()
        sigma = float(
            np.clip(np.median(gaps), config["sigma_min"], config["sigma_max"])
        )
        before = base(f, c, mask)
    result = {
        "seed": seed,
        "sigma": sigma,
        "top_two_gap_quantiles": np.quantile(gaps, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "calibration_input": "first replay TRAIN batch, no validation/test/reward search",
        "trial_counts_by_corpus": {
            name: int((groups == i).sum())
            for i, name in enumerate(config["reward_corpora"])
        },
        "arms": {},
    }
    for arm in ("weight", "subset"):
        model = build(config, seed, arm, sigma)
        with torch.no_grad():
            actual = model(f, c, mask)
            for a, b in zip(actual, before, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        seed_all(config["finetuning_seed"])
        loss, details = objective(model, f, c, mask, labels, config, arm, groups)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        assert norm > 0 and details["nonzero_advantage_fraction"] > 0
        assert all(p.grad is None for p in model.reference.parameters())
        result["arms"][arm] = {
            **details,
            "gradient_norm": float(norm),
            "initial_deterministic_scores_identical": True,
        }
    write(path, result, immutable=True)
    return result
