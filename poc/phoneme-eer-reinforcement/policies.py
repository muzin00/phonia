"""One-step stochastic policies; exact densities for score-function gradients."""

import copy

from common import create_model, np, prior, starting_path, torch


def phone_logits(model, features, mask):
    if not mask.any(-1).all():
        raise ValueError("Every trial must have an available phone")
    clean = torch.where(mask[..., None], features, 0)
    state = model.input(clean) + model.phone.weight[None]
    state = model.context(state, src_key_padding_mask=~mask)
    return model.output(state).squeeze(-1)


def log_nonempty(logits, mask):
    # log(1 - product(1-p)); double precision avoids cancellation near zero.
    log_empty = (torch.nn.functional.logsigmoid(-logits) * mask).sum(-1)
    return torch.log(-torch.expm1(log_empty))


def subset_log_prob(logits, action, mask):
    if (action & ~mask).any() or not action.any(-1).all():
        raise ValueError("Actions must be nonempty available subsets")
    terms = torch.where(
        action,
        torch.nn.functional.logsigmoid(logits),
        torch.nn.functional.logsigmoid(-logits),
    )
    return (terms * mask).sum(-1) - log_nonempty(logits, mask)


def sample_subsets(logits, mask, count):
    """Exact rejection sampling, never replace empty samples by a different law."""
    probabilities = logits.detach().sigmoid().expand(count, -1, -1)
    available = mask.expand_as(probabilities)
    action = (torch.rand_like(probabilities) < probabilities) & available
    for _ in range(1024):
        empty = ~action.any(-1)
        if not empty.any():
            return action
        action[empty] = (
            torch.rand_like(probabilities[empty]) < probabilities[empty]
        ) & available[empty]
    raise RuntimeError("Nonempty subset rejection sampler exceeded its bound")


def subset_kl(logits, reference, mask):
    logz, logqz = log_nonempty(logits, mask), log_nonempty(reference, mask)
    expected = logits.sigmoid() / logz.exp()[..., None]
    positive = torch.nn.functional.logsigmoid(logits) - torch.nn.functional.logsigmoid(
        reference
    )
    negative = torch.nn.functional.logsigmoid(-logits) - torch.nn.functional.logsigmoid(
        -reference
    )
    return (
        ((expected * positive + (1 - expected) * negative) * mask).sum(-1)
        + logqz
        - logz
    )


def deterministic_subset(logits, mask):
    chosen = (logits >= 0) & mask
    empty = ~chosen.any(-1)
    top = logits.masked_fill(~mask, -torch.inf).argmax(-1)
    chosen[empty, top[empty]] = True
    return chosen


class Policy(torch.nn.Module):
    def __init__(self, base, arm, sigma=1.0, logit_limit=4.0):
        super().__init__()
        if arm not in ("bce", "weight", "subset"):
            raise ValueError(arm)
        self.arm, self.logit_limit = arm, logit_limit
        self.actor = copy.deepcopy(base)
        self.reference = copy.deepcopy(base).requires_grad_(False)
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float64))
        if arm != "bce":
            self.actor.log_scale.requires_grad_(False)
            self.actor.bias.requires_grad_(False)
        if arm == "subset":
            torch.nn.init.zeros_(self.actor.output.weight)
            torch.nn.init.zeros_(self.actor.output.bias)
        self.eval()

    def gate_logits(self, features, mask):
        return (
            phone_logits(self.actor, features, mask)
            .double()
            .clamp(-self.logit_limit, self.logit_limit)
        )

    def selected_mask(self, features, mask):
        if self.arm != "subset":
            return mask
        return deterministic_subset(self.gate_logits(features, mask), mask)

    def forward(self, features, components, mask):
        if self.arm == "subset":
            return self.reference(
                features, components, self.selected_mask(features, mask)
            )
        return self.actor(features, components, mask)

    def rollouts(self, features, components, mask, count, batch_size=128):
        if self.arm == "weight":
            means = phone_logits(self.actor, features, mask).double()
            law = torch.distributions.Normal(means, self.sigma)
            actions = law.sample(
                (count,)
            )  # Detached sample is essential for REINFORCE.
            logp = (law.log_prob(actions) * mask).sum(-1)
            with torch.no_grad():
                weights = actions.masked_fill(~mask, -torch.inf).softmax(-1)
                scores = (weights * torch.where(mask, components, 0)).sum(-1)
                initial = phone_logits(self.reference, features, mask).double()
            kl = (((means - initial) / self.sigma).square() * mask).sum(-1) / 2
            size = mask.sum(-1).double().mean()
        elif self.arm == "subset":
            logits = self.gate_logits(features, mask)
            actions = sample_subsets(logits, mask, count)
            logp = subset_log_prob(logits, actions, mask)
            kl = subset_kl(logits, torch.zeros_like(logits), mask)
            scores = torch.empty((count, len(features)), dtype=torch.float64)
            # No gradients through the reward or frozen scorer.
            with torch.no_grad():
                for k in range(count):
                    for start in range(0, len(features), batch_size):
                        end = start + batch_size
                        chosen = actions[k, start:end]
                        _, weights = self.reference(
                            features[start:end], components[start:end], chosen
                        )
                        w = weights.double()
                        w = w / w.sum(-1, keepdim=True)
                        scores[k, start:end] = (
                            w * torch.where(chosen, components[start:end], 0)
                        ).sum(-1)
            size = actions.sum(-1).double().mean()
        else:
            raise ValueError("BCE is not a stochastic policy arm")
        return scores, logp, kl.mean(), size


def build(config, seed, arm, sigma=1.0):
    base = create_model(config, seed)
    bundle = torch.load(
        starting_path(config, seed), weights_only=True, map_location="cpu"
    )
    base.load_state_dict(bundle["state_dict"])
    return Policy(base, arm, sigma, config["subset_logit_limit"])


class Scorer:
    """Independent manual-layer inference versus the public forward path."""

    def __init__(self, model, arrays, ids, batch_size=128):
        self.model, self.arrays, self.ids, self.batch_size = (
            model,
            arrays,
            ids,
            batch_size,
        )
        self.diagnostics = {}

    def score(self, direct=False):
        model, a = self.model.eval(), self.arrays
        result, sizes = [], []
        with torch.inference_mode():
            for start in range(0, len(self.ids), self.batch_size):
                ids = self.ids[start : start + self.batch_size]
                n = len(a["enrollment"])
                f, c = prior.backend.pair_features(
                    torch.from_numpy(a["enrollment"][ids % n]),
                    torch.from_numpy(a["emeta"][ids % n]),
                    torch.from_numpy(a["queries"][ids // n]),
                    torch.from_numpy(a["qmeta"][ids // n]),
                )
                available = torch.from_numpy(a["masks"][ids].copy())
                active = model.selected_mask(f, available)
                if direct:
                    _, weights = model(f, c, available)
                else:
                    fusion = model.reference if model.arm == "subset" else model.actor
                    logits = phone_logits(fusion, f, active).masked_fill(
                        ~active, -torch.inf
                    )
                    weights = logits.softmax(-1)
                w = weights.numpy().astype(np.float64)
                assert not np.any(w[~active.numpy()])
                assert not (active & ~available).any() and active.any(-1).all()
                w /= w.sum(-1, keepdims=True)
                result.append((w * a["components"][ids]).sum(-1))
                sizes.extend(active.sum(-1).tolist())
        self.diagnostics = {
            "mean_selected_phones": float(np.mean(sizes)),
            "minimum_selected_phones": int(min(sizes)),
            "maximum_selected_phones": int(max(sizes)),
        }
        values = np.concatenate(result)
        assert np.isfinite(values).all()
        return values


def validation_eers(model, validation):
    return {
        corpus: prior.eer(Scorer(model, arrays, ids).score(), labels)
        for corpus, (arrays, ids, labels) in validation.items()
    }
