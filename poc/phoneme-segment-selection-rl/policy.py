"""Shared contextual pair policy with an exact categorical exploration mixture."""

from runtime import torch

FEATURES = 269  # 128 mean + 128 difference + 4 duration/RMS + 9 student/context.


class Teacher(torch.nn.Module):
    def __init__(self, phones=36, hidden=32, exploration=0.2):
        super().__init__()
        self.exploration = exploration
        self.phone = torch.nn.Embedding(phones, 8)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(FEATURES + 8, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def probabilities(self, features, phones, mask):
        # Dummy choice only for padding; padding contributes neither reward gradient nor KL.
        mask = mask.clone()
        mask[~mask.any(-1), 0] = True
        logits = self.network(torch.cat((features, self.phone(phones)), -1)).squeeze(-1)
        learned = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
        uniform = mask / mask.sum(-1, keepdim=True)
        return (1 - self.exploration) * learned + self.exploration * uniform


def sample(probabilities, draws):
    cumulative = probabilities.detach().double().numpy().cumsum(-1)
    cumulative /= cumulative[:, -1:]
    cumulative[:, -1] = 1.0
    return (draws[:, None] >= cumulative).sum(-1)


def uniform_probabilities(mask):
    mask = torch.as_tensor(mask).clone()
    mask[~mask.any(-1), 0] = True
    return mask / mask.sum(-1, keepdim=True)


def statistics(probabilities, choices, mask):
    active = torch.as_tensor(mask).sum(-1) > 1
    if not active.any():
        zero = probabilities.sum() * 0
        return zero, zero, 0
    chosen = probabilities[torch.arange(len(choices)), torch.as_tensor(choices)]
    logp = chosen[active].clamp_min(1e-12).log().sum()
    count = torch.as_tensor(mask).sum(-1).clamp_min(1)
    kl = (
        probabilities * (probabilities.clamp_min(1e-12).log() + count[:, None].log())
    ).sum(-1)
    return logp, kl[active].sum(), int(active.sum())


def teacher_step(
    config, teacher, optimizer, log_gradient, kl_gradient, decisions, reward, scale
):
    """Gradients were collected under one unchanged behavior policy, then detached."""
    if not decisions:
        return {"loss_gradient_norm": 0.0, "reward": reward, "decisions": 0}
    optimizer.zero_grad(set_to_none=True)
    for parameter, policy_grad, regularizer_grad in zip(
        teacher.parameters(), log_gradient, kl_gradient, strict=True
    ):
        parameter.grad = (
            -float(reward) * scale * policy_grad
            + config["teacher_uniform_kl"] * regularizer_grad
        ) / decisions
    norm = torch.nn.utils.clip_grad_norm_(
        teacher.parameters(), config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "loss_gradient_norm": float(norm),
        "reward": float(reward),
        "decisions": decisions,
    }


def gradients(teacher, logp, kl):
    parameters = tuple(teacher.parameters())
    a = torch.autograd.grad(logp, parameters, retain_graph=True)
    b = torch.autograd.grad(kl, parameters)
    return [v.detach() for v in a], [v.detach() for v in b]


def add_gradients(destination, source):
    if destination is None:
        return [v.clone() for v in source]
    for total, value in zip(destination, source, strict=True):
        total.add_(value)
    return destination
