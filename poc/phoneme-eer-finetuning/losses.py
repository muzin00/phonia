"""Sigmoid soft-EER with the exact first derivative of its implicit threshold."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


class _SoftEER(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, genuine, temperature, iterations):
        if scores.ndim != 1 or genuine.shape != scores.shape:
            raise ValueError("Expected equally sized one-dimensional scores and labels")
        if genuine.dtype != torch.bool or not genuine.any() or genuine.all():
            raise ValueError("Both genuine and impostor trials are required")
        if not torch.isfinite(scores).all() or not math.isfinite(temperature):
            raise ValueError("Nonfinite input")
        if temperature <= 0 or iterations < 40:
            raise ValueError(
                "Positive temperature and at least 40 root iterations required"
            )
        x = scores.detach().to(torch.float64)
        positive, negative = x[genuine], x[~genuine]
        # At these bounds FAR-FRR has opposite signs, including tied/all-equal scores.
        lo, hi = x.min() - 40 * temperature, x.max() + 40 * temperature
        for _ in range(iterations):
            threshold = (lo + hi) / 2
            far = torch.sigmoid((negative - threshold) / temperature).mean()
            frr = torch.sigmoid((threshold - positive) / temperature).mean()
            lo, hi = (
                torch.where(far > frr, threshold, lo),
                torch.where(far > frr, hi, threshold),
            )
        threshold = (lo + hi) / 2
        zp = (threshold - positive) / temperature
        zn = (negative - threshold) / temperature
        far, frr = torch.sigmoid(zn).mean(), torch.sigmoid(zp).mean()
        if abs(float(far - frr)) > 1e-9:
            raise ValueError("Soft equal-error threshold failed to converge")
        # log slopes remain stable even when sigmoid(z) rounds to exactly one.
        lp = F.logsigmoid(zp) + F.logsigmoid(-zp) - math.log(temperature)
        ln = F.logsigmoid(zn) + F.logsigmoid(-zn) - math.log(temperature)
        log_b = torch.logsumexp(lp, 0) - math.log(len(positive))
        log_a = torch.logsumexp(ln, 0) - math.log(len(negative))
        denominator = torch.logaddexp(log_a, log_b)
        gradient = torch.empty_like(x)
        # A=FAR, R=FRR. dt/dx follows from A(t,x)-R(t,x)=0.
        # dE/dpositive = -a/(a+b) * sigmoid'(zp)/N_positive/temperature.
        gradient[genuine] = -(lp + log_a - denominator).exp() / len(positive)
        gradient[~genuine] = (ln + log_b - denominator).exp() / len(negative)
        ctx.save_for_backward(gradient.to(scores.dtype))
        return ((far + frr) / 2).to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return grad_output * gradient, None, None, None


def soft_eer(scores, genuine, temperature=0.03, iterations=60):
    """Return a rate in [0,1]; the training threshold is never an inference input."""
    return _SoftEER.apply(scores, genuine.bool(), float(temperature), iterations)


def objective(model, features, components, mask, labels, config, arm):
    if arm not in ("bce", "soft_eer"):
        raise ValueError("Unknown training arm")
    scores, weights = model(features, components, mask)
    logits = model.log_scale.exp().clamp(1, 100) * scores + model.bias
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    kl = (
        (weights * (weights.clamp_min(1e-12).log() + mask.sum(dim=1)[:, None].log()))
        .sum(dim=1)
        .mean()
    )
    # Compute in both arms so diagnostics and random-number consumption are matched.
    eer = soft_eer(
        scores, labels, config["soft_eer_temperature"], config["root_iterations"]
    )
    coefficient = config["soft_eer_coefficient"] if arm == "soft_eer" else 0.0
    loss = bce + config["uniform_kl_coefficient"] * kl + coefficient * eer
    return loss, {
        "bce": float(bce.detach()),
        "kl": float(kl.detach()),
        "soft_eer": float(eer.detach()),
        "loss": float(loss.detach()),
    }
