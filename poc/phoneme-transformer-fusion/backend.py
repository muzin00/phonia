"""Explicit per-phone contribution weights, with a frozen acoustic encoder."""

from __future__ import annotations

import math

import torch
from torch import nn

FEATURE_DIM = 520


class Fusion(nn.Module):
    def __init__(self, kind, phone_count=36, dimension=64, layers=2, heads=4):
        super().__init__()
        self.kind = kind
        self.input = nn.Sequential(
            nn.Linear(FEATURE_DIM, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        self.phone = nn.Embedding(phone_count, dimension)
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                dimension,
                heads,
                dim_feedforward=2 * dimension,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = nn.TransformerEncoder(
                layer, layers, enable_nested_tensor=False
            )
            # TransformerEncoder clones its layer: independently initialize each copy.
            for block in self.context.layers:
                for parameter in block.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        elif kind == "mlp":
            self.context = nn.Sequential(
                nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.1)
            )
        else:
            raise ValueError("unknown learned fusion")
        self.output = nn.Linear(dimension, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.bias = nn.Parameter(torch.tensor(-12.0))

    def forward(self, features, components, mask, phone_ids=None):
        if not mask.any(dim=1).all():
            raise ValueError("at least one shared phone is required")
        phone_ids = (
            torch.arange(mask.shape[1], device=features.device)
            if phone_ids is None
            else phone_ids
        )
        clean = torch.where(mask[..., None], features, 0)
        state = self.input(clean) + self.phone(phone_ids)[None]
        state = (
            self.context(state, src_key_padding_mask=~mask)
            if self.kind == "transformer"
            else self.context(state)
        )
        logits = self.output(state).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        score = (weights * torch.where(mask, components, 0)).sum(dim=1)
        return score, weights

    def loss(self, features, components, mask, genuine, regularization=0.001):
        scores, weights = self(features, components, mask)
        logits = self.log_scale.exp().clamp(1, 100) * scores + self.bias
        classification = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, genuine
        )
        kl = (
            (
                weights
                * (weights.clamp_min(1e-12).log() + mask.sum(dim=1)[:, None].log())
            )
            .sum(dim=1)
            .mean()
        )
        return classification + regularization * kl


def pair_features(enrollment, enrollment_metadata, queries, query_metadata):
    """No speaker ID, corpus ID, label, or evaluation split enters the network."""
    features = torch.cat(
        (
            enrollment,
            queries,
            (enrollment - queries).abs(),
            enrollment * queries,
            enrollment_metadata,
            query_metadata,
        ),
        dim=-1,
    )
    components = (enrollment * queries).sum(dim=-1)
    return features, components
