"""Equal-parameter split/join encoders with paired dropout draws."""

from pair_common import torch
from phase3_train.models import create_encoder
from torch import nn
from torch.nn import functional as F


class PairEncoder(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ("split", "joined")
        self.arm = arm
        original = create_encoder("statistics_mlp")
        self.projection = original.projection
        self.epsilon = original.l2_epsilon

    def forward(self, features):
        assert features.ndim == 3 and features.shape[1:] == (3, 128)
        # Two branches in both arms preserve the exact random tensor shape.
        values = (
            features[:, :2]
            if self.arm == "split"
            else features[:, 2:3].expand(-1, 2, -1)
        )
        hidden = self.projection[1](self.projection[0](values))
        dropout = F.dropout(
            torch.ones(
                (len(values), 1, hidden.shape[-1]),
                device=hidden.device,
                dtype=hidden.dtype,
            ),
            p=self.projection[2].p,
            training=self.training,
        )
        projected = self.projection[3](hidden * dropout)
        individual = F.normalize(projected, dim=-1, eps=self.epsilon)
        return F.normalize(individual.mean(dim=1), dim=-1, eps=self.epsilon)


def supcon(embeddings):
    values = F.normalize(embeddings.reshape(5, 20, 128), dim=-1, eps=1e-12)
    logits = values @ values.transpose(1, 2) / 0.07
    logits = logits.masked_fill(torch.eye(20, dtype=torch.bool)[None], -torch.inf)
    positive = (torch.arange(20) ^ 1)[None, :, None].expand(5, -1, -1)
    return (
        torch.logsumexp(logits, dim=-1) - logits.gather(2, positive).squeeze(-1)
    ).mean()
