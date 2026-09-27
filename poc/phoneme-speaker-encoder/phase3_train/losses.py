"""AAM-Softmax and within-vowel supervised contrastive loss."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AAMSoftmax(nn.Module):
    def __init__(
        self,
        speaker_count: int,
        embedding_dimension: int = 128,
        scale: float = 30.0,
        margin: float = 0.2,
    ):
        super().__init__()
        if speaker_count < 2 or scale <= 0 or not 0 <= margin < math.pi / 2:
            raise ValueError("invalid AAM-Softmax settings")
        self.weight = nn.Parameter(torch.empty(speaker_count, embedding_dimension))
        nn.init.xavier_normal_(self.weight)
        self.scale, self.margin = scale, margin

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] != self.weight.shape[1]
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("invalid AAM embeddings")
        return (
            F.normalize(embeddings, dim=1, eps=1e-12)
            @ F.normalize(self.weight, dim=1, eps=1e-12).T
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self.cosine(embeddings)
        if (
            labels.shape != (embeddings.shape[0],)
            or labels.dtype != torch.long
            or (labels < 0).any()
            or (labels >= self.weight.shape[0]).any()
        ):
            raise ValueError("invalid AAM labels")
        target = cosine.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
        margin_cosine = torch.cos(torch.acos(target) + self.margin)
        logits = cosine.scatter(1, labels[:, None], margin_cosine) * self.scale
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite AAM loss")
        return loss


def within_vowel_supcon(
    embeddings: torch.Tensor,
    speaker_ids: Sequence[str] | torch.Tensor,
    *,
    temperature: float = 0.07,
    vowels: Sequence[str] | None = None,
) -> torch.Tensor:
    if (
        temperature <= 0
        or embeddings.ndim != 2
        or embeddings.shape[0] < 2
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("invalid SupCon input")
    if vowels is not None and (
        len(vowels) != embeddings.shape[0] or len(set(vowels)) != 1
    ):
        raise ValueError("SupCon batch must contain one complete vowel group")
    if isinstance(speaker_ids, torch.Tensor):
        labels = speaker_ids.to(embeddings.device)
    else:
        if len(speaker_ids) != embeddings.shape[0]:
            raise ValueError("speaker labels do not match embeddings")
        lookup = {
            speaker: index for index, speaker in enumerate(sorted(set(speaker_ids)))
        }
        labels = torch.tensor(
            [lookup[speaker] for speaker in speaker_ids], device=embeddings.device
        )
    if labels.shape != (embeddings.shape[0],):
        raise ValueError("speaker labels do not match embeddings")
    eye = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    if not positive.any(dim=1).all():
        raise ValueError("empty SupCon positive set")
    logits = (
        F.normalize(embeddings, dim=1, eps=1e-12)
        @ F.normalize(embeddings, dim=1, eps=1e-12).T
    ) / temperature
    logits = logits.masked_fill(eye, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    log_probability = logits - log_denominator[:, None]
    loss = -torch.where(positive, log_probability, 0).sum(dim=1) / positive.sum(dim=1)
    result = loss.mean()
    if not torch.isfinite(result):
        raise ValueError("nonfinite SupCon loss")
    return result
