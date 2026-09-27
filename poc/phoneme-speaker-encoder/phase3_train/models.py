"""Config-driven, padding-safe speaker encoders from design version 2.0.0."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _right_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != length:
        raise ValueError("expected bool[B,T] mask matching input length")
    lengths = mask.sum(dim=1)
    if (lengths <= 0).any() or not torch.equal(
        mask, torch.arange(length, device=mask.device)[None] < lengths[:, None]
    ):
        raise ValueError("mask must contain a nonempty right-padded prefix")
    return lengths


def masked_mean_std(
    values: torch.Tensor, mask: torch.Tensor, variance_floor: float = 1e-5
) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] != mask.shape[0]:
        raise ValueError("expected values[B,C,T] and mask[B,T]")
    lengths = _right_mask(mask, values.shape[-1])
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite encoder input")
    weights = mask[:, None, :].to(values.dtype)
    mean = (values * weights).sum(-1) / lengths[:, None]
    variance = ((values - mean[..., None]).square() * weights).sum(-1) / lengths[
        :, None
    ]
    result = torch.cat(
        (mean, torch.sqrt(torch.clamp(variance, min=variance_floor))), dim=1
    )
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite masked pooling")
    return result


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ConvBlock(nn.Module):
    def __init__(self, block: dict, dropout: float, epsilon: float):
        super().__init__()
        self.kernel = block["kernel_size"]
        self.stride = block.get("stride", 1)
        self.padding = block.get("padding", 0)
        self.dilation = block.get("dilation", 1)
        self.conv = nn.Conv1d(
            block["input_channels"],
            block["output_channels"],
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=block.get("groups", 1),
            bias=False,
        )
        self.norm = ChannelLayerNorm(block["output_channels"], epsilon)
        self.dropout = nn.Dropout(dropout)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return (
            lengths + 2 * self.padding - self.dilation * (self.kernel - 1) - 1
        ) // self.stride + 1

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = _right_mask(mask, values.shape[-1])
        values = values.masked_fill(~mask[:, None, :], 0)
        output_lengths = self.output_lengths(lengths)
        if (output_lengths < 1).any():
            raise ValueError("convolution has no valid output frames")
        values = self.dropout(F.gelu(self.norm(self.conv(values))))
        output_mask = (
            torch.arange(values.shape[-1], device=values.device)[None]
            < output_lengths[:, None]
        )
        return values.masked_fill(~output_mask[:, None, :], 0), output_mask


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        candidate: dict,
        *,
        input_kind: str,
        common: dict,
        blocks: list[dict],
        input_channels: int,
        dropout_override: float | None = None,
    ):
        super().__init__()
        self.name = name
        self.input_kind = input_kind
        self.input_channels = input_channels
        self.minimum_length = 2 if input_kind == "log_mel" else 720
        dropout = candidate.get(
            "dropout",
            common.get("dropout", candidate.get("projection", {}).get("dropout", 0.1)),
        )
        if dropout_override is not None:
            if not 0 <= dropout_override < 1:
                raise ValueError("dropout override must be in [0,1)")
            dropout = dropout_override
        epsilon = candidate.get(
            "normalization_epsilon", common.get("normalization_epsilon", 1e-5)
        )
        self.blocks = nn.ModuleList(
            ConvBlock(block, dropout, epsilon) for block in blocks
        )
        output_channels = blocks[-1]["output_channels"] if blocks else input_channels
        hidden = candidate.get(
            "projection_hidden_dimension",
            common.get(
                "projection_hidden_dimension",
                candidate.get("projection", {}).get("hidden_dimension", 256),
            ),
        )
        embedding = common.get("embedding_dimension", 128)
        self.variance_floor = common["pooling_population_variance_floor"]
        self.l2_epsilon = common["output_normalization_epsilon"]
        self.projection = nn.Sequential(
            nn.Linear(output_channels * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
        )
        self.embedding_dimension = embedding

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, *, return_sequence: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.input_kind == "waveform" and values.ndim == 2:
            values = values[:, None, :]
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("input channel count does not match encoder")
        lengths = _right_mask(mask, values.shape[-1])
        if (lengths < self.minimum_length).any() or not torch.isfinite(values).all():
            raise ValueError("invalid encoder input")
        values = values.masked_fill(~mask[:, None, :], 0)
        for block in self.blocks:
            values, mask = block(values, mask)
        pooled = masked_mean_std(values, mask, self.variance_floor)
        projected = self.projection(pooled)
        embedding = F.normalize(projected, dim=1, eps=self.l2_epsilon)
        if not torch.isfinite(embedding).all():
            raise ValueError("nonfinite embedding")
        return (embedding, values, mask) if return_sequence else embedding


def create_encoder(
    name: str, config_dir: Path = CONFIG_DIR, *, dropout_override: float | None = None
) -> SpeakerEncoder:
    config_dir = Path(config_dir)
    mel = json.loads((config_dir / "log-mel-encoders.json").read_text(encoding="utf-8"))
    wave = json.loads(
        (config_dir / "waveform-encoders.json").read_text(encoding="utf-8")
    )
    if mel["design_version"] != "2.0.0" or wave["design_version"] != "2.0.0":
        raise ValueError("unsupported encoder design version")
    for candidate in mel["candidates"]:
        if candidate["id"] == name:
            encoder = SpeakerEncoder(
                name,
                candidate,
                input_kind="log_mel",
                common=mel,
                blocks=candidate.get("blocks", []),
                input_channels=mel["input_channels"],
                dropout_override=dropout_override,
            )
            break
    else:
        for candidate in wave["candidates"]:
            if candidate["id"] == name:
                common = wave["common_model"]
                blocks = [
                    candidate["first_block"],
                    *common["blocks_after_first"],
                    *candidate.get("context_blocks", []),
                ]
                encoder = SpeakerEncoder(
                    name,
                    candidate,
                    input_kind="waveform",
                    common=common,
                    blocks=blocks,
                    input_channels=1,
                    dropout_override=dropout_override,
                )
                break
        else:
            raise ValueError(f"unknown encoder: {name}")
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    if parameter_count != candidate["expected_parameter_count"]:
        raise ValueError(
            f"{name} parameter count {parameter_count} != {candidate['expected_parameter_count']}"
        )
    return encoder
