"""Source-slice reading, waveform preprocessing, log-Mel and right padding."""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import Segment, json_sha256, seeded_hash
from .sampling import SampleRequest


def read_pcm_slice(root: Path, segment: Segment) -> torch.Tensor:
    root = Path(root).resolve()
    source = (root / segment.source_file).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"source outside repository: {segment.source_file}")
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError(f"expected 24 kHz mono signed 16-bit PCM: {source}")
        if segment.end_frame > wav.getnframes():
            raise ValueError(f"segment exceeds source WAV: {segment.segment_id}")
        wav.setpos(segment.start_frame)
        pcm = wav.readframes(segment.length)
    if len(pcm) != segment.length * 2:
        raise ValueError(f"short WAV read: {segment.segment_id}")
    return torch.from_numpy(
        np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    )


def crop_start(
    length: int,
    segment_id: str,
    mode: str,
    run_seed: int = 0,
    logical_update: int = 0,
    maximum: int = 6000,
) -> int:
    if length <= maximum:
        return 0
    if mode == "center":
        return (length - maximum) // 2
    if mode == "random":
        if logical_update < 0:
            raise ValueError("logical_update must be nonnegative")
        return seeded_hash("crop", run_seed, logical_update, segment_id) % (
            length - maximum + 1
        )
    raise ValueError(f"invalid crop mode: {mode}")


def _hz_to_mel(frequencies: torch.Tensor) -> torch.Tensor:
    linear = frequencies / (200.0 / 3.0)
    log = 15.0 + torch.log(frequencies / 1000.0) / (math.log(6.4) / 27.0)
    return torch.where(frequencies >= 1000.0, log, linear)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    linear = mels * (200.0 / 3.0)
    log = 1000.0 * torch.exp((mels - 15.0) * (math.log(6.4) / 27.0))
    return torch.where(mels >= 15.0, log, linear)


def slaney_filterbank(
    sample_rate: int = 24000,
    fft_size: int = 600,
    mel_bins: int = 64,
    min_hz: float = 50,
    max_hz: float = 12000,
) -> torch.Tensor:
    if not 0 <= min_hz < max_hz <= sample_rate / 2:
        raise ValueError("invalid Mel frequency range")
    endpoints = _mel_to_hz(
        torch.linspace(
            _hz_to_mel(torch.tensor(float(min_hz))),
            _hz_to_mel(torch.tensor(float(max_hz))),
            mel_bins + 2,
            dtype=torch.float64,
        )
    )
    fft_frequencies = torch.linspace(
        0, sample_rate / 2, fft_size // 2 + 1, dtype=torch.float64
    )
    lower = (fft_frequencies[None, :] - endpoints[:-2, None]) / (
        endpoints[1:-1, None] - endpoints[:-2, None]
    )
    upper = (endpoints[2:, None] - fft_frequencies[None, :]) / (
        endpoints[2:, None] - endpoints[1:-1, None]
    )
    filters = torch.clamp(torch.minimum(lower, upper), min=0)
    filters *= 2.0 / (endpoints[2:] - endpoints[:-2])[:, None]
    if torch.any(filters.sum(dim=1) <= 0):
        raise ValueError("empty Mel filter")
    return filters.to(torch.float32)


class InputPipeline:
    def __init__(
        self, config: dict, *, rms_enabled: bool, statistics: dict | None = None
    ):
        self.config = config
        self.rms_enabled = rms_enabled
        self.maximum = config["maximum_samples"]
        self.minimum = config["minimum_samples"]
        mel = config["log_mel"]
        if (
            config["sample_rate_hz"] != 24000
            or config["channels"] != 1
            or config["dtype"] != "float32"
            or config["remove_dc"] is not True
            or config["peak_clipping_after_normalization"] is not False
            or self.minimum != 720
            or self.maximum != 6000
            or mel["window_samples"] != 600
            or mel["hop_samples"] != 120
            or mel["fft_size"] != 600
            or mel["mel_bins"] != 64
            or mel["center"] is not False
            or mel["window"] != "hann_periodic"
            or mel["power"] != 2.0
            or mel["log_floor"] != 1e-10
            or mel["mel_scale"] != "slaney"
            or mel["mel_normalization"] != "slaney"
        ):
            raise ValueError("unsupported Phase 3 input config")
        self.mel = mel
        self.window = torch.hann_window(600, periodic=True)
        self.filterbank = slaney_filterbank(
            24000, 600, 64, mel["minimum_frequency_hz"], mel["maximum_frequency_hz"]
        )
        self.statistics = statistics
        self.preprocessing_sha256 = json_sha256(
            {"input": config, "rms_enabled": rms_enabled}
        )
        if statistics is not None:
            contents = {
                key: value for key, value in statistics.items() if key != "sha256"
            }
            if statistics.get("sha256") != json_sha256(contents):
                raise ValueError("feature statistics checksum mismatch")
            if (
                statistics["preprocessing_sha256"] != self.preprocessing_sha256
                or statistics["rms_enabled"] != rms_enabled
            ):
                raise ValueError("feature statistics preprocessing checksum mismatch")
            mean = torch.tensor(statistics["mean"], dtype=torch.float32)
            variance = torch.tensor(
                statistics["population_variance"], dtype=torch.float32
            )
            if (
                mean.shape != (64,)
                or variance.shape != (64,)
                or not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or (variance < 0).any()
            ):
                raise ValueError("invalid feature statistics")
            self.mean = mean[:, None]
            self.std = torch.clamp(
                torch.sqrt(variance),
                min=config["feature_normalization"]["standard_deviation_floor"],
            )[:, None]

    def waveform(
        self,
        pcm: torch.Tensor,
        segment_id: str,
        *,
        mode: str,
        run_seed: int = 0,
        logical_update: int = 0,
    ) -> tuple[torch.Tensor, int]:
        if pcm.ndim != 1 or pcm.numel() < self.minimum or not torch.isfinite(pcm).all():
            raise ValueError(f"invalid waveform: {segment_id}")
        start = crop_start(
            pcm.numel(), segment_id, mode, run_seed, logical_update, self.maximum
        )
        signal = pcm[start : start + self.maximum].to(torch.float32).clone()
        signal -= signal.mean()
        if self.rms_enabled:
            rms = torch.sqrt(torch.mean(signal.square())).item()
            rms = max(rms, self.config["rms_normalization"]["minimum_amplitude"])
            target = 10 ** (self.config["rms_normalization"]["target_dbfs"] / 20)
            gain_db = 20 * math.log10(target / rms)
            bounds = self.config["rms_normalization"]
            gain_db = min(
                max(gain_db, bounds["minimum_gain_db"]), bounds["maximum_gain_db"]
            )
            signal *= 10 ** (gain_db / 20)
        if not torch.isfinite(signal).all():
            raise ValueError(f"nonfinite waveform: {segment_id}")
        return signal, start

    def log_mel(self, signal: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        if (
            signal.ndim != 1
            or signal.numel() < self.minimum
            or not torch.isfinite(signal).all()
        ):
            raise ValueError("invalid signal for log-Mel")
        spectrum = torch.stft(
            signal,
            n_fft=600,
            hop_length=120,
            win_length=600,
            window=self.window,
            center=False,
            return_complex=True,
        )
        mel_power = self.filterbank @ spectrum.abs().square()
        features = torch.log(torch.clamp(mel_power, min=self.mel["log_floor"]))
        if normalize and self.statistics is not None:
            features = (features - self.mean) / self.std
        if not torch.isfinite(features).all() or features.shape[1] < 2:
            raise ValueError("invalid log-Mel features")
        return features

    def prepare(
        self,
        pcm: torch.Tensor,
        segment_id: str,
        *,
        mode: str,
        run_seed: int = 0,
        logical_update: int = 0,
        kind: str = "log_mel",
        normalize: bool = True,
    ) -> dict:
        waveform, start = self.waveform(
            pcm, segment_id, mode=mode, run_seed=run_seed, logical_update=logical_update
        )
        value = (
            self.log_mel(waveform, normalize=normalize)
            if kind == "log_mel"
            else waveform
        )
        if kind not in ("log_mel", "waveform"):
            raise ValueError(f"invalid input kind: {kind}")
        return {
            "segment_id": segment_id,
            "input": value,
            "length": value.shape[-1],
            "crop_start": start,
            "original_length": pcm.numel(),
        }


class SegmentDataset(Dataset):
    def __init__(
        self,
        root: Path,
        segments: list[Segment],
        pipeline: InputPipeline,
        *,
        kind: str = "log_mel",
        run_seed: int = 0,
        mode: str = "center",
    ):
        self.root, self.segments, self.pipeline = Path(root), segments, pipeline
        self.kind, self.run_seed, self.mode = kind, run_seed, mode
        self.by_id = {s.segment_id: s for s in segments}
        if len(self.by_id) != len(segments):
            raise ValueError("duplicate segment IDs")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int | SampleRequest) -> dict:
        segment = (
            self.by_id[index.segment_id]
            if isinstance(index, SampleRequest)
            else self.segments[index]
        )
        update = index.logical_update if isinstance(index, SampleRequest) else 0
        pcm = read_pcm_slice(self.root, segment)
        item = self.pipeline.prepare(
            pcm,
            segment.segment_id,
            mode=self.mode,
            run_seed=self.run_seed,
            logical_update=update,
            kind=self.kind,
        )
        item.update(
            {
                "speaker_id": segment.speaker_id,
                "vowel": segment.vowel,
                "source_file": segment.source_file,
            }
        )
        return item


def collate_segments(items: list[dict], *, padding_value: float = 0.0) -> dict:
    if not items:
        raise ValueError("empty batch")
    shape = items[0]["input"].shape[:-1]
    lengths = torch.tensor([item["length"] for item in items], dtype=torch.int64)
    if (lengths <= 0).any() or any(
        item["input"].shape[:-1] != shape or item["input"].shape[-1] != item["length"]
        for item in items
    ):
        raise ValueError("invalid batch input or empty mask")
    batch = torch.full(
        (len(items), *shape, int(lengths.max())), padding_value, dtype=torch.float32
    )
    mask = torch.arange(batch.shape[-1])[None, :] < lengths[:, None]
    for row, item in enumerate(items):
        if not torch.isfinite(item["input"]).all():
            raise ValueError("nonfinite batch input")
        batch[row, ..., : item["length"]] = item["input"]
    return {
        "input": batch,
        "mask": mask,
        "lengths": lengths,
        "segment_ids": [item["segment_id"] for item in items],
        "speaker_ids": [item.get("speaker_id") for item in items],
        "vowels": [item.get("vowel") for item in items],
        "crop_starts": [item["crop_start"] for item in items],
    }


def load_feature_statistics(
    path: Path,
    *,
    manifest_sha256: str,
    cohort: int,
    speaker_ids_sha256: str,
    segment_ids_sha256: str,
    preprocessing_sha256: str,
) -> dict:
    raw = Path(path).read_bytes()
    statistics = json.loads(raw)
    expected = {
        "manifest_sha256": manifest_sha256,
        "cohort": cohort,
        "speaker_ids_sha256": speaker_ids_sha256,
        "segment_ids_sha256": segment_ids_sha256,
        "preprocessing_sha256": preprocessing_sha256,
    }
    for key, value in expected.items():
        if statistics.get(key) != value:
            raise ValueError(f"feature statistics {key} checksum mismatch")
    if (
        statistics.get("schema_version") != 1
        or statistics.get("design_version") != "2.0.0"
    ):
        raise ValueError("unsupported feature statistics schema")
    if statistics.get("frame_count", 0) <= 0 or statistics.get("segment_count", 0) <= 0:
        raise ValueError("empty feature statistics")
    if "sha256" not in statistics:
        raise ValueError("missing feature statistics checksum")
    if not isinstance(statistics["sha256"], str):
        raise TypeError("feature statistics checksum must be a string")
    checksum = statistics.pop("sha256")
    if json_sha256(statistics) != checksum:
        raise ValueError("feature statistics checksum mismatch")
    statistics["sha256"] = checksum
    return statistics
