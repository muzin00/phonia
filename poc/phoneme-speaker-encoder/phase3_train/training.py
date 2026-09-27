"""Deterministic logical-update training and optimizer-boundary checkpoints."""

from __future__ import annotations

import math
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from phase3_data.input import SegmentDataset, collate_segments
from phase3_data.manifest import json_sha256, seeded_hash
from phase3_data.sampling import BalancedSampler, VowelMicrobatchSampler
from torch.utils.data import DataLoader

from .losses import AAMSoftmax, within_vowel_supcon
from .models import SpeakerEncoder


@dataclass(frozen=True)
class TrainSettings:
    encoder: str
    cohort: int
    rms_enabled: bool
    supcon_enabled: bool
    seed: int
    maximum_updates: int
    warmup_updates: int
    validation_interval: int
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    workers: int = 0
    device: str = "cpu"
    overfit: bool = False

    def __post_init__(self):
        if (
            self.cohort not in (10, 25, 50, 70)
            or self.maximum_updates < 1
            or self.warmup_updates < 0
        ):
            raise ValueError("invalid cohort or update budget")
        if (
            self.warmup_updates >= self.maximum_updates
            or self.workers < 0
            or self.validation_interval < 1
        ):
            raise ValueError("invalid warmup, workers or validation interval")
        if (
            not 0 < self.minimum_learning_rate <= self.learning_rate
            or self.gradient_clip <= 0
        ):
            raise ValueError("invalid learning rate or gradient clip")

    @property
    def sha256(self) -> str:
        return json_sha256(asdict(self))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, run_seed: int) -> None:
    seed = seeded_hash("worker", run_seed, worker_id) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, settings: TrainSettings):
        self.optimizer = optimizer
        self.settings = settings
        self.completed_steps = 0
        self._set_lr()

    def _set_lr(self) -> None:
        update = self.completed_steps
        if self.settings.overfit:
            lr = self.settings.learning_rate
        elif self.settings.warmup_updates and update < self.settings.warmup_updates:
            lr = (
                self.settings.learning_rate
                * (update + 1)
                / self.settings.warmup_updates
            )
        else:
            progress = (update - self.settings.warmup_updates) / (
                self.settings.maximum_updates - self.settings.warmup_updates
            )
            progress = min(max(progress, 0.0), 1.0)
            lr = self.settings.minimum_learning_rate + 0.5 * (
                self.settings.learning_rate - self.settings.minimum_learning_rate
            ) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        self.completed_steps += 1
        self._set_lr()

    def state_dict(self) -> dict:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state: dict) -> None:
        completed = state["completed_steps"]
        if not 0 <= completed <= self.settings.maximum_updates:
            raise ValueError("invalid scheduler state")
        self.completed_steps = completed
        self._set_lr()


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "torch_mps": torch.mps.get_rng_state()
        if torch.backends.mps.is_available()
        else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_mps") is not None:
        if not torch.backends.mps.is_available():
            raise ValueError("checkpoint requires MPS RNG state")
        torch.mps.set_rng_state(state["torch_mps"])


class Trainer:
    def __init__(
        self,
        model: SpeakerEncoder,
        head: AAMSoftmax,
        settings: TrainSettings,
        dataset: SegmentDataset,
        sampler: BalancedSampler | None,
        *,
        manifest_sha256: str,
        feature_statistics_sha256: str | None,
    ):
        if (
            model.name != settings.encoder
            or dataset.kind != model.input_kind
            or dataset.run_seed != settings.seed
        ):
            raise ValueError("encoder and dataset do not match training settings")
        if settings.overfit != (sampler is None):
            raise ValueError("overfit mode requires fixed batches instead of sampler")
        self.model, self.head, self.settings = model, head, settings
        self.dataset, self.sampler = dataset, sampler
        self.manifest_sha256 = manifest_sha256
        self.feature_statistics_sha256 = feature_statistics_sha256
        self.device = torch.device(settings.device)
        self.model.to(self.device)
        self.head.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [*model.parameters(), *head.parameters()],
            lr=settings.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=settings.weight_decay,
        )
        self.scheduler = WarmupCosine(self.optimizer, settings)
        self.update = 0
        self.early_stopping = {"reference_eer": None, "patience": 0}
        self.best = {"eer": None, "update": None}
        self.speaker_to_class = {
            speaker: index
            for index, speaker in enumerate(
                sorted({s.speaker_id for s in dataset.segments})
            )
        }
        if len(self.speaker_to_class) != head.weight.shape[0]:
            raise ValueError("AAM class count differs from training speakers")

    def train_update(self, microbatches: list[dict]) -> dict:
        expected = 4 if self.settings.overfit else 5
        if len(microbatches) != expected:
            raise ValueError(
                f"logical update requires {expected} complete vowel groups"
            )
        self.model.train()
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        correct = 0
        total_examples = 0
        for batch in microbatches:
            vowels = batch["vowels"]
            if (
                len(vowels) != (8 if self.settings.overfit else 20)
                or len(set(vowels)) != 1
            ):
                raise ValueError("microbatch must contain one complete vowel group")
            values = batch["input"].to(self.device)
            mask = batch["mask"].to(self.device)
            speakers = batch["speaker_ids"]
            labels = torch.tensor(
                [self.speaker_to_class[s] for s in speakers],
                device=self.device,
                dtype=torch.long,
            )
            embeddings = self.model(values, mask)
            if not torch.isfinite(embeddings).all():
                raise ValueError("nonfinite training embedding")
            aam = self.head(embeddings, labels)
            loss = aam + (
                0.5 * within_vowel_supcon(embeddings, speakers, vowels=vowels)
                if self.settings.supcon_enabled
                else 0
            )
            (loss / expected).backward()
            total_loss += float(loss.detach()) / expected
            correct += int((self.head.cosine(embeddings).argmax(dim=1) == labels).sum())
            total_examples += len(speakers)
        parameters = [*self.model.parameters(), *self.head.parameters()]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("nonfinite or missing gradient")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, self.settings.gradient_clip, error_if_nonfinite=True
            )
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.update += 1
        return {
            "update": self.update,
            "loss": total_loss,
            "accuracy": correct / total_examples,
            "gradient_norm": gradient_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _loader(self, updates: int) -> DataLoader:
        if self.sampler is None:
            raise ValueError("overfit mode uses fixed microbatches")
        loader_generator = torch.Generator().manual_seed(
            seeded_hash("loader", self.settings.seed, self.update) & ((1 << 63) - 1)
        )
        return DataLoader(
            self.dataset,
            batch_sampler=VowelMicrobatchSampler(
                self.sampler, start_update=self.update, updates=updates
            ),
            collate_fn=collate_segments,
            num_workers=self.settings.workers,
            worker_init_fn=partial(seed_worker, run_seed=self.settings.seed)
            if self.settings.workers
            else None,
            generator=loader_generator,
            **({"prefetch_factor": 2} if self.settings.workers else {}),
        )

    def train_until(
        self,
        target_update: int,
        *,
        on_validation: Callable[[Trainer], float] | None = None,
        checkpoint_dir: Path | None = None,
        on_update: Callable[[dict], None] | None = None,
    ) -> list[dict]:
        if (
            self.settings.overfit
            or not self.update <= target_update <= self.settings.maximum_updates
        ):
            raise ValueError("invalid training target")
        history = []
        group = []
        for batch in self._loader(target_update - self.update):
            group.append(batch)
            if len(group) != 5:
                continue
            record = self.train_update(group)
            group = []
            history.append(record)
            if on_update is not None:
                on_update(record)
            if (
                on_validation is not None
                and self.update % self.settings.validation_interval == 0
            ):
                eer = on_validation(self)
                if not math.isfinite(eer):
                    raise ValueError("nonfinite validation macro EER")
                improved = self.best["eer"] is None or eer < self.best["eer"]
                if improved:
                    self.best = {"eer": eer, "update": self.update}
                if (
                    self.early_stopping["reference_eer"] is None
                    or eer <= self.early_stopping["reference_eer"] - 0.001
                ):
                    self.early_stopping = {"reference_eer": eer, "patience": 0}
                else:
                    self.early_stopping["patience"] += 1
                if improved and checkpoint_dir is not None:
                    self.save_checkpoint(Path(checkpoint_dir) / "best.pt")
                if self.early_stopping["patience"] >= 8:
                    break
            if checkpoint_dir is not None and (
                self.update % self.settings.validation_interval == 0
                or self.update == target_update
            ):
                self.save_checkpoint(Path(checkpoint_dir) / "last.pt")
        return history

    def checkpoint(self) -> dict:
        if any(
            p.grad is not None
            for p in (*self.model.parameters(), *self.head.parameters())
        ):
            raise ValueError("checkpoint may only be saved after zero_grad")
        return {
            "schema_version": 1,
            "design_version": "2.0.0",
            "update": self.update,
            "model": self.model.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "run_seed": self.settings.seed,
            "config_sha256": self.settings.sha256,
            "manifest_sha256": self.manifest_sha256,
            "feature_statistics_sha256": self.feature_statistics_sha256,
            "rng": _rng_state(),
            "sampler": self.sampler.state_at(self.update)
            if self.sampler is not None
            else None,
            "early_stopping": dict(self.early_stopping),
            "best": dict(self.best),
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        os.close(descriptor)
        try:
            torch.save(self.checkpoint(), temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "run_seed": self.settings.seed,
            "config_sha256": self.settings.sha256,
            "manifest_sha256": self.manifest_sha256,
            "feature_statistics_sha256": self.feature_statistics_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"checkpoint {key} mismatch")
        if self.sampler is not None:
            self.sampler.load_state_dict(state["sampler"])
        elif state["sampler"] is not None:
            raise ValueError("overfit checkpoint unexpectedly has sampler state")
        self.model.load_state_dict(state["model"])
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.update = state["update"]
        if self.scheduler.completed_steps != self.update:
            raise ValueError("checkpoint scheduler/update mismatch")
        self.early_stopping = dict(state["early_stopping"])
        self.best = dict(state["best"])
        _restore_rng(state["rng"])
