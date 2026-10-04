"""Inference-only loading of the Phase 3 selected distribution bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from phase3_data.input import InputPipeline, collate_segments
from phase3_data.manifest import json_sha256, sha256_file
from phase3_train.models import create_encoder

REGISTRATION_CONFIG = Path(__file__).resolve().parents[1] / "config"
PHASE3_CONFIG = REGISTRATION_CONFIG.parents[1] / "phoneme-speaker-encoder/config"
POLICY = json.loads((REGISTRATION_CONFIG / "registration-policy.json").read_text())


class FrozenEncoder:
    def __init__(self, model, pipeline: InputPipeline, identity: dict):
        if model.embedding_dimension != 128 or model.input_kind != "log_mel":
            raise ValueError("registration requires a 128-dimensional log-Mel encoder")
        self.model = model.to(device="cpu", dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.pipeline = pipeline
        self.identity = identity

    @classmethod
    def from_bundle(cls, bundle: Path) -> FrozenEncoder:
        bundle = Path(bundle)
        for name, expected in POLICY["bundle_sha256"].items():
            if sha256_file(bundle / name) != expected:
                raise ValueError(f"selected bundle checksum mismatch: {name}")
        run = json.loads((bundle / "run.json").read_text())
        statistics = json.loads((bundle / "feature-statistics.json").read_text())
        settings = run["settings"]
        if (
            settings["encoder"] != POLICY["selected_encoder"]
            or settings["seed"] != POLICY["selected_seed"]
            or settings["cohort"] != 70
            or settings["rms_enabled"] is not False
            or settings["supcon_enabled"] is not True
            or settings["overfit"] is not False
            or json_sha256(settings) != run["configuration_sha256"]
        ):
            raise ValueError("bundle is not the selected Phase 3 encoder")
        config_hash = json_sha256(
            {
                name: sha256_file(PHASE3_CONFIG / name)
                for name in ("log-mel-encoders.json", "waveform-encoders.json")
            }
        )
        if config_hash != run["encoder_config_sha256"]:
            raise ValueError("encoder configuration differs from the selected bundle")
        baseline = json.loads((PHASE3_CONFIG / "baseline-log-mel.json").read_text())
        pipeline = InputPipeline(
            baseline["input"], rms_enabled=False, statistics=statistics
        )
        # The checksum-pinned checkpoint contains historical optimizer/RNG state.
        # Only the model tensors are loaded; no Trainer or optimizer is constructed.
        checkpoint = torch.load(
            bundle / "best.pt", map_location="cpu", weights_only=False
        )
        expected = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "run_seed": POLICY["selected_seed"],
            "config_sha256": run["configuration_sha256"],
            "manifest_sha256": statistics["manifest_sha256"],
            "feature_statistics_sha256": POLICY["bundle_sha256"][
                "feature-statistics.json"
            ],
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("selected checkpoint metadata mismatch")
        model = create_encoder(POLICY["selected_encoder"])
        model.load_state_dict(checkpoint["model"], strict=True)
        identity = {
            "design_version": "2.0.0",
            "encoder": POLICY["selected_encoder"],
            "seed": POLICY["selected_seed"],
            "checkpoint_sha256": POLICY["bundle_sha256"]["best.pt"],
            "encoder_config_sha256": config_hash,
            "feature_statistics_sha256": POLICY["bundle_sha256"][
                "feature-statistics.json"
            ],
            "preprocessing_sha256": pipeline.preprocessing_sha256,
            "implementation_sha256": json_sha256(
                {
                    name: sha256_file(PHASE3_CONFIG.parent / name)
                    for name in ("phase3_data/input.py", "phase3_train/models.py")
                }
            ),
        }
        return cls(model, pipeline, identity)

    def embed(self, pcm: torch.Tensor, segment_id: str) -> np.ndarray:
        item = self.pipeline.prepare(pcm, segment_id, mode="center")
        batch = collate_segments([item])
        self.model.eval()
        with torch.inference_mode():
            vector = self.model(batch["input"], batch["mask"])[0]
            vector = vector.to(dtype=torch.float64).numpy().copy()
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm < 1e-12:
            raise ValueError(f"invalid embedding: {segment_id}")
        return vector / norm
