#!/usr/bin/env python3
"""Validate the FP32 ONNX export against its pinned PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from run_wav2vec2_alignment import (
    DEFAULT_ONNX_MODEL,
    DEFAULT_SOURCE_DIR,
    MODEL_ID,
    MODEL_REVISION,
    PROJECT_DIR,
    load_audio,
    sha256,
)
from transformers import Wav2Vec2ForCTC

DEFAULT_AUDIO = PROJECT_DIR / "data" / "samples" / "jvs001_VOICEACTRESS100_001.wav"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "export-validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX logits for one pilot recording."
    )
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_directory = args.source_directory.expanduser().resolve()
    onnx_model = args.onnx_model.expanduser().resolve()
    audio = args.audio.expanduser().resolve()
    samples, _ = load_audio(audio)

    pytorch_model = Wav2Vec2ForCTC.from_pretrained(
        source_directory, local_files_only=True
    ).eval()
    with torch.no_grad():
        started_at = time.perf_counter()
        pytorch_logits = pytorch_model(
            torch.from_numpy(samples[None, :])
        ).logits.numpy()
        pytorch_duration = time.perf_counter() - started_at

    session = ort.InferenceSession(
        str(onnx_model), providers=["CPUExecutionProvider"]
    )
    started_at = time.perf_counter()
    onnx_logits = session.run(None, {"input_values": samples[None, :]})[0]
    onnx_duration = time.perf_counter() - started_at
    if pytorch_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            f"Logit shapes differ: {pytorch_logits.shape} != {onnx_logits.shape}"
        )

    absolute_difference = np.abs(pytorch_logits - onnx_logits)
    argmax_agreement = float(
        np.mean(pytorch_logits.argmax(axis=-1) == onnx_logits.argmax(axis=-1))
    )
    result = {
        "model_id": MODEL_ID,
        "model_version": MODEL_REVISION,
        "onnx_sha256": sha256(onnx_model),
        "audio": str(audio.relative_to(PROJECT_DIR)),
        "logit_shape": list(pytorch_logits.shape),
        "max_absolute_logit_difference": float(absolute_difference.max()),
        "mean_absolute_logit_difference": float(absolute_difference.mean()),
        "argmax_agreement_rate": argmax_agreement,
        "pytorch_duration_sec": round(pytorch_duration, 6),
        "onnx_duration_sec": round(onnx_duration, 6),
    }
    if argmax_agreement != 1.0:
        raise RuntimeError("PyTorch and ONNX argmax outputs differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
