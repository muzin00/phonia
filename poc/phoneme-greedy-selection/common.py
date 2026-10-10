"""Shared frozen data and original encoder primitives for the greedy study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PHASE3 = ROOT / "poc/phoneme-speaker-encoder"
sys.path.insert(0, str(PHASE3))
sys.path.insert(0, str(BASE.parent / "phoneme-training-data-expansion"))
import numpy as np
import torch
import training as expanded
from phase3_data import artifacts, manifest
from phase3_data.input import InputPipeline, collate_segments
from phase3_train import models

write_json = artifacts.write_json
json_sha256 = manifest.json_sha256
sha256_file = manifest.sha256_file
create_encoder = models.create_encoder
masked_mean_std = models.masked_mean_std

spec = importlib.util.spec_from_file_location(
    "greedy_review_shared", BASE.parent / "phoneme-consonant-review/sample_review.py"
)
review_shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_shared)
VOWELS = ("a", "i", "u", "e", "o")
CONFIG = BASE / "config/protocol.json"
read_json, checked = expanded.read_json, expanded.checked


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pin(files, path):
    name, sha = relative(path), sha256_file(path)
    if files.setdefault(name, sha) != sha:
        raise ValueError(f"input changed: {path}")


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def canonical(phone):
    if phone in ("pau", "sp", "silB", "silE"):
        return None
    if phone in ("A", "I", "U", "E", "O"):
        return phone.lower()
    return phone


def ordered(phones):
    phones = set(phones)
    return [*VOWELS, *sorted(phones - set(VOWELS))]


def rank(seed, name):
    return hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()


def pipeline(config):
    checked(ROOT / config["feature_statistics"], config["feature_statistics_sha256"])
    return InputPipeline(
        expanded.BASELINE["input"],
        rms_enabled=False,
        statistics=read_json(ROOT / config["feature_statistics"]),
    )


def pool_features(pipe, samples, identifier):
    pcm = torch.from_numpy(np.asarray(samples, dtype=np.float32).copy())
    item = pipe.prepare(pcm, identifier, mode="center", kind="log_mel")
    values = item["input"][None]
    mask = torch.ones((1, values.shape[-1]), dtype=torch.bool)
    return masked_mean_std(values, mask)[0].numpy()


def model_embeddings(model, features, batch_size=1024):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for first in range(0, len(features), batch_size):
            values = torch.from_numpy(
                np.asarray(features[first : first + batch_size]).copy()
            )
            result = torch.nn.functional.normalize(
                model.projection(values), dim=1, eps=model.l2_epsilon
            )
            if not torch.isfinite(result).all():
                raise ValueError("nonfinite cached embedding")
            chunks.append(result.numpy())
    return np.concatenate(chunks)


def check_cache_equivalence(model, pipe, probes):
    items, features = [], []
    for identifier, samples in probes:
        item = pipe.prepare(torch.from_numpy(samples.copy()), identifier, mode="center")
        items.append(item)
        features.append(pool_features(pipe, samples, identifier))
    batch = collate_segments(items)
    model.eval()
    with torch.inference_mode():
        regular = model(batch["input"], batch["mask"]).numpy()
        cached = model_embeddings(model, np.array(features))
    error = float(np.max(np.abs(regular - cached)))
    if error > 2e-6:
        raise ValueError(f"cache changes encoder arithmetic: {error}")
    return error
