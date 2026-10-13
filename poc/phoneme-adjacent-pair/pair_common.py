"""Immutable inputs and original acoustic primitives for adjacent-pair trials."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CONFIG = BASE / "config.json"
PHASE3 = BASE.parent / "phoneme-speaker-encoder"
sys.path.insert(0, str(PHASE3))
from phase3_data.input import InputPipeline
from phase3_train.models import masked_mean_std

VOWELS = frozenset("aiueo")
ARMS = ("split", "joined")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


metrics = module("adjacent_metrics", PHASE3 / "phase3_train/metrics.py")
bootstrap = module("adjacent_bootstrap", PHASE3 / "scripts/weighted_bootstrap.py")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        yield from map(json.loads, stream)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def check(files):
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        assert read(path) == value, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def state(directory, phase, **details):
    write(directory / "state.json", {"status": phase, "time": time.time(), **details})
    print(phase, details, flush=True)


def rank(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def corpus(speaker):
    for prefix, name in (
        ("jvs", "jvs"),
        ("cv17_", "common_voice"),
        ("jtube_", "jtube"),
        ("ext_src4vc", "src4vc"),
    ):
        if speaker.startswith(prefix):
            return name
    raise ValueError(speaker)


def sources(config, split):
    source = ROOT / config["source_run"]
    if split == "train":
        report = read(source / "preparation-report.json")
        return {
            "train": {
                "manifest": source / "train-segments.jsonl",
                "features": source / "train-features.f32",
                "count": report["train_feature_count"],
                "inputs": None,
            }
        }
    prior = read(source / "design-freeze.json")["reference_design"]["config"]
    result = {}
    for c in config["corpora"]:
        inp = source / f"acoustic-{config['seeds'][0]}/fusion/{c}-{split}-inputs.json"
        data = read(inp)
        if c == "common_voice":
            path = ROOT / prior["cv_evaluation_run"] / split
            matrix = path / "features.npy"
            manifest = path / "segments.jsonl"
            count = len(np.load(matrix, mmap_mode="r", allow_pickle=False))
        else:
            path = ROOT / prior["source_run" if c == "jvs" else "corpus_run"]
            matrix, manifest = (
                path / f"{split}-features.f32",
                path / f"{split}-segments.jsonl",
            )
            count = (
                data.get("feature_count")
                if c == "jvs"
                else read(path / "preparation-report.json")["splits"][split]["segments"]
            )
        result[c] = {
            "manifest": manifest,
            "features": matrix,
            "count": count,
            "inputs": inp,
        }
    return result


def features(spec):
    path = spec["features"]
    return (
        np.load(path, mmap_mode="r", allow_pickle=False)
        if path.suffix == ".npy"
        else np.memmap(path, dtype="<f4", mode="r", shape=(spec["count"], 128))
    )


def pipeline(config):
    prior = read(ROOT / config["source_run"] / "design-freeze.json")[
        "reference_design"
    ]["config"]
    path = ROOT / prior["feature_statistics"]
    assert sha(path) == prior["feature_statistics_sha256"]
    return InputPipeline(
        read(PHASE3 / "config/baseline-log-mel.json")["input"],
        rms_enabled=False,
        statistics=read(path),
    ), path


def pooled(pipe, pcm, identifier):
    item = pipe.prepare(
        torch.from_numpy(np.asarray(pcm, dtype=np.float32).copy()),
        identifier,
        mode="center",
        kind="log_mel",
    )
    x = item["input"][None]
    return masked_mean_std(x, torch.ones((1, x.shape[-1]), dtype=torch.bool))[0].numpy()


def model_hash(state_dict):
    h = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_torch(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)
