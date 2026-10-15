"""Local, immutable reuse of the acoustic loss and verification backend."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-greedy-selection"))
import learner as acoustic
from phase3_train.losses import AAMSoftmax
from phase3_train.models import create_encoder

__all__ = ["AAMSoftmax", "acoustic", "create_encoder"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


backend = module(
    "segment_fusion", BASE.parent / "phoneme-transformer-fusion/backend.py"
)
metrics = module(
    "segment_metrics", BASE.parent / "phoneme-speaker-encoder/phase3_train/metrics.py"
)
ARMS = ("uniform", "proxy", "eer", "hybrid")
# Legacy imports prepend their own paths; our entrypoints must resolve locally.
sys.path.insert(0, str(BASE))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(paths):
    return {
        str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in sorted(set(paths))
    }


def check(files):
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def tree_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                visit(part)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return digest.hexdigest()


def rank(seed, name):
    return hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()


def seed_for(seed, *parts):
    return int(rank(seed, "/".join(map(str, parts)))[:15], 16)


def runtime():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def clone(value):
    return copy.deepcopy(value)


def corpus(speaker):
    for prefix, name in (
        ("ext_src4vc", "src4vc"),
        ("cv17_", "common_voice"),
        ("jtube_", "jtube"),
        ("jvs", "jvs"),
    ):
        if speaker.startswith(prefix):
            return name
    raise ValueError(f"Unknown corpus: {speaker}")
