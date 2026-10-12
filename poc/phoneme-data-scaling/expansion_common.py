"""Namespaced I/O helpers for the next immutable data expansion run."""

import hashlib
import json
import os
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CONFIG = BASE / "expansion-config.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def fixed(path, value):
    if path.exists():
        assert read(path) == value, f"Frozen output changed: {path}"
    else:
        atomic(path, value)


def pin(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def check_files(files):
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name


def verify(run):
    frozen = read(run / "preparation-freeze.json")
    check_files(frozen["files"])
    return frozen


def progress(run, stage, processed, total, counts, started):
    state = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "counts": dict(counts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "updated_at_epoch": time.time(),
        "pid": os.getpid(),
    }
    atomic(run / f"{stage}-progress.json", state)
    if processed % 25 == 0 or processed == total:
        print(json.dumps(state), flush=True)
