"""Read-only reuse of audited SRC490 data and verification routines."""

from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
REUSE = BASE.parent / "phoneme-subset-retraining/common.py"
spec = importlib.util.spec_from_file_location("eer_retraining_common", REUSE)
reused = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reused)
prior = reused.prior
np, torch = reused.np, reused.torch
read, write, sha, pin, check = (
    reused.read,
    reused.write,
    reused.sha,
    reused.pin,
    reused.check,
)
seed_all, model_hash, atomic_torch = (
    reused.seed_all,
    reused.model_hash,
    reused.atomic_torch,
)
assert_nested_equal, state = reused.assert_nested_equal, reused.state
create_model, family = reused.create_model, reused.family
load_validation, validation_eers = reused.load_validation, reused.validation_eers
CONFIG = BASE / "config.json"
ARMS = ("bce", "soft_eer")
VARIANTS = (
    "initial",
    "bce-selected",
    "soft_eer-selected",
    "bce-final",
    "soft_eer-final",
)
ALL_PHONES = tuple(range(36))


def runtime():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def starting_path(config, seed):
    return (
        family(config, seed)
        / "continued"
        / f"seed-{config['fusion_seed']}"
        / f"budget-{config['source_budget']}/selected.pt"
    )


def model_path(config, run, seed, variant):
    if variant == "initial":
        return starting_path(config, seed)
    arm, mode = variant.split("-")
    return run / f"acoustic-{seed}" / arm / f"{mode}.pt"


def training_data(config, seed):
    full = {**config, "updates": config["source_budget"], "prefix_updates": 4000}
    arrays, owners, original = reused.load_training(full, seed)
    start = config["source_schedule_start"]
    end = start + config["source_batches_per_update"] * config["updates"]
    assert 4000 <= start < end <= config["source_budget"]
    schedule = {
        k: v[start:end].reshape(config["updates"], -1).copy()
        for k, v in original.items()
    }
    assert schedule["queries"].shape == (
        config["updates"],
        32 * config["source_batches_per_update"],
    )
    return arrays, owners, schedule


def batch(arrays, owners, schedule, update):
    return reused.training_batch(
        arrays, owners, schedule, update, torch.ones(36, dtype=torch.bool)
    )
