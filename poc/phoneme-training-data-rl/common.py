"""Read-only reuse of audited SRC490 data and verification routines."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
REUSE = BASE.parent / "phoneme-subset-retraining/common.py"
spec = importlib.util.spec_from_file_location("data_rl_retraining_common", REUSE)
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
load_validation = reused.load_validation
CONFIG = BASE / "config.json"
ARMS = ("uniform", "random", "rl")
VARIANTS = (
    "initial",
    *(f"{arm}-{mode}" for mode in ("selected", "final") for arm in ARMS),
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


def partitions(config, speakers, corpus):
    ordered = sorted(
        speakers,
        key=lambda s: (
            hashlib.sha256(
                f"{config['partition_seed']}:{corpus}:{s}".encode()
            ).hexdigest(),
            s,
        ),
    )
    middle = len(ordered) // 2
    assert middle >= 2
    return {"reward": ordered[:middle], "selection": ordered[middle:]}


def evaluation_inputs(config, run, seed, corpus, split):
    arrays, data, rows = prior.load_inputs(
        config,
        seed,
        corpus,
        "validation" if split in ("reward", "selection") else split,
    )
    if split in ("reward", "selection"):
        allowed = set(read(run / "partitions.json")[corpus][split])
        rows = [
            dict(r, _source_index=i)
            for i, r in enumerate(rows)
            if r["speaker_id"] in allowed and r["claimed_speaker_id"] in allowed
        ]
    else:
        rows = [dict(r, _source_index=i) for i, r in enumerate(rows)]
    return arrays, data, rows


def positions(rows, normal_only=False):
    return np.array(
        [
            r["_source_index"]
            for r in rows
            if r["score"] is not None
            and (not normal_only or r["role"] == "verification")
        ],
        dtype=np.int64,
    )


def evaluation_data(config, run, seed, split):
    result = {}
    for corpus in config["corpora"]:
        arrays, _, rows = evaluation_inputs(config, run, seed, corpus, split)
        keep = [
            r for r in rows if r["score"] is not None and r["role"] == "verification"
        ]
        result[corpus] = (
            arrays,
            np.array([r["_source_index"] for r in keep]),
            np.array([r["is_genuine"] for r in keep], bool),
        )
    return result


def evaluate_eers(model, data):
    return reused.validation_eers(model, data, ALL_PHONES)


def initial_model(config, seed):
    seed_all(config["initialization_seed"])
    return create_model(config, seed)


def model_path(config, run, seed, variant):
    root = run / f"acoustic-{seed}"
    if variant == "initial":
        return root / "initial.pt"
    arm, mode = variant.split("-")
    return root / arm / f"{mode}.pt"


def comparisons():
    return [
        (f"{arm}-{mode}", f"{control}-{mode}")
        for mode in ("selected", "final")
        for arm, control in (("rl", "uniform"), ("rl", "random"), ("random", "uniform"))
    ]


def training_data(config, seed):
    full = {**config, "updates": config["source_budget"], "prefix_updates": 4000}
    arrays, owners, original = reused.load_training(full, seed)
    steps, size = config["updates"], config["queries_per_update"]
    assert size == 32
    count = steps * config["candidate_multiplier"]
    assert count <= config["source_budget"]
    pools = {k: v[:count].reshape(steps, -1).copy() for k, v in original.items()}
    q, neg = pools["queries"], pools["negative_claims"]
    common = (
        arrays["qmask"].numpy()[q]
        & arrays["emask"].numpy()[owners.numpy()[q]]
        & arrays["emask"].numpy()[neg]
    )
    assert common[..., :5].all()
    data = read(family(config, seed) / "fusion/train-inputs.json")
    groups = np.array(
        [config["training_corpora"].index(q["corpus"]) for q in data["queries"]]
    )
    return arrays, owners, {**pools, "common": common, "corpora": groups[q]}
