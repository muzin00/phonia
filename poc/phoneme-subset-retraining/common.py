"""Reuse immutable subset inference and source embeddings in a new experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import random
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
_spec = importlib.util.spec_from_file_location(
    "retraining_subset_core", BASE.parent / "phoneme-subset-selection/core.py"
)
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)

np, torch = prior.np, prior.torch
read, write, sha, pin, check = (
    prior.read,
    prior.write,
    prior.sha,
    prior.pin,
    prior.check,
)
CONFIG = BASE / "config.json"


def runtime():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
    }


def state(directory, status, **details):
    write(directory / "state.json", {"status": status, "time": time.time(), **details})
    print(status, details, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def model_hash(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def family(config, seed):
    return ROOT / config["source_run"] / f"acoustic-{seed}"


def create_model(config, seed):
    options = read(family(config, seed) / "fusion/design-freeze.json")["config"]
    return prior.backend.Fusion(
        "transformer",
        dimension=options["dimension"],
        layers=options["transformer_layers"],
        heads=options["attention_heads"],
    )


def old_model_path(config, seed, mode):
    return (
        family(config, seed)
        / "continued"
        / f"seed-{config['fusion_seed']}/budget-{config['updates']}/{mode}.pt"
    )


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def load_training(config, seed):
    source = family(config, seed)
    with np.load(source / "fusion/train.npz", allow_pickle=False) as saved:
        arrays = {k: torch.from_numpy(saved[k].copy()) for k in saved.files}
    data = read(source / "fusion/train-inputs.json")
    index = {v: i for i, v in enumerate(data["speakers"])}
    owners = torch.tensor([index[q["speaker_id"]] for q in data["queries"]])
    with np.load(
        source / "continued" / f"schedule-{config['fusion_seed']}.npz",
        allow_pickle=False,
    ) as saved:
        schedule = {k: saved[k].copy() for k in saved.files}
    assert (
        schedule["queries"].shape
        == schedule["negative_claims"].shape
        == (config["updates"], 32)
    )
    with np.load(
        source / "fusion" / f"schedule-{config['fusion_seed']}.npz", allow_pickle=False
    ) as saved:
        for key in schedule:
            np.testing.assert_array_equal(
                schedule[key][: config["prefix_updates"]], saved[key]
            )
    return arrays, owners, schedule


def training_batch(arrays, owners, schedule, update, allowed):
    queries = torch.from_numpy(schedule["queries"][update].astype(np.int64))
    negative = torch.from_numpy(schedule["negative_claims"][update].astype(np.int64))
    genuine = owners[queries]
    assert not torch.any(genuine == negative)
    shared = (
        arrays["qmask"][queries] & arrays["emask"][genuine] & arrays["emask"][negative]
    )
    assert shared[:, :5].all()
    shared = shared & allowed
    q, c = torch.cat((queries, queries)), torch.cat((genuine, negative))
    f, components = prior.backend.pair_features(
        arrays["enrollment"][c],
        arrays["emeta"][c],
        arrays["queries"][q],
        arrays["qmeta"][q],
    )
    mask = arrays["emask"][c] & arrays["qmask"][q] & torch.cat((shared, shared))
    assert torch.equal(mask[: len(queries)], mask[len(queries) :])
    return (
        f,
        components,
        mask,
        torch.cat((torch.ones(len(queries)), torch.zeros(len(queries)))),
    )


def load_validation(config, seed):
    result = {}
    for corpus in config["corpora"]:
        arrays, _data, rows = prior.load_inputs(config, seed, corpus, "validation")
        ids = prior.positions(rows)
        result[corpus] = (
            arrays,
            ids,
            np.array([rows[i]["is_genuine"] for i in ids], bool),
        )
    return result


def validation_eers(model, validation, subset):
    model.eval()
    result = {}
    for corpus, (arrays, ids, labels) in validation.items():
        scorer = prior.Scorer(model, arrays, ids)
        result[corpus] = prior.eer(scorer.score(subset), labels)
    return result
