"""Frozen Transformer inference and deterministic, bounded subset search."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
VOWELS = frozenset("aiueo")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BACKEND_PATH = BASE.parent / "phoneme-transformer-fusion/backend.py"
METRICS_PATH = BASE.parent / "phoneme-speaker-encoder/phase3_train/metrics.py"
BOOTSTRAP_PATH = BASE.parent / "phoneme-speaker-encoder/scripts/weighted_bootstrap.py"
backend = module("subset_backend", BACKEND_PATH)
metrics = module("subset_metrics", METRICS_PATH)
bootstrap = module("subset_bootstrap", BOOTSTRAP_PATH)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def check(files):
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen file changed: {name}")


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing to change fixed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def partition(speakers, seed):
    ordered = sorted(
        speakers,
        key=lambda s: (hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(), s),
    )
    middle = (len(ordered) + 1) // 2
    if len(ordered) - middle < 2:
        raise ValueError("At least two confirmation speakers are required")
    return {"discovery": ordered[:middle], "confirmation": ordered[middle:]}


def positions(rows, speakers=None, normal_only=True):
    allowed = None if speakers is None else set(speakers)
    return np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if row["score"] is not None
            and (not normal_only or row["role"] == "verification")
            and (
                allowed is None
                or (
                    row["speaker_id"] in allowed
                    and row["claimed_speaker_id"] in allowed
                )
            )
        ],
        dtype=np.int64,
    )


def subset_id(subset):
    return "-".join(f"{p:02d}" for p in subset)


def neighbors(subset, phones, swap=False):
    current = set(subset)
    missing = set(range(len(phones))) - current
    candidates = (
        [current - {a} | {b} for a in current for b in missing]
        if swap
        else [current - {a} for a in current] + [current | {b} for b in missing]
    )
    return sorted(
        {tuple(sorted(v)) for v in candidates if any(phones[p] in VOWELS for p in v)}
    )


def search(phones, evaluate, config, on_step=None):
    current = tuple(range(len(phones)))
    score = evaluate([current])[current]
    history = [{"move": "start", "subset": list(current), "objective": score}]
    single_count = swap_count = 0
    while True:
        accepted = False
        for swap in (False, True):
            if (swap and swap_count >= config["swap_move_limit"]) or (
                not swap and single_count >= config["single_move_limit"]
            ):
                continue
            candidates = neighbors(current, phones, swap)
            if not candidates:
                continue
            results = evaluate(candidates)
            best = min(candidates, key=lambda v: (results[v], v))
            if results[best] >= score - config["minimum_improvement"]:
                continue
            removed, added = set(current) - set(best), set(best) - set(current)
            current, score = best, results[best]
            single_count += int(not swap)
            swap_count += int(swap)
            history.append(
                {
                    "move": "swap" if swap else "delete" if removed else "add",
                    "removed": [phones[p] for p in sorted(removed)],
                    "added": [phones[p] for p in sorted(added)],
                    "subset": list(current),
                    "objective": score,
                }
            )
            if on_step:
                on_step(history)
            accepted = True
            break
        if not accepted:
            break
    return current, history


def load_model(config, seed):
    family = ROOT / config["source_run"] / f"acoustic-{seed}"
    options = read(family / "fusion/design-freeze.json")["config"]
    model = backend.Fusion(
        "transformer",
        dimension=options["dimension"],
        layers=options["transformer_layers"],
        heads=options["attention_heads"],
    )
    path = (
        family
        / "continued"
        / f"seed-{config['fusion_seed']}"
        / f"budget-{config['fusion_budget']}"
        / "selected.pt"
    )
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    )
    return model.eval()


def load_inputs(config, seed, corpus, split):
    family = ROOT / config["source_run"] / f"acoustic-{seed}/fusion"
    with np.load(family / f"{corpus}-{split}.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    data = read(family / f"{corpus}-{split}-inputs.json")
    rows = read(family / f"{corpus}-{split}-baseline.json")
    n = len(data["speakers"])
    assert len(rows) == len(data["queries"]) * n
    assert arrays["masks"].shape == arrays["components"].shape == (len(rows), 36)
    for i, row in enumerate(rows):
        query = data["queries"][i // n]
        assert row["query_id"] == query["query_id"]
        assert row["speaker_id"] == query["speaker_id"]
        assert row["claimed_speaker_id"] == data["speakers"][i % n]
        assert row["is_genuine"] == (row["speaker_id"] == row["claimed_speaker_id"])
        assert (row["score"] is not None) == bool(arrays["masks"][i, :5].all())
    # The available-phone mask must not reveal whether a claim is genuine.
    masks = arrays["masks"].reshape(-1, n, 36)
    assert np.all(masks == masks[:, :1])
    return arrays, data, rows


class Scorer:
    """Cache only the independent token projection, never attention or weights.

    Removed query-token states cannot affect retained tokens: they are masked
    as keys in every attention layer. Their outputs receive zero softmax weight.
    Keeping 36 positions also preserves the original numerical operation order.
    """

    def __init__(self, model, arrays, ids, batch_size=128):
        self.model, self.arrays, self.ids = model, arrays, ids
        self.batch_size = batch_size
        self.mask = torch.from_numpy(arrays["masks"][ids].copy())
        self.components = arrays["components"][ids].copy()
        self.state = torch.empty((len(ids), 36, model.phone.embedding_dim))
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                features, _ = self.features(ids[start:end])
                clean = torch.where(self.mask[start:end, :, None], features, 0)
                self.state[start:end] = model.input(clean) + model.phone.weight[None]

    def features(self, ids):
        a = self.arrays
        n = len(a["enrollment"])
        return backend.pair_features(
            torch.from_numpy(a["enrollment"][ids % n]),
            torch.from_numpy(a["emeta"][ids % n]),
            torch.from_numpy(a["queries"][ids // n]),
            torch.from_numpy(a["qmeta"][ids // n]),
        )

    def score(self, subset, direct=False):
        chosen = torch.zeros(36, dtype=torch.bool)
        chosen[list(subset)] = True
        mask = self.mask & chosen
        if not mask.any(dim=1).all():
            raise ValueError("Subset changes the set of scorable trials")
        result = np.empty(len(self.ids), np.float64)
        with torch.inference_mode():
            for start in range(0, len(self.ids), self.batch_size):
                end = start + self.batch_size
                active = mask[start:end]
                if direct:
                    features, components = self.features(self.ids[start:end])
                    _, weights = self.model(features, components, active)
                else:
                    state = self.model.context(
                        self.state[start:end], src_key_padding_mask=~active
                    )
                    logits = (
                        self.model.output(state)
                        .squeeze(-1)
                        .masked_fill(~active, -torch.inf)
                    )
                    weights = torch.softmax(logits, dim=1)
                w = weights.numpy().astype(np.float64)
                assert not np.any(w[~active.numpy()])
                w /= w.sum(axis=1, keepdims=True)
                result[start:end] = (w * self.components[start:end]).sum(axis=1)
        assert np.isfinite(result).all()
        return result


def eer(values, labels):
    result = metrics.roc_eer(values, labels)["eer"]
    if result is None:
        raise ValueError("No genuine or impostor trials")
    return result


def independent_eer(values, labels):
    levels, inverse = np.unique(values, return_inverse=True)
    p = np.bincount(inverse, weights=labels, minlength=len(levels))
    n = np.bincount(inverse, weights=~labels, minlength=len(levels))
    frr, far = np.r_[0, p.cumsum() / p.sum()], np.r_[1, 1 - n.cumsum() / n.sum()]
    return float(np.interp(0, frr - far, far))


def summarize(values, labels, threshold, all_genuine, all_impostor):
    rates = metrics.error_rates(values, labels, threshold)
    return {
        "eer": eer(values, labels),
        "far": rates["false_accepts"] / all_impostor,
        "frr": (rates["false_rejects"] + all_genuine - int(labels.sum())) / all_genuine,
        "coverage": int(labels.sum()) / all_genuine,
        "scored_genuine": int(labels.sum()),
        "scored_impostor": int((~labels).sum()),
        "all_genuine": all_genuine,
        "all_impostor": all_impostor,
        "false_accepts": rates["false_accepts"],
        "false_rejects": rates["false_rejects"] + all_genuine - int(labels.sum()),
        "threshold": metrics.json_threshold(threshold),
    }
