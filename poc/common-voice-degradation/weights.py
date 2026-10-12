"""Frozen-model weight interventions and descriptive validation stability."""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-data-budget"))
import analyze as a
import data_budget as d

OUT = a.ROOT / "artifacts/common-voice-degradation/weights-20261012-v1"
VARIANTS = ("u16000-selected", "u16000-final", "u64000-selected", "u64000-final")
PAIRS = ((0, 2), (1, 3), (2, 3))
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)


def error(scores, genuine, mask=None):
    if mask is not None:
        scores, genuine = scores[mask], genuine[mask]
    assert np.isfinite(scores).all()
    return a.roc_eer(scores, genuine)["eer"] * 100


def infer(model, arrays, rows):
    active = np.flatnonzero([r["score"] is not None for r in rows])
    weights = np.zeros_like(arrays["components"])
    n = len(arrays["enrollment"])
    with d.s.torch.inference_mode():
        for start in range(0, len(active), 128):
            ids = active[start : start + 128]
            features, components, mask = d.e.batch(
                arrays,
                d.s.torch.from_numpy(ids // n),
                d.s.torch.from_numpy(ids % n),
                d.s.torch.from_numpy(arrays["masks"][ids]),
            )
            _, w = model(features, components, mask)
            w = w.numpy().astype(np.float64)
            w /= w.sum(axis=1, keepdims=True)
            weights[ids] = w
    assert (weights[~arrays["masks"]] == 0).all()
    np.testing.assert_allclose(weights[active].sum(axis=1), 1, atol=1e-12)
    score = (weights * arrays["components"]).sum(axis=1)
    difference = float(
        np.max(np.abs(score[active] - [rows[i]["score"] for i in active]))
    )
    assert difference < 1e-12, difference
    return weights[active], active, difference


def summarize(weights, labels, phones):
    result = {}
    for name, mask in (("genuine", labels), ("impostor", ~labels)):
        w = weights[mask]
        result[name] = {
            "pairs": len(w),
            "effective_phone_count": a.describe(1 / np.square(w).sum(axis=1)),
            "maximum_phone_weight": a.describe(w.max(axis=1)),
            "vowel_mass": a.describe(w[:, :5].sum(axis=1)),
            "phone_mean_weight": dict(
                zip(phones, w.mean(axis=0).tolist(), strict=True)
            ),
        }
    return result


def stability(scores, labels, rows):
    speakers = sorted({r["speaker_id"] for r in rows})
    assert len(speakers) == 30
    owner = np.array([speakers.index(r["speaker_id"]) for r in rows])
    claim = np.array([speakers.index(r["claimed_speaker_id"]) for r in rows])
    rng = np.random.default_rng(20261012)
    halves = []
    for _ in range(200):
        order = rng.permutation(30)
        pair = []
        for ids in (order[:15], order[15:]):
            mask = np.isin(owner, ids) & np.isin(claim, ids)
            assert labels[mask].any() and (~labels[mask]).any()
            pair.append([error(s, labels, mask) for s in scores])
        assert not set(order[:15]) & set(order[15:])
        halves.append(pair)
    halves = np.array(halves)
    # These checkpoints already used the full validation set. This is sensitivity,
    # not independent selection/confirmation or a new estimate of generalization.
    result = {}
    for before, after in PAIRS:
        delta = halves[:, :, after] - halves[:, :, before]
        result[f"{VARIANTS[before]}_to_{VARIANTS[after]}"] = {
            "full_validation_delta_pp": error(scores[after], labels)
            - error(scores[before], labels),
            "half_delta_pp": a.describe(delta.ravel()),
            "after_better_halves": int((delta < -1e-10).sum()),
            "tied_halves": int((abs(delta) <= 1e-10).sum()),
            "opposite_direction_partition_pairs": int(
                ((delta[:, 0] * delta[:, 1]) < -1e-20).sum()
            ),
            "partition_pairs": 200,
        }
    return result


def main():
    d.s.torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    results = a.read(a.RUN / "results.json")
    frozen = a.read(a.RUN / "design-freeze.json")
    phones = frozen["phones"]
    assert phones[:5] == ["a", "i", "u", "e", "o"]
    # Freeze the exploratory procedure before new interventions. Prior test
    # observations informed this diagnosis, so this is not a confirmatory study.
    code = [
        Path(__file__),
        BASE / "analyze.py",
        d.BASE / "data_budget.py",
        d.e.BASE / "experiment.py",
        d.e.BASE / "inputs.py",
        d.e.BACKEND
        if hasattr(d.e, "BACKEND")
        else BASE.parent / "phoneme-transformer-fusion/backend.py",
        BASE.parent / "phoneme-speaker-encoder/phase3_train/metrics.py",
    ]
    for path in code:
        a.pin(path)
    protocol = {
        "scope": "combined420, all three acoustic seeds, both encoder budgets, four stored Transformer checkpoints; CV validation and test",
        "uniform_mix_fraction": MIXES,
        "validation_partitions": {
            "seed": 20261012,
            "repetitions": 200,
            "speakers_per_half": 15,
            "gallery_and_queries_disjoint_between_halves": True,
        },
        "limitations": "Previously observed test; interventions are descriptive, no model promotion. Stored checkpoints were selected on the full validation set; partition analysis is not independent cross-validation. Phone weight is not attention or causal importance. RMS is not SNR.",
        "files": dict(a.FILES),
    }
    d.fixed(OUT / "protocol.json", protocol)
    output, total, maximum = {}, 0, 0.0
    for key, source in results["sources"].items():
        if source["arm"] != "combined420":
            continue
        original = a.read(a.ROOT / source["fusion"] / "design-freeze.json")["config"]
        paths = d.model_paths(results["config"], source)
        models = {}
        for variant, path in paths.items():
            a.pin(path)
            models[variant] = d.e.create_model(original, "transformer")
            models[variant].load_state_dict(
                d.s.torch.load(path, weights_only=True)["state_dict"]
            )
            models[variant].eval()
        output[key] = {}
        for split in ("validation", "test"):
            path = a.ROOT / source["fusion"] / f"common_voice-{split}.npz"
            a.pin(path)
            arrays = d.e.load_arrays(path)
            data = a.read(
                a.ROOT / source["fusion"] / f"common_voice-{split}-inputs.json"
            )
            values, scores, weight_cache = {}, [], []
            for variant in VARIANTS:
                rows = a.read(
                    a.RUN / "scores" / f"common_voice-{key}-{variant}-{split}.json"
                )
                weights, active, diff = infer(models[variant], arrays, rows)
                maximum = max(maximum, diff)
                total += len(active)
                selected_rows = [rows[i] for i in active]
                labels = np.array([r["is_genuine"] for r in selected_rows])
                assert all(
                    rows[i]["query_id"]
                    == data["queries"][i // len(data["speakers"])]["query_id"]
                    and rows[i]["claimed_speaker_id"]
                    == data["speakers"][i % len(data["speakers"])]
                    for i in active
                )
                mask = arrays["masks"][active]
                uniform = mask / mask.sum(axis=1, keepdims=True)
                components = arrays["components"][active]
                intervention = {}
                for fraction in MIXES:
                    w = (1 - fraction) * weights + fraction * uniform
                    new_scores = (w * components).sum(axis=1)
                    intervention[str(fraction)] = error(new_scores, labels)
                equal_rows = a.read(
                    a.RUN / "scores" / f"common_voice-{key}-equal-{split}.json"
                )
                np.testing.assert_allclose(
                    (uniform * components).sum(axis=1),
                    [equal_rows[i]["score"] for i in active],
                    atol=1e-12,
                    rtol=0,
                )
                values[variant] = {
                    "weights": summarize(weights, labels, phones),
                    "uniform_mix_eer_percent": intervention,
                }
                scores.append((weights * components).sum(axis=1))
                weight_cache.append(weights)
                print(key, split, variant, "verified", flush=True)
            if split == "validation":
                values["speaker_partition_stability"] = stability(
                    scores, labels, selected_rows
                )
            # Save per-pair weights for deeper diagnosis without new inference.
            np.savez_compressed(
                OUT / f"{key}-{split}-weights.npz",
                active=active,
                **dict(zip(VARIANTS, weight_cache, strict=True)),
            )
            a.pin(OUT / f"{key}-{split}-weights.npz")
            output[key][split] = values
            (OUT / "progress.json").write_text(
                json.dumps({"last_completed": [key, split], "verified_scores": total})
                + "\n"
            )
    audited = a.read(a.RUN / "complete-audit.json")["files"]
    model_pins = a.read(a.RUN / "all-models-freeze.json")["files"]
    for path, digest in a.FILES.items():
        assert d.sha(a.ROOT / path) == digest
        if path in audited:
            assert digest == audited[path]
        if path in model_pins:
            assert digest == model_pins[path]
    result = {
        "protocol": protocol,
        "conditions": output,
        "audit": {
            "verified_scores": total,
            "max_score_difference": maximum,
            "files": a.FILES,
        },
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("complete", total, maximum, flush=True)


if __name__ == "__main__":
    main()
