"""Controlled frontend and nested-duration diagnosis on frozen data."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-recording-quality"))
import report as qr

t = qr.t
s, np, a = t.s, t.np, t.a
CONFIG = BASE / "config.json"


def normalize(x, config):
    x = np.asarray(x, np.float64)
    level = a.quality(x)["active_rms_dbfs"]
    if level <= -100:
        return x.astype(np.float32), {"gain_db": 0.0, "limited": True, "silence": True}
    desired = config["target_active_rms_dbfs"] - level
    gain_db = float(
        np.clip(desired, -config["maximum_gain_db"], config["maximum_gain_db"])
    )
    gain = 10 ** (gain_db / 20)
    peak = np.max(np.abs(x))
    gain = min(gain, config["peak_limit"] / peak) if peak else gain
    y = (np.rint(x * gain * 32768) / 32768).astype(np.float32)
    assert np.max(np.abs(y)) <= config["peak_limit"] + 1 / 32768
    return y, {
        "gain_db": float(20 * np.log10(gain)),
        "limited": bool(abs(20 * np.log10(gain) - desired) > 1e-6),
        "silence": False,
    }


def window_indices(query, segments, seconds):
    original = sorted(i for ids in query["groups"].values() for i in ids)
    if not original:
        return [], 0, 0
    first = min(segments[i]["raw_start_frame"] for i in original)
    last = first + round(seconds * a.RATE)
    return (
        [
            i
            for i in original
            if segments[i]["raw_start_frame"] >= first
            and segments[i]["raw_end_frame"] <= last
        ],
        first,
        last,
    )


def verify(config, run):
    design = s.read_json(run / "design-freeze.json")
    assert config == design["config"]
    for name, digest in design["files"].items():
        s.checked(ROOT / name, digest)
    return design


def freeze(config, run):
    if run.exists():
        raise ValueError("fresh run required")
    source = ROOT / config["quality_run"]
    previous = t.verify(source)
    files = dict(previous["files"])
    for path in [
        CONFIG,
        BASE / "README.md",
        BASE / "experiment.py",
        Path(qr.__file__),
        source / "selection.json",
        source / "threshold-freeze.json",
    ]:
        t.pin(files, path)
    # Source perturbations and raw feature caches are immutable inputs to this comparison.
    for split in ("validation", "test"):
        for path in (source / split).rglob("*"):
            if path.is_file():
                t.pin(files, path)
        old = previous["config"]
        for path in [
            ROOT / old["encoder_run"] / f"{split}-features.f32",
            ROOT / old["cv_run"] / split / "features.npy",
            ROOT / old["cv_run"] / split / "segments.jsonl",
        ]:
            t.pin(files, path)
    run.mkdir(parents=True)
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "source_config": previous["config"],
            "extract_config": previous["extract_config"],
            "files": files,
            "status": "fixed_before_normalization_processing",
        },
    )
    print("Frozen normalization and duration comparisons", flush=True)


def normalized_dataset(config, run, split, corpus, data, segments, pipe, model):
    output = run / split / corpus
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "normalized.npz"
    if saved.exists():
        for name, digest in s.read_json(output / "cache.json").items():
            s.checked(ROOT / name, digest)
        return np.load(saved, allow_pickle=False)["embeddings"]
    indices = {
        i
        for phone_map in data["profiles"].values()
        for ids in phone_map.values()
        for i in ids
    }
    indices.update(
        i for q in data["queries"] for ids in q["groups"].values() for i in ids
    )
    by_source = defaultdict(list)
    for i in sorted(indices):
        by_source[segments[i]["source_file"]].append(i)
    features = np.zeros((len(segments), 128), np.float32)
    diagnostics = []
    for number, (source, chosen) in enumerate(sorted(by_source.items()), 1):
        y, info = normalize(a.read_wave(ROOT / source), config)
        for i in chosen:
            row = segments[i]
            features[i] = t.h.primitives.pool_features(
                pipe, y[row["start_frame"] : row["end_frame"]], row["segment_id"]
            )
        diagnostics.append({"source_file": source, **info})
        if number % 200 == 0:
            print(
                f"{split}/{corpus} normalized sources {number}/{len(by_source)}",
                flush=True,
            )
    embeddings = np.zeros((len(segments), 128), np.float32)
    ordered = sorted(indices)
    embeddings[ordered] = s.original.model_embeddings(model, features[ordered])
    np.savez(
        saved, features=features, embeddings=embeddings, indices=np.asarray(ordered)
    )
    s.write_json(output / "normalization.json", diagnostics)
    s.write_json(
        output / "cache.json",
        {t.rel(p): s.sha256_file(p) for p in (saved, output / "normalization.json")},
    )
    return embeddings


def durations(config, run, split, data, segments, vectors):
    old = ROOT / config["quality_run"]
    selection = s.read_json(old / "selection.json")["stress"][split]
    meta = {r["query"]["query_id"]: r["metadata"] for r in selection}
    queries = []
    for q in data["queries"]:
        _, first, last = window_indices(q, segments, max(config["duration_seconds"]))
        if (
            any(q["groups"].values())
            and round(meta[q["query_id"]]["duration_sec"] * a.RATE) >= last
        ):
            queries.append(q)
    selected = {**data, "queries": queries}
    registration = t.profiles(data, vectors)
    for seconds in ["full", *config["duration_seconds"]]:
        rows = []
        details = []
        for q in queries:
            if seconds == "full":
                indices = sorted(i for ids in q["groups"].values() for i in ids)
                first, last = 0, round(meta[q["query_id"]]["duration_sec"] * a.RATE)
            else:
                indices, first, last = window_indices(q, segments, seconds)
            chosen = [segments[i] for i in indices]
            rows.extend(
                t.score(
                    q, selected, vectors[indices], chosen, registration, "fixed", split
                )
            )
            details.append(
                {
                    "query_id": q["query_id"],
                    "first_frame": first,
                    "last_frame": last,
                    "indices": indices,
                }
            )
        s.write_rows(run / split / f"duration-{seconds}-scores.jsonl", rows)
        s.write_json(run / split / f"duration-{seconds}-windows.json", details)
    print(f"{split} duration common cohort: {len(queries)}", flush=True)


def evaluate(config, run, split):
    design = verify(config, run)
    if split == "test":
        frozen = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", frozen["design_sha256"])
        for name, digest in frozen["files"].items():
            s.checked(ROOT / name, digest)
    old = design["source_config"]
    source = ROOT / config["quality_run"]
    selected = s.read_json(source / "selection.json")["stress"][split]
    selected_ids = {r["query"]["query_id"] for r in selected}
    pipe = t.h.primitives.pipeline(design["extract_config"])
    model, _ = s.original.load_model(
        ROOT / old["encoder_run"] / "trials" / old["encoder_trial"]
    )
    points = {}
    for corpus in ("JVS", "CV"):
        if corpus == "JVS":
            data = t.jvs_data(old, split)
            data = {
                **data,
                "queries": [
                    q for q in data["queries"] if q["query_id"] in selected_ids
                ],
            }
            segments = list(
                s.rows(ROOT / old["encoder_run"] / f"{split}-segments.jsonl")
            )
            original = np.load(
                ROOT / old["jvs_run"] / f"{split}-embeddings.npy", allow_pickle=False
            )
        else:
            data = s.read_json(ROOT / old["cv_run"] / split / "inputs.json")
            segments = list(s.rows(ROOT / old["cv_run"] / split / "segments.jsonl"))
            original = np.load(
                ROOT / old["cv_run"] / split / "embeddings.npy", allow_pickle=False
            )
        normalized = normalized_dataset(
            config, run, split, corpus, data, segments, pipe, model
        )
        for arm, vectors in [("original", original), ("normalized", normalized)]:
            values = t.h.candidates(data, vectors)
            s.write_rows(run / split / corpus / f"{arm}-scores.jsonl", values)
            if split == "validation":
                points[f"{corpus}/{arm}"] = {
                    p["name"]: s.original.thresholds(s.apply_policy(values, p))
                    for p in old["policies"]
                }
        if corpus == "JVS":
            registration = t.profiles(data, normalized)
            for condition in old["conditions"]:
                values = []
                dest = run / split / "stress" / condition
                dest.mkdir(parents=True, exist_ok=True)
                for number, path in enumerate(
                    sorted((source / split / condition).glob("*/prepared.json")), 1
                ):
                    prepared = s.read_json(path)
                    cache = dest / f"{prepared['query']['query_id']}.npz"
                    y, _info = normalize(
                        a.read_wave(ROOT / prepared["metadata"]["source_file"]), config
                    )
                    features = np.asarray(
                        [
                            t.h.primitives.pool_features(
                                pipe,
                                y[r["start_frame"] : r["end_frame"]],
                                r["segment_id"],
                            )
                            for r in prepared["fixed"]
                        ],
                        np.float32,
                    )
                    vectors = s.original.model_embeddings(model, features)
                    np.savez(cache, features=features, embeddings=vectors)
                    values.extend(
                        t.score(
                            prepared["query"],
                            data,
                            vectors,
                            prepared["fixed"],
                            registration,
                            "fixed",
                            split,
                        )
                    )
                    if number % 75 == 0:
                        print(
                            f"{split}/{condition} normalized queries {number}/150",
                            flush=True,
                        )
                s.write_rows(dest / "scores.jsonl", values)
            durations(config, run, split, data, segments, original)
    verify(config, run)
    if split == "validation":
        files = {
            t.rel(p): s.sha256_file(p) for p in (run / split).rglob("*") if p.is_file()
        }
        s.write_json(
            run / "threshold-freeze.json",
            {
                "thresholds": points,
                "files": files,
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "status": "frozen_before_test_normalization",
            },
        )
    print(f"Completed {split}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "validation", "test"))
    args = parser.parse_args()
    config = s.read_json(CONFIG)
    run = ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(config, run)
    else:
        evaluate(config, run, args.stage)


if __name__ == "__main__":
    main()
