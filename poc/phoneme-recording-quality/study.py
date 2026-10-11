"""Frozen recording-quality diagnosis; no retraining or untouched-test claims."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
import acoustics as a

spec = importlib.util.spec_from_file_location(
    "quality_holdout", BASE.parent / "phoneme-missing-vowel-holdout/study.py"
)
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
s = h.s
np = s.np
CONFIG = BASE / "config.json"


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def pin(files, path):
    files[rel(path)] = s.sha256_file(path)


def verify(run):
    design = s.read_json(run / "design-freeze.json")
    for path, checksum in design["files"].items():
        s.checked(ROOT / path, checksum)
    return design


def jvs_data(config, split):
    return s.read_json(ROOT / config["jvs_run"] / f"{split}-inputs.json")


def freeze(config, run):
    if run.exists():
        raise ValueError("fresh run required")
    files = {}
    jvs_meta = list(
        s.rows(
            ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
        )
    )
    by_source = {r["source_file"]: r for r in jvs_meta}
    cv_train = list(
        s.rows(ROOT / config["cv_training"] / "additional-utterances.jsonl")
    )
    quality, stress = [], {}
    for corpus, values in (("JVS", jvs_meta), ("CV", cv_train)):
        grouped = defaultdict(list)
        for r in values:
            if r["split"] == "train":
                grouped[r["speaker_id"]].append(r)
        assert len(grouped) == 70
        for speaker, rows in sorted(grouped.items()):
            selected = sorted(
                rows, key=lambda r: h.digest(config["selection_seed"], r["source_file"])
            )[: config["training_clips_per_speaker"]]
            for r in selected:
                quality.append({**r, "corpus": corpus, "role": "training"})
    for split in ("validation", "test"):
        data = jvs_data(config, split)
        segments = list(
            s.rows(ROOT / config["encoder_run"] / f"{split}-segments.jsonl")
        )
        enrollment_sources = {
            segments[i]["source_file"]
            for v in data["profiles"].values()
            for indices in v.values()
            for i in indices
        }
        queries = [q for q in data["queries"] if q["role"] == "verification"]
        assert len(queries) == 750
        assert not enrollment_sources & {q["source_file"] for q in queries}
        for source in sorted(enrollment_sources):
            quality.append({**by_source[source], "corpus": "JVS", "role": "enrollment"})
        for q in queries:
            quality.append(
                {
                    **by_source[q["source_file"]],
                    "corpus": "JVS",
                    "role": "verification",
                    "query_id": q["query_id"],
                }
            )
        chosen = []
        for speaker in data["speakers"]:
            selected = sorted(
                [q for q in queries if q["speaker_id"] == speaker],
                key=lambda q: h.digest(config["selection_seed"], q["query_id"]),
            )[: config["stress_queries_per_speaker"]]
            for q in selected:
                chosen.append({"query": q, "metadata": by_source[q["source_file"]]})
                indices = [i for indices in q["groups"].values() for i in indices]
                for path in {segments[i]["alignment_file"] for i in indices}:
                    pin(files, ROOT / path)
        stress[split] = chosen
        cv_values = list(s.rows(ROOT / config["cv_run"] / split / "utterances.jsonl"))
        quality.extend(
            {**r, "corpus": "CV", "query_id": r["utterance_id"]} for r in cv_values
        )
        for name in (
            "inputs.json",
            "embeddings.npy",
            "candidates.jsonl",
            "utterances.jsonl",
        ):
            pin(files, ROOT / config["cv_run"] / split / name)
        for suffix in ("inputs.json", "embeddings.npy"):
            pin(files, ROOT / config["jvs_run"] / f"{split}-{suffix}")
        pin(files, ROOT / config["encoder_run"] / f"{split}-segments.jsonl")
    for r in quality:
        path = ROOT / r["source_file"]
        s.checked(path, r["source_sha256"])
        pin(files, path)
    model_dir = ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
    _, bundle = s.original.load_model(model_dir)
    old_config = s.read_json(ROOT / config["encoder_run"] / "design-freeze.json")[
        "config"
    ]
    extract = {
        **old_config,
        **{
            k: config[k]
            for k in ("minimum_frames", "maximum_frames", "minimum_rms_dbfs")
        },
    }
    align = h.preparation.alignment
    for path in [
        CONFIG,
        BASE / "README.md",
        BASE / "acoustics.py",
        BASE / "study.py",
        model_dir / "encoder.pt",
        model_dir / "training-summary.json",
        ROOT / old_config["feature_statistics"],
        ROOT / config["missing_run"] / "threshold-freeze.json",
        ROOT / config["cv_run"] / "threshold-freeze.json",
        align.DEFAULT_JULIUS,
        align.DEFAULT_SEGMENTATION_KIT / "models/hmmdefs_monof_mix16_gid.binhmm",
    ]:
        pin(files, path)
    for loaded in tuple(sys.modules.values()):
        filename = getattr(loaded, "__file__", None)
        if (
            filename
            and filename.endswith(".py")
            and Path(filename).resolve().is_relative_to(ROOT / "poc")
        ):
            pin(files, Path(filename))
    run.mkdir(parents=True)
    s.write_json(run / "selection.json", {"quality": quality, "stress": stress})
    pin(files, run / "selection.json")
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "extract_config": extract,
            "phones": bundle["phonemes"],
            "runtime": s.original.expanded.runtime(),
            "status": "frozen_before_quality_measurement_and_waveform_intervention",
        },
    )
    print(
        f"Frozen {len(quality)} quality recordings; stress queries {[len(v) for v in stress.values()]}",
        flush=True,
    )


def measure_quality(config, run):
    verify(run)
    selection = s.read_json(run / "selection.json")["quality"]

    def process(row):
        return {
            k: row[k]
            for k in (
                "corpus",
                "role",
                "split",
                "speaker_id",
                "source_file",
                "source_sha256",
            )
        } | {
            "query_id": row.get("query_id"),
            **a.quality(a.read_wave(ROOT / row["source_file"])),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(process, selection))
    s.write_rows(run / "quality.jsonl", results)
    cv = [
        r
        for r in results
        if r["corpus"] == "CV"
        and r["split"] == "validation"
        and r["role"] == "verification"
    ]
    metrics = (
        "energy_contrast_db",
        "active_rms_dbfs",
        "high_frequency_fraction",
        "active_seconds",
    )
    s.write_json(
        run / "quality-strata-freeze.json",
        {
            "boundaries": {
                m: np.quantile([r[m] for r in cv], [1 / 3, 2 / 3]).tolist()
                for m in metrics
            },
            "source": "CV validation queries only",
            "quality_sha256": s.sha256_file(run / "quality.jsonl"),
        },
    )
    print(
        f"Quality measurements: {len(results)}; strata frozen from {len(cv)} validation queries",
        flush=True,
    )


def profiles(data, embeddings):
    result = {}
    for p in data["universally_registered"]:
        means = np.stack(
            [
                embeddings[data["profiles"][sp][p]].mean(axis=0)
                for sp in data["speakers"]
            ]
        )
        result[p] = means / np.linalg.norm(means, axis=1)[:, None]
    return result


def score(query, data, vectors, segments, registration, mode, split):
    use = data["universally_registered"]
    groups = defaultdict(list)
    for i, r in enumerate(segments):
        if mode == "fixed_qc" and r["rms_dbfs"] < -50:
            continue
        if r["phoneme"] in use:
            groups[r["phoneme"]].append(i)
    components = {
        p: registration[p] @ vectors[indices].mean(axis=0)
        for p, indices in groups.items()
    }
    available = sorted(components)
    result = []
    for c, speaker in enumerate(data["speakers"]):
        values = {p: float(components[p][c]) for p in available}
        result.append(
            {
                "query_id": query["query_id"],
                "speaker_id": query["speaker_id"],
                "claimed_speaker_id": speaker,
                "is_genuine": speaker == query["speaker_id"],
                "role": "verification",
                "split": split,
                "status": "scored" if available else "no_score",
                "score": float(np.mean(list(values.values()))) if available else None,
                "used_phones": available,
                "phone_scores": values,
            }
        )
    return result


def transformed_query(item, condition, directory, config, original_segments):
    query, meta = item["query"], dict(item["metadata"])
    output = directory / meta["utterance_id"]
    saved = output / "prepared.json"
    if saved.exists():
        result = s.read_json(saved)
        for path, digest in result["files"].items():
            s.checked(ROOT / path, digest)
        return result
    x = a.read_wave(ROOT / meta["source_file"])
    y, diagnostics = a.transform(
        x, condition, a.seed_for(config["noise_seed"], query["query_id"])
    )
    path = output / "audio.wav"
    a.write_wave(path, y)
    assert np.array_equal(y, a.read_wave(path))
    meta.update(source_file=rel(path), source_sha256=s.sha256_file(path))
    files = {}
    pin(files, path)
    fixed = []
    for i in sorted(i for indices in query["groups"].values() for i in indices):
        row = dict(original_segments[i])
        clip = y[row["start_frame"] : row["end_frame"]]
        row["rms_dbfs"] = float(a.db(np.mean(clip.astype(np.float64) ** 2)))
        fixed.append(row)
    alignment = h.preparation.alignment
    raw_path = output / "alignment.json"
    failure = None
    try:
        alignment.align_record(
            meta,
            raw_path,
            output / "failure.log",
            alignment.DEFAULT_JULIUS,
            alignment.DEFAULT_SEGMENTATION_KIT
            / "models/hmmdefs_monof_mix16_gid.binhmm",
        )
    except RuntimeError as error:
        failure = str(error)
    if failure is None:
        extracted = h.primitives.extract(raw_path, meta, config, files, Counter())
        realigned = [r for r, _ in extracted if not r["quality_flags"]]
    else:
        realigned = []
    result = {
        "query": query,
        "metadata": meta,
        "fixed": fixed,
        "realigned": realigned,
        "failure": failure,
        "quality": a.quality(y),
        "transform": diagnostics,
        "files": files,
    }
    s.write_json(saved, result)
    return result


def stress(config, run, split):
    design = verify(run)
    if split == "test":
        threshold_freeze = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", threshold_freeze["design_sha256"])
        for path, digest in threshold_freeze["files"].items():
            s.checked(ROOT / path, digest)
    selection = s.read_json(run / "selection.json")["stress"][split]
    data = jvs_data(config, split)
    clean_vectors = np.load(
        ROOT / config["jvs_run"] / f"{split}-embeddings.npy", allow_pickle=False
    )
    original_segments = list(
        s.rows(ROOT / config["encoder_run"] / f"{split}-segments.jsonl")
    )
    registration = profiles(data, clean_vectors)
    pipe = h.primitives.pipeline(design["extract_config"])
    model, _ = s.original.load_model(
        ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
    )
    all_thresholds = {}
    for condition in config["conditions"]:
        directory = run / split / condition
        directory.mkdir(parents=True, exist_ok=True)
        prepared = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [
                pool.submit(
                    transformed_query,
                    item,
                    condition,
                    directory,
                    {**config, **design["extract_config"]},
                    original_segments,
                )
                for item in selection
            ]
            for i, future in enumerate(as_completed(pending), 1):
                prepared.append(future.result())
                if i % 50 == 0:
                    print(
                        f"{split}/{condition} alignment {i}/{len(pending)}", flush=True
                    )
        prepared.sort(key=lambda r: r["query"]["query_id"])
        rows_by_mode = {m: [] for m in config["modes"]}
        diagnostics = []
        for number, result in enumerate(prepared, 1):
            output = directory / result["metadata"]["utterance_id"]
            embedding_path = output / "vectors.npz"
            feature_path = output / "features.npz"
            cache_pin = output / "cache.json"
            if cache_pin.exists():
                for path, checksum in s.read_json(cache_pin).items():
                    s.checked(ROOT / path, checksum)
                with np.load(embedding_path, allow_pickle=False) as saved:
                    vectors = {m: saved[m] for m in ("fixed", "realigned")}
            else:
                y = a.read_wave(ROOT / result["metadata"]["source_file"])
                features = {
                    m: np.asarray(
                        [
                            h.primitives.pool_features(
                                pipe,
                                y[r["start_frame"] : r["end_frame"]],
                                r["segment_id"],
                            )
                            for r in result[m]
                        ],
                        np.float32,
                    )
                    for m in ("fixed", "realigned")
                }
                vectors = {
                    m: s.original.model_embeddings(model, f)
                    if len(f)
                    else np.zeros((0, 128), np.float32)
                    for m, f in features.items()
                }
                np.savez(feature_path, **features)
                np.savez(embedding_path, **vectors)
                s.write_json(
                    cache_pin,
                    {rel(p): s.sha256_file(p) for p in (feature_path, embedding_path)},
                )
            if condition == "clean":
                expected = clean_vectors[[r["cache_index"] for r in result["fixed"]]]
                if not np.allclose(vectors["fixed"], expected, atol=1e-6, rtol=0):
                    raise ValueError("clean waveform embedding replay mismatch")
            for mode in config["modes"]:
                source = "fixed" if mode == "fixed_qc" else mode
                rows_by_mode[mode].extend(
                    score(
                        result["query"],
                        data,
                        vectors[source],
                        result[source],
                        registration,
                        mode,
                        split,
                    )
                )
            original = {r["interval_index"]: r for r in result["fixed"]}
            shifts = [
                abs(r["start_frame"] - original[r["interval_index"]]["start_frame"])
                / 24000
                for r in result["realigned"]
                if r["interval_index"] in original
            ]
            diagnostics.append(
                {
                    "query_id": result["query"]["query_id"],
                    "speaker_id": result["query"]["speaker_id"],
                    "quality": result["quality"],
                    "transform": result["transform"],
                    "alignment_failure": result["failure"],
                    "fixed_intervals": len(result["fixed"]),
                    "realigned_intervals": len(result["realigned"]),
                    "matched_crop_shift_seconds": shifts,
                }
            )
            if number % 50 == 0:
                print(
                    f"{split}/{condition} embeddings {number}/{len(prepared)}",
                    flush=True,
                )
        for mode, values in rows_by_mode.items():
            s.write_rows(directory / f"{mode}-scores.jsonl", values)
            if split == "validation":
                all_thresholds[f"{condition}/{mode}"] = {
                    p["name"]: s.original.thresholds(s.apply_policy(values, p))
                    for p in config["policies"]
                }
        s.write_json(directory / "diagnostics.json", diagnostics)
    verify(run)
    if split == "validation":
        files = {
            rel(p): s.sha256_file(p)
            for p in (run / "validation").rglob("*")
            if p.is_file()
        }
        s.write_json(
            run / "threshold-freeze.json",
            {
                "thresholds": all_thresholds,
                "files": files,
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "status": "frozen_before_test_waveform_processing",
            },
        )
    print(f"Completed {split}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "quality", "validation", "test"))
    args = parser.parse_args()
    config = s.read_json(CONFIG)
    run = ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(config, run)
    elif args.stage == "quality":
        measure_quality(config, run)
    else:
        stress(config, run, args.stage)


if __name__ == "__main__":
    main()
