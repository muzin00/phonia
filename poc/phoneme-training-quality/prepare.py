"""Apply frozen quality rules and materialize a separate, traceable training view."""

import argparse
import gzip
import hashlib
import json
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
from data import subset_bank, subset_pool, validate_bank
from quality import REASONS, STATUS, decide, reason_names, source_quality

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def checked(files):
    for name, digest in files.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen file changed: {name}")


def pin(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def coverage(info, pool, status):
    shape = (len(info["speakers"]), len(info["phones"]))
    key = pool["segment_speaker"].astype(np.int64) * shape[1] + pool["segment_phone"]
    before = np.bincount(key, minlength=np.prod(shape)).reshape(shape)
    after = np.bincount(key[status == 0], minlength=np.prod(shape)).reshape(shape)
    rows = [
        {
            "speaker_id": s,
            "phoneme": p,
            "before": int(before[i, j]),
            "after": int(after[i, j]),
        }
        for i, s in enumerate(info["speakers"])
        for j, p in enumerate(info["phones"])
        if before[i, j] != after[i, j]
    ]
    report = {
        "groups_emptied": int(((before > 0) & (after == 0)).sum()),
        "pair_groups_becoming_singletons": int(((before >= 2) & (after == 1)).sum()),
        "speakers_emptied": [
            s for s, n in zip(info["speakers"], after.sum(1)) if n == 0
        ],
        "per_phone": {
            p: {
                "before": int(before[:, j].sum()),
                "after": int(after[:, j].sum()),
                "speakers_before": int((before[:, j] > 0).sum()),
                "speakers_after": int((after[:, j] > 0).sum()),
                "positive_pair_groups_after": int((after[:, j] >= 2).sum()),
            }
            for j, p in enumerate(info["phones"])
        },
        "changed_speaker_phone_groups": rows,
    }
    return report


def verify_dc_decisions(measures, sources, threshold):
    """Check rejected and near-threshold rows using the actual float32 torch operations."""
    import torch

    rms = measures["dc_removed_rms_dbfs"]
    selected = np.flatnonzero((rms < threshold) | (np.abs(rms - threshold) < 0.01))
    grouped = defaultdict(list)
    for i in selected:
        grouped[int(measures["source"][i])].append(int(i))
    max_error = 0.0
    for sid, ids in grouped.items():
        path = ROOT / sources[sid]["source_file"]
        if sha(path) != sources[sid]["source_sha256"]:
            raise ValueError("DC verification source changed")
        with wave.open(str(path)) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        for i in ids:
            signal = torch.from_numpy(
                pcm[measures["start"][i] : measures["end"][i]].astype(np.float32)
                / 32768
            )
            signal -= signal.mean()
            actual = float(10 * torch.log10(signal.square().mean().clamp_min(1e-20)))
            max_error = max(max_error, abs(actual - rms[i]))
            if (actual < threshold) != (rms[i] < threshold):
                raise ValueError(f"Float32/DC threshold disagreement at {i}")
    return {
        "checked_intervals": len(selected),
        "maximum_rms_difference_db": max_error,
        "threshold_disagreements": 0,
    }


def prepare(config_path):
    config = read(config_path)
    out = ROOT / config["run_directory"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        seal = read(out / "complete.json")
        if seal["config"] != config:
            raise ValueError("Completed run configuration changed; use a new run")
        checked(seal["inputs"])
        checked(seal["outputs"])
        print("Completed quality preprocessing verified")
        return
    if any(out.iterdir()):
        raise ValueError("Incomplete output exists; inspect it and use a new run name")
    started = time.time()

    def state(phase, **extra):
        value = {"phase": phase, "elapsed_seconds": time.time() - started, **extra}
        write(out / "state.json", value)
        print(value, flush=True)

    try:
        state("verifying_inputs")
        audit = ROOT / config["audit_run"]
        source = ROOT / config["source_run"]
        original = ROOT / config["reference_prepared"]
        seal = read(audit / "complete.json")
        checked(seal["inputs"])
        checked(
            {str((audit / p).relative_to(ROOT)): h for p, h in seal["files"].items()}
        )
        checked(read(audit / "analysis-complete.json")["files"])
        audited = read(audit / "results.json")
        if any(audited["structural_issues"].values()) or audited["duplicate_pcm"]:
            raise ValueError("Resolve structural audit findings first")
        if audited["segments"] != config["expected_segments"]:
            raise ValueError("Unexpected training population")
        for name in ("train-segments.jsonl", "train-features.f32"):
            key = str((source / name).relative_to(ROOT))
            if key not in seal["inputs"]:
                raise ValueError("Requested source was not audited")
        old_inputs = read(original / "inputs.json")
        bank_path = ROOT / old_inputs["training_inputs"]
        files = [
            config_path,
            *BASE.glob("*.py"),
            *BASE.glob("tests/*.py"),
            audit / "complete.json",
            audit / "analysis-complete.json",
            audit / "segment-measures.npz",
            audit / "source-measures.jsonl",
            audit / "asr-diagnostics.json",
            source / "train-segments.jsonl",
            source / "train-features.f32",
            original / "freeze.json",
            bank_path,
            *(
                original / n
                for n in ("pool.json", "pool.npz", "train-metadata.npy", "inputs.json")
            ),
        ]
        reference = read(original / "freeze.json")["files"]
        ref_files = [
            bank_path,
            *(
                original / n
                for n in ("pool.json", "pool.npz", "train-metadata.npy", "inputs.json")
            ),
        ]
        checked(
            {
                str(p.relative_to(ROOT)): reference[str(p.relative_to(ROOT))]
                for p in ref_files
            }
        )
        inputs = pin(files)
        write(out / "design-freeze.json", {"config": config, "files": inputs})
        with np.load(audit / "segment-measures.npz", allow_pickle=False) as data:
            measures = {k: data[k] for k in data.files}
        with np.load(original / "pool.npz", allow_pickle=False) as data:
            pool = {k: data[k] for k in data.files}
        info = read(original / "pool.json")
        sources = [
            json.loads(line) for line in (audit / "source-measures.jsonl").open()
        ]
        asr = read(audit / "asr-diagnostics.json")
        asr_lookup = {r["source_file"]: r for r in asr}
        if len(asr_lookup) != len(asr):
            raise ValueError("Duplicate ASR source rows")
        if len(measures["source"]) != info["segments"] or not np.array_equal(
            measures["phone"], pool["segment_phone"]
        ):
            raise ValueError("Audit and training pool differ")
        source_speakers = np.array(
            [info["speakers"].index(s["speaker_id"]) for s in sources]
        )
        if not np.array_equal(
            source_speakers[measures["source"]], pool["segment_speaker"]
        ):
            raise ValueError("Audit and training speaker ownership differ")
        source_rows = [
            source_quality(s, asr_lookup.get(s["source_file"]), config["policy"])
            for s in sources
        ]
        with (out / "source-quality.jsonl").open("w") as f:
            for i, row in enumerate(source_rows):
                f.write(
                    json.dumps(
                        {
                            "source_index": i,
                            **row,
                            "reasons": reason_names(row["reason_bits"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        source_bits = np.array([s["reason_bits"] for s in source_rows], np.uint16)
        bits, status = decide(
            measures,
            source_bits[measures["source"]],
            config["policy"],
            config["sample_rate"],
        )
        np.savez_compressed(out / "decisions.npz", reason_bits=bits, status=status)
        dc_check = verify_dc_decisions(
            measures, sources, config["policy"]["minimum_dc_removed_rms_dbfs"]
        )
        state(
            "building_training_views",
            accepted=int((status == 0).sum()),
            held=int((status == 1).sum()),
            rejected=int((status == 2).sum()),
        )
        stats, filtered, ids, mapping = subset_pool(info, pool, status == 0)
        bank, dropped = subset_bank(read(bank_path), mapping, info["phones"][:5])
        bank_audit = validate_bank(bank, stats, filtered)
        directory = out / "screened"
        directory.mkdir()
        np.savez(directory / "pool.npz", **filtered)
        np.save(directory / "source-segment-indices.npy", ids)
        np.save(out / "dc-only-source-indices.npy", np.flatnonzero(status != 2))
        write(directory / "pool.json", stats)
        write(directory / "train-inputs.json", bank)
        metadata = np.load(original / "train-metadata.npy", allow_pickle=False)
        np.save(directory / "train-metadata.npy", metadata[ids])
        old_features = np.memmap(
            source / "train-features.f32",
            mode="r",
            dtype="<f4",
            shape=(len(status), 128),
        )
        features = np.memmap(
            directory / "train-features.f32",
            mode="w+",
            dtype="<f4",
            shape=(len(ids), 128),
        )
        for start in range(0, len(ids), 65536):
            features[start : start + 65536] = old_features[ids[start : start + 65536]]
        features.flush()
        del features
        loaded = np.memmap(
            directory / "train-features.f32",
            mode="r",
            dtype="<f4",
            shape=(len(ids), 128),
        )
        for start in range(0, len(ids), 65536):
            if not np.array_equal(
                loaded[start : start + 65536], old_features[ids[start : start + 65536]]
            ):
                raise ValueError("Feature/manifest remapping failed")
        cov = coverage(info, pool, status)
        write(out / "coverage.json", cov)
        corpora = np.array([s["corpus"] for s in sources])[measures["source"]]
        by_corpus = {}
        candidates = set()
        rng = np.random.default_rng(config["review_seed"])
        for c in sorted(set(corpora)):
            keep = corpora == c
            by_corpus[c] = {
                name: int((keep & (status == k)).sum()) for k, name in STATUS.items()
            }
            by_corpus[c]["total"] = int(keep.sum())
            for p in range(len(info["phones"])):
                for k in STATUS:
                    group = np.flatnonzero(
                        keep & (measures["phone"] == p) & (status == k)
                    )
                    if len(group):
                        candidates.update(
                            rng.choice(
                                group,
                                min(
                                    len(group), config["review_per_corpus_phone_status"]
                                ),
                                replace=False,
                            ).tolist()
                        )
        state("writing_manifests")
        examples, written, accepted_written = [], 0, 0
        metrics = (
            "measured_rms_dbfs",
            "dc_removed_rms_dbfs",
            "dc_offset",
            "near_clip_fraction",
            "rail_fraction",
            "zero_fraction",
            "alignment_score",
        )
        with (
            (source / "train-segments.jsonl").open() as src,
            gzip.open(
                out / "segment-quality.jsonl.gz", "wt", compresslevel=1
            ) as quality,
            (directory / "train-segments.jsonl").open("w") as dst,
        ):
            for i, line in enumerate(src):
                row = json.loads(line)
                if (
                    row["cache_index"] != i
                    or row["split"] != "train"
                    or row["role"] != "training"
                ):
                    raise ValueError("Unexpected training row")
                if (
                    row["source_file"]
                    != sources[int(measures["source"][i])]["source_file"]
                ):
                    raise ValueError("Source index mismatch")
                item = {
                    "source_cache_index": i,
                    "segment_id": row["segment_id"],
                    "source_index": int(measures["source"][i]),
                    "decision": STATUS[int(status[i])],
                    "reasons": reason_names(bits[i]),
                    "raw_duration_seconds": float(
                        (measures["raw_end"][i] - measures["raw_start"][i])
                        / config["sample_rate"]
                    ),
                    **{k: float(measures[k][i]) for k in metrics},
                }
                quality.write(
                    json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                )
                if status[i] == 0:
                    dst.write(
                        json.dumps(
                            {
                                **row,
                                "cache_index": int(mapping[i]),
                                "source_cache_index": i,
                                "quality_version": config["policy"]["version"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted_written += 1
                if i in candidates:
                    examples.append(
                        {
                            "segment": row,
                            "quality": item,
                            "source_quality": source_rows[int(measures["source"][i])],
                            "human_review": None,
                        }
                    )
                written += 1
                if written % 200000 == 0:
                    state("writing_manifests", rows=written, total=len(status))
        if written != len(status) or accepted_written != len(ids):
            raise ValueError("Output population mismatch")
        write(out / "review-samples.json", examples)
        report = {
            "status": "complete",
            "config": config,
            "total_segments": len(status),
            "decisions": {name: int((status == k).sum()) for k, name in STATUS.items()},
            "reason_counts_overlapping": {
                name: int(((bits & bit) != 0).sum()) for name, bit in REASONS.items()
            },
            "by_corpus": by_corpus,
            "screened_pool": stats,
            "screened_bank": bank_audit,
            "bank_items_removed": dropped,
            "coverage": {
                k: v for k, v in cov.items() if k != "changed_speaker_phone_groups"
            },
            "float32_dc_check": dc_check,
            "review_samples": len(examples),
            "review_sampling": "One per available corpus/phoneme/decision stratum, fixed seed; not population-uniform; no human answers",
            "feature_rows_verified": len(ids),
            "training_started": False,
            "limitations": "Heuristic held rows are not proven bad. ASR flags apply to their entire recording. Original feature normalization and model metadata are retained to isolate filtering. Existing rejected data were not reintroduced. Evaluation is unchanged and has not been used to select thresholds.",
            "elapsed_seconds": time.time() - started,
        }
        write(out / "results.json", report)
        state("verifying_outputs")
        checked(inputs)
        outputs = pin(
            p for p in out.rglob("*") if p.is_file() and p.name != "state.json"
        )
        write(
            out / "complete.json",
            {"config": config, "inputs": inputs, "outputs": outputs},
        )
        state("complete", **report["decisions"])
    except BaseException as error:
        write(
            out / "failed.json",
            {"error": repr(error), "elapsed_seconds": time.time() - started},
        )
        state("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "status"))
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.command == "prepare":
        prepare(config_path)
    else:
        out = ROOT / read(config_path)["run_directory"]
        for name in ("state.json", "failed.json"):
            if (out / name).exists():
                print(name, json.dumps(read(out / name), ensure_ascii=False))
