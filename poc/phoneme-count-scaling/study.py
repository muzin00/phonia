"""Freeze equal-time common14 inputs, then evaluate a phoneme-count curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import count_math as cm
import numpy as np
import torch
from bridge import BASE, ROOT, checked, fixed, pin, read_json, sha256_file, write_json
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import Segment
from voicing_audio import capacity, pcm_rms_dbfs, plan

CONFIG = BASE / "config/protocol.json"
ROLES = fixed.ROLES


def rank(row, seed):
    return hashlib.sha256(f"{seed}/{row['segment_id']}".encode()).hexdigest()


def candidates(source, metadata, files):
    meta = metadata[source]
    path = (
        ROOT
        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
        / meta["speaker_id"]
        / f"{meta['utterance_id']}.json"
    )
    raw = read_json(path)
    if any(
        raw[k] != meta[k]
        for k in ("source_file", "source_sha256", "speaker_id", "raw_phonemes")
    ):
        raise ValueError("alignment metadata mismatch")
    pin(files, path)
    checked(ROOT / source, meta["source_sha256"])
    files[source] = meta["source_sha256"]
    with wave.open(str(ROOT / source), "rb") as audio:
        if (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError("unexpected PCM format")
        length = audio.getnframes()
    result = []
    for i, interval in enumerate(raw["intervals"]):
        phone = interval["phoneme"]
        if phone not in fixed.PHONEMES:
            continue
        lo, hi = (
            round(interval["start_sec"] * 24000),
            round(interval["end_sec"] * 24000),
        )
        if not 0 <= lo < hi <= length:
            raise ValueError("invalid original boundary")
        size = min(6000, hi - lo)
        if size < 720:
            continue
        first = lo + (hi - lo - size) // 2
        row = {
            "segment_id": f"{meta['utterance_id']}--raw-{i:03d}-{phone}",
            "vowel": phone,
            "source_file": source,
            "source_sha256": meta["source_sha256"],
            "speaker_id": meta["speaker_id"],
            "split": meta["split"],
            "role": meta["evaluation_role"],
            "start_frame": first,
            "end_frame": first + size,
            "raw_index": i,
            "raw_start_frame": lo,
            "raw_end_frame": hi,
        }
        if pcm_rms_dbfs(ROOT, row) >= -50:
            result.append(row)
    return result


def matched_plans(pool, conditions, cap, config):
    if {r["vowel"] for r in pool} != set(config["phones"]):
        return None
    upper = min(
        cap,
        *(
            sum(capacity(r) for r in pool if r["vowel"] in phones)
            for phones in conditions.values()
        ),
    )
    upper -= upper % config["budget_step_frames"]
    for budget in range(upper, 720 * 14 - 1, -config["budget_step_frames"]):
        plans = {key: plan(pool, phones, budget) for key, phones in conditions.items()}
        if any(rows is None for rows in plans.values()):
            continue
        if all(
            pcm_rms_dbfs(ROOT, r) >= config["minimum_rms_dbfs"]
            for rows in plans.values()
            for r in rows
        ):
            return {"budget_frames": budget, "conditions": plans}
    return None


def enrollment_candidates(pool, config):
    selected = []
    for phone in config["phones"]:
        rows = [r for r in pool if r["vowel"] == phone]
        if not rows:
            raise ValueError("missing enrollment phone")
        count = min(config["enrollment_candidates_per_phone_maximum"], len(rows))
        selected.extend(fixed.prior.diversify(rows, count, config["selection_seed"]))
    return selected


def validate(data, config, design, training_hashes):
    enrollment = {
        r["source_sha256"]
        for p in data["profiles"].values()
        for rows in p["conditions"].values()
        for r in rows
    }
    total = 0
    for item in [*data["profiles"].values(), *data["queries"]]:
        if set(item["conditions"]) != set(design["conditions"]):
            raise ValueError("incomplete condition support")
        for condition, rows in item["conditions"].items():
            if {r["vowel"] for r in rows} != set(design["conditions"][condition]):
                raise ValueError("phone set changed")
            if (
                sum(r["end_frame"] - r["start_frame"] for r in rows)
                != item["budget_frames"]
            ):
                raise ValueError("unequal actual input duration")
            if len({r["original_segment_id"] for r in rows}) != len(rows):
                raise ValueError("repeated interval")
            ranges = defaultdict(list)
            for r in rows:
                if r["source_sha256"] in training_hashes or (
                    "query_id" in item and r["source_sha256"] in enrollment
                ):
                    raise ValueError("training/enrollment/query overlap")
                if "query_id" in item and r["source_file"] != item["source_file"]:
                    raise ValueError("query no longer a single original utterance")
                if r["speaker_id"] != item["speaker_id"] or r["split"] != data["split"]:
                    raise ValueError("wrong split/speaker")
                size = r["end_frame"] - r["start_frame"]
                if not (
                    720 <= size <= 6000
                    and r["raw_start_frame"]
                    <= r["start_frame"]
                    < r["end_frame"]
                    <= r["raw_end_frame"]
                ):
                    raise ValueError("invalid boundary or duration")
                if pcm_rms_dbfs(ROOT, r) < config["minimum_rms_dbfs"]:
                    raise ValueError("post-crop RMS failure")
                ranges[r["source_file"]].append((r["start_frame"], r["end_frame"]))
                total += 1
            if any(
                a[1] > b[0] for rs in ranges.values() for a, b in pairwise(sorted(rs))
            ):
                raise ValueError("overlapping source slices")
    support = Counter(
        q["speaker_id"] for q in data["queries"] if q["role"] == "verification"
    )
    if set(support) != set(data["speakers"]) or min(support.values()) < 2:
        raise ValueError("insufficient normal speaker coverage before inference")
    return total


def prepare(config, run):
    if run.exists():
        raise ValueError("unused run directory required")
    if config["phones"] != list(fixed.PHONEMES):
        raise ValueError("wrong fixed inventory")
    design = cm.schedule(config["phones"], config["selection_seed"])
    files = {}
    meta_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = {r["source_file"]: r for r in fixed.old.previous.read_rows(meta_path)}
    pin(files, meta_path)
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    labels = read_json(split_path)["speaker_splits"]
    pin(files, split_path)
    training = ROOT / config["training_run"]
    summary = read_json(training / "training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["selected_update"] != 30000
        or summary["test_used"]
    ):
        raise ValueError("wrong immutable training snapshot")
    checked(training / "training-freeze.json", summary["freeze_sha256"])
    training_freeze = read_json(training / "training-freeze.json")
    manifest = training / "training-segments.jsonl"
    checked(manifest, training_freeze["files"][str(manifest.relative_to(ROOT))])
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        checked(training / name, summary["outputs_sha256"][name])
        pin(files, training / name)
    for name in (
        "training/summary.json",
        "training-freeze.json",
        "training-segments.jsonl",
    ):
        pin(files, training / name)
    train_hashes = {
        r["source_sha256"]
        for r in fixed.old.previous.read_rows(training / "training-segments.jsonl")
    }
    datasets = {}
    for split in ("validation", "test"):
        ref = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
        original = read_json(ref)
        pin(files, ref)
        if set(original["speakers"]) != set(labels[split]) or set(labels[split]) & set(
            labels["train"]
        ):
            raise ValueError("speaker split mismatch")
        cache = {}

        def get(source, cache=cache):
            if source not in cache:
                cache[source] = sorted(
                    candidates(source, metadata, files),
                    key=lambda candidate: rank(candidate, config["selection_seed"]),
                )
            return cache[source]

        profiles = {}
        for original_profile in original["enrollment"]:
            sources = sorted({r["source_file"] for r in original_profile["segments"]})
            pool = [r for source in sources for r in get(source)]
            chosen = enrollment_candidates(pool, config)
            profile = matched_plans(
                chosen, design["conditions"], config["enrollment_cap_frames"], config
            )
            if profile is None:
                raise ValueError("no common enrollment budget")
            profiles[original_profile["user_id"]] = {
                **profile,
                "speaker_id": original_profile["user_id"],
                "candidate_source_files": sources,
            }
        queries, excluded = [], Counter()
        raw_common = Counter()
        for q in original["queries"]:
            pool = get(q["source_file"])
            if {r["vowel"] for r in pool} != set(config["phones"]):
                excluded[f"{q['role']}/missing_phone"] += 1
                continue
            raw_common[q["role"]] += 1
            match = matched_plans(
                pool, design["conditions"], config["query_cap_frames"], config
            )
            if match is None:
                excluded[f"{q['role']}/no_legal_common_budget"] += 1
                continue
            queries.append(
                {
                    **{
                        k: q[k]
                        for k in (
                            "query_id",
                            "speaker_id",
                            "source_file",
                            "source_sha256",
                            "role",
                            "split",
                        )
                    },
                    **match,
                }
            )
        data = {
            "split": split,
            "speakers": original["speakers"],
            "profiles": profiles,
            "queries": queries,
            "preparation": {
                "raw_common14": dict(raw_common),
                "excluded": dict(excluded),
                "roles": dict(Counter(q["role"] for q in queries)),
                "by_speaker": {
                    role: dict(
                        Counter(q["speaker_id"] for q in queries if q["role"] == role)
                    )
                    for role in ROLES
                },
                "selection_uses_embeddings": False,
            },
        }
        validate(data, config, design, train_hashes)
        datasets[split] = data
        print(f"{split}: {json.dumps(data['preparation'])}", flush=True)
    for path in BASE.rglob("*"):
        if (
            path.is_file()
            and path.suffix in (".py", ".json")
            and "__pycache__" not in path.parts
        ):
            pin(files, path)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.is_file()
                and path.is_relative_to(ROOT)
                and path.suffix == ".py"
                and ".venv" not in path.parts
            ):
                pin(files, path)
    pin(files, ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json")
    pin(files, BASE.parent / "phoneme-voicing-comparison/audit.py")
    run.mkdir(parents=True)
    for split, data in datasets.items():
        write_json(run / f"{split}-inputs.json", data)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "frozen_before_new_inference",
            "config": config,
            "schedule": design,
            "files": files,
            "runtime": fixed.old.previous.runtime(),
        },
    )


def verify(config, run, test=False):
    freeze = read_json(run / "design-freeze.json")
    if freeze["config"] != config or freeze["runtime"] != fixed.old.previous.runtime():
        raise ValueError("frozen design/runtime changed")
    for name, sha in freeze["files"].items():
        checked(ROOT / name, sha)
    if test:
        for name, sha in read_json(run / "evaluation-freeze.json")["files"].items():
            checked(ROOT / name, sha)
    return freeze["schedule"]


def infer(config, run, split):
    verify(config, run, split == "test")
    output = run / split
    if output.exists():
        raise ValueError("inference already performed")
    data = read_json(run / f"{split}-inputs.json")
    training = ROOT / config["training_run"]
    bundle = torch.load(
        training / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if (
        list(bundle["phonemes"]) != config["phones"]
        or bundle["checkpoint_update"] != 30000
    ):
        raise ValueError("wrong encoder")
    torch.set_num_threads(1)
    model = fixed.create_encoder("statistics_mlp").eval().requires_grad_(False)
    model.load_state_dict(bundle["model"], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    settings = read_json(
        ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json"
    )
    pipeline = InputPipeline(
        settings["input"],
        rms_enabled=False,
        statistics=read_json(training / "bundle/feature-statistics.json"),
    )
    unique = {}
    for item in [*data["profiles"].values(), *data["queries"]]:
        for selected in item["conditions"].values():
            for r in selected:
                if r["segment_id"] in unique and unique[r["segment_id"]] != r:
                    raise ValueError("conflicting slice")
                unique[r["segment_id"]] = r
    segments = [
        Segment(
            segment_id=r["segment_id"],
            speaker_id=r["speaker_id"],
            vowel=r["vowel"],
            split=split,
            role=r["role"],
            cohorts=(),
            source_file=r["source_file"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            source_sha256=r["source_sha256"],
        )
        for r in sorted(unique.values(), key=lambda r: r["segment_id"])
    ]
    dataset = SegmentDataset(ROOT, segments, pipeline, mode="center")
    vectors = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(segments), 64):
            batch = collate_segments(
                [dataset[i] for i in range(offset, min(offset + 64, len(segments)))]
            )
            embeddings = model(batch["input"], batch["mask"])
            if not torch.isfinite(embeddings).all() or not torch.equal(
                embeddings, model(batch["input"], batch["mask"])
            ):
                raise ValueError("nonfinite/nonrepeatable inference")
            values = embeddings.numpy().astype(np.float64)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            vectors.update(zip(batch["segment_ids"], values, strict=True))
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise ValueError("encoder changed")
    scores = cm.fused_scores(data, vectors)
    output.mkdir()
    np.savez_compressed(
        output / "embedding-vectors.npz",
        segment_ids=np.array(list(vectors)),
        vectors=np.array(list(vectors.values())),
    )
    with (output / "scores.jsonl").open("x") as stream:
        for r in scores:
            stream.write(json.dumps(r, allow_nan=False) + "\n")
    write_json(
        output / "inference.json",
        {
            "unique_embeddings": len(vectors),
            "score_rows": len(scores),
            "elapsed_sec": time.perf_counter() - start,
            "bitwise_repeat_equal": True,
            "encoder_unchanged": True,
            "scores_sha256": sha256_file(output / "scores.jsonl"),
            "embedding_vectors_sha256": sha256_file(output / "embedding-vectors.npz"),
        },
    )
    verify(config, run, split == "test")
    return data, scores


def cells_for(data, scores, design, thresholds):
    cells, groups = {}, {}
    for condition in design["conditions"]:
        for role in ROLES:
            selected = [
                r for r in scores if r["condition"] == condition and r["role"] == role
            ]
            actual = sorted({r["speaker_id"] for r in selected})
            key = f"{condition}/{role}"
            cell, _ = fixed.old.evaluate_cell(
                selected, thresholds[condition], actual or data["speakers"]
            )
            cell["query_speaker_count"] = len(actual)
            cell["enrolled_speaker_count"] = len(data["speakers"])
            cell["descriptive_partial_speaker_support"] = role != "verification"
            cells[key], groups[key] = cell, selected
    return cells, groups


def validation(config, run):
    design = verify(config, run)
    data, scores = infer(config, run, "validation")
    thresholds = {
        k: fixed.old.calibrate_cell(
            [r for r in scores if r["condition"] == k and r["role"] == "verification"],
            data["speakers"],
        )
        for k in design["conditions"]
    }
    if any(t["status"] != "calibrated" for t in thresholds.values()):
        raise ValueError("incomplete calibration support")
    cells, _ = cells_for(data, scores, design, thresholds)
    write_json(run / "validation-thresholds.json", thresholds)
    write_json(run / "validation-metrics.json", cells)
    draws, counts = fixed.old.speaker_draws(
        15, config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    np.savez_compressed(run / "bootstrap-speakers.npz", draws=draws, counts=counts)
    files = {}
    for name in (
        "design-freeze.json",
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-speakers.npz",
        "validation/scores.jsonl",
        "validation/inference.json",
        "validation/embedding-vectors.npz",
    ):
        pin(files, run / name)
    write_json(
        run / "evaluation-freeze.json",
        {"status": "thresholds_frozen_before_test_inference", "files": files},
    )
    print("twenty validation thresholds frozen", flush=True)


def test(config, run):
    design = verify(config, run, True)
    data, scores = infer(config, run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    cells, groups = cells_for(data, scores, design, thresholds)
    with np.load(run / "bootstrap-speakers.npz", allow_pickle=False) as saved:
        counts = saved["counts"]
    replicas, arrays = {}, {}
    for condition in design["conditions"]:
        key = f"{condition}/verification"
        cis, replica = fixed.old.bootstrap_cell(
            groups[key], cells[key], thresholds[condition], data["speakers"], counts
        )
        cells[key]["ci95"] = cis
        replicas[condition] = replica
        for field, values in replica.items():
            arrays[f"cell/{condition}/{field}"] = values
        print(f"test bootstrap {condition}", flush=True)
    fields = (
        "pooled_eer",
        "far_1pct/all_input_far",
        "far_1pct/all_input_frr",
        "far_0_1pct/all_input_far",
        "far_0_1pct/all_input_frr",
    )
    summaries, differences = {}, {}
    for role in ROLES:
        summaries[role] = {}
        for field in fields:
            summary = cm.count_means(cells, design["by_count"], role, field)
            summaries[role][field] = summary
            if role != "verification":
                continue
            means = {}
            for count in config["counts"]:
                values = np.mean(
                    [replicas[k][field] for k in design["by_count"][str(count)]], axis=0
                )
                means[count] = values
                summary[str(count)]["ci95"] = fixed.old.interval(values)
                arrays[f"count/{count}/{field}"] = values
            for lo, hi in [(5, 7), (7, 10), (10, 14), (5, 14)]:
                delta = means[hi] - means[lo]
                name = f"{hi}_minus_{lo}/{field}"
                differences[name] = {
                    "difference": summary[str(hi)]["mean"] - summary[str(lo)]["mean"],
                    "ci95": fixed.old.interval(delta),
                    "primary": field == "pooled_eer" and (lo, hi) == (5, 14),
                }
                arrays[f"difference/{name}"] = delta
    np.savez_compressed(run / "bootstrap-metrics.npz", **arrays)
    result = {
        "cells": cells,
        "count_summary": summaries,
        "differences": differences,
        "schedule": design,
        "primary": differences["14_minus_5/pooled_eer"],
        "threshold_recalibrated_on_test": False,
        "scope": config["bootstrap_scope"],
    }
    write_json(run / "test-results.json", result)
    verify(config, run, True)
    print(
        json.dumps(
            {
                "normal_EER": summaries["verification"]["pooled_eer"],
                "primary": result["primary"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validation", "test"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    {"prepare": prepare, "validation": validation, "test": test}[args.stage](
        config, ROOT / config["run_directory"]
    )


if __name__ == "__main__":
    main()
