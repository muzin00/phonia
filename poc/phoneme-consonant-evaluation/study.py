"""Equal-duration phoneme comparison with validation-frozen thresholds."""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-training-data-expansion"))

import evaluation_study as old
import numpy as np
import torch
from matched_audio import (
    MINIMUM,
    PHONEMES,
    VOWELS,
    capacity,
    matched_query,
    plan,
)
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import Segment, seeded_hash
from phase3_train.models import create_encoder

CONFIG = BASE / "config/protocol.json"
CONDITIONS = ("vowels5", "vowels_mn_available")
SUPPORTS = ("common7", "native")
ROLES = old.ROLES
read_json, checked, sha256_file, write_json = (
    old.read_json,
    old.checked,
    old.sha256_file,
    old.write_json,
)


def pin(files, path):
    name = str(path.relative_to(ROOT))
    checksum = sha256_file(path)
    if name in files and files[name] != checksum:
        raise ValueError("conflicting source checksum")
    files[name] = checksum


def diversify(rows, count, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_file"]].append(row)
    sources = sorted(grouped, key=lambda s: seeded_hash("nasal_source", seed, s))
    for source in sources:
        grouped[source].sort(
            key=lambda r: seeded_hash("nasal_segment", seed, r["segment_id"])
        )
    result, index = [], 0
    while len(result) < count:
        added = False
        for source in sources:
            if index < len(grouped[source]):
                result.append(grouped[source][index])
                added = True
                if len(result) == count:
                    break
        if not added:
            raise ValueError("insufficient nasal enrollment candidates")
        index += 1
    return result


def prepare_split(config, split, files, metadata):
    prior = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
    original = read_json(prior)
    pin(files, prior)
    labels = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    if set(original["speakers"]) != set(labels[split]) or set(
        original["speakers"]
    ) & set(labels["train"]):
        raise ValueError("training/evaluation label leak")
    nasal_cache, qc = {}, Counter()

    def nasals(source, speaker, role):
        if source in nasal_cache:
            return nasal_cache[source]
        meta = metadata[source]
        if (
            meta["split"] != split
            or meta["speaker_id"] != speaker
            or meta["evaluation_role"] != role
        ):
            raise ValueError("unexpected source metadata")
        path = (
            ROOT
            / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
            / speaker
            / f"{meta['utterance_id']}.json"
        )
        raw = read_json(path)
        if any(
            raw[k] != meta[k]
            for k in ("utterance_id", "speaker_id", "source_file", "source_sha256")
        ):
            raise ValueError("nasal alignment/source mismatch")
        pin(files, path)
        checked(ROOT / source, meta["source_sha256"])
        files[source] = meta["source_sha256"]
        with wave.open(str(ROOT / source), "rb") as audio:
            if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
                24000,
                1,
                2,
            ):
                raise ValueError("unexpected PCM format")
            samples = (
                np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                    np.float64
                )
                / 32768
            )
        result = []
        for index, interval in enumerate(raw["intervals"]):
            phone = interval["phoneme"]
            if phone not in ("m", "n"):
                continue
            first, last = (
                round(interval["start_sec"] * 24000),
                round(interval["end_sec"] * 24000),
            )
            if not 0 <= first < last <= len(samples):
                raise ValueError("invalid nasal interval")
            qc[f"{role}/{phone}/raw"] += 1
            rms = float(np.sqrt(np.mean(samples[first:last] ** 2)))
            if last - first < MINIMUM or rms < 10 ** (
                config["minimum_nasal_rms_dbfs"] / 20
            ):
                qc[f"{role}/{phone}/excluded"] += 1
                continue
            qc[f"{role}/{phone}/eligible"] += 1
            result.append(
                {
                    "segment_id": f"{meta['utterance_id']}--phone-{index:03d}-{phone}",
                    "vowel": phone,
                    "source_file": source,
                    "source_sha256": meta["source_sha256"],
                    "start_frame": first,
                    "end_frame": last,
                }
            )
        nasal_cache[source] = result
        return result

    profiles = {}
    enrollment_hashes = set()
    for item in original["enrollment"]:
        speaker = item["user_id"]
        vowels = item["segments"]
        if Counter(s["vowel"] for s in vowels) != Counter({p: 10 for p in VOWELS}):
            raise ValueError("expected frozen ten-vowel candidates")
        sources = sorted({s["source_file"] for s in vowels})
        nasal_pool = [
            s for source in sources for s in nasals(source, speaker, "enrollment")
        ]
        enrollment_hashes.update(
            metadata[source]["source_sha256"] for source in sources
        )
        extra = [
            s
            for p in ("m", "n")
            for s in diversify(
                [r for r in nasal_pool if r["vowel"] == p],
                config["nasal_enrollment_candidates_per_phone"],
                config["selection_seed"],
            )
        ]
        budget = min(config["enrollment_cap_frames"], sum(capacity(s) for s in vowels))
        a, b = plan(vowels, VOWELS, budget), plan([*vowels, *extra], PHONEMES, budget)
        if a is None or b is None:
            raise ValueError("enrollment cannot meet matched budget")
        profiles[speaker] = {
            "budget_frames": budget,
            "candidate_source_files": sources,
            "conditions": dict(zip(CONDITIONS, (a, b))),
        }
    queries = []
    for item in original["queries"]:
        if item["source_sha256"] in enrollment_hashes:
            raise ValueError("enrollment/query audio content overlap")
        extra = nasals(item["source_file"], item["speaker_id"], item["role"])
        plans, common = matched_query(
            item["segments"], extra, config["query_cap_frames"]
        )
        query = {
            k: item[k]
            for k in (
                "query_id",
                "utterance_id",
                "speaker_id",
                "source_file",
                "source_sha256",
                "role",
                "split",
            )
        }
        query.update(
            {
                "conditions": plans,
                "common7": common,
                "available_phones": sorted(
                    {s["vowel"] for s in [*item["segments"], *extra]}
                ),
                "budget_frames": sum(
                    s["end_frame"] - s["start_frame"] for s in plans[CONDITIONS[0]]
                )
                if plans[CONDITIONS[0]]
                else 0,
            }
        )
        queries.append(query)
    if len(queries) != 1200 or len(profiles) != 15:
        raise ValueError("changed query/profile population")
    report = {
        "nasal_qc": dict(qc),
        "roles": {
            role: {
                "all_queries": sum(q["role"] == role for q in queries),
                "common7_queries": sum(
                    q["role"] == role and q["common7"] for q in queries
                ),
                "scored_queries": sum(
                    q["role"] == role and q["conditions"][CONDITIONS[0]] is not None
                    for q in queries
                ),
                "extra_phone_counts": dict(
                    Counter(
                        "".join(
                            p
                            for p in ("m", "n")
                            if p
                            in {
                                s["vowel"] for s in q["conditions"][CONDITIONS[1]] or []
                            }
                        )
                        or "none"
                        for q in queries
                        if q["role"] == role
                    )
                ),
            }
            for role in ROLES
        },
    }
    return {
        "split": split,
        "speakers": original["speakers"],
        "profiles": profiles,
        "queries": queries,
        "preparation": report,
    }


def prepare(config, run):
    if run.exists():
        raise ValueError("unused evaluation directory required")
    training = ROOT / config["training_run"]
    summary_path = training / "training/summary.json"
    checked(summary_path, config["training_summary_sha256"])
    summary = read_json(summary_path)
    if (
        summary["status"] != "completed"
        or summary["test_used"]
        or summary["selected_update"] != 30000
    ):
        raise ValueError("requires fixed completed seven-phone training")
    files = {}
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        checked(training / name, summary["outputs_sha256"][name])
        pin(files, training / name)
    pin(files, summary_path)
    pin(files, training / "training-freeze.json")
    train_freeze = read_json(training / "training-freeze.json")
    checked(training / "training-freeze.json", summary["freeze_sha256"])
    meta_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    with meta_path.open() as source:
        metadata = {r["source_file"]: r for r in map(json.loads, source)}
    pin(files, meta_path)
    pin(files, ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")
    pin(files, CONFIG)
    run.mkdir(parents=True)
    for split in ("validation", "test"):
        inputs = prepare_split(config, split, files, metadata)
        validate_plans(inputs)
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
        print(
            f"{split} input preparation: {inputs['preparation']['roles']}", flush=True
        )
    paths = {
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        *(ROOT / "poc/phoneme-speaker-encoder/config").glob("*.json"),
    }
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_absolute():
            path = Path(filename).resolve()
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.is_relative_to(ROOT)
                and ".venv" not in path.parts
            ):
                paths.add(path)
    for path in paths:
        name = str(path.relative_to(ROOT))
        if name in train_freeze["files"]:
            checked(path, train_freeze["files"][name])
        pin(files, path)
    write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "runtime": old.previous.runtime(),
            "status": "frozen_before_new_inference",
        },
    )


def frozen(config, run, split):
    freeze = read_json(run / "design-freeze.json")
    if freeze["config"] != config or freeze["runtime"] != old.previous.runtime():
        raise ValueError("design/runtime changed")
    for path, sha in freeze["files"].items():
        checked(ROOT / path, sha)
    if split == "test":
        thresholds = read_json(run / "evaluation-freeze.json")
        if thresholds["status"] != "thresholds_frozen_before_test_inference":
            raise ValueError("test requires frozen validation thresholds")
        for path, sha in thresholds["files"].items():
            checked(run / path, sha)
    return read_json(run / f"{split}-inputs.json")


def infer(config, run, split):
    inputs = frozen(config, run, split)
    training = ROOT / config["training_run"]
    export = torch.load(
        training / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if tuple(export["phonemes"]) != PHONEMES or export["checkpoint_update"] != 30000:
        raise ValueError("unexpected encoder")
    model = create_encoder("statistics_mlp").eval().requires_grad_(False)
    model.load_state_dict(export["model"], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    settings = read_json(
        ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json"
    )
    pipeline = InputPipeline(
        settings["input"],
        rms_enabled=False,
        statistics=read_json(training / "bundle/feature-statistics.json"),
    )
    rows = {}
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        for selected in item["conditions"].values():
            for row in selected or []:
                if row["segment_id"] in rows and rows[row["segment_id"]] != row:
                    raise ValueError("conflicting source slice")
                rows[row["segment_id"]] = row
    segments = [
        Segment(
            segment_id=r["segment_id"],
            speaker_id="unused",
            vowel=r["vowel"],
            split=split,
            role="verification",
            cohorts=(),
            source_file=r["source_file"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            source_sha256=r["source_sha256"],
        )
        for r in sorted(rows.values(), key=lambda r: r["segment_id"])
    ]
    dataset = SegmentDataset(ROOT, segments, pipeline, mode="center")
    vectors, repeats = {}, 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(segments), 64):
            batch = collate_segments(
                [dataset[i] for i in range(offset, min(offset + 64, len(segments)))]
            )
            embeddings = model(batch["input"], batch["mask"])
            if not torch.isfinite(embeddings).all():
                raise ValueError("nonfinite embedding")
            for _ in range(2):
                if not torch.equal(embeddings, model(batch["input"], batch["mask"])):
                    raise ValueError("inference repeat mismatch")
                repeats += len(batch["segment_ids"])
            values = embeddings.numpy().astype(np.float64)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            vectors.update(zip(batch["segment_ids"], values))
    profiles = {}
    for speaker, item in inputs["profiles"].items():
        for condition, selected in item["conditions"].items():
            for phone in PHONEMES:
                chosen = [
                    vectors[s["segment_id"]] for s in selected if s["vowel"] == phone
                ]
                if chosen:
                    mean = np.mean(chosen, axis=0)
                    profiles[speaker, condition, phone] = mean / np.linalg.norm(mean)
    scores = []
    for query in inputs["queries"]:
        for condition, selected in query["conditions"].items():
            phones = [
                p for p in PHONEMES if any(s["vowel"] == p for s in selected or [])
            ]
            query_vectors = (
                {
                    p: np.mean(
                        [vectors[s["segment_id"]] for s in selected if s["vowel"] == p],
                        axis=0,
                    )
                    for p in phones
                }
                if selected
                else {}
            )
            for claimed in inputs["speakers"]:
                components = {
                    p: float(query_vectors[p] @ profiles[claimed, condition, p])
                    for p in phones
                }
                score = (
                    float(np.mean(list(components.values()))) if components else None
                )
                scores.append(
                    {
                        "query_id": query["query_id"],
                        "speaker_id": query["speaker_id"],
                        "claimed_speaker_id": claimed,
                        "split": split,
                        "role": query["role"],
                        "condition": condition,
                        "is_genuine": claimed == query["speaker_id"],
                        "common7": query["common7"],
                        "status": "scored" if score is not None else "no_score",
                        "score": score,
                        "phone_scores": components,
                        "used_frames": query["budget_frames"]
                        if score is not None
                        else 0,
                    }
                )
    if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError("encoder modified by inference")
    output = run / split
    output.mkdir(exist_ok=True)
    with (output / "scores.jsonl").open("w") as stream:
        for row in scores:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(
        output / "inference.json",
        {
            "status": "completed",
            "unique_embeddings": len(vectors),
            "repeated_embeddings": repeats,
            "all_repeats_bitwise_equal": True,
            "encoder_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "scores_sha256": sha256_file(output / "scores.jsonl"),
        },
    )
    frozen(config, run, split)
    print(f"{split}: {len(scores)} score slots, {len(vectors)} embeddings", flush=True)


def validate_plans(inputs):
    """Check exact time equality, legal source slices and shared support."""
    checks = 0
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        plans = item["conditions"]
        if (plans[CONDITIONS[0]] is None) != (plans[CONDITIONS[1]] is None):
            raise ValueError("native score coverage must match")
        for condition, rows in plans.items():
            if rows is None:
                continue
            if (
                sum(s["end_frame"] - s["start_frame"] for s in rows)
                != item["budget_frames"]
            ):
                raise ValueError("unequal actual PCM budgets")
            phones = {s["vowel"] for s in rows}
            expected = set(VOWELS) if condition == CONDITIONS[0] else set(PHONEMES)
            if not set(VOWELS) <= phones or not phones <= expected:
                raise ValueError("invalid selected phonemes")
            if "common7" not in item and phones != expected:
                raise ValueError("incomplete enrollment phonemes")
            if len({s["original_segment_id"] for s in rows}) != len(rows):
                raise ValueError("repeated interval")
            grouped = defaultdict(list)
            for row in rows:
                if (
                    not MINIMUM <= row["end_frame"] - row["start_frame"] <= 6000
                    or row["start_frame"] < 0
                ):
                    raise ValueError("invalid duration")
                grouped[row["source_file"]].append(
                    (row["start_frame"], row["end_frame"])
                )
            for ranges in grouped.values():
                ordered = sorted(ranges)
                if any(a[1] > b[0] for a, b in pairwise(ordered)):
                    raise ValueError("overlapping PCM slices")
            checks += 1
        if "common7" in item:
            actual = plans[CONDITIONS[1]]
            common = actual is not None and {s["vowel"] for s in actual} == set(
                PHONEMES
            )
            if item["common7"] != common:
                raise ValueError("incorrect common support")
    return checks


def validate_scores(inputs, rows):
    queries = {q["query_id"]: q for q in inputs["queries"]}
    seen = set()
    for row in rows:
        query = queries.get(row["query_id"])
        key = row["query_id"], row["condition"], row["claimed_speaker_id"]
        if (
            query is None
            or key in seen
            or row["condition"] not in CONDITIONS
            or row["claimed_speaker_id"] not in inputs["speakers"]
        ):
            raise ValueError("invalid trial identity")
        seen.add(key)
        if any(
            row[k] != query[k] for k in ("speaker_id", "role", "split", "common7")
        ) or row["is_genuine"] != (row["speaker_id"] == row["claimed_speaker_id"]):
            raise ValueError("trial label/metadata mismatch")
        selected = query["conditions"][row["condition"]]
        if row["status"] != ("scored" if selected else "no_score"):
            raise ValueError("unexpected score coverage")
        if selected:
            components = row["phone_scores"]
            if set(components) != {r["vowel"] for r in selected} or not all(
                np.isfinite(v) and -1 <= v <= 1 for v in components.values()
            ):
                raise ValueError("invalid phoneme scores")
            if (
                row["score"] != float(np.mean(list(components.values())))
                or row["used_frames"] != query["budget_frames"]
            ):
                raise ValueError("invalid fused score/budget")
        elif row["score"] is not None or row["phone_scores"] or row["used_frames"]:
            raise ValueError("missing score mismatch")
    if len(seen) != len(queries) * len(CONDITIONS) * len(inputs["speakers"]):
        raise ValueError("incomplete trial matrix")


def audit_source_slices(inputs, config):
    prior = read_json(
        ROOT / config["reference_inputs"] / f"{inputs['split']}-inputs.json"
    )
    originals = {
        s["segment_id"]: s
        for item in [*prior["enrollment"], *prior["queries"]]
        for s in item["segments"]
    }
    raw_cache = {}
    count = 0
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        for selected in item["conditions"].values():
            for row in selected or []:
                origin = row["original_segment_id"]
                original = originals.get(origin)
                if original is None:
                    utterance, suffix = origin.split("--phone-")
                    speaker = utterance.split("_")[0]
                    path = (
                        ROOT
                        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
                        / speaker
                        / f"{utterance}.json"
                    )
                    if path not in raw_cache:
                        raw_cache[path] = read_json(path)
                    raw = raw_cache[path]
                    interval = raw["intervals"][int(suffix.split("-")[0])]
                    original = {
                        "vowel": interval["phoneme"],
                        "start_frame": round(interval["start_sec"] * 24000),
                        "end_frame": round(interval["end_sec"] * 24000),
                        "source_file": raw["source_file"],
                        "source_sha256": raw["source_sha256"],
                    }
                size = row["end_frame"] - row["start_frame"]
                expected = (
                    original["start_frame"]
                    + (original["end_frame"] - original["start_frame"] - size) // 2
                )
                if (
                    row["start_frame"] != expected
                    or row["end_frame"] > original["end_frame"]
                    or any(
                        row[k] != original[k]
                        for k in ("vowel", "source_file", "source_sha256")
                    )
                ):
                    raise ValueError(
                        "selected PCM differs from original labeled center slice"
                    )
                count += 1
    return count


def load_groups(run, split):
    with (run / split / "scores.jsonl").open() as stream:
        rows = list(map(json.loads, stream))
    checked(
        run / split / "scores.jsonl",
        read_json(run / split / "inference.json")["scores_sha256"],
    )
    validate_scores(read_json(run / f"{split}-inputs.json"), rows)
    groups = {}
    for support in SUPPORTS:
        for condition in CONDITIONS:
            for role in ROLES:
                groups[support, condition, role] = [
                    r
                    for r in rows
                    if r["condition"] == condition
                    and r["role"] == role
                    and (support == "native" or r["common7"])
                ]
    return groups


def calibrate(config, run):
    inputs = frozen(config, run, "validation")
    groups = load_groups(run, "validation")
    thresholds, cells = {}, {}
    for support in SUPPORTS:
        for condition in CONDITIONS:
            key = f"{support}/{condition}"
            thresholds[key] = old.calibrate_cell(
                groups[support, condition, "verification"], inputs["speakers"]
            )
            if thresholds[key]["status"] != "calibrated":
                raise ValueError("insufficient validation speaker support")
            for role in ROLES:
                cells[f"{key}/{role}"], _ = old.evaluate_cell(
                    groups[support, condition, role],
                    thresholds[key],
                    inputs["speakers"],
                )
    write_json(run / "validation-thresholds.json", thresholds)
    write_json(run / "validation-metrics.json", {"conditions": cells})
    _, counts = old.speaker_draws(
        15, config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    names = (
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-counts.npy",
        "validation/scores.jsonl",
        "validation/inference.json",
    )
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_test_inference",
            "files": {name: sha256_file(run / name) for name in names},
        },
    )
    print("validation thresholds frozen for both supports; starting test", flush=True)


def evaluate(config, run):
    inputs = frozen(config, run, "test")
    groups = load_groups(run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, replicas, differences, delta_arrays = {}, {}, {}, {}
    for (support, condition, role), rows in groups.items():
        key = f"{support}/{condition}/{role}"
        threshold = thresholds[f"{support}/{condition}"]
        cells[key], _ = old.evaluate_cell(rows, threshold, inputs["speakers"])
        cells[key]["ci95"], replicas[key] = old.bootstrap_cell(
            rows, cells[key], threshold, inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    names = [
        f"{point}/{metric}" for point, _ in old.POINTS for metric in old.METRICS
    ] + ["pooled_eer", "query_coverage"]
    for support in SUPPORTS:
        for role in ROLES:
            a, b = (
                f"{support}/{CONDITIONS[0]}/{role}",
                f"{support}/{CONDITIONS[1]}/{role}",
            )
            for name in names:
                key = f"{support}/{role}/{name}"
                delta = 100 * (replicas[b][name] - replicas[a][name])
                delta_arrays[key] = delta
                differences[key] = {
                    "difference_percentage_points": 100
                    * (
                        old.metric_value(cells[b], name)
                        - old.metric_value(cells[a], name)
                    ),
                    "ci95_percentage_points": old.interval(delta),
                    "direction": "extended_minus_vowels5",
                }
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{f"{k}/{n}": v for k, items in replicas.items() for n, v in items.items()},
    )
    np.savez_compressed(run / "bootstrap-differences.npz", **delta_arrays)
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "paired_differences": differences,
            "threshold_recalibrated_on_test": False,
            "bootstrap_replicates": config["bootstrap_replicates"],
        },
    )
    frozen(config, run, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validation", "test"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    if (
        config["conditions"] != list(CONDITIONS)
        or config["phonemes"] != list(PHONEMES)
        or config["required_query_phonemes"] != list(VOWELS)
        or config["supports"] != list(SUPPORTS)
        or config["roles"] != list(ROLES)
        or config["minimum_frames"] != MINIMUM
        or config["maximum_frames_per_segment"] != 6000
        or config["retrain"]
        or config["independent_holdout"]
    ):
        raise ValueError("unsupported matched-duration protocol")
    torch.set_num_threads(1)
    run = ROOT / config["run_directory"]
    if args.stage == "prepare":
        prepare(config, run)
    elif args.stage == "validation":
        infer(config, run, "validation")
        calibrate(config, run)
    else:
        infer(config, run, "test")
        evaluate(config, run)


if __name__ == "__main__":
    main()
