"""Equal-duration pair and triple comparison using one fixed eight-phone model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import wave
from collections import Counter, defaultdict
from itertools import combinations, pairwise
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR_BASE = BASE.parent / "phoneme-consonant-evaluation"
sys.path.insert(0, str(PRIOR_BASE))
spec = importlib.util.spec_from_file_location(
    "pair_prior_study", PRIOR_BASE / "study.py"
)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
import numpy as np
import torch
from matched_audio import MINIMUM, VOWELS, capacity, plan
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import Segment
from phase3_train.models import create_encoder

old = prior.old
read_json, write_json, checked, sha256_file, pin = (
    prior.read_json,
    prior.write_json,
    prior.checked,
    prior.sha256_file,
    prior.pin,
)
PHONEMES = (*VOWELS, "m", "n", "s")
BASELINE_CONDITIONS = ("vowels5", "vowels_mn", "vowels_ms", "vowels_ns")
CONDITIONS = (*BASELINE_CONDITIONS, "vowels_mns")
PHONE_SETS = dict(
    zip(
        CONDITIONS,
        (
            VOWELS,
            (*VOWELS, "m", "n"),
            (*VOWELS, "m", "s"),
            (*VOWELS, "n", "s"),
            PHONEMES,
        ),
    )
)
ROLES = old.ROLES
PAIRS = tuple(combinations(CONDITIONS, 2))
LABELS = dict(
    zip(CONDITIONS, ("5母音", "5母音＋m/n", "5母音＋m/s", "5母音＋n/s", "5母音＋m/n/s"))
)
CONFIG = BASE / "config/protocol.json"


def extract_consonants(source, metadata, split, speaker, role, config, files, qc):
    meta = metadata[source]
    if (meta["split"], meta["speaker_id"], meta["evaluation_role"]) != (
        split,
        speaker,
        role,
    ):
        raise ValueError("source metadata mismatch")
    path = (
        ROOT
        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
        / speaker
        / f"{meta['utterance_id']}.json"
    )
    raw = read_json(path)
    if any(
        raw[k] != meta[k]
        for k in (
            "utterance_id",
            "speaker_id",
            "source_file",
            "source_sha256",
            "raw_phonemes",
        )
    ):
        raise ValueError("alignment metadata/text mismatch")
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
        samples = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                np.float64
            )
            / 32768
        )
    rows = []
    for index, interval in enumerate(raw["intervals"]):
        phone = interval["phoneme"]
        if phone not in ("m", "n", "s"):
            continue
        first, last = (
            round(interval["start_sec"] * 24000),
            round(interval["end_sec"] * 24000),
        )
        if not 0 <= first < last <= len(samples):
            raise ValueError("invalid consonant bounds")
        qc[f"{role}/{phone}/raw"] += 1
        if phone == "s":
            size = min(6000, last - first)
            first += (last - first - size) // 2
            last = first + size
        rms = float(np.sqrt(np.mean(samples[first:last] ** 2)))
        if last - first < MINIMUM or rms < 10 ** (
            config["minimum_consonant_rms_dbfs"] / 20
        ):
            qc[f"{role}/{phone}/excluded"] += 1
            continue
        qc[f"{role}/{phone}/eligible"] += 1
        identifier = f"{meta['utterance_id']}--phone-{index:03d}-{phone}"
        if phone == "s":
            identifier += f"--center-{first}-{last - first}"
        rows.append(
            {
                "segment_id": identifier,
                "vowel": phone,
                "source_file": source,
                "source_sha256": meta["source_sha256"],
                "start_frame": first,
                "end_frame": last,
            }
        )
    return rows


def condition_plans(candidates, budget):
    return {
        condition: plan(candidates, phones, budget)
        for condition, phones in PHONE_SETS.items()
    }


def prepare_split(config, split, files, metadata):
    reference = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
    original = read_json(reference)
    pin(files, reference)
    labels = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    if set(original["speakers"]) != set(labels[split]) or set(
        original["speakers"]
    ) & set(labels["train"]):
        raise ValueError("speaker support/leak mismatch")
    cache, qc = {}, Counter()

    def consonants(source, speaker, role):
        if source not in cache:
            cache[source] = extract_consonants(
                source, metadata, split, speaker, role, config, files, qc
            )
        return cache[source]

    profiles = {}
    enrollment_hashes = set()
    for enrollment in original["enrollment"]:
        speaker, vowels = enrollment["user_id"], enrollment["segments"]
        if Counter(r["vowel"] for r in vowels) != Counter({p: 10 for p in VOWELS}):
            raise ValueError("frozen ten-per-vowel candidates changed")
        sources = sorted({r["source_file"] for r in vowels})
        pool = [
            row
            for source in sources
            for row in consonants(source, speaker, "enrollment")
        ]
        extras = [
            row
            for phone in ("m", "n", "s")
            for row in prior.diversify(
                [r for r in pool if r["vowel"] == phone],
                config["enrollment_candidates_per_consonant"],
                config["selection_seed"],
            )
        ]
        budget = min(config["enrollment_cap_frames"], sum(capacity(r) for r in vowels))
        plans = condition_plans(vowels + extras, budget)
        if any(rows is None for rows in plans.values()):
            raise ValueError("incomplete enrollment/time budget")
        profiles[speaker] = {
            "budget_frames": budget,
            "candidate_source_files": sources,
            "conditions": plans,
        }
        enrollment_hashes.update(metadata[s]["source_sha256"] for s in sources)
    queries = []
    excluded = Counter()
    per_role = Counter()
    for item in original["queries"]:
        if item["source_sha256"] in enrollment_hashes:
            raise ValueError("enrollment/query content overlap")
        per_role[item["role"]] += 1
        candidates = item["segments"] + consonants(
            item["source_file"], item["speaker_id"], item["role"]
        )
        present = {r["vowel"] for r in candidates}
        if not set(PHONEMES) <= present:
            missing = "".join(p for p in PHONEMES if p not in present)
            excluded[f"{item['role']}/missing_{missing}"] += 1
            continue
        budget = min(
            config["query_cap_frames"], sum(capacity(r) for r in item["segments"])
        )
        plans = condition_plans(candidates, budget)
        if any(rows is None for rows in plans.values()):
            excluded[f"{item['role']}/cannot_fit_equal_budget"] += 1
            continue
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
        queries.append(
            {
                **query,
                "conditions": plans,
                "common8": True,
                "available_phones": sorted(present),
                "budget_frames": budget,
            }
        )
    return {
        "split": split,
        "speakers": original["speakers"],
        "profiles": profiles,
        "queries": queries,
        "preparation": {
            "roles": dict(Counter(q["role"] for q in queries)),
            "original_query_counts": dict(per_role),
            "excluded_query_reasons": dict(excluded),
            "consonant_qc": dict(qc),
            "selection_uses_embeddings": False,
        },
    }


def prepare_inputs(config, run):
    if run.exists():
        raise ValueError("unused evaluation directory required")
    files = {}
    meta_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = {r["source_file"]: r for r in old.previous.read_rows(meta_path)}
    pin(files, meta_path)
    pin(files, ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")
    run.mkdir(parents=True)
    for split in ("validation", "test"):
        inputs = prepare_split(config, split, files, metadata)
        validate_plans(inputs)
        audit_source_slices(inputs, config)
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
        print(f"{split}: common8 {inputs['preparation']['roles']}", flush=True)
    compare_prior_inputs(config, run, files)
    for path in [CONFIG, *BASE.glob("*.py"), *BASE.glob("tests/*.py")]:
        pin(files, path)
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
                pin(files, path)
    write_json(
        run / "input-freeze.json",
        {
            "status": "inputs_frozen_before_inference",
            "config": config,
            "files": files,
            "runtime": old.previous.runtime(),
        },
    )


def baseline_item(item):
    return {
        **item,
        "conditions": {name: item["conditions"][name] for name in BASELINE_CONDITIONS},
    }


def compare_prior_inputs(config, run, files):
    previous = ROOT / config["prior_pair_run"]
    previous_freeze = read_json(previous / "design-freeze.json")
    for name, checksum in previous_freeze["files"].items():
        checked(ROOT / name, checksum)
        files[name] = checksum
    pin(files, previous / "design-freeze.json")
    checks = {}
    for split in ("validation", "test"):
        original = read_json(previous / f"{split}-inputs.json")
        extended = read_json(run / f"{split}-inputs.json")
        restored = {
            **extended,
            "profiles": {
                speaker: baseline_item(item)
                for speaker, item in extended["profiles"].items()
            },
            "queries": [baseline_item(item) for item in extended["queries"]],
        }
        if restored != original:
            raise ValueError("prior utterance support or baseline PCM plans changed")
        for name in (
            f"{split}-inputs.json",
            f"{split}/scores.jsonl",
            f"{split}/inference.json",
            f"{split}-metrics.json",
        ):
            pin(files, previous / name)
        checks[split] = {
            "queries": len(original["queries"]),
            "speakers": len(original["speakers"]),
            "baseline_plans_exactly_equal": True,
        }
    for name in ("validation-thresholds.json", "bootstrap-counts.npy"):
        pin(files, previous / name)
    write_json(run / "baseline-input-audit.json", {"status": "passed", **checks})
    pin(files, run / "baseline-input-audit.json")


def compare_prior_scores(config, run, split, scores):
    previous = ROOT / config["prior_pair_run"]
    original = {
        trial_key(row): row for row in read_scores(previous / split / "scores.jsonl")
    }
    restored = {
        trial_key(row): row for row in scores if row["condition"] in BASELINE_CONDITIONS
    }
    if set(original) != set(restored):
        raise ValueError("prior trial support changed")
    maximum = 0.0
    exact = 0
    for key, row in restored.items():
        old_row = original[key]
        if any(
            row[name] != old_row[name]
            for name in row
            if name not in ("score", "phone_scores")
        ) or set(row["phone_scores"]) != set(old_row["phone_scores"]):
            raise ValueError("prior labels or score components changed")
        errors = [abs(row["score"] - old_row["score"])] + [
            abs(value - old_row["phone_scores"][phone])
            for phone, value in row["phone_scores"].items()
        ]
        maximum = max(maximum, max(errors))
        exact += row == old_row
    if maximum > 1e-6:
        raise ValueError("re-inferred baseline differs from prior model scores")
    write_json(
        run / split / "baseline-score-audit.json",
        {
            "status": "passed",
            "baseline_score_rows": len(restored),
            "exactly_equal_rows": exact,
            "maximum_absolute_score_or_component_error": maximum,
            "baseline_scores_reused": False,
        },
    )


def prepare(config, run):
    fixed = read_json(run / "input-freeze.json")
    if fixed["config"] != config or fixed["runtime"] != old.previous.runtime():
        raise ValueError("input protocol/runtime changed")
    files = dict(fixed["files"])
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    if (run / "design-freeze.json").exists():
        raise ValueError("inference inputs already frozen")
    training = ROOT / config["training_run"]
    summary = read_json(training / "training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["test_used"]
        or summary["selected_update"] != 30000
        or summary["export_reload_bitwise_equal_phonemes"] != list(PHONEMES)
    ):
        raise ValueError("requires fixed completed eight-phone training")
    checked(training / "training-freeze.json", summary["freeze_sha256"])
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        checked(training / name, summary["outputs_sha256"][name])
        pin(files, training / name)
    for path in (
        training / "training/summary.json",
        training / "training-freeze.json",
        run / "input-freeze.json",
    ):
        pin(files, path)
    write_json(
        run / "design-freeze.json",
        {
            "status": "frozen_before_new_inference",
            "config": config,
            "files": files,
            "runtime": old.previous.runtime(),
        },
    )


def validate_plans(inputs):
    checks = 0
    enrollment_hashes = {
        row["source_sha256"]
        for profile in inputs["profiles"].values()
        for rows in profile["conditions"].values()
        for row in rows
    }
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        if (
            set(item["conditions"]) != set(CONDITIONS)
            or item.get("common8", True) is not True
        ):
            raise ValueError("incorrect fixed support/conditions")
        for condition, rows in item["conditions"].items():
            if not rows or {r["vowel"] for r in rows} != set(PHONE_SETS[condition]):
                raise ValueError("incomplete/wrong phonemes")
            if (
                sum(r["end_frame"] - r["start_frame"] for r in rows)
                != item["budget_frames"]
            ):
                raise ValueError("unequal actual PCM budgets")
            if len({r["original_segment_id"] for r in rows}) != len(rows):
                raise ValueError("repeated interval")
            grouped = defaultdict(list)
            for row in rows:
                if (
                    row["start_frame"] < 0
                    or not MINIMUM <= row["end_frame"] - row["start_frame"] <= 6000
                ):
                    raise ValueError("invalid source duration")
                if "common8" in item and row["source_sha256"] in enrollment_hashes:
                    raise ValueError("enrollment/query content leak")
                if (
                    "common8" not in item
                    and row["source_file"] not in item["candidate_source_files"]
                ):
                    raise ValueError("enrollment candidate source changed")
                grouped[row["source_file"]].append(
                    (row["start_frame"], row["end_frame"])
                )
            if any(
                a[1] > b[0]
                for ranges in grouped.values()
                for a, b in pairwise(sorted(ranges))
            ):
                raise ValueError("overlapping source slices")
            checks += 1
    return checks


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
                if original["vowel"] == "s":
                    candidate_size = min(
                        6000, original["end_frame"] - original["start_frame"]
                    )
                    candidate_first = (
                        original["start_frame"]
                        + (
                            original["end_frame"]
                            - original["start_frame"]
                            - candidate_size
                        )
                        // 2
                    )
                    original = {
                        **original,
                        "start_frame": candidate_first,
                        "end_frame": candidate_first + candidate_size,
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
                        "common8": query["common8"],
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
    compare_prior_scores(config, run, split, scores)
    frozen(config, run, split)
    print(f"{split}: {len(scores)} score slots, {len(vectors)} embeddings", flush=True)


def read_scores(path):
    with path.open() as stream:
        return list(map(json.loads, stream))


def trial_key(row):
    return row["query_id"], row["condition"], row["claimed_speaker_id"]


def validate_scores(inputs, rows):
    queries = {q["query_id"]: q for q in inputs["queries"]}
    seen = set()
    for row in rows:
        key = trial_key(row)
        query = queries.get(row["query_id"])
        if (
            query is None
            or key in seen
            or row["condition"] not in CONDITIONS
            or row["claimed_speaker_id"] not in inputs["speakers"]
        ):
            raise ValueError("invalid/duplicate trial identity")
        seen.add(key)
        if any(
            row[k] != query[k] for k in ("speaker_id", "role", "split", "common8")
        ) or row["is_genuine"] != (row["speaker_id"] == row["claimed_speaker_id"]):
            raise ValueError("trial labels changed")
        components = row["phone_scores"]
        if (
            row["status"] != "scored"
            or set(components) != set(PHONE_SETS[row["condition"]])
            or not all(np.isfinite(v) and -1 <= v <= 1 for v in components.values())
        ):
            raise ValueError("invalid component scores")
        if (
            row["score"] != float(np.mean(list(components.values())))
            or row["used_frames"] != query["budget_frames"]
        ):
            raise ValueError("score fusion/time mismatch")
    if len(seen) != len(queries) * len(CONDITIONS) * len(inputs["speakers"]):
        raise ValueError("incomplete five-condition matrix")


def load_groups(config, run, split):
    inputs = frozen(config, run, split)
    path = run / split / "scores.jsonl"
    checked(path, read_json(run / split / "inference.json")["scores_sha256"])
    rows = read_scores(path)
    validate_scores(inputs, rows)
    return {
        (condition, role): [
            r for r in rows if r["condition"] == condition and r["role"] == role
        ]
        for condition in CONDITIONS
        for role in ROLES
    }


def calibrate(config, run):
    inputs = frozen(config, run, "validation")
    groups = load_groups(config, run, "validation")
    thresholds, cells = {}, {}
    for condition in CONDITIONS:
        threshold = old.calibrate_cell(
            groups[condition, "verification"], inputs["speakers"]
        )
        if threshold["status"] != "calibrated":
            raise ValueError("missing validation support")
        thresholds[condition] = threshold
        for role in ROLES:
            cells[f"{condition}/{role}"], _ = old.evaluate_cell(
                groups[condition, role], threshold, inputs["speakers"]
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
    print("five validation thresholds frozen before new test inference", flush=True)


def evaluate(config, run):
    inputs = frozen(config, run, "test")
    groups = load_groups(config, run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, replicas, differences, arrays = {}, {}, {}, {}
    for (condition, role), rows in groups.items():
        key = f"{condition}/{role}"
        cells[key], _ = old.evaluate_cell(
            rows, thresholds[condition], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = old.bootstrap_cell(
            rows, cells[key], thresholds[condition], inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    names = [f"{p}/{m}" for p, _ in old.POINTS for m in old.METRICS] + [
        "pooled_eer",
        "query_coverage",
    ]
    for a, b in PAIRS:
        for role in ROLES:
            ka, kb = f"{a}/{role}", f"{b}/{role}"
            for name in names:
                key = f"{b}_minus_{a}/{role}/{name}"
                delta = 100 * (replicas[kb][name] - replicas[ka][name])
                arrays[key] = delta
                differences[key] = {
                    "difference_percentage_points": 100
                    * (
                        old.metric_value(cells[kb], name)
                        - old.metric_value(cells[ka], name)
                    ),
                    "ci95_percentage_points": old.interval(delta),
                    "minuend": b,
                    "subtrahend": a,
                }
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{f"{k}/{n}": v for k, items in replicas.items() for n, v in items.items()},
    )
    np.savez_compressed(run / "bootstrap-differences.npz", **arrays)
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "paired_differences": differences,
            "threshold_recalibrated_on_test": False,
            "bootstrap_replicates": config["bootstrap_replicates"],
            "shared_new_eight_phone_model": True,
        },
    )
    frozen(config, run, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "prepare", "validation", "test"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    if (
        config["conditions"] != list(CONDITIONS)
        or config["condition_phones"] != {k: list(v) for k, v in PHONE_SETS.items()}
        or config["phonemes"] != list(PHONEMES)
        or config["reuse_prior_baseline_scores"]
        or config["independent_holdout"]
    ):
        raise ValueError("unsupported pair/triple protocol")
    torch.set_num_threads(1)
    run = ROOT / config["run_directory"]
    if args.stage == "inputs":
        prepare_inputs(config, run)
    elif args.stage == "prepare":
        prepare(config, run)
    elif args.stage == "validation":
        infer(config, run, "validation")
        calibrate(config, run)
    else:
        infer(config, run, "test")
        evaluate(config, run)


if __name__ == "__main__":
    main()
