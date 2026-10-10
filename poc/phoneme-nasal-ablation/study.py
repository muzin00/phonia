"""Four fixed-time input ablations of the same frozen seven-phone encoder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import wave
from collections import Counter, defaultdict
from itertools import combinations, pairwise
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR_BASE = BASE.parent / "phoneme-consonant-evaluation"
sys.path.insert(0, str(PRIOR_BASE))
spec = importlib.util.spec_from_file_location(
    "consonant_study", PRIOR_BASE / "study.py"
)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

import numpy as np
import torch
from matched_audio import MINIMUM, PHONEMES, VOWELS, plan

old = prior.old
read_json, write_json, checked, sha256_file, pin = (
    prior.read_json,
    prior.write_json,
    prior.checked,
    prior.sha256_file,
    prior.pin,
)
CONFIG = BASE / "config/protocol.json"
CONDITIONS = ("vowels5", "vowels_m", "vowels_n", "vowels_mn_available")
PHONE_SETS = dict(zip(CONDITIONS, (VOWELS, (*VOWELS, "m"), (*VOWELS, "n"), PHONEMES)))
BASELINES = (CONDITIONS[0], CONDITIONS[-1])
ROLES = old.ROLES
PAIRS = tuple(combinations(CONDITIONS, 2))
LABELS = dict(zip(CONDITIONS, ("5母音", "5母音＋m", "5母音＋n", "5母音＋m/n")))
frozen = prior.frozen


def extract_nasals(source, metadata, split, speaker, role, config, files):
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
        for k in ("utterance_id", "speaker_id", "source_file", "source_sha256")
    ):
        raise ValueError("alignment metadata mismatch")
    if str(path.relative_to(ROOT)) not in files or source not in files:
        raise ValueError("candidate was not in prior frozen sources")
    checked(path, files[str(path.relative_to(ROOT))])
    checked(ROOT / source, meta["source_sha256"])
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
    rows = []
    for index, interval in enumerate(raw["intervals"]):
        phone = interval["phoneme"]
        if phone not in ("m", "n"):
            continue
        first, last = (
            round(interval["start_sec"] * 24000),
            round(interval["end_sec"] * 24000),
        )
        if not 0 <= first < last <= len(samples):
            raise ValueError("invalid nasal bounds")
        rms = float(np.sqrt(np.mean(samples[first:last] ** 2)))
        if last - first < MINIMUM or rms < 10 ** (
            config["minimum_nasal_rms_dbfs"] / 20
        ):
            continue
        rows.append(
            {
                "segment_id": f"{meta['utterance_id']}--phone-{index:03d}-{phone}",
                "vowel": phone,
                "source_file": source,
                "source_sha256": meta["source_sha256"],
                "start_frame": first,
                "end_frame": last,
            }
        )
    return rows


def four_plans(candidates, budget, baseline):
    result = {
        condition: plan(candidates, phones, budget)
        for condition, phones in PHONE_SETS.items()
    }
    if any(rows is None for rows in result.values()):
        raise ValueError(
            "all four conditions must fit unchanged support and time budget"
        )
    for condition in BASELINES:
        if result[condition] != baseline[condition]:
            raise ValueError("prior baseline source slices changed")
    return result


def prepare_split(config, split, files, metadata):
    reference = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
    original = read_json(reference)
    baseline_path = ROOT / config["prior_run"] / f"{split}-inputs.json"
    baseline = read_json(baseline_path)
    pin(files, reference)
    pin(files, baseline_path)
    labels = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    if (
        original["speakers"] != baseline["speakers"]
        or set(baseline["speakers"]) != set(labels[split])
        or set(baseline["speakers"]) & set(labels["train"])
    ):
        raise ValueError("speaker support/leak mismatch")
    cache = {}

    def nasals(source, speaker, role):
        if source not in cache:
            cache[source] = extract_nasals(
                source, metadata, split, speaker, role, config, files
            )
        return cache[source]

    profiles = {}
    for enrollment in original["enrollment"]:
        speaker, vowels = enrollment["user_id"], enrollment["segments"]
        base = baseline["profiles"][speaker]
        sources = sorted({s["source_file"] for s in vowels})
        if sources != base["candidate_source_files"]:
            raise ValueError("enrollment WAV candidates changed")
        pool = [
            row for source in sources for row in nasals(source, speaker, "enrollment")
        ]
        extras = [
            row
            for phone in ("m", "n")
            for row in prior.diversify(
                [r for r in pool if r["vowel"] == phone],
                config["nasal_enrollment_candidates_per_phone"],
                config["selection_seed"],
            )
        ]
        profiles[speaker] = {
            **base,
            "conditions": four_plans(
                vowels + extras, base["budget_frames"], base["conditions"]
            ),
        }
    original_queries = {q["query_id"]: q for q in original["queries"]}
    queries = []
    for base in baseline["queries"]:
        if not base["common7"]:
            continue
        source = original_queries[base["query_id"]]
        candidates = source["segments"] + nasals(
            source["source_file"], source["speaker_id"], source["role"]
        )
        queries.append(
            {
                **base,
                "conditions": four_plans(
                    candidates, base["budget_frames"], base["conditions"]
                ),
            }
        )
    counts = Counter(q["role"] for q in queries)
    if dict(counts) != config["expected_queries"][split]:
        raise ValueError("prior common7 query support changed")
    return {
        "split": split,
        "speakers": baseline["speakers"],
        "profiles": profiles,
        "queries": queries,
        "preparation": {
            "roles": dict(counts),
            "prior_query_ids_unchanged": True,
            "prior_baseline_slices_identical": True,
            "no_fallback_or_new_cohort": True,
        },
    }


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
            or item.get("common7", True) is not True
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
                if "common7" in item and row["source_sha256"] in enrollment_hashes:
                    raise ValueError("enrollment/query content leak")
                if (
                    "common7" not in item
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


def prepare(config, run):
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused artifact directory required")
    prior_run = ROOT / config["prior_run"]
    checked(prior_run / "design-freeze.json", config["prior_design_sha256"])
    prior_config = read_json(prior_run / "design-freeze.json")["config"]
    for key in (
        "training_run",
        "reference_inputs",
        "enrollment_cap_frames",
        "query_cap_frames",
        "minimum_nasal_rms_dbfs",
        "nasal_enrollment_candidates_per_phone",
        "selection_seed",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        if config[key] != prior_config[key]:
            raise ValueError(f"prior condition changed: {key}")
    prior.frozen(prior_config, prior_run, "test")
    files = dict(read_json(prior_run / "design-freeze.json")["files"])
    pin(files, prior_run / "design-freeze.json")
    pin(files, prior_run / "evaluation-freeze.json")
    for name in (
        "validation-thresholds.json",
        "validation-metrics.json",
        "test-metrics.json",
        "bootstrap-counts.npy",
    ):
        pin(files, prior_run / name)
    for split in ("validation", "test"):
        for name in ("scores.jsonl", "inference.json"):
            path = prior_run / split / name
            if name == "scores.jsonl":
                checked(
                    path,
                    read_json(prior_run / split / "inference.json")["scores_sha256"],
                )
            pin(files, path)
    meta_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    with meta_path.open() as stream:
        metadata = {r["source_file"]: r for r in map(json.loads, stream)}
    run.mkdir(parents=True)
    for split in ("validation", "test"):
        inputs = prepare_split(config, split, files, metadata)
        validate_plans(inputs)
        prior.audit_source_slices(inputs, config)
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
        print(f"{split}: fixed support {inputs['preparation']['roles']}", flush=True)
    for path in [CONFIG, *BASE.glob("*.py"), *BASE.glob("tests/*.py")]:
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


def read_scores(path):
    with path.open() as stream:
        return list(map(json.loads, stream))


def trial_key(row):
    return row["query_id"], row["condition"], row["claimed_speaker_id"]


def infer(config, run, split):
    # The prior inference function is condition-agnostic: it consumes the frozen plans.
    prior.infer(config, run, split)
    inputs = frozen(config, run, split)
    query_ids = {q["query_id"] for q in inputs["queries"]}
    baseline = {
        trial_key(r): r
        for r in read_scores(ROOT / config["prior_run"] / split / "scores.jsonl")
        if r["query_id"] in query_ids and r["common7"]
    }
    path = run / split / "scores.jsonl"
    rows = read_scores(path)
    maximum, replaced = 0.0, 0
    for i, row in enumerate(rows):
        if row["condition"] not in BASELINES:
            continue
        reference = baseline[trial_key(row)]
        if any(
            row[k] != reference[k] for k in row if k not in ("score", "phone_scores")
        ) or set(row["phone_scores"]) != set(reference["phone_scores"]):
            raise ValueError("baseline trial metadata changed")
        error = max(
            abs(row["score"] - reference["score"]),
            *(
                abs(row["phone_scores"][p] - reference["phone_scores"][p])
                for p in reference["phone_scores"]
            ),
        )
        maximum = max(maximum, error)
        if error > config["baseline_reproduction_absolute_tolerance"]:
            raise ValueError("prior baseline inference not reproduced")
        rows[i] = reference
        replaced += 1
    if replaced != len(baseline):
        raise ValueError("incomplete baseline score reuse")
    validate_scores(inputs, rows)
    raw_scores = path.with_name("reproduced-scores.jsonl")
    path.rename(raw_scores)
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    inference_path = run / split / "inference.json"
    audit = read_json(inference_path)
    audit.update(
        scores_sha256=sha256_file(path),
        reused_prior_score_rows=replaced,
        baseline_reproduction_max_absolute_error=maximum,
        baseline_reproduction_tolerance=config[
            "baseline_reproduction_absolute_tolerance"
        ],
        reproduced_scores_sha256=sha256_file(raw_scores),
    )
    inference_path.rename(inference_path.with_name("raw-inference.json"))
    write_json(inference_path, audit)
    frozen(config, run, split)


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
            row[k] != query[k] for k in ("speaker_id", "role", "split", "common7")
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
        raise ValueError("incomplete four-condition matrix")


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
    reference = read_json(ROOT / config["prior_run"] / "validation-thresholds.json")
    for condition in CONDITIONS:
        threshold = old.calibrate_cell(
            groups[condition, "verification"], inputs["speakers"]
        )
        if threshold["status"] != "calibrated":
            raise ValueError("missing validation support")
        if condition in BASELINES and threshold != reference[f"common7/{condition}"]:
            raise ValueError("prior baseline calibration changed")
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
    if not np.array_equal(
        counts,
        np.load(
            ROOT / config["prior_run"] / "bootstrap-counts.npy", allow_pickle=False
        ),
    ):
        raise ValueError("paired bootstrap draws changed")
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
    print("four validation thresholds frozen before new test inference", flush=True)


def evaluate(config, run):
    inputs = frozen(config, run, "test")
    groups = load_groups(config, run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    reference = read_json(ROOT / config["prior_run"] / "test-metrics.json")
    cells, replicas, differences, arrays = {}, {}, {}, {}
    for (condition, role), rows in groups.items():
        key = f"{condition}/{role}"
        cells[key], _ = old.evaluate_cell(
            rows, thresholds[condition], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = old.bootstrap_cell(
            rows, cells[key], thresholds[condition], inputs["speakers"], counts
        )
        if (
            condition in BASELINES
            and cells[key] != reference["conditions"][f"common7/{key}"]
        ):
            raise ValueError("prior baseline test metrics/CI changed")
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
            "prior_baseline_metrics_and_ci_identical": True,
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
        or config["condition_phones"] != {k: list(v) for k, v in PHONE_SETS.items()}
        or config["operating_points"] != [p for p, _ in old.POINTS]
        or config["retrain"]
        or config["independent_holdout"]
        or not config["reuse_prior_baseline_scores"]
    ):
        raise ValueError("unsupported ablation protocol")
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
