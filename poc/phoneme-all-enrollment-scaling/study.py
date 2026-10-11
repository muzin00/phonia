"""Measure enrollment limits 10/20/30 with the fixed, trained all36 encoder."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
GREEDY = BASE.parent / "phoneme-greedy-selection"
sys.path.insert(0, str(GREEDY))

from common import (
    PHASE3,
    checked,
    expanded,
    model_embeddings,
    np,
    pin,
    read_json,
    relative,
    rows,
    sha256_file,
    torch,
    write_json,
    write_rows,
)
from evaluate import metrics, scores_for, thresholds
from learner import load_model
from prepare import diverse

CONFIG = BASE / "config/protocol.json"
SPLITS = ("validation", "test")
ROLES = ("verification", "cross_text_verification")


def expanded_inputs(original, segments, limit, seed):
    """Retain all original phone/query support and expand real enrollment only."""
    query_sources = {q["source_file"] for q in original["queries"]}
    query_hashes = {
        row["source_sha256"] for row in segments if row["source_file"] in query_sources
    }
    pool = defaultdict(list)
    for row in segments:
        if row["role"] != "enrollment":
            continue
        if row["source_file"] in query_sources or row["source_sha256"] in query_hashes:
            raise ValueError("enrollment/query audio overlap")
        if row["split"] != original["split"] or row["quality_flags"]:
            raise ValueError("enrollment split or quality changed")
        pool[row["speaker_id"], row["phoneme"]].append(row)
    result = copy.deepcopy(original)
    for speaker, phones in original["profiles"].items():
        for phone in phones:
            candidates = pool[speaker, phone]
            if len({r["segment_id"] for r in candidates}) != len(candidates):
                raise ValueError("duplicate enrollment segment")
            result["profiles"][speaker][phone] = [
                r["cache_index"] for r in diverse(candidates, limit, seed)
            ]
    registered = {
        p
        for p in next(iter(result["profiles"].values()))
        if all(result["profiles"][s][p] for s in result["speakers"])
    }
    if registered != set(original["universally_registered"]):
        raise ValueError("registered phone support changed")
    if limit == 10 and result != original:
        raise ValueError("10-interval enrollment is not the original baseline")
    return result


def condition(run, limit):
    return run / f"enrollment-{limit}"


def verify(config, run):
    frozen = read_json(run / "design-freeze.json")
    if frozen["config"] != config or frozen["runtime"] != expanded.runtime():
        raise ValueError("fixed settings or numerical runtime changed")
    for name, checksum in frozen["files"].items():
        checked(ROOT / name, checksum)
    return frozen


def enrollment_summary(data, segments, phones, limit):
    used = set(phones) & set(data["universally_registered"])
    per_phone = {}
    for phone in phones:
        sizes = [len(data["profiles"][s][phone]) for s in data["speakers"]]
        per_phone[phone] = {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "mean": float(np.mean(sizes)),
            "speakers_below_limit": sum(n < limit for n in sizes),
            "used_for_scoring": phone in used,
        }
    seconds, counts, sources = [], [], []
    for speaker in data["speakers"]:
        indices = [i for p in used for i in data["profiles"][speaker][p]]
        seconds.append(
            sum(segments[i]["end_frame"] - segments[i]["start_frame"] for i in indices)
            / 24000
        )
        counts.append(len(indices))
        sources.append(len({segments[i]["source_file"] for i in indices}))
    return {
        "limit": limit,
        "used_registered_phonemes": len(used),
        "per_phone": per_phone,
        "used_intervals_total": sum(counts),
        "used_intervals_per_speaker_median": float(np.median(counts)),
        "used_audio_seconds_per_speaker_median": float(np.median(seconds)),
        "used_source_wavs_per_speaker_median": float(np.median(sources)),
    }


def prepare(config, run):
    if run.exists():
        raise ValueError("prepare requires an unused run directory")
    if config["enrollment_limits"] != [10, 20, 30] or config["retrain_encoder"]:
        raise ValueError("only fixed-model 10/20/30 comparison is supported")
    source = ROOT / config["source_run"]
    trial = source / "trials" / config["source_trial"]
    old_selection = read_json(source / "selection-freeze.json")
    checked(source / "design-freeze.json", old_selection["design_freeze_sha256"])
    old_design = read_json(source / "design-freeze.json")
    if old_design["runtime"] != expanded.runtime():
        raise ValueError("use the original verified numerical runtime")
    model, bundle = load_model(trial)
    if (
        len(bundle["phonemes"]) != 36
        or sha256_file(trial / "encoder.pt") != config["encoder_sha256"]
        or old_selection["models"][config["source_trial"]]["encoder_sha256"]
        != config["encoder_sha256"]
    ):
        raise ValueError("wrong frozen all36 encoder")
    del model
    files = {}
    for name, checksum in old_design["files"].items():
        path = ROOT / name
        if path.suffix == ".py" or (path.suffix == ".json" and "/config/" in name):
            checked(path, checksum)
            pin(files, path)
    for path in (
        source / "design-freeze.json",
        source / "selection-freeze.json",
        source / "completion-verification.json",
        source / "independent-audit.json",
        source / "publication-freeze.json",
        trial / "encoder.pt",
        trial / "training-summary.json",
        trial / "validation-thresholds.json",
        CONFIG,
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        PHASE3 / "scripts/weighted_bootstrap.py",
    ):
        pin(files, path)
    publication = read_json(source / "publication-freeze.json")
    checked(source / "independent-audit.json", publication["audit_sha256"])
    if read_json(source / "completion-verification.json")["status"] != "passed":
        raise ValueError("source study incomplete")
    run.mkdir(parents=True)
    counts = {}
    for split in SPLITS:
        original = read_json(source / f"{split}-inputs.json")
        segments = list(rows(source / f"{split}-segments.jsonl"))
        for name in ("inputs.json", "segments.jsonl", "features.f32"):
            path = source / f"{split}-{name}"
            checked(path, old_design["files"][relative(path)])
            pin(files, path)
        if len(segments) != original["feature_count"] or any(
            r["cache_index"] != i or r["split"] != split or r["quality_flags"]
            for i, r in enumerate(segments)
        ):
            raise ValueError("cached segment identities changed")
        wave_hashes = {}
        for row in segments:
            if (
                wave_hashes.setdefault(row["source_file"], row["source_sha256"])
                != row["source_sha256"]
            ):
                raise ValueError("source hash identity differs")
        for name in {r["alignment_file"] for r in segments} | set(wave_hashes):
            checked(ROOT / name, old_design["files"][name])
            pin(files, ROOT / name)
        if any(files[name] != sha for name, sha in wave_hashes.items()):
            raise ValueError("raw waveform hash differs")
        old_metrics = read_json(trial / f"{split}-metrics.json")
        if old_metrics["encoder_sha256"] != config["encoder_sha256"]:
            raise ValueError("baseline scores use a different encoder")
        for path, checksum in (
            (source / f"{split}-inputs.json", old_metrics["input_sha256"]),
            (trial / f"{split}-scores.jsonl", old_metrics["scores_sha256"]),
            (trial / f"{split}-embeddings.npy", old_metrics["embeddings_sha256"]),
        ):
            checked(path, checksum)
            pin(files, path)
        pin(files, trial / f"{split}-metrics.json")
        counts[split] = {}
        previous = None
        for limit in config["enrollment_limits"]:
            data = expanded_inputs(original, segments, limit, config["enrollment_seed"])
            if previous:
                for speaker, ps in previous["profiles"].items():
                    for phone, ids in ps.items():
                        if data["profiles"][speaker][phone][: len(ids)] != ids:
                            raise ValueError("enrollment sets are not nested")
            previous = data
            directory = condition(run, limit)
            directory.mkdir(exist_ok=True)
            write_json(directory / f"{split}-inputs.json", data)
            pin(files, directory / f"{split}-inputs.json")
            counts[split][str(limit)] = enrollment_summary(
                data, segments, bundle["phonemes"], limit
            )
    write_json(
        run / "design-freeze.json",
        {
            "status": "all_enrollment_limits_and_inputs_frozen_before_new_scores",
            "config": config,
            "runtime": expanded.runtime(),
            "phonemes": bundle["phonemes"],
            "untrained_phonemes": ["ty"],
            "model_retrained": False,
            "original_test_previously_observed": True,
            "enrollment": counts,
            "files": files,
        },
    )
    print(json.dumps({"stage": "prepared", "pinned_files": len(files)}), flush=True)


def score_distribution(values):
    result = {}
    for role in ROLES:
        result[role] = {}
        for genuine, label in ((True, "genuine"), (False, "impostor")):
            scores = np.array(
                [
                    r["score"]
                    for r in values
                    if r["role"] == role
                    and r["status"] == "scored"
                    and r["is_genuine"] == genuine
                ]
            )
            result[role][label] = {
                "count": len(scores),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "p05": float(np.percentile(scores, 5)),
                "p95": float(np.percentile(scores, 95)),
            }
        result[role]["mean_separation"] = (
            result[role]["genuine"]["mean"] - result[role]["impostor"]["mean"]
        )
    return result


def evaluate(config, run):
    frozen = verify(config, run)
    source = ROOT / config["source_run"]
    trial = source / "trials" / config["source_trial"]
    model, bundle = load_model(trial)
    for split in SPLITS:
        if split == "test":
            selection = read_json(run / "threshold-freeze.json")
            checked(run / "design-freeze.json", selection["design_freeze_sha256"])
            for name, sha in selection["files"].items():
                checked(ROOT / name, sha)
        data = read_json(source / f"{split}-inputs.json")
        features = np.memmap(
            source / f"{split}-features.f32",
            mode="r",
            dtype="<f4",
            shape=(data["feature_count"], 128),
        )
        vectors = model_embeddings(model, features)
        if not np.array_equal(
            vectors, np.load(trial / f"{split}-embeddings.npy", allow_pickle=False)
        ):
            raise ValueError("fixed encoder inference differs from original embeddings")
        for limit in config["enrollment_limits"]:
            directory = condition(run, limit)
            data = read_json(directory / f"{split}-inputs.json")
            values = scores_for(data, vectors, bundle["phonemes"])
            points = (
                thresholds(values)
                if split == "validation"
                else read_json(directory / "validation-thresholds.json")["thresholds"]
            )
            result = metrics(values, points)
            if limit == 10:
                if values != list(rows(trial / f"{split}-scores.jsonl")):
                    raise ValueError("10-interval scores do not reproduce the baseline")
                if result != read_json(trial / f"{split}-metrics.json")["metrics"]:
                    raise ValueError(
                        "10-interval metrics do not reproduce the baseline"
                    )
                if (
                    points
                    != read_json(trial / "validation-thresholds.json")["thresholds"]
                ):
                    raise ValueError(
                        "10-interval thresholds do not reproduce the baseline"
                    )
            np.save(directory / f"{split}-embeddings.npy", vectors, allow_pickle=False)
            score_path = directory / f"{split}-scores.jsonl"
            if score_path.exists():
                if list(rows(score_path)) != values:
                    raise ValueError("existing scores differ on repeat")
            else:
                write_rows(score_path, values)
            if split == "validation":
                write_json(
                    directory / "validation-thresholds.json",
                    {"split": split, "role": "verification", "thresholds": points},
                )
            write_json(
                directory / f"{split}-metrics.json",
                {
                    "split": split,
                    "enrollment_limit": limit,
                    "phonemes": frozen["phonemes"],
                    "encoder_sha256": config["encoder_sha256"],
                    "input_sha256": sha256_file(directory / f"{split}-inputs.json"),
                    "scores_sha256": sha256_file(score_path),
                    "embeddings_sha256": sha256_file(
                        directory / f"{split}-embeddings.npy"
                    ),
                    "metrics": result,
                    "score_distribution": score_distribution(values),
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "evaluated",
                        "split": split,
                        "limit": limit,
                        "eer_pct": {role: 100 * result[role]["eer"] for role in ROLES},
                    }
                ),
                flush=True,
            )
        if split == "validation":
            files = {}
            for limit in config["enrollment_limits"]:
                for name in ("validation-thresholds.json", "validation-metrics.json"):
                    pin(files, condition(run, limit) / name)
            write_json(
                run / "threshold-freeze.json",
                {
                    "status": "all_validation_thresholds_frozen_before_new_test_scores",
                    "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
                    "test_used_for_selection": False,
                    "files": files,
                },
            )


def audit(config, run):
    from audit import audit_trial, independent_eer, speaker_draws, weighted_eers

    frozen = verify(config, run)
    selection = read_json(run / "threshold-freeze.json")
    checked(run / "design-freeze.json", selection["design_freeze_sha256"])
    for name, sha in selection["files"].items():
        checked(ROOT / name, sha)
    totals = {"scores": 0, "eer_checks": 0, "rate_checks": 0}
    max_error, score_distribution_checks = 0.0, 0
    source = ROOT / config["source_run"]
    for split in SPLITS:
        original = read_json(source / f"{split}-inputs.json")
        segments = list(rows(source / f"{split}-segments.jsonl"))
        for limit in config["enrollment_limits"]:
            directory = condition(run, limit)
            data = read_json(directory / f"{split}-inputs.json")
            if data != expanded_inputs(
                original, segments, limit, config["enrollment_seed"]
            ):
                raise ValueError("enrollment input changed")
            if (
                enrollment_summary(data, segments, frozen["phonemes"], limit)
                != frozen["enrollment"][split][str(limit)]
            ):
                raise ValueError("actual enrollment counts or durations changed")
            result = read_json(directory / f"{split}-metrics.json")
            checked(directory / f"{split}-inputs.json", result["input_sha256"])
            if (
                result["phonemes"] != frozen["phonemes"]
                or result["encoder_sha256"] != config["encoder_sha256"]
            ):
                raise ValueError("scoring encoder/phones changed")
            inspected = audit_trial(directory, directory, split, result)
            for key in totals:
                totals[key] += inspected[key]
            max_error = max(max_error, inspected["max_score_error"])
            values = list(rows(directory / f"{split}-scores.jsonl"))
            if score_distribution(values) != result["score_distribution"]:
                raise ValueError("score distributions changed")
            score_distribution_checks += 2
    speakers = read_json(condition(run, 10) / "test-inputs.json")["speakers"]
    indices, counts = speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    lookup = {s: i for i, s in enumerate(speakers)}
    estimates, checks, intervals = {}, 0, {}
    for limit in config["enrollment_limits"]:
        values = list(rows(condition(run, limit) / "test-scores.jsonl"))
        for role in ROLES:
            scored = [
                r for r in values if r["role"] == role and r["status"] == "scored"
            ]
            scores = np.array([r["score"] for r in scored])
            q = np.array([lookup[r["speaker_id"]] for r in scored])
            c = np.array([lookup[r["claimed_speaker_id"]] for r in scored])
            eers = weighted_eers(scores, q, c, counts)
            for sample in range(0, len(counts), max(1, len(counts) // 40)):
                weights = np.where(
                    q == c, counts[sample, q], counts[sample, q] * counts[sample, c]
                )
                if abs(independent_eer(scores, q == c, weights) - eers[sample]) > 1e-12:
                    raise ValueError("independent bootstrap differs")
                checks += 1
            estimates[f"{limit}/{role}"] = eers
    for later, earlier in ((20, 10), (30, 10), (30, 20)):
        for role in ROLES:
            key = f"{later}-{earlier}/{role}"
            delta = estimates[f"{later}/{role}"] - estimates[f"{earlier}/{role}"]
            estimates[key] = delta
            intervals[key] = {
                "eer_difference": read_json(
                    condition(run, later) / "test-metrics.json"
                )["metrics"][role]["eer"]
                - read_json(condition(run, earlier) / "test-metrics.json")["metrics"][
                    role
                ]["eer"],
                "eer_difference95": np.percentile(delta, [2.5, 97.5]).tolist(),
            }
    np.savez_compressed(
        run / "bootstrap.npz",
        speaker_indices=indices,
        speaker_counts=counts,
        **estimates,
    )
    write_json(run / "bootstrap-results.json", intervals)
    report = {
        "status": "passed",
        "model_retrained": False,
        "limits": config["enrollment_limits"],
        "baseline_10_exactly_reproduced": True,
        "nested_enrollment_and_fixed_queries_verified": True,
        "independent_max_score_error": max_error,
        "independent_bootstrap_checks": checks,
        "score_distribution_checks": score_distribution_checks,
        **totals,
    }
    write_json(run / "independent-audit.json", report)
    outputs = {}
    for directory in [condition(run, n) for n in config["enrollment_limits"]]:
        for path in directory.iterdir():
            if path.is_file():
                pin(outputs, path)
    for name in (
        "threshold-freeze.json",
        "independent-audit.json",
        "bootstrap.npz",
        "bootstrap-results.json",
    ):
        pin(outputs, run / name)
    write_json(
        run / "completion-verification.json", {**report, "output_sha256": outputs}
    )
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "audit", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    if args.stage == "verify":
        verify(config, run)
        completion = read_json(run / "completion-verification.json")
        for name, sha in completion["output_sha256"].items():
            checked(ROOT / name, sha)
        if (run / "publication-freeze.json").exists():
            publication = read_json(run / "publication-freeze.json")
            for name, sha in publication["files"].items():
                checked(ROOT / name, sha)
        print(
            "fixed model, source audio, inputs, scores and publication hashes verified",
            flush=True,
        )
    else:
        globals()[args.stage](config, run)


if __name__ == "__main__":
    main()
