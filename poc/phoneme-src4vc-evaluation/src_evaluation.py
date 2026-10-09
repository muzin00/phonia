"""Evaluate a fixed SRC4VC endpoint against two checksum-pinned references."""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-training-exposure-ablation"))
import exposure_evaluation as reference_study
import numpy as np

old = reference_study.old
ROOT = old.ROOT
read_json, checked, sha256_file, write_json = (
    old.read_json,
    old.checked,
    old.sha256_file,
    old.write_json,
)
CONFIG = BASE / "config/protocol.json"
CONDITIONS = ("jvs70_u15000", "cv70_u30000", "src70_u30000")
REFERENCES = {"jvs70_u15000": "jvs70_u15000", "cv70_u30000": "cv70_u30000"}
ROLES, POINTS, METRICS = old.ROLES, old.POINTS, old.METRICS


def pin(files, path):
    files[str(path.relative_to(ROOT))] = sha256_file(path)


def validate_protocol(config):
    if (
        config["conditions"] != list(CONDITIONS)
        or config["primary_comparison"] != "src70_u30000_minus_cv70_u30000"
        or config["enrollment_count_per_vowel"] != 10
        or config["required_vowels"] != list(old.VOWELS)
        or config["roles"] != list(ROLES)
        or config["query_window"] != "full"
        or config["calibration"] != "pooled_validation_verification_per_model"
        or config["operating_points"] != [p for p, _ in POINTS]
        or config["acceptance"] != "score_gte_threshold"
        or config["no_score"] != "reject_and_keep_in_all_input_denominators"
        or config["fusion"]
        != "mean_segment_cosine_within_vowel_then_equal_mean_over_five_vowels"
        or config["retrain_encoder"]
        or config["update_feature_statistics"]
        or config["test_threshold_recalibration"]
        or config["independent_holdout"]
        or not config["reuse_baseline_scores"]
        or config["repeats_per_unique_embedding"] != 3
        or config["bootstrap_replicates"] != 10000
        or config["bootstrap_seed"] != 20260929
        or config["new_inference_conditions"] != 1
        or config["checkpoint_selection"] != "fixed_30000_selected_before_training"
    ):
        raise ValueError("unsupported fixed SRC4VC utterance evaluation protocol")


def prepare(run):
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused artifact evaluation directory required")
    config = read_json(CONFIG)
    validate_protocol(config)
    prior = ROOT / config["source_evaluation_run"]
    checked(prior / "study-report.json", config["source_evaluation_report_sha256"])
    report = read_json(prior / "study-report.json")
    freeze = read_json(prior / "design-freeze.json")
    if report["status"] != "completed" or freeze["runtime"] != old.previous.runtime():
        raise ValueError("reference evaluation incomplete/runtime changed")
    files = dict(freeze["files"])
    for name, checksum in report["outputs_sha256"].items():
        checked(prior / name, checksum)
        files[str((prior / name).relative_to(ROOT))] = checksum
    pin(files, prior / "study-report.json")
    training = ROOT / config["training_run"]
    summary_path = training / "expanded/training/summary.json"
    checked(summary_path, config["training_summary_sha256"])
    summary = read_json(summary_path)
    if (
        summary["status"] != "completed"
        or summary["selected_update"] != 30000
        or summary["test_used"]
        or summary["selection_rule"] != config["checkpoint_selection"]
    ):
        raise ValueError("requires completed fixed 30,000 SRC4VC training endpoint")
    training_freeze = read_json(training / "training-freeze.json")
    descriptor = read_json(training / "expanded/run.json")
    if descriptor["freeze_sha256"] != sha256_file(training / "training-freeze.json"):
        raise ValueError("training descriptor/freeze disagree")
    for name, checksum in training_freeze["files"].items():
        if name in files and files[name] != checksum:
            raise ValueError("reference/new training input checksum conflict")
        files[name] = checksum
    for name, checksum in summary["outputs_sha256"].items():
        checked(training / "expanded" / name, checksum)
        files[str((training / "expanded" / name).relative_to(ROOT))] = checksum
    for path in (
        CONFIG,
        summary_path,
        training / "training-freeze.json",
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
    ):
        pin(files, path)
    prepared = {}
    split_labels = read_json(
        ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    )["speaker_splits"]
    for split in ("validation", "test"):
        inputs = read_json(prior / f"{split}-inputs.json")
        inputs["config"] = config
        old.validate_population(inputs)
        if set(inputs["speakers"]) != set(split_labels[split]) or set(
            inputs["speakers"]
        ) & set(split_labels["train"]):
            raise ValueError("JVS train/evaluation split mismatch")
        prepared[split] = inputs
    if set(prepared["validation"]["speakers"]) & set(prepared["test"]["speakers"]):
        raise ValueError("speaker split overlap")
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    run.mkdir(parents=True)
    for split, inputs in prepared.items():
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "design_frozen_before_new_inference",
            "config": config,
            "runtime": old.previous.runtime(),
            "files": files,
            "references_reused": list(REFERENCES),
            "new_inference_conditions": ["src70_u30000"],
            "primary_comparison": config["primary_comparison"],
            "test_previously_observed": True,
            "src4vc_reserved_speakers_evaluated": False,
        },
    )


def infer(run, split):
    from src_inference import infer_new

    inputs = frozen_inputs(run, split)
    infer_new(run, split, inputs)
    report = read_json(run / split / "jvs70_src4vc70/report.json")
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    original = read_json(prior / split / "jvs70_cv70/report.json")
    a, b = report["encoder_identity"], original["encoder_identity"]
    if (
        a["preprocessing_sha256"] != b["preprocessing_sha256"]
        or a["implementation_sha256"] != b["implementation_sha256"]
        or a["checkpoint_sha256"] == b["checkpoint_sha256"]
        or a["feature_statistics_sha256"] == b["feature_statistics_sha256"]
    ):
        raise ValueError(
            "expected new weights/statistics with unchanged preprocessing and encoder implementation"
        )
    frozen_inputs(run, split)


def score_groups(run, split):
    inputs = read_json(run / f"{split}-inputs.json")
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    old_inputs, reference_groups = reference_study.score_groups(prior, split)
    if any(
        inputs[k] != old_inputs[k]
        for k in ("speakers", "enrollment", "queries", "sources", "trials")
    ):
        raise ValueError("reference/new input populations differ")
    groups = {
        (condition, role): reference_groups[condition, role]
        for condition in REFERENCES
        for role in ROLES
    }
    rows = new_rows(run, split, inputs)
    for role in ROLES:
        groups["src70_u30000", role] = [r for r in rows if r["role"] == role]
        expected = {r["trial_id"]: r["status"] for r in groups["cv70_u30000", role]}
        actual = {r["trial_id"]: r["status"] for r in groups["src70_u30000", role]}
        if expected != actual:
            raise ValueError("model comparison has different score coverage")
    return inputs, groups


def frozen_inputs(run, split):
    if split not in ("validation", "test"):
        raise ValueError("validation/test only")
    freeze = read_json(run / "design-freeze.json")
    if (
        freeze["status"] != "design_frozen_before_new_inference"
        or freeze["runtime"] != old.previous.runtime()
    ):
        raise ValueError("frozen runtime/design changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    inputs = read_json(run / f"{split}-inputs.json")
    if inputs["split"] != split:
        raise ValueError("input split mismatch")
    if split == "test":
        evaluation = read_json(run / "evaluation-freeze.json")
        if evaluation["status"] != "thresholds_frozen_before_new_test_inference":
            raise ValueError("validation threshold freeze required")
        for name, checksum in evaluation["files"].items():
            checked(run / name, checksum)
    return inputs


def new_rows(run, split, inputs):
    output = run / split / "jvs70_src4vc70"
    report = read_json(output / "report.json")
    if report["status"] != "completed":
        raise ValueError("incomplete new inference")
    for name, checksum in report["outputs_sha256"].items():
        checked(output / name, checksum)
    expected = {r["trial_id"]: r for r in inputs["trials"]}
    seen = set()
    result = []
    for row in old.iter_rows(output / "scores.jsonl.gz"):
        trial = expected.get(row["trial_id"])
        if (
            trial is None
            or row["trial_id"] in seen
            or row["model_id"] != "jvs70_src4vc70"
            or any(row[k] != v for k, v in trial.items())
        ):
            raise ValueError("new trial identity/label mismatch")
        seen.add(row["trial_id"])
        values = row["scores"]
        if row["status"] not in ("scored", "no_score") or (values is None) != (
            row["status"] == "no_score"
        ):
            raise ValueError("invalid new score state")
        if values is not None and (
            set(values) != {"fused", *old.VOWELS}
            or not all(np.isfinite(v) and -1 <= v <= 1 for v in values.values())
            or values["fused"]
            != float(np.mean([values[v] for v in old.VOWELS], dtype=np.float64))
        ):
            raise ValueError("invalid new fused score")
        result.append({**row, "score": values["fused"] if values else None})
    if seen != set(expected):
        raise ValueError("missing new trial")
    return result


def calibrate(run):
    inputs, groups = score_groups(run, "validation")
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    original = read_json(prior / "validation-thresholds.json")["conditions"]
    thresholds, cells, curves = {}, {}, {}
    for condition in CONDITIONS:
        threshold = old.calibrate_cell(
            groups[condition, "verification"], inputs["speakers"]
        )
        if threshold["status"] != "calibrated":
            raise ValueError("validation support insufficient")
        if (
            condition in REFERENCES
            and threshold["operating_points"] != original[condition]["operating_points"]
        ):
            raise ValueError("reference threshold parity failed")
        thresholds[condition] = threshold
        for role in ROLES:
            key = f"{condition}/{role}"
            cells[key], curves[key] = old.evaluate_cell(
                groups[condition, role], threshold, inputs["speakers"]
            )
    write_json(
        run / "validation-thresholds.json",
        {"split": "validation", "role": "verification", "conditions": thresholds},
    )
    write_json(run / "validation-metrics.json", {"conditions": cells, "curves": curves})
    counts = np.load(prior / "bootstrap-counts.npy", allow_pickle=False)
    _, expected = old.speaker_draws(15, 10000, 20260929)
    if not np.array_equal(counts, expected):
        raise ValueError("bootstrap draws changed")
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    names = [
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-counts.npy",
        "validation/jvs70_src4vc70/report.json",
    ]
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_new_test_inference",
            "test_previously_observed": True,
            "files": {n: sha256_file(run / n) for n in names},
        },
    )


def paired(cells, replicas):
    differences, arrays = {}, {}
    for reference in REFERENCES:
        comparison = f"src70_u30000_minus_{reference}"
        for role in ROLES:
            a, b = f"src70_u30000/{role}", f"{reference}/{role}"
            for name in [f"{p}/{m}" for p, _ in POINTS for m in METRICS] + [
                "pooled_eer",
                "query_coverage",
            ]:
                key = f"{comparison}/{role}/{name}"
                delta = 100 * (replicas[a][name] - replicas[b][name])
                arrays[key] = delta
                differences[key] = {
                    "difference_percentage_points": 100
                    * (
                        old.metric_value(cells[a], name)
                        - old.metric_value(cells[b], name)
                    ),
                    "ci95_percentage_points": old.interval(delta),
                    "primary": reference == "cv70_u30000",
                }
    return differences, arrays


def evaluate(run):
    frozen_inputs(run, "test")
    inputs, groups = score_groups(run, "test")
    thresholds = read_json(run / "validation-thresholds.json")["conditions"]
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, curves, replicas = {}, {}, {}
    for (condition, role), rows in sorted(groups.items()):
        key = f"{condition}/{role}"
        cells[key], curves[key] = old.evaluate_cell(
            rows, thresholds[condition], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = old.bootstrap_cell(
            rows, cells[key], thresholds[condition], inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    original = read_json(prior / "test-metrics.json")["conditions"]
    for condition in REFERENCES:
        for role in ROLES:
            if cells[f"{condition}/{role}"] != original[f"{condition}/{role}"]:
                raise ValueError("reference metrics and CI parity failed")
    differences, arrays = paired(cells, replicas)
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{
            f"{key}/{name}": value
            for key, items in replicas.items()
            for name, value in items.items()
        },
    )
    np.savez_compressed(run / "bootstrap-paired-differences.npz", **arrays)
    _, validation_groups = score_groups(run, "validation")
    validation_cells = read_json(run / "validation-metrics.json")["conditions"]
    audits = 0
    for condition in CONDITIONS:
        for rows, measures in ((groups, cells), (validation_groups, validation_cells)):
            audits += old.previous.audit_rates(
                {(10, role): rows[condition, role] for role in ROLES},
                {f"n10/{role}": measures[f"{condition}/{role}"] for role in ROLES},
                {"n10": thresholds[condition]},
            )
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "curves": curves,
            "paired_differences": differences,
            "threshold_recalibrated": False,
            "bootstrap_replicates": 10000,
            "reference_metric_ci_parity_cells": 4,
            "independent_rate_audits": audits,
        },
    )
    frozen_inputs(run, "test")


def audit_intervals(run):
    metrics = read_json(run / "test-metrics.json")
    checks = 0
    with np.load(run / "bootstrap-metrics.npz", allow_pickle=False) as arrays:
        for key, cell in metrics["conditions"].items():
            names = ["pooled_eer", "query_coverage"] + [
                f"{p}/{m}" for p, _ in POINTS for m in METRICS
            ]
            for name in names:
                if "/" in name:
                    point, metric = name.split("/")
                    expected = cell["ci95"]["operating_points"][point][metric]
                else:
                    expected = cell["ci95"][name]
                if old.interval(arrays[f"{key}/{name}"]) != expected:
                    raise ValueError("CI percentile audit failed")
                checks += 1
        with np.load(
            run / "bootstrap-paired-differences.npz", allow_pickle=False
        ) as paired_arrays:
            for key, value in metrics["paired_differences"].items():
                comparison, role, name = key.split("/", 2)
                a, b = comparison.split("_minus_")
                delta = 100 * (
                    arrays[f"{a}/{role}/{name}"] - arrays[f"{b}/{role}/{name}"]
                )
                if (
                    not np.array_equal(delta, paired_arrays[key])
                    or old.interval(delta) != value["ci95_percentage_points"]
                ):
                    raise ValueError("paired interval audit failed")
                checks += 1
    return checks
