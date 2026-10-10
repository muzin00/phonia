"""Validation-first utterance evaluation of six preregistered budget snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-src4vc-evaluation"))
import budget_training as training
import numpy as np
import src_evaluation as prior_study
import torch

old = prior_study.old
ROOT = training.ROOT
CONFIG = BASE / "config/evaluation.json"
read_json, checked, sha256_file, write_json = (
    training.read_json,
    training.checked,
    training.sha256_file,
    training.write_json,
)
BASELINE = "jvs70_u15000"
NEW_CONDITIONS = tuple(
    f"{corpus}_u{update}_s60k"
    for corpus in ("cv", "src")
    for update in (30000, 45000, 60000)
)
CONDITIONS = (BASELINE, *NEW_CONDITIONS)
ROLES, POINTS, METRICS = old.ROLES, old.POINTS, old.METRICS
NAMES = [f"{p}/{m}" for p, _ in POINTS for m in METRICS] + [
    "pooled_eer",
    "query_coverage",
]


def specification(condition):
    if condition not in NEW_CONDITIONS:
        raise ValueError("unknown budget condition")
    corpus, token, _ = condition.split("_")
    return corpus, int(token[1:])


def validate_protocol(config, evaluation):
    training.validate_protocol(config)
    comparisons = [
        f"{c}_u{u}_s60k_minus_{c}_u30000_s60k"
        for c in ("cv", "src")
        for u in (45000, 60000)
    ]
    primary = [f"{c}_u60000_s60k_minus_{c}_u30000_s60k" for c in ("cv", "src")]
    expected = {
        "enrollment_count_per_vowel": 10,
        "query_window": "full",
        "required_vowels": list(old.VOWELS),
        "calibration": "pooled_validation_verification_per_model",
        "operating_points": [p for p, _ in POINTS],
        "no_score": "reject_and_keep_in_all_input_denominators",
        "fusion": "mean_segment_cosine_within_vowel_then_equal_mean_over_five_vowels",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20260929,
        "embedding_repeats": 3,
        "independent_holdout": False,
    }
    if (
        any(config.get(k) != v for k, v in expected.items())
        or evaluation["new_conditions"] != list(NEW_CONDITIONS)
        or evaluation["baseline_condition"] != BASELINE
        or evaluation["paired_comparisons"] != comparisons
        or evaluation["primary_comparisons"] != primary
        or evaluation["checkpoint_selection"]
        != "all_six_fixed_before_training_no_selection_on_validation_or_test"
        or evaluation["independent_holdout"]
        or not evaluation["test_previously_observed"]
    ):
        raise ValueError("unsupported fixed six-snapshot utterance protocol")


def pin(files, path):
    training.merge_files(files, {str(path.relative_to(ROOT)): sha256_file(path)})


def prepare(run):
    config = read_json(training.CONFIG)
    evaluation = read_json(CONFIG)
    validate_protocol(config, evaluation)
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused evaluation directory required")
    study = ROOT / config["run_directory"]
    freeze = training.frozen(config, study)
    files = dict(freeze["files"])
    pin(files, study / "training-freeze.json")
    heads, encoder_hashes, settings = [], [], []
    for corpus in config["corpora"]:
        output = study / corpus
        summary_path = output / "training/summary.json"
        summary = read_json(summary_path)
        if (
            summary["status"] != "completed"
            or summary["completed_updates"] != 60000
            or summary["test_used"]
        ):
            raise ValueError("both full train/validation-only trajectories required")
        for name, digest in summary["outputs_sha256"].items():
            checked(output / name, digest)
            training.merge_files(
                files, {str((output / name).relative_to(ROOT)): digest}
            )
        pin(files, summary_path)
        descriptor = read_json(output / "run.json")
        heads.append(descriptor["initial_head_sha256"])
        encoder_hashes.append(descriptor["initial_encoder_sha256"])
        settings.append(descriptor["settings"])
        for update in config["snapshot_updates"]:
            snapshot = read_json(
                output / f"snapshots/update-{update:06d}/snapshot.json"
            )
            if (
                snapshot["update"] != update
                or snapshot["maximum_updates"] != 60000
                or snapshot["test_used"]
            ):
                raise ValueError("snapshot adoption mismatch")
    if (
        heads[0] != heads[1]
        or encoder_hashes[0] != encoder_hashes[1]
        or settings[0] != settings[1]
    ):
        raise ValueError("initialization or settings differ across corpora")
    prior = ROOT / config["source_evaluation_run"]
    checked(prior / "study-report.json", evaluation["source_evaluation_report_sha256"])
    reference = read_json(prior / "study-report.json")
    if reference["status"] != "completed":
        raise ValueError("reference evaluation incomplete")
    training.merge_files(files, read_json(prior / "design-freeze.json")["files"])
    for name, digest in reference["outputs_sha256"].items():
        checked(prior / name, digest)
        training.merge_files(files, {str((prior / name).relative_to(ROOT)): digest})
    pin(files, prior / "study-report.json")
    for p in (
        CONFIG,
        *BASE.glob("budget_evaluation.py"),
        *BASE.glob("budget_inference.py"),
        *BASE.glob("budget_report.py"),
        *BASE.glob("run_evaluation.py"),
        *BASE.glob("tests/test_evaluation.py"),
        *BASE.glob("tests/test_report.py"),
        *(
            BASE / name
            for name in (
                "training-results.json",
                "training-results.md",
                "training-results.html",
                "training-snapshots.csv",
                "learning-curves.svg",
                "learning-curves.png",
            )
        ),
    ):
        pin(files, p)
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
            raise ValueError("evaluation speaker split mismatch")
        prepared[split] = inputs
    if set(prepared["validation"]["speakers"]) & set(prepared["test"]["speakers"]):
        raise ValueError("evaluation speaker overlap")
    training.check_files(files)
    run.mkdir(parents=True)
    for split, inputs in prepared.items():
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "design_frozen_before_new_inference",
            "config": config,
            "evaluation": evaluation,
            "runtime": old.previous.runtime(),
            "files": files,
            "new_inference_conditions": list(NEW_CONDITIONS),
            "test_previously_observed": True,
            "src4vc_reserved_speakers_evaluated": False,
        },
    )


def frozen_inputs(run, split):
    if split not in ("validation", "test"):
        raise ValueError("validation/test only")
    freeze = read_json(run / "design-freeze.json")
    if (
        freeze["status"] != "design_frozen_before_new_inference"
        or freeze["runtime"] != old.previous.runtime()
    ):
        raise ValueError("evaluation runtime/design changed")
    training.check_files(freeze["files"])
    inputs = read_json(run / f"{split}-inputs.json")
    if inputs["split"] != split:
        raise ValueError("input split mismatch")
    if split == "test":
        gate = read_json(run / "evaluation-freeze.json")
        if (
            gate["status"]
            != "all_validation_thresholds_frozen_before_any_new_test_inference"
        ):
            raise ValueError("all validation thresholds must be frozen before test")
        for name, digest in gate["files"].items():
            checked(run / name, digest)
    return inputs


def load_encoder(run, condition):
    corpus, update = specification(condition)
    config = read_json(run / "design-freeze.json")["config"]
    output = ROOT / config["run_directory"] / corpus
    descriptor = read_json(output / "run.json")
    snapshot_dir = output / f"snapshots/update-{update:06d}"
    snapshot = read_json(snapshot_dir / "snapshot.json")
    for name, digest in snapshot["outputs_sha256"].items():
        checked(snapshot_dir / name, digest)
    export = torch.load(
        snapshot_dir / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if (
        snapshot["update"] != update
        or export["checkpoint_update"] != update
        or export["configuration_sha256"] != descriptor["configuration_sha256"]
        or descriptor["configuration_sha256"]
        != training.cv.json_sha256(descriptor["settings"])
        or descriptor["settings"]["maximum_updates"] != 60000
        or export["encoder_parameters"] != 65920
        or export["embedding_dimension"] != 128
        or export["run_seed"] != 20260926
        or export["encoder"] != "statistics_mlp"
    ):
        raise ValueError("snapshot bundle metadata mismatch")
    statistics = read_json(snapshot_dir / "bundle/feature-statistics.json")
    if statistics["speaker_count"] != 140 or statistics["cohort"] != 140:
        raise ValueError("statistics population mismatch")
    pipeline = old.InputPipeline(
        training.cv.BASELINE["input"], rms_enabled=False, statistics=statistics
    )
    model = old.create_encoder("statistics_mlp")
    model.load_state_dict(export["model"], strict=True)
    phase3 = training.cv.PHASE3
    identity = {
        "design_version": "2.0.0",
        "encoder": "statistics_mlp",
        "seed": 20260926,
        "checkpoint_sha256": sha256_file(snapshot_dir / "bundle/encoder.pt"),
        "encoder_config_sha256": old.json_sha256(
            {
                name: sha256_file(phase3 / "config" / name)
                for name in ("log-mel-encoders.json", "waveform-encoders.json")
            }
        ),
        "feature_statistics_sha256": sha256_file(
            snapshot_dir / "bundle/feature-statistics.json"
        ),
        "preprocessing_sha256": pipeline.preprocessing_sha256,
        "implementation_sha256": old.json_sha256(
            {
                name: sha256_file(phase3 / name)
                for name in ("phase3_data/input.py", "phase3_train/models.py")
            }
        ),
    }
    return old.FrozenEncoder(model, pipeline, identity)


def infer_split(run, split):
    from budget_inference import infer_new

    inputs = frozen_inputs(run, split)
    for condition in NEW_CONDITIONS:
        infer_new(run, split, inputs, condition)
    frozen_inputs(run, split)


def new_rows(run, split, inputs, condition):
    output = run / split / condition
    report = read_json(output / "report.json")
    if report["status"] != "completed":
        raise ValueError("incomplete inference")
    for name, digest in report["outputs_sha256"].items():
        checked(output / name, digest)
    expected = {r["trial_id"]: r for r in inputs["trials"]}
    result, seen = [], set()
    for row in old.iter_rows(output / "scores.jsonl.gz"):
        trial = expected.get(row["trial_id"])
        if (
            trial is None
            or row["trial_id"] in seen
            or row["model_id"] != condition
            or any(row[k] != v for k, v in trial.items())
        ):
            raise ValueError("trial identity, label or condition mismatch")
        seen.add(row["trial_id"])
        values = row["scores"]
        if row["status"] not in ("scored", "no_score") or (values is None) != (
            row["status"] == "no_score"
        ):
            raise ValueError("invalid score state")
        if values is not None and (
            set(values) != {"fused", *old.VOWELS}
            or not all(np.isfinite(v) and -1 <= v <= 1 for v in values.values())
            or values["fused"]
            != float(np.mean([values[v] for v in old.VOWELS], dtype=np.float64))
        ):
            raise ValueError("invalid fused scores")
        result.append({**row, "score": values["fused"] if values else None})
    if seen != set(expected):
        raise ValueError("missing score trials")
    return result


def score_groups(run, split):
    inputs = read_json(run / f"{split}-inputs.json")
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    reference_inputs, reference_groups = prior_study.score_groups(prior, split)
    if any(
        inputs[k] != reference_inputs[k]
        for k in ("speakers", "enrollment", "queries", "sources", "trials")
    ):
        raise ValueError("input populations differ from reference")
    groups = {(BASELINE, role): reference_groups[BASELINE, role] for role in ROLES}
    for condition in NEW_CONDITIONS:
        rows = new_rows(run, split, inputs, condition)
        for role in ROLES:
            groups[condition, role] = [r for r in rows if r["role"] == role]
            before = {r["trial_id"]: r["status"] for r in groups[BASELINE, role]}
            after = {r["trial_id"]: r["status"] for r in groups[condition, role]}
            if before != after:
                raise ValueError("score coverage differs across conditions")
    return inputs, groups


def calibrate(run):
    inputs, groups = score_groups(run, "validation")
    prior = ROOT / inputs["config"]["source_evaluation_run"]
    original = read_json(prior / "validation-thresholds.json")["conditions"][BASELINE]
    thresholds, cells, curves = {}, {}, {}
    for condition in CONDITIONS:
        threshold = old.calibrate_cell(
            groups[condition, "verification"], inputs["speakers"]
        )
        if threshold["status"] != "calibrated":
            raise ValueError("validation support insufficient")
        if (
            condition == BASELINE
            and threshold["operating_points"] != original["operating_points"]
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
        {"conditions": thresholds, "split": "validation", "role": "verification"},
    )
    write_json(run / "validation-metrics.json", {"conditions": cells, "curves": curves})
    _, counts = old.speaker_draws(15, 10000, 20260929)
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    paths = [
        run / "validation-thresholds.json",
        run / "validation-metrics.json",
        run / "bootstrap-counts.npy",
    ]
    paths += [run / "validation" / c / "report.json" for c in NEW_CONDITIONS]
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "all_validation_thresholds_frozen_before_any_new_test_inference",
            "files": {str(p.relative_to(run)): sha256_file(p) for p in paths},
        },
    )


def paired(cells, replicas, evaluation):
    differences, arrays = {}, {}
    for comparison in evaluation["paired_comparisons"]:
        a, b = comparison.split("_minus_")
        for role in ROLES:
            for name in NAMES:
                key = f"{comparison}/{role}/{name}"
                delta = 100 * (
                    replicas[f"{a}/{role}"][name] - replicas[f"{b}/{role}"][name]
                )
                arrays[key] = delta
                differences[key] = {
                    "difference_percentage_points": 100
                    * (
                        old.metric_value(cells[f"{a}/{role}"], name)
                        - old.metric_value(cells[f"{b}/{role}"], name)
                    ),
                    "ci95_percentage_points": old.interval(delta),
                    "primary": comparison in evaluation["primary_comparisons"],
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
    reference = read_json(prior / "test-metrics.json")["conditions"]
    for role in ROLES:
        if cells[f"{BASELINE}/{role}"] != reference[f"{BASELINE}/{role}"]:
            raise ValueError("reference metric/CI parity failed")
    differences, arrays = paired(cells, replicas, read_json(CONFIG))
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{
            f"{k}/{n}": value
            for k, items in replicas.items()
            for n, value in items.items()
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
            "reference_metric_ci_parity_cells": 2,
            "independent_rate_audits": audits,
        },
    )


def audit_intervals(run):
    metrics = read_json(run / "test-metrics.json")
    checks = 0
    with np.load(run / "bootstrap-metrics.npz", allow_pickle=False) as arrays:
        for key, cell in metrics["conditions"].items():
            for name in NAMES:
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
            for key, result in metrics["paired_differences"].items():
                comparison, role, name = key.split("/", 2)
                a, b = comparison.split("_minus_")
                delta = 100 * (
                    arrays[f"{a}/{role}/{name}"] - arrays[f"{b}/{role}/{name}"]
                )
                if (
                    not np.array_equal(delta, paired_arrays[key])
                    or old.interval(delta) != result["ci95_percentage_points"]
                ):
                    raise ValueError("paired CI audit failed")
                checks += 1
    return checks
