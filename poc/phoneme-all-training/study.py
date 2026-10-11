"""Train one fixed all-phone encoder and compare its final checkpoint with vowels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
GREEDY = BASE.parent / "phoneme-greedy-selection"
sys.path.insert(0, str(GREEDY))

from common import (
    PHASE3,
    VOWELS,
    checked,
    expanded,
    ordered,
    pin,
    read_json,
    relative,
    rows,
    sha256_file,
    torch,
    write_json,
)
from evaluate import evaluate_trial
from learner import TrainingPool, load_model, train_trial

CONFIG = BASE / "config/protocol.json"
INPUT_FILES = (
    "inventory.json",
    "preparation-report.json",
    "train-features.f32",
    "train-segments.jsonl",
    "train-excluded.jsonl",
    "validation-features.f32",
    "validation-inputs.json",
    "validation-segments.jsonl",
    "test-features.f32",
    "test-inputs.json",
    "test-segments.jsonl",
)


def all_trainable(report):
    support = report["phone_support"]
    expected = set(VOWELS) | set(report["candidates"])
    if set(support) != expected or not all(support[p]["trainable"] for p in VOWELS):
        raise ValueError("full speech inventory or fixed vowels missing")
    phones = ordered(p for p in expected if support[p]["trainable"])
    unavailable = sorted(expected - set(phones))
    return phones, unavailable


def verify(config, run, full=False):
    frozen = read_json(run / "design-freeze.json")
    if frozen["config"] != config or frozen["runtime"] != expanded.runtime():
        raise ValueError("frozen settings/runtime changed")
    for name, checksum in frozen["files"].items():
        if full or name.startswith(
            (relative(BASE) + "/", relative(GREEDY) + "/", relative(run) + "/")
        ):
            checked(ROOT / name, checksum)


def prepare(config, run):
    if run.exists():
        raise ValueError("prepare requires an unused run directory")
    source = ROOT / config["source_run"]
    former = read_json(source / "design-freeze.json")
    if former["runtime"] != expanded.runtime():
        raise ValueError("use the original verified numerical runtime")
    for name, checksum in former["files"].items():
        checked(ROOT / name, checksum)
    phones, unavailable = all_trainable(read_json(source / "preparation-report.json"))
    if config["maximum_updates"] * 5 != config["baseline_updates"] * len(phones):
        raise ValueError("each vowel must retain the baseline group exposure")
    baseline_source = source / "trials" / config["baseline_trial"]
    baseline = read_json(baseline_source / "training-summary.json")
    if (
        baseline["phonemes"] != list(VOWELS)
        or baseline["completed_updates"] != config["baseline_updates"]
        or baseline["normalization_sha256"] != config["feature_statistics_sha256"]
    ):
        raise ValueError("baseline does not match the fixed comparison")
    files = dict(former["files"])
    run.mkdir(parents=True)
    for name in INPUT_FILES:
        os.link(source / name, run / name)
        pin(files, run / name)
    baseline_target = run / "trials" / config["baseline_trial"]
    baseline_target.mkdir(parents=True)
    for path in baseline_source.iterdir():
        if path.is_file() and path.name != "last.pt":
            os.link(path, baseline_target / path.name)
            pin(files, path)
            pin(files, baseline_target / path.name)
    for path in (
        CONFIG,
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        PHASE3 / "scripts/weighted_bootstrap.py",
        source / "design-freeze.json",
        source / "independent-audit.json",
        source / "completion-verification.json",
    ):
        pin(files, path)
    frozen = {
        "status": "full_phone_set_and_budget_frozen_before_new_training",
        "config": config,
        "runtime": expanded.runtime(),
        "phonemes": phones,
        "unavailable_phonemes": unavailable,
        "source_design_sha256": sha256_file(source / "design-freeze.json"),
        "source_inputs_reused_byte_for_byte": True,
        "baseline_already_test_evaluated": True,
        "files": files,
    }
    write_json(run / "design-freeze.json", frozen)
    print(
        json.dumps(
            {
                "stage": "prepared",
                "phonemes": phones,
                "unavailable": unavailable,
                "updates": config["maximum_updates"],
            }
        ),
        flush=True,
    )


def train(config, run):
    verify(config, run)
    frozen = read_json(run / "design-freeze.json")
    pool = TrainingPool(config, run)
    train_trial(
        config, run, run / "trials" / config["final_trial"], frozen["phonemes"], pool
    )
    verify(config, run)


def evaluate(config, run):
    verify(config, run)
    design = read_json(run / "design-freeze.json")
    trial = run / "trials" / config["final_trial"]
    model, bundle = load_model(trial)
    if (
        bundle["phonemes"] != design["phonemes"]
        or bundle["checkpoint_update"] != config["maximum_updates"]
    ):
        raise ValueError("evaluate only the fixed final full-phone encoder")
    evaluate_trial(run, trial, model, bundle["phonemes"], "validation")
    path = run / "selection-freeze.json"
    if not path.exists():
        models = {}
        for name in (config["baseline_trial"], config["final_trial"]):
            directory = run / "trials" / name
            summary = read_json(directory / "training-summary.json")
            models[name] = {
                "phonemes": summary["phonemes"],
                "encoder_sha256": summary["encoder_sha256"],
                "thresholds_sha256": sha256_file(
                    directory / "validation-thresholds.json"
                ),
                "validation_metrics_sha256": sha256_file(
                    directory / "validation-metrics.json"
                ),
            }
        write_json(
            path,
            {
                "status": "fixed_final_encoder_and_thresholds_frozen_before_new_test_inference",
                "baseline_trial": config["baseline_trial"],
                "final_trial": config["final_trial"],
                "selected_phonemes": design["phonemes"],
                "unavailable_candidates": design["unavailable_phonemes"],
                "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
                "models": models,
                "test_used_for_selection": False,
                "phonemes_chosen_by": "training_support_only_before_training; no_EER_selection",
            },
        )
    frozen = read_json(path)
    baseline_model, baseline_bundle = load_model(
        run / "trials" / config["baseline_trial"]
    )
    baseline = evaluate_trial(
        run,
        run / "trials" / config["baseline_trial"],
        baseline_model,
        baseline_bundle["phonemes"],
        "test",
    )
    final = evaluate_trial(run, trial, model, bundle["phonemes"], "test")
    output = run / "final-test.json"
    if not output.exists():
        write_json(
            output,
            {
                "selection_freeze_sha256": sha256_file(path),
                "baseline": baseline,
                "final": final,
                "test_used_for_selection": False,
            },
        )
    print(
        json.dumps(
            {
                "stage": "evaluated",
                "frozen_phonemes": len(frozen["selected_phonemes"]),
                "normal_test_baseline_eer_pct": 100
                * baseline["metrics"]["verification"]["eer"],
                "normal_test_all_eer_pct": 100
                * final["metrics"]["verification"]["eer"],
            }
        ),
        flush=True,
    )


def audit(config, run):
    from audit import audit_bootstrap, audit_slices, audit_trial

    verify(config, run)
    design = read_json(run / "design-freeze.json")
    frozen = read_json(run / "selection-freeze.json")
    final = read_json(run / "final-test.json")
    checked(run / "selection-freeze.json", final["selection_freeze_sha256"])
    checked(run / "design-freeze.json", frozen["design_freeze_sha256"])
    prep = read_json(run / "preparation-report.json")
    phones, unavailable = all_trainable(prep)
    if (
        phones != design["phonemes"]
        or phones != frozen["selected_phonemes"]
        or unavailable != design["unavailable_phonemes"]
    ):
        raise ValueError("a trainable phoneme was dropped or selected by score")
    sums, summaries = {"scores": 0, "eer_checks": 0, "rate_checks": 0}, {}
    for label, name, updates in (
        ("baseline", config["baseline_trial"], config["baseline_updates"]),
        ("final", config["final_trial"], config["maximum_updates"]),
    ):
        trial = run / "trials" / name
        summary = read_json(trial / "training-summary.json")
        if (
            summary["completed_updates"] != updates
            or summary["encoder_parameters"] != 65920
        ):
            raise ValueError("wrong model or completed update budget")
        load_model(trial)
        for file, key in (
            ("history.jsonl", "history_sha256"),
            ("training-freeze.json", "training_freeze_sha256"),
        ):
            checked(trial / file, summary[key])
        for split in ("validation", "test"):
            metrics = read_json(trial / f"{split}-metrics.json")
            checked(run / f"{split}-inputs.json", metrics["input_sha256"])
            if metrics["phonemes"] != summary["phonemes"]:
                raise ValueError("score phonemes differ from trained set")
            inspected = audit_trial(run, trial, split, metrics)
            for key in sums:
                sums[key] += inspected[key]
            for role, count in (
                ("verification", 735),
                ("cross_text_verification", 392),
            ):
                if metrics["metrics"][role]["scored_queries"] != count:
                    raise ValueError("evaluation support changed")
            if split == "test" and metrics != final[label]:
                raise ValueError("final report differs from saved trial")
        summaries[label] = summary
    if any(
        summaries["baseline"][k] != summaries["final"][k]
        for k in (
            "initial_encoder_sha256",
            "initial_head_sha256",
            "normalization_sha256",
        )
    ):
        raise ValueError("initial weights or normalization changed")
    if summaries["final"]["phonemes"] != phones:
        raise ValueError("trained model is not the full fixed inventory")
    total_groups = config["maximum_updates"] * 5
    if total_groups % len(phones):
        raise ValueError("unequal phone exposure")
    groups_per_phone = total_groups // len(phones)
    counts = {
        p: groups_per_phone
        * 2
        * min(10, prep["phone_support"][p]["speakers_with_two_intervals"])
        for p in phones
    }
    if sum(counts.values()) != summaries["final"]["training_examples"]:
        raise ValueError("sparse slots counted as real examples")
    for p in VOWELS:
        if counts[p] != summaries["baseline"]["phone_slot_maximum_example_counts"][p]:
            raise ValueError("vowel training exposure differs from baseline")
    history = list(rows(run / "trials" / config["final_trial"] / "history.jsonl"))
    if [h["update"] for h in history] != list(
        range(1000, config["maximum_updates"] + 1, 1000)
    ):
        raise ValueError("missing or repeated completed training updates")
    slice_checks = audit_slices(run, config, frozen)
    bootstrap_checks = audit_bootstrap(config, run, frozen, final)
    report = {
        "status": "passed",
        "new_training_runs": 1,
        "phoneme_count": len(phones),
        "completed_updates": config["maximum_updates"],
        "real_training_examples": counts,
        "vowel_exposure_matched": True,
        "test_used_for_selection": False,
        "independent_bootstrap_eer_checks": bootstrap_checks,
        **sums,
        **slice_checks,
    }
    write_json(run / "independent-audit.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "train", "evaluate", "audit", "verify")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    if args.stage == "verify":
        verify(config, run, full=True)
        print("all frozen sources and inputs verified", flush=True)
    else:
        globals()[args.stage](config, run)


if __name__ == "__main__":
    main()
