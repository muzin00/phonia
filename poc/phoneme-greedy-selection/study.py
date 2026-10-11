"""Run the user's random forward search and retry unselected phones."""

from __future__ import annotations

import argparse
import json
import os

from common import (
    CONFIG,
    ROOT,
    VOWELS,
    checked,
    expanded,
    ordered,
    read_json,
    relative,
    sha256_file,
    torch,
    write_json,
)
from evaluate import evaluate_trial
from learner import TrainingPool, load_model, train_trial
from selection import baseline_state, draw_order, next_candidate, record_attempt


def save_state(run, state):
    path = run / "search-state.json"
    temporary = run / ".search-state.tmp"
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def verify_design(config, run, full=False):
    frozen = read_json(run / "design-freeze.json")
    if frozen["config"] != config or frozen["runtime"] != expanded.runtime():
        raise ValueError("frozen protocol/runtime changed")
    for name, checksum in frozen["files"].items():
        if full or name.startswith(
            ("poc/phoneme-greedy-selection/", relative(run) + "/")
        ):
            checked(ROOT / name, checksum)


def selection_freeze(run, state):
    path = run / "selection-freeze.json"
    if path.exists():
        return read_json(path)
    names = list(dict.fromkeys(["trial-000-baseline", state["best_trial"]]))
    models = {}
    for name in names:
        trial = run / "trials" / name
        summary = read_json(trial / "training-summary.json")
        models[name] = {
            "phonemes": summary["phonemes"],
            "encoder_sha256": summary["encoder_sha256"],
            "thresholds_sha256": sha256_file(trial / "validation-thresholds.json"),
            "validation_metrics_sha256": sha256_file(trial / "validation-metrics.json"),
        }
    frozen = {
        "status": "selection_and_thresholds_frozen_before_test_inference",
        "baseline_trial": "trial-000-baseline",
        "final_trial": state["best_trial"],
        "selected_phonemes": state["accepted"],
        "selection_validation_eer": state["best_eer"],
        "models": models,
        "search_state_sha256": sha256_file(run / "search-state.json"),
        "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
        "unavailable_candidates": sorted(
            {e["phoneme"] for e in state["attempts"] if e["status"] == "unavailable"}
        ),
        "test_used_for_selection": False,
    }
    write_json(path, frozen)
    return frozen


def search(config, run):
    verify_design(config, run)
    if (run / "selection-freeze.json").exists():
        print("selection already completed and frozen", flush=True)
        return
    report = read_json(run / "preparation-report.json")
    order = draw_order(report["candidates"], config["draw_seed"])
    run.joinpath("trials").mkdir(exist_ok=True)
    pool = TrainingPool(config, run)
    state_path = run / "search-state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["order"] != order:
            raise ValueError("random box order changed")
    else:
        baseline = run / "trials/trial-000-baseline"
        model = train_trial(config, run, baseline, VOWELS, pool)
        result = evaluate_trial(run, baseline, model, list(VOWELS), "validation")
        eer = result["metrics"]["verification"]["eer"]
        state = baseline_state(order, eer)
        save_state(run, state)
        print(
            json.dumps(
                {"stage": "baseline", "validation_eer_pct": eer * 100, "order": order}
            ),
            flush=True,
        )
    while not state["completed"]:
        if "pending" in state:
            phone = state["pending"]["phoneme"]
        else:
            phone = next_candidate(state)
            if phone is None:
                save_state(run, state)
                break
            state["pending"] = {"phoneme": phone, "step": len(state["attempts"]) + 1}
            save_state(run, state)
        if not report["phone_support"][phone]["trainable"]:
            entry = record_attempt(
                state,
                phone,
                None,
                None,
                "no_speaker_has_two_distinct_post_QC_training_intervals; candidate_retained_as_unavailable",
            )
        else:
            phones = ordered([*state["accepted"], phone])
            prior = next(
                (
                    e
                    for e in state["attempts"]
                    if e["phonemes"] == phones and e["validation_eer"] is not None
                ),
                None,
            )
            if prior:
                name, eer = prior["trial"], prior["validation_eer"]
            else:
                name = f"trial-{len(state['attempts']) + 1:03d}-add-{phone}"
                trial = run / "trials" / name
                print(
                    json.dumps(
                        {
                            "stage": "candidate_start",
                            "pass": state["pass"],
                            "trial": name,
                            "phoneme": phone,
                            "phonemes": phones,
                            "previous_best_eer_pct": state["best_eer"] * 100,
                        }
                    ),
                    flush=True,
                )
                model = train_trial(config, run, trial, phones, pool)
                result = evaluate_trial(run, trial, model, phones, "validation")
                eer = result["metrics"]["verification"]["eer"]
            entry = record_attempt(state, phone, name, eer)
        del state["pending"]
        save_state(run, state)
        print(
            json.dumps({"stage": "decision", **entry}, ensure_ascii=False), flush=True
        )
    verify_design(config, run)
    frozen = selection_freeze(run, state)
    print(
        json.dumps(
            {
                "stage": "search_completed",
                "selected": frozen["selected_phonemes"],
                "validation_eer_pct": frozen["selection_validation_eer"] * 100,
                "attempts": len(state["attempts"]),
                "unavailable": frozen["unavailable_candidates"],
            }
        ),
        flush=True,
    )


def final_test(config, run):
    verify_design(config, run)
    frozen = read_json(run / "selection-freeze.json")
    checked(run / "search-state.json", frozen["search_state_sha256"])
    results = {}
    for name in frozen["models"]:
        trial = run / "trials" / name
        model, bundle = load_model(trial)
        results[name] = evaluate_trial(run, trial, model, bundle["phonemes"], "test")
    path = run / "final-test.json"
    if not path.exists():
        write_json(
            path,
            {
                "selection_freeze_sha256": sha256_file(run / "selection-freeze.json"),
                "baseline": results[frozen["baseline_trial"]],
                "final": results[frozen["final_trial"]],
                "test_used_for_selection": False,
            },
        )
    print(
        json.dumps(
            {
                "stage": "final_test",
                "baseline_normal_eer_pct": results[frozen["baseline_trial"]]["metrics"][
                    "verification"
                ]["eer"]
                * 100,
                "final_normal_eer_pct": results[frozen["final_trial"]]["metrics"][
                    "verification"
                ]["eer"]
                * 100,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify-inputs", "search", "test"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    if args.stage == "verify-inputs":
        verify_design(config, run, full=True)
        print("all frozen sources and caches verified", flush=True)
    elif args.stage == "search":
        search(config, run)
    else:
        final_test(config, run)


if __name__ == "__main__":
    main()
