"""Run the selected Phase 3 configuration on the 10/25/50 speaker cohorts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import sha256_file
from run_full_matrix import _monitor

FULL_BUDGET = BASE / "config/full-execution-budget.json"
TRAIN_SCRIPT = BASE / "scripts/run_phase3.py"
FULL_ROOT = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2"
)
SELECTION = FULL_ROOT / "selection-evaluation/selection.json"
VALIDATION_SELECTIONS = ROOT / "artifacts/phoneme-speaker-encoder/fixed-validation"
OUTPUT = ROOT / "artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2"
CONFIG_ID = "log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel"
COHORTS = (10, 25, 50)


def _sha_json(data: dict) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code_files() -> list[Path]:
    return [
        *sorted((BASE / "phase3_data").glob("*.py")),
        *sorted((BASE / "phase3_train").glob("*.py")),
        TRAIN_SCRIPT,
        BASE / "config/baseline-log-mel.json",
        BASE / "config/experiment-protocol.json",
        BASE / "config/log-mel-encoders.json",
    ]


def _plan() -> dict:
    budget = json.loads(FULL_BUDGET.read_text())
    selection = json.loads(SELECTION.read_text())
    if selection["selected_config_id"] != CONFIG_ID or selection["test_used"]:
        raise ValueError("selected configuration or test-use state differs")
    if budget["protocol"]["cohort"] != 70:
        raise ValueError("expected the 70-speaker full-training protocol")
    seeds = budget["protocol"]["seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("expected three distinct seeds")
    rows = []
    for cohort in COHORTS:
        stats = (
            ROOT
            / f"artifacts/phoneme-speaker-encoder/cohort-{cohort}-rms-off/feature-statistics.json"
        )
        if not stats.is_file():
            raise FileNotFoundError(stats)
        for seed in seeds:
            rows.append(
                {
                    "cohort": cohort,
                    "seed": seed,
                    "run_id": f"curve-c{cohort}-s{seed}__{CONFIG_ID}",
                    "statistics_sha256": sha256_file(stats),
                }
            )
    if not (VALIDATION_SELECTIONS / "selections/trials.jsonl").is_file():
        raise FileNotFoundError("fixed validation trials are missing")
    return {
        "schema_version": 1,
        "selected_config_id": CONFIG_ID,
        "source_selection_sha256": sha256_file(SELECTION),
        "full_budget_sha256": sha256_file(FULL_BUDGET),
        "code_sha256": _sha_json(
            {str(path.relative_to(ROOT)): sha256_file(path) for path in _code_files()}
        ),
        "maximum_updates": budget["protocol"]["maximum_updates"],
        "warmup_updates": budget["protocol"]["warmup_updates"],
        "validation_interval": budget["protocol"]["validation_interval"],
        "data_loader_workers": budget["environment"]["data_loader_workers"],
        "eval_batch_size": budget["environment"]["evaluation_batch_size"],
        "device": budget["environment"]["device"],
        "maximum_parallel_runs": budget["environment"]["maximum_parallel_runs"],
        "limits": budget["limits"],
        "runs": rows,
    }


def _saved_plan(plan: dict) -> None:
    path = OUTPUT / "plan.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("learning-curve plan differs from the frozen plan")
    else:
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")


def _run(row: dict, plan: dict) -> dict:
    run_dir = OUTPUT / "runs" / row["run_id"]
    summary = run_dir / "training/summary.json"
    result_path = run_dir / "execution.json"
    if summary.exists():
        result = json.loads(summary.read_text())
        if not result.get("checkpoint_sha256"):
            raise ValueError(f"incomplete summary: {summary}")
        return {"run_id": row["run_id"], "status": "completed", **result}

    stats = (
        ROOT
        / f"artifacts/phoneme-speaker-encoder/cohort-{row['cohort']}-rms-off/feature-statistics.json"
    )
    if sha256_file(stats) != row["statistics_sha256"]:
        raise ValueError(f"feature statistics changed: {stats}")
    if shutil.disk_usage(ROOT).free < plan["limits"]["minimum_free_disk_bytes"]:
        raise RuntimeError("free disk below the full-comparison safety limit")

    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "train",
        "--encoder",
        "statistics_mlp",
        "--cohort",
        str(row["cohort"]),
        "--rms",
        "off",
        "--supcon",
        "on",
        "--seed",
        str(row["seed"]),
        "--maximum-updates",
        str(plan["maximum_updates"]),
        "--warmup-updates",
        str(plan["warmup_updates"]),
        "--validation-interval",
        str(plan["validation_interval"]),
        "--workers",
        str(plan["data_loader_workers"]),
        "--device",
        plan["device"],
        "--eval-batch-size",
        str(plan["eval_batch_size"]),
        "--statistics",
        str(stats),
        "--selections",
        str(VALIDATION_SELECTIONS),
        "--compact-validation",
        "--output",
        str(run_dir),
    ]
    if (run_dir / "checkpoints/last.pt").exists():
        command.append("--resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    observed = _monitor(
        command,
        run_dir,
        run_dir / "logs/train.log",
        {"limits": plan["limits"]},
        plan["limits"]["maximum_run_wall_seconds"],
    )
    status = (
        "completed"
        if observed["exit_code"] == 0 and summary.exists()
        else observed["limit_reason"] or "failed"
    )
    result = {
        "run_id": row["run_id"],
        "status": status,
        "exit_code": observed["exit_code"],
        "sampled_peak_rss_bytes": observed["sampled_peak_rss_bytes"],
        "started_at_unix": started,
        "finished_at_unix": time.time(),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _report(plan: dict) -> list[dict]:
    rows = []
    for row in plan["runs"]:
        run_dir = OUTPUT / "runs" / row["run_id"]
        summary_path = run_dir / "training/summary.json"
        execution_path = run_dir / "execution.json"
        record = {"cohort": row["cohort"], "seed": row["seed"]}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            record.update(
                status="completed",
                completed_updates=summary["completed_updates"],
                best_eer=summary["best"]["eer"],
                best_update=summary["best"]["update"],
                elapsed_seconds=summary["elapsed_seconds"],
            )
        elif execution_path.exists():
            record["status"] = json.loads(execution_path.read_text())["status"]
        elif (run_dir / "run.json").exists():
            record["status"] = "running_or_interrupted"
        else:
            record["status"] = "pending"
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    args = parser.parse_args()
    plan = _plan()
    _saved_plan(plan)
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
    elif args.command == "report":
        print(json.dumps(_report(plan), indent=2))
    else:
        with (OUTPUT / "runner.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("learning-curve runner is already active") from error
            with ThreadPoolExecutor(max_workers=plan["maximum_parallel_runs"]) as pool:
                futures = {pool.submit(_run, row, plan): row for row in plan["runs"]}
                for future in as_completed(futures):
                    print(json.dumps(future.result()), flush=True)
        print(json.dumps(_report(plan), indent=2))


if __name__ == "__main__":
    main()
