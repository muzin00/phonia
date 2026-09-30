"""Finalize runs whose training ended before the full-matrix runner was interrupted.

This recovery path does not resume training: a checkpoint saved after an early stop
must not be advanced by another update. It validates the saved training evidence,
repeats the selected checkpoint evaluation, and then marks the run complete.
"""

from __future__ import annotations

import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from run_full_matrix import (
    BUDGET_PATH,
    ROOT,
    TRAIN_SCRIPT,
    _check_history,
    _code_sha256,
    _disk_bytes,
    _monitor,
)
from run_phase3 import _trim_history_to_checkpoint


def _training_seconds(run_dir: Path, summary: dict) -> float:
    previous = run_dir / "logs/train-before-20260928-resume.log"
    if previous.exists():
        last_line = previous.read_text(encoding="utf-8").splitlines()[-1]
        record = json.loads(last_line)
        if record.get("completed_updates") == summary["completed_updates"]:
            return float(record["elapsed_seconds"])
    return float(summary["elapsed_seconds"])


def _finalize(row: dict, root: Path, budget: dict, matrix_sha: str) -> dict:
    run_dir = root / "runs" / row["run_id"]
    result_path = run_dir / "full-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["matrix_sha256"] != matrix_sha or result["status"] != "running":
        raise ValueError(f"run is not recoverable: {row['run_id']}")
    summary = json.loads(
        (run_dir / "training/summary.json").read_text(encoding="utf-8")
    )
    completed = summary["completed_updates"]
    if (
        completed != budget["protocol"]["maximum_updates"]
        and not summary["early_stopped"]
    ):
        raise ValueError(f"training is not terminal: {row['run_id']}")
    checkpoint = run_dir / "checkpoints/last.pt"
    if summary["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError(f"training checkpoint differs: {row['run_id']}")
    _trim_history_to_checkpoint(run_dir / "training/history.jsonl", completed)
    _check_history(run_dir / "training/history.jsonl", completed)

    selected = summary["best"]["update"]
    if selected is None:
        raise ValueError(f"no selected checkpoint: {row['run_id']}")
    metric_path = run_dir / f"validation/update-{selected:06d}/metrics/validation.json"
    score_path = run_dir / f"validation/update-{selected:06d}/scores/validation.jsonl"
    before = json.loads(metric_path.read_text(encoding="utf-8"))
    if before["partial"] or before["split"] != "validation":
        raise ValueError(f"selected validation is incomplete: {row['run_id']}")
    if before["score_sha256"] != sha256_file(score_path):
        raise ValueError(f"selected score checksum differs: {row['run_id']}")
    if before["macro_eer"] != summary["best"]["eer"]:
        raise ValueError(f"selected EER differs: {row['run_id']}")

    training_seconds = _training_seconds(run_dir, summary)
    remaining = budget["limits"]["maximum_run_wall_seconds"] - training_seconds
    if remaining <= 0:
        raise ValueError(f"no wall budget left for repeat evaluation: {row['run_id']}")
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "evaluate",
        "--run-dir",
        str(run_dir),
        "--batch-size",
        str(budget["environment"]["evaluation_batch_size"]),
    ]
    observed = _monitor(
        command,
        run_dir,
        run_dir / "logs/repeat-evaluation-recovery.log",
        budget,
        remaining,
    )
    if observed["limit_reason"] or observed["exit_code"] != 0:
        raise RuntimeError(f"repeat evaluation failed: {row['run_id']}: {observed}")
    after = json.loads(metric_path.read_text(encoding="utf-8"))
    after.pop("checkpoint_sha256", None)
    after.pop("checkpoint_update", None)
    expected = dict(before)
    expected.pop("checkpoint_sha256", None)
    expected.pop("checkpoint_update", None)
    if expected != after or before["score_sha256"] != sha256_file(score_path):
        raise ValueError(f"repeat evaluation differs: {row['run_id']}")

    result["stages"].extend(
        [
            {
                "stage": "training-recovered-from-checkpoint",
                "wall_seconds": training_seconds,
                "exit_code": None,
                "evidence": "terminal summary, checkpoint checksum, contiguous history",
            },
            {"stage": "repeat-evaluation-recovery", **observed},
        ]
    )
    result.update(
        {
            "status": "completed",
            "completed_updates": completed,
            "early_stopped": summary["early_stopped"],
            "selected_checkpoint_update": selected,
            "selected_checkpoint_sha256": sha256_file(run_dir / "checkpoints/best.pt"),
            "validation_macro_eer": before["macro_eer"],
            "validation_score_sha256": before["score_sha256"],
            "validation_trial_count": before["trial_count"],
            "parameter_count": summary["parameter_count"],
            "repeat_evaluation_identical": True,
            "recovered_after_runner_interruption": True,
            "wall_seconds_this_invocation": observed["wall_seconds"],
            "sampled_peak_rss_bytes": observed["sampled_peak_rss_bytes"],
            "artifact_bytes": _disk_bytes(run_dir),
        }
    )
    write_json(result_path, result)
    return result


def main() -> None:
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    root = (
        ROOT / "artifacts/phoneme-speaker-encoder/comparisons" / budget["comparison_id"]
    )
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    if matrix["execution_budget_sha256"] != sha256_file(BUDGET_PATH):
        raise ValueError("frozen execution budget differs")
    if matrix["training_code_sha256"] != _code_sha256():
        raise ValueError("frozen training code differs")
    if matrix["runner_sha256"] != sha256_file(BASE / "scripts/run_full_matrix.py"):
        raise ValueError("frozen full runner differs")
    selected = []
    for row in matrix["runs"]:
        run_dir = root / "runs" / row["run_id"]
        result_path = run_dir / "full-result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result["status"] == "running"
            and (run_dir / "training/summary.json").exists()
        ):
            selected.append(row)
    if not selected:
        print("No finished training runs need recovery.", flush=True)
        return
    with (root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(
            max_workers=budget["environment"]["maximum_parallel_runs"]
        ) as pool:
            futures = {
                pool.submit(_finalize, row, root, budget, matrix["sha256"]): row
                for row in selected
            }
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                print(
                    json.dumps(
                        {
                            "run_id": row["run_id"],
                            "status": result["status"],
                            "macro_eer": result["validation_macro_eer"],
                        }
                    ),
                    flush=True,
                )
    results = []
    for row in matrix["runs"]:
        path = root / "runs" / row["run_id"] / "full-result.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {
                "run_id": row["run_id"],
                "config_id": row["config_id"],
                "status": "not_started",
            }
        )
    write_json(
        root / "results.json", {"matrix_sha256": matrix["sha256"], "results": results}
    )


if __name__ == "__main__":
    main()
