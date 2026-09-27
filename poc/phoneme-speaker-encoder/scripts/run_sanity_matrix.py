"""Freeze and execute the 18 CPU-only Phase 3 ten-speaker sanity runs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import torch
from phase3_data import sha256_file
from phase3_data.artifacts import write_json

SEARCH_PATH = BASE / "config/search-space.json"
BUDGET_PATH = BASE / "config/execution-budget.json"
BASELINE_PATH = BASE / "config/baseline-log-mel.json"
TRAIN_SCRIPT = BASE / "scripts/run_phase3.py"


def expand_matrix(search: dict, budget: dict) -> list[dict]:
    if search["design_version"] != budget["design_version"]:
        raise ValueError("design versions differ")
    seed = budget["protocol"]["seed"]
    cohort = budget["protocol"]["cohort"]
    if seed != search["sanity_seed"] or cohort != 10:
        raise ValueError("sanity seed or cohort differs from search plan")
    rows = []
    for family in search["families"]:
        axes = family["axes"]
        count = 0
        for encoder, rms_enabled, loss in itertools.product(
            axes["encoder"], axes["rms_normalization"], axes["loss"]
        ):
            rows.append(
                _row(family["input"], encoder, rms_enabled, loss, "main", cohort, seed)
            )
            count += 1
        if count != family["combination_count"]:
            raise ValueError(f"{family['input']} combination count mismatch")
    if len(rows) != search["main_combination_count"]:
        raise ValueError("main combination count mismatch")
    for limited in search["limited_comparisons"]:
        rows.append(
            _row(
                limited["input"],
                limited["encoder"],
                limited["rms_normalization"],
                limited["loss"],
                limited["id"],
                cohort,
                seed,
            )
        )
    ids = [row["config_id"] for row in rows]
    run_ids = [row["run_id"] for row in rows]
    if (
        len(rows) != search["total_combination_count"]
        or len(rows) != 18
        or len(ids) != len(set(ids))
        or len(run_ids) != len(set(run_ids))
    ):
        raise ValueError("missing or duplicate sanity configurations")
    return rows


def _row(
    input_kind: str,
    encoder: str,
    rms_enabled: bool,
    loss: str,
    comparison: str,
    cohort: int,
    seed: int,
) -> dict:
    if input_kind not in ("log_mel", "waveform"):
        raise ValueError(f"unknown input kind: {input_kind}")
    if loss not in ("aam_softmax", "aam_softmax_plus_supcon_within_vowel"):
        raise ValueError(f"unknown loss: {loss}")
    if not isinstance(rms_enabled, bool):
        raise TypeError("RMS axis must be boolean")
    config_id = f"{input_kind}__{encoder}__rms-{'on' if rms_enabled else 'off'}__{loss}"
    return {
        "config_id": config_id,
        "run_id": f"sanity-c{cohort}-s{seed}__{config_id}",
        "comparison": comparison,
        "input": input_kind,
        "encoder": encoder,
        "rms_enabled": rms_enabled,
        "supcon_enabled": loss.endswith("plus_supcon_within_vowel"),
        "loss": loss,
        "cohort": cohort,
        "seed": seed,
    }


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _disk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sample_rss(pid: int) -> int:
    sampled = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(sampled.stdout.strip()) * 1024
    except ValueError:
        return 0


def _monitored_command(
    command: list[str],
    *,
    run_dir: Path,
    log_path: Path,
    budget: dict,
    seconds_remaining: float,
) -> dict:
    limits = budget["limits"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak_rss = 0
    limit_reason = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(limits["poll_interval_seconds"])
            peak_rss = max(peak_rss, _sample_rss(process.pid))
            if time.monotonic() - start > seconds_remaining:
                limit_reason = "run_wall_limit"
            elif peak_rss > limits["maximum_process_rss_bytes"]:
                limit_reason = "process_rss_limit"
            elif _disk_bytes(run_dir) > limits["maximum_run_artifact_bytes"]:
                limit_reason = "run_artifact_limit"
            elif shutil.disk_usage(ROOT).free < limits["minimum_free_disk_bytes"]:
                limit_reason = "free_disk_limit"
            if limit_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
    return {
        "command": command,
        "log": str(log_path.relative_to(ROOT)),
        "wall_seconds": time.monotonic() - start,
        "sampled_peak_rss_bytes": peak_rss,
        "exit_code": process.returncode,
        "limit_reason": limit_reason,
    }


def _check_history(path: Path) -> dict:
    updates = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if any(
                not math.isfinite(record[key])
                for key in ("loss", "gradient_norm", "learning_rate", "accuracy")
            ):
                raise ValueError("nonfinite training history")
            updates.append(record["update"])
    if updates != list(range(1, len(updates) + 1)):
        raise ValueError("training update sequence is incomplete or duplicated")
    return {"update_count": len(updates), "all_finite": True}


def _validation_metrics(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics.pop("checkpoint_sha256", None)
    metrics.pop("checkpoint_update", None)
    return metrics


def _archive_fixed_validation(run_dir: Path, selection_root: Path) -> None:
    for relative in (
        "data.json",
        "selections/enrollment-segments.jsonl",
        "selections/trials.jsonl",
    ):
        source = selection_root / relative
        destination = run_dir / "fixed-validation" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"archived fixed validation differs: {relative}")


def _command(row: dict, run_dir: Path, budget: dict) -> list[str]:
    protocol = budget["protocol"]
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "train",
        "--encoder",
        row["encoder"],
        "--cohort",
        str(row["cohort"]),
        "--rms",
        "on" if row["rms_enabled"] else "off",
        "--supcon",
        "on" if row["supcon_enabled"] else "off",
        "--seed",
        str(row["seed"]),
        "--maximum-updates",
        str(protocol["maximum_updates"]),
        "--warmup-updates",
        str(protocol["warmup_updates"]),
        "--validation-interval",
        str(protocol["validation_interval"]),
        "--workers",
        str(budget["environment"]["data_loader_workers"]),
        "--device",
        budget["environment"]["device"],
        "--eval-batch-size",
        str(budget["environment"]["evaluation_batch_size"]),
        "--output",
        str(run_dir),
    ]
    if row["input"] == "log_mel":
        statistics = (
            ROOT
            / f"artifacts/phoneme-speaker-encoder/cohort-10-rms-{'on' if row['rms_enabled'] else 'off'}/feature-statistics.json"
        )
        if not statistics.is_file():
            raise FileNotFoundError(statistics)
        command.extend(("--statistics", str(statistics)))
    return command


def _execute_one(row: dict, root: Path, budget: dict, matrix_sha: str) -> dict:
    run_dir = root / "runs" / row["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "sanity-result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["matrix_sha256"] != matrix_sha:
            raise ValueError("matrix checksum differs from existing run")
        if result["status"] in ("completed", "excluded", "failed"):
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            selection = ROOT / baseline["artifacts_root"] / "fixed-validation"
            _archive_fixed_validation(run_dir, selection)
            result["fixed_validation_archived"] = True
            result["artifact_bytes"] = _disk_bytes(run_dir)
            write_json(result_path, result)
            return result
    else:
        result = {
            "config_id": row["config_id"],
            "run_id": row["run_id"],
            "comparison": row["comparison"],
            "matrix_sha256": matrix_sha,
            "status": "running",
            "stages": [],
        }
    write_json(result_path, result)
    write_json(run_dir / "config.json", row)
    shutil.copyfile(BUDGET_PATH, run_dir / "execution-budget.json")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    manifest = ROOT / baseline["data"]["manifest"]
    selection = ROOT / baseline["artifacts_root"] / "fixed-validation"
    selection_data = json.loads((selection / "data.json").read_text(encoding="utf-8"))
    if selection_data["split"] != "validation":
        raise ValueError("only validation is allowed in sanity run")
    _archive_fixed_validation(run_dir, selection)
    data = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selection_root": str(selection),
        "selection_data_sha256": sha256_file(selection / "data.json"),
        "trial_sha256": sha256_file(selection / "selections/trials.jsonl"),
        "statistics_sha256": None,
    }
    if row["input"] == "log_mel":
        statistics = Path(_command(row, run_dir, budget)[-1])
        shutil.copyfile(statistics, run_dir / "feature-statistics.json")
        data["statistics_sha256"] = sha256_file(statistics)
    write_json(run_dir / "data.json", data)
    write_json(
        run_dir / "environment.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
            "precision": "float32",
            "mps_used": False,
        },
    )
    started = time.monotonic()

    def launch(command: list[str], stage: str) -> None:
        used = sum(item["wall_seconds"] for item in result["stages"])
        remaining = budget["limits"]["maximum_run_wall_seconds"] - used
        if remaining <= 0:
            raise RuntimeError("run_wall_limit")
        observed = _monitored_command(
            command,
            run_dir=run_dir,
            log_path=run_dir / "logs" / f"{stage}.log",
            budget=budget,
            seconds_remaining=remaining,
        )
        result["stages"].append({"stage": stage, **observed})
        write_json(result_path, result)
        if observed["limit_reason"]:
            raise RuntimeError(observed["limit_reason"])
        if observed["exit_code"] != 0:
            raise RuntimeError(f"{stage} exited {observed['exit_code']}")

    try:
        summary_path = run_dir / "training/summary.json"
        incomplete_summary = (
            summary_path.exists()
            and json.loads(summary_path.read_text(encoding="utf-8"))[
                "completed_updates"
            ]
            < budget["protocol"]["maximum_updates"]
        )
        if not summary_path.exists() or incomplete_summary:
            command = _command(row, run_dir, budget)
            checkpoint = run_dir / "checkpoints/last.pt"
            if checkpoint.exists():
                launch(command + ["--resume"], "resume")
            elif (
                row
                == expand_matrix(
                    json.loads(SEARCH_PATH.read_text(encoding="utf-8")), budget
                )[0]
            ):
                # One formal run crosses a checkpoint boundary to exercise resume.
                launch(command + ["--target-update", "250"], "staged-250")
                launch(command + ["--resume"], "resume")
            else:
                launch(command, "train")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["history"] = _check_history(run_dir / "training/history.jsonl")
        if result["history"]["update_count"] != summary["completed_updates"]:
            raise ValueError("history and checkpoint update differ")
        if summary["best"]["update"] is None:
            raise ValueError("no selected validation checkpoint")
        selected_update = summary["best"]["update"]
        checkpoint = run_dir / "checkpoints/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics_path = (
            run_dir / f"validation/update-{selected_update:06d}/metrics/validation.json"
        )
        before = _validation_metrics(metrics_path)
        if before["partial"] or before["split"] != "validation":
            raise ValueError("partial or non-validation metrics in formal run")
        evaluation = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "evaluate",
            "--run-dir",
            str(run_dir),
            "--batch-size",
            str(budget["environment"]["evaluation_batch_size"]),
        ]
        launch(evaluation, "repeat-evaluation")
        after = _validation_metrics(metrics_path)
        if before != after:
            raise ValueError("same-checkpoint validation metrics or scores differ")
        result.update(
            {
                "status": "completed",
                "completed_updates": summary["completed_updates"],
                "early_stopped": summary["completed_updates"]
                < budget["protocol"]["maximum_updates"],
                "selected_checkpoint_update": selected_update,
                "selected_checkpoint_sha256": sha256_file(checkpoint),
                "validation_macro_eer": after["macro_eer"],
                "validation_score_sha256": after["score_sha256"],
                "validation_trial_count": after["trial_count"],
                "repeat_evaluation_identical": True,
                "parameter_count": summary["parameter_count"],
            }
        )
    except Exception as error:  # noqa: BLE001 - record every formal-run failure
        result["status"] = "excluded" if str(error).endswith("_limit") else "failed"
        result["reason"] = f"{type(error).__name__}: {error}"
    result["wall_seconds_this_invocation"] = time.monotonic() - started
    result["sampled_peak_rss_bytes"] = max(
        (stage["sampled_peak_rss_bytes"] for stage in result["stages"]), default=0
    )
    result["artifact_bytes"] = _disk_bytes(run_dir)
    result["fixed_validation_archived"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    args = parser.parse_args()
    search = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    rows = expand_matrix(search, budget)
    root = (
        ROOT / "artifacts/phoneme-speaker-encoder/comparisons" / budget["comparison_id"]
    )
    root.mkdir(parents=True, exist_ok=True)
    matrix = {
        "schema_version": 1,
        "design_version": search["design_version"],
        "search_space_sha256": sha256_file(SEARCH_PATH),
        "execution_budget_sha256": sha256_file(BUDGET_PATH),
        "runs": rows,
    }
    matrix_sha = _digest(matrix)
    matrix["sha256"] = matrix_sha
    matrix_path = root / "matrix.json"
    if (
        matrix_path.exists()
        and json.loads(matrix_path.read_text(encoding="utf-8")) != matrix
    ):
        raise ValueError("frozen matrix differs from current source")
    write_json(matrix_path, matrix)
    shutil.copyfile(BUDGET_PATH, root / "execution-budget.json")
    if args.command == "plan":
        print(json.dumps({"matrix": str(matrix_path), "runs": len(rows)}))
        return
    results = []
    for index, row in enumerate(rows, 1):
        result_path = root / "runs" / row["run_id"] / "sanity-result.json"
        if args.command == "run":
            print(f"[{index}/18] {row['run_id']}", flush=True)
            result = _execute_one(row, root, budget, matrix_sha)
            print(
                json.dumps(
                    {
                        "config_id": row["config_id"],
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "macro_eer": result.get("validation_macro_eer"),
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        else:
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {"config_id": row["config_id"], "status": "not_started"}
            )
        results.append(result)
        write_json(
            root / "results.json",
            {"matrix_sha256": matrix_sha, "results": results},
        )
    counts = {
        status: sum(r["status"] == status for r in results)
        for status in ("completed", "excluded", "failed", "not_started")
    }
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
