"""Freeze and execute the 54 CPU-only Phase 3 seventy-speaker comparisons."""

from __future__ import annotations

import argparse
import fcntl
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import torch
from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from run_phase3 import _code_sha256

SEARCH_PATH = BASE / "config/search-space.json"
BUDGET_PATH = BASE / "config/full-execution-budget.json"
BASELINE_PATH = BASE / "config/baseline-log-mel.json"
TRAIN_SCRIPT = BASE / "scripts/run_phase3.py"

# The same-condition 10-speaker run times only determine scheduling order.
# Neither their EER nor the 70-speaker pilot EER is consulted.
SCHEDULING_SECONDS_BY_ENCODER = {
    "waveform_cnn_k240_context27": 23.3 * 60,
    "framewise_cnn": 21.5 * 60,
    "waveform_cnn_k80": 18.8 * 60,
    "waveform_cnn_k240": 18.6 * 60,
    "tdnn": 9.4 * 60,
    "statistics_mlp": 7.4 * 60,
}


def expand_matrix(search: dict, budget: dict) -> list[dict]:
    protocol = budget["protocol"]
    if search["design_version"] != budget["design_version"]:
        raise ValueError("design versions differ")
    if protocol["cohort"] != 70 or protocol["seeds"] != search["seeds"]:
        raise ValueError("full comparison cohort or seeds differ from search plan")
    configs = []
    for family in search["families"]:
        axes = family["axes"]
        combinations = list(
            itertools.product(axes["encoder"], axes["rms_normalization"], axes["loss"])
        )
        if len(combinations) != family["combination_count"]:
            raise ValueError("main family count mismatch")
        configs.extend(
            (family["input"], encoder, rms, loss, "main")
            for encoder, rms, loss in combinations
        )
    if len(configs) != search["main_combination_count"]:
        raise ValueError("main configuration count mismatch")
    configs.extend(
        (
            item["input"],
            item["encoder"],
            item["rms_normalization"],
            item["loss"],
            item["id"],
        )
        for item in search["limited_comparisons"]
    )
    rows = []
    for input_kind, encoder, rms, loss, comparison in configs:
        if input_kind not in ("log_mel", "waveform"):
            raise ValueError("unknown input kind")
        if loss not in ("aam_softmax", "aam_softmax_plus_supcon_within_vowel"):
            raise ValueError("unknown loss")
        if not isinstance(rms, bool):
            raise TypeError("RMS axis must be boolean")
        config_id = f"{input_kind}__{encoder}__rms-{'on' if rms else 'off'}__{loss}"
        for seed in protocol["seeds"]:
            rows.append(
                {
                    "config_id": config_id,
                    "run_id": f"full-c70-s{seed}__{config_id}",
                    "comparison": comparison,
                    "input": input_kind,
                    "encoder": encoder,
                    "rms_enabled": rms,
                    "supcon_enabled": loss.endswith("plus_supcon_within_vowel"),
                    "loss": loss,
                    "cohort": 70,
                    "seed": seed,
                }
            )
    ids = [row["run_id"] for row in rows]
    config_ids = {row["config_id"] for row in rows}
    if (
        len(rows)
        != search["runs"]["main_full_runs_70_speakers"]
        + search["runs"]["limited_full_runs_70_speakers"]
        or len(rows) != 54
        or len(config_ids) != search["total_combination_count"]
        or len(ids) != len(set(ids))
    ):
        raise ValueError("missing or duplicate full comparison runs")
    return sorted(
        rows,
        key=lambda row: (
            -SCHEDULING_SECONDS_BY_ENCODER[row["encoder"]],
            row["config_id"],
            row["seed"],
        ),
    )


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _archive_source(root: Path) -> None:
    paths = [
        *sorted((BASE / "phase3_data").glob("*.py")),
        *sorted((BASE / "phase3_train").glob("*.py")),
        TRAIN_SCRIPT,
        Path(__file__),
        SEARCH_PATH,
        BUDGET_PATH,
        BASELINE_PATH,
        BASE / "config/experiment-protocol.json",
        BASE / "config/log-mel-encoders.json",
        BASE / "config/waveform-encoders.json",
    ]
    for source in paths:
        destination = root / "source-snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"frozen source differs: {source}")


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


def _command(row: dict, run_dir: Path, budget: dict, selection: Path) -> list[str]:
    protocol = budget["protocol"]
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "train",
        "--encoder",
        row["encoder"],
        "--cohort",
        "70",
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
        "--selections",
        str(selection),
        "--compact-validation",
        "--output",
        str(run_dir),
    ]
    if row["input"] == "log_mel":
        statistics = (
            ROOT
            / f"artifacts/phoneme-speaker-encoder/cohort-70-rms-{'on' if row['rms_enabled'] else 'off'}/feature-statistics.json"
        )
        if not statistics.is_file():
            raise FileNotFoundError(statistics)
        command.extend(("--statistics", str(statistics)))
    return command


def _monitor(
    command: list[str],
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
        checked_storage_at = start
        while process.poll() is None:
            time.sleep(limits["poll_interval_seconds"])
            peak_rss = max(peak_rss, _sample_rss(process.pid))
            elapsed = time.monotonic() - start
            if elapsed > seconds_remaining:
                limit_reason = "run_wall_limit"
            elif peak_rss > limits["maximum_process_rss_bytes"]:
                limit_reason = "process_rss_limit"
            elif time.monotonic() - checked_storage_at >= 30:
                checked_storage_at = time.monotonic()
                if _disk_bytes(run_dir) > limits["maximum_run_artifact_bytes"]:
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


def _check_history(path: Path, completed: int) -> None:
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
    if updates != list(range(1, completed + 1)):
        raise ValueError("training history differs from completed updates")


def _execute_one(row: dict, root: Path, budget: dict, matrix_sha: str) -> dict:
    run_dir = root / "runs" / row["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "full-result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["matrix_sha256"] != matrix_sha:
            raise ValueError("matrix checksum differs from existing run")
        if result["status"] in ("completed", "excluded", "failed"):
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
    selection = root / "fixed-validation"
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    manifest = ROOT / baseline["data"]["manifest"]
    selection_data = json.loads((selection / "data.json").read_text(encoding="utf-8"))
    if selection_data["split"] != "validation":
        raise ValueError("only validation split is allowed")
    data = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selection_root": str(selection),
        "selection_data_sha256": sha256_file(selection / "data.json"),
        "trial_sha256": sha256_file(selection / "selections/trials.jsonl"),
        "statistics_sha256": None,
    }
    command = _command(row, run_dir, budget, selection)
    if row["input"] == "log_mel":
        statistics = Path(command[command.index("--statistics") + 1])
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
    start = time.monotonic()
    try:

        def launch(command: list[str], stage: str) -> None:
            used = sum(item["wall_seconds"] for item in result["stages"])
            remaining = budget["limits"]["maximum_run_wall_seconds"] - used
            if remaining <= 0:
                raise RuntimeError("run_wall_limit")
            observed = _monitor(
                command,
                run_dir,
                run_dir / "logs" / f"{stage}.log",
                budget,
                remaining,
            )
            result["stages"].append({"stage": stage, **observed})
            write_json(result_path, result)
            if observed["limit_reason"]:
                raise RuntimeError(observed["limit_reason"])
            if observed["exit_code"] != 0:
                raise RuntimeError(f"{stage} exited {observed['exit_code']}")

        checkpoint = run_dir / "checkpoints/last.pt"
        if checkpoint.exists():
            command.append("--resume")
        launch(command, f"train-{len(result['stages']) + 1}")
        summary = json.loads((run_dir / "training/summary.json").read_text())
        completed = summary["completed_updates"]
        if not (completed == 30000 or summary["early_stopped"]):
            raise ValueError("run stopped without maximum updates or early stopping")
        _check_history(run_dir / "training/history.jsonl", completed)
        selected = summary["best"]["update"]
        if selected is None:
            raise ValueError("no validation checkpoint selected")
        metric_path = (
            run_dir / f"validation/update-{selected:06d}/metrics/validation.json"
        )
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        score_path = (
            run_dir / f"validation/update-{selected:06d}/scores/validation.jsonl"
        )
        if metrics["partial"] or metrics["split"] != "validation":
            raise ValueError("partial or non-validation selected metrics")
        if metrics["score_sha256"] != sha256_file(score_path):
            raise ValueError("selected validation score checksum mismatch")
        if metrics["macro_eer"] != summary["best"]["eer"]:
            raise ValueError("selected EER differs from training summary")
        before = dict(metrics)
        launch(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "evaluate",
                "--run-dir",
                str(run_dir),
                "--batch-size",
                str(budget["environment"]["evaluation_batch_size"]),
            ],
            f"repeat-evaluation-{len(result['stages']) + 1}",
        )
        after = json.loads(metric_path.read_text(encoding="utf-8"))
        after.pop("checkpoint_sha256", None)
        after.pop("checkpoint_update", None)
        if before != after or before["score_sha256"] != sha256_file(score_path):
            raise ValueError("same-checkpoint validation differs on repeat")
        result.update(
            {
                "status": "completed",
                "completed_updates": completed,
                "early_stopped": summary["early_stopped"],
                "selected_checkpoint_update": selected,
                "selected_checkpoint_sha256": sha256_file(
                    run_dir / "checkpoints/best.pt"
                ),
                "validation_macro_eer": metrics["macro_eer"],
                "validation_score_sha256": metrics["score_sha256"],
                "validation_trial_count": metrics["trial_count"],
                "parameter_count": summary["parameter_count"],
                "repeat_evaluation_identical": True,
            }
        )
    except Exception as error:  # noqa: BLE001 - preserve the full matrix failure record
        result["status"] = "excluded" if str(error).endswith("_limit") else "failed"
        result["reason"] = f"{type(error).__name__}: {error}"
    result["wall_seconds_this_invocation"] = time.monotonic() - start
    result["sampled_peak_rss_bytes"] = max(
        (stage["sampled_peak_rss_bytes"] for stage in result["stages"]), default=0
    )
    result["artifact_bytes"] = _disk_bytes(run_dir)
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
    _archive_source(root)
    matrix = {
        "schema_version": 1,
        "design_version": search["design_version"],
        "search_space_sha256": sha256_file(SEARCH_PATH),
        "execution_budget_sha256": sha256_file(BUDGET_PATH),
        "training_code_sha256": _code_sha256(),
        "runner_sha256": sha256_file(Path(__file__)),
        "runs": rows,
    }
    matrix_sha = _digest(matrix)
    matrix["sha256"] = matrix_sha
    matrix_path = root / "matrix.json"
    if matrix_path.exists() and json.loads(matrix_path.read_text()) != matrix:
        raise ValueError("frozen matrix differs from current source")
    write_json(matrix_path, matrix)
    shutil.copyfile(BUDGET_PATH, root / "execution-budget.json")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    canonical_selection = ROOT / baseline["artifacts_root"] / "fixed-validation"
    selection = root / "fixed-validation"
    for relative in (
        "data.json",
        "selections/enrollment-segments.jsonl",
        "selections/trials.jsonl",
    ):
        source = canonical_selection / relative
        destination = selection / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise ValueError(f"fixed validation differs: {relative}")
    if args.command == "plan":
        print(json.dumps({"matrix": str(matrix_path), "runs": len(rows)}))
        return
    if args.command == "run":
        with (root / "runner.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "full comparison runner is already active"
                ) from error
            with ThreadPoolExecutor(
                max_workers=budget["environment"]["maximum_parallel_runs"]
            ) as pool:
                futures = {
                    pool.submit(_execute_one, row, root, budget, matrix_sha): row
                    for row in rows
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as error:  # noqa: BLE001 - never lose a matrix row
                        result_path = root / "runs" / row["run_id"] / "full-result.json"
                        result = (
                            json.loads(result_path.read_text(encoding="utf-8"))
                            if result_path.exists()
                            else {
                                "config_id": row["config_id"],
                                "run_id": row["run_id"],
                                "comparison": row["comparison"],
                                "matrix_sha256": matrix_sha,
                                "stages": [],
                            }
                        )
                        result["status"] = "failed"
                        result["reason"] = f"{type(error).__name__}: {error}"
                        write_json(result_path, result)
                    print(
                        json.dumps(
                            {
                                "run_id": row["run_id"],
                                "status": result["status"],
                                "reason": result.get("reason"),
                                "macro_eer": result.get("validation_macro_eer"),
                            }
                        ),
                        flush=True,
                    )
    results = []
    for row in rows:
        result_path = root / "runs" / row["run_id"] / "full-result.json"
        results.append(
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {
                "run_id": row["run_id"],
                "config_id": row["config_id"],
                "status": "not_started",
            }
        )
    write_json(root / "results.json", {"matrix_sha256": matrix_sha, "results": results})
    print(
        json.dumps(
            {
                status: sum(result["status"] == status for result in results)
                for status in (
                    "completed",
                    "running",
                    "excluded",
                    "failed",
                    "not_started",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
