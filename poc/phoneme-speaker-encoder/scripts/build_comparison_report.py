"""Build a standalone browser report from the completed validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
VOWELS = ("a", "i", "u", "e", "o")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def average(values: list) -> float | None:
    if any(value is None for value in values):
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite report metric")
    return statistics.mean(values)


def build(comparison: Path) -> dict:
    matrix = read(comparison / "matrix.json")
    summary = read(comparison / "results.json")
    rows = matrix["runs"]
    if len(rows) != 54 or len({row["run_id"] for row in rows}) != 54:
        raise ValueError("expected 54 unique matrix runs")
    recorded = {item["run_id"]: item for item in summary["results"]}
    if (
        len(summary["results"]) != 54
        or len(recorded) != 54
        or set(recorded) != {row["run_id"] for row in rows}
    ):
        raise ValueError("summary does not match matrix")
    if summary["matrix_sha256"] != matrix["sha256"]:
        raise ValueError("summary matrix hash differs")
    groups = {}
    for row in rows:
        run_dir = comparison / "runs" / row["run_id"]
        result = read(run_dir / "full-result.json")
        if result != recorded[row["run_id"]]:
            raise ValueError(f"summary differs: {row['run_id']}")
        if result["status"] != "completed" or not result["repeat_evaluation_identical"]:
            raise ValueError(f"incomplete run: {row['run_id']}")
        if result["matrix_sha256"] != matrix["sha256"]:
            raise ValueError("run matrix hash differs")
        update = result["selected_checkpoint_update"]
        metric_path = (
            run_dir / f"validation/update-{update:06d}/metrics/validation.json"
        )
        metric = read(metric_path)
        if metric["partial"] or metric["split"] != "validation":
            raise ValueError("full validation metrics required")
        if (
            metric["primary_metric"]
            != "validation_verification_10_enrollment_macro_eer"
        ):
            raise ValueError("unexpected ranking metric")
        if metric["macro_eer"] != result["validation_macro_eer"]:
            raise ValueError("selected metric differs from result")
        if metric["score_sha256"] != result["validation_score_sha256"]:
            raise ValueError("selected score hash differs from result")
        if metric["trial_count"] != result["validation_trial_count"]:
            raise ValueError("selected trial count differs from result")
        if not (run_dir / "checkpoints/best.pt").is_file():
            raise FileNotFoundError(run_dir / "checkpoints/best.pt")
        training = read(run_dir / "training/summary.json")
        if training["completed_updates"] != result["completed_updates"]:
            raise ValueError("training updates differ")
        if not (result["completed_updates"] == 30000 or result["early_stopped"]):
            raise ValueError("training did not reach a defined stopping condition")
        if any(
            stage.get("command") and stage["exit_code"] != 0
            for stage in result["stages"]
        ):
            raise ValueError("failed recorded command stage")
        primary = metric["roles"]["verification"]["10"]
        item = {
            "seed": row["seed"],
            "run_id": row["run_id"],
            "eer": result["validation_macro_eer"],
            "cross_text_eer": metric["roles"]["cross_text_verification"]["10"][
                "macro_eer"
            ],
            "short_eer": metric["strata"]["duration"]["30-49ms"]["macro_eer"],
            "vowels": {vowel: primary["vowels"][vowel]["eer"] for vowel in VOWELS},
            "enrollment": {
                count: metric["roles"]["verification"][count]["macro_eer"]
                for count in ("1", "5", "10")
            },
            "strata": {
                dimension: {
                    value: report["macro_eer"] for value, report in values.items()
                }
                for dimension, values in metric["strata"].items()
            },
            "completed_updates": result["completed_updates"],
            "selected_update": update,
            "early_stopped": result["early_stopped"],
            "recorded_seconds": sum(
                stage["wall_seconds"] for stage in result["stages"]
            ),
            "train_seconds": sum(
                stage["wall_seconds"]
                for stage in result["stages"]
                if stage["stage"].startswith("train")
            ),
            "peak_rss_bytes": result["sampled_peak_rss_bytes"],
            "artifact_bytes": result["artifact_bytes"],
            "trial_count": result["validation_trial_count"],
            "checkpoint_sha256": result["selected_checkpoint_sha256"],
            "score_sha256": result["validation_score_sha256"],
            "recovered": result.get("recovered_after_runner_interruption", False),
            "history": [],
        }
        for path in sorted(
            (run_dir / "validation").glob("update-*/metrics/validation.json")
        ):
            history_update = int(path.parents[1].name.removeprefix("update-"))
            if history_update == update:
                history_eer = item["eer"]
            else:
                history_eer = read(path)["macro_eer"]
            item["history"].append({"update": history_update, "eer": history_eer})
        group = groups.setdefault(
            row["config_id"],
            {
                "config_id": row["config_id"],
                "input": row["input"],
                "encoder": row["encoder"],
                "rms": row["rms_enabled"],
                "supcon": row["supcon_enabled"],
                "comparison": row["comparison"],
                "parameter_count": result["parameter_count"],
                "runs": [],
            },
        )
        if group["parameter_count"] != result["parameter_count"]:
            raise ValueError("encoder parameter counts differ between seeds")
        group["runs"].append(item)
        del metric
    expected_seeds = set(
        read(comparison / "execution-budget.json")["protocol"]["seeds"]
    )
    for group in groups.values():
        runs = sorted(group["runs"], key=lambda run: run["seed"])
        if len(runs) != 3 or {run["seed"] for run in runs} != expected_seeds:
            raise ValueError("missing seed")
        group["runs"] = runs
        group["mean_eer"] = average([run["eer"] for run in runs])
        group["std_eer"] = statistics.pstdev(run["eer"] for run in runs)
        for field in ("cross_text_eer", "short_eer", "train_seconds"):
            group[field] = average([run[field] for run in runs])
        group["vowels"] = {
            vowel: average([run["vowels"][vowel] for run in runs]) for vowel in VOWELS
        }
        group["enrollment"] = {
            count: average([run["enrollment"][count] for run in runs])
            for count in ("1", "5", "10")
        }
        group["strata"] = {
            dimension: {
                value: average([run["strata"][dimension].get(value) for run in runs])
                for value in values
            }
            for dimension, values in runs[0]["strata"].items()
        }
        group["recorded_seconds"] = sum(run["recorded_seconds"] for run in runs)
        group["peak_rss_bytes"] = max(run["peak_rss_bytes"] for run in runs)
        group["artifact_bytes"] = sum(run["artifact_bytes"] for run in runs)
    ranked = sorted(
        groups.values(), key=lambda group: (group["mean_eer"], group["config_id"])
    )
    if len(ranked) != 18:
        raise ValueError("expected 18 configurations")
    for rank, group in enumerate(ranked, 1):
        group["rank"] = rank
        group["delta_eer"] = group["mean_eer"] - ranked[0]["mean_eer"]
        group["eligible"] = group["mean_eer"] <= ranked[0]["mean_eer"] + 0.001
    return {
        "comparison_id": comparison.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": matrix["sha256"],
        "results_sha256": digest(comparison / "results.json"),
        "source_directory": str(comparison),
        "seed_count": 3,
        "run_count": 54,
        "config_count": 18,
        "eligible_count": sum(group["eligible"] for group in ranked),
        "selection_finalized": False,
        "missing_selection_evidence": [
            "共通benchmarkの推論時間・最大メモリ",
            "対応付きbootstrapの差の95%区間（報告用）",
            "採用構成の正式記録",
        ],
        "groups": ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    budget = read(BASE / "config/full-execution-budget.json")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=ROOT
        / "artifacts/phoneme-speaker-encoder/comparisons"
        / budget["comparison_id"],
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    comparison = args.comparison_dir.resolve()
    output = args.output_dir or comparison / "browser-report"
    data = build(comparison)
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    template = (BASE / "reports/comparison.html").read_text(encoding="utf-8")
    embedded = (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranking.json").write_text(serialized + "\n", encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("__REPORT_DATA__", embedded), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output / "index.html"),
                "configurations": len(data["groups"]),
                "runs": data["run_count"],
                "eligible": data["eligible_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
