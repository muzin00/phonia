"""Collect validation quality-stratum evidence for all 54 selected checkpoints."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMPARISON = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2"
)
OUTPUT = ROOT / "artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2"
SELECTED = "log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel"
DIMENSIONS = ("is_devoiced", "quality_flags", "is_long")
VOWELS = ("a", "i", "u", "e", "o")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(row: dict) -> dict:
    if row["status"] != "completed" or not row["repeat_evaluation_identical"]:
        raise ValueError(f"incomplete comparison run: {row['run_id']}")
    seed_match = re.search(r"-s(\d+)__", row["run_id"])
    if seed_match is None:
        raise ValueError(f"invalid run ID: {row['run_id']}")
    path = (
        COMPARISON
        / "runs"
        / row["run_id"]
        / f"validation/update-{row['selected_checkpoint_update']:06d}/metrics/validation.json"
    )
    metrics = _read(path)
    if (
        metrics["split"] != "validation"
        or metrics["partial"]
        or metrics["macro_eer"] != row["validation_macro_eer"]
    ):
        raise ValueError(f"comparison metric mismatch: {row['run_id']}")
    strata = {}
    for dimension in DIMENSIONS:
        strata[dimension] = {}
        for value, report in metrics["strata"].get(dimension, {}).items():
            strata[dimension][value] = {
                "macro_eer": report["macro_eer"],
                "vowels": {
                    vowel: {
                        "eer": report.get(vowel, {}).get("eer"),
                        "query_count": report.get(vowel, {}).get("query_count", 0),
                        "speaker_count": report.get(vowel, {}).get("speaker_count", 0),
                        "trial_count": report.get(vowel, {}).get("trial_count", 0),
                        "fixed_threshold_rates": report.get(vowel, {}).get(
                            "fixed_threshold_rates"
                        ),
                    }
                    for vowel in VOWELS
                },
            }
    return {
        "run_id": row["run_id"],
        "config_id": row["config_id"],
        "seed": int(seed_match.group(1)),
        "macro_eer": row["validation_macro_eer"],
        "strata": strata,
        "metric_path": str(path.relative_to(ROOT)),
    }


def _selected_summary(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["config_id"] == SELECTED]
    if len(selected) != 3 or len({row["seed"] for row in selected}) != 3:
        raise ValueError("expected three selected configuration seeds")
    result = {}
    for dimension in DIMENSIONS:
        values = sorted(
            set().union(*(set(row["strata"][dimension]) for row in selected))
        )
        result[dimension] = {}
        for value in values:
            reports = [row["strata"][dimension].get(value) for row in selected]
            eers = [report["macro_eer"] for report in reports if report]
            result[dimension][value] = {
                "seed_count": len(eers),
                "mean_macro_eer": statistics.mean(eers)
                if len(eers) == 3 and all(eer is not None for eer in eers)
                else None,
                "mean_eer_by_vowel": {
                    vowel: statistics.mean(vowel_eers) if len(vowel_eers) == 3 else None
                    for vowel in VOWELS
                    if (
                        vowel_eers := [
                            report["vowels"][vowel]["eer"]
                            for report in reports
                            if report and report["vowels"][vowel]["eer"] is not None
                        ]
                    )
                },
                "mean_query_count_by_vowel": {
                    vowel: statistics.mean(
                        report["vowels"][vowel]["query_count"]
                        for report in reports
                        if report
                    )
                    for vowel in VOWELS
                },
            }
    return result


def main() -> None:
    results = _read(COMPARISON / "results.json")["results"]
    if len(results) != 54:
        raise ValueError("expected 54 comparison runs")
    rows = [_run(row) for row in results]
    selected = _selected_summary(rows)
    result = {
        "schema_version": 1,
        "dimensions": list(DIMENSIONS),
        "runs": rows,
        "selected_config_id": SELECTED,
        "selected_summary": selected,
        "test_used": False,
    }
    output = OUTPUT / "quality-strata-results.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(rows)} validation runs to {output}")
    for dimension, values in selected.items():
        print(
            dimension, {key: report["mean_macro_eer"] for key, report in values.items()}
        )


if __name__ == "__main__":
    main()
