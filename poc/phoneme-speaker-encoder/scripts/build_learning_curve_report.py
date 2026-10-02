"""Summarize the selected encoder's nested speaker-cohort validation runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CURVE_ROOT = (
    ROOT / "artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2"
)
FULL_ROOT = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2"
)
CONFIG_ID = "log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel"
COHORTS = (10, 25, 50, 70)
VOWELS = ("a", "i", "u", "e", "o")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(cohort: int, seed: int) -> Path:
    if cohort == 70:
        return FULL_ROOT / "runs" / f"full-c70-s{seed}__{CONFIG_ID}"
    return CURVE_ROOT / "runs" / f"curve-c{cohort}-s{seed}__{CONFIG_ID}"


def _record(cohort: int, seed: int) -> dict:
    run_dir = _run_dir(cohort, seed)
    summary = _read(run_dir / "training/summary.json")
    selected = summary["best"]["update"]
    metrics = _read(
        run_dir / f"validation/update-{selected:06d}/metrics/validation.json"
    )
    if (
        metrics["partial"]
        or metrics["split"] != "validation"
        or metrics["macro_eer"] != summary["best"]["eer"]
        or (summary["completed_updates"] != 30000 and not summary["early_stopped"])
    ):
        raise ValueError(f"incomplete or inconsistent run: {run_dir}")
    primary = metrics["roles"]["verification"]["10"]
    return {
        "cohort": cohort,
        "seed": seed,
        "selected_update": selected,
        "completed_updates": summary["completed_updates"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "macro_eer": metrics["macro_eer"],
        "vowel_eer": {v: primary["vowels"][v]["eer"] for v in VOWELS},
        "cross_text_macro_eer": metrics["roles"]["cross_text_verification"]["10"][
            "macro_eer"
        ],
        "short_macro_eer": metrics["strata"]["duration"]["30-49ms"]["macro_eer"],
        "metric_path": str(
            (
                run_dir / f"validation/update-{selected:06d}/metrics/validation.json"
            ).relative_to(ROOT)
        ),
    }


def _aggregate(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    if any(value is None for value in values):
        raise ValueError(f"missing {key} for one of the three seeds")
    return {
        "mean": statistics.mean(values),
        "std_population": statistics.pstdev(values),
    }


def build() -> dict:
    plan = _read(CURVE_ROOT / "plan.json")
    if plan["selected_config_id"] != CONFIG_ID:
        raise ValueError("learning-curve plan selected a different configuration")
    seeds = sorted({row["seed"] for row in plan["runs"]})
    if len(seeds) != 3 or len(plan["runs"]) != 9:
        raise ValueError("expected nine additional runs and three seeds")
    results = []
    cohorts = {}
    for cohort in COHORTS:
        rows = [_record(cohort, seed) for seed in seeds]
        results.extend(rows)
        cohorts[str(cohort)] = {
            "macro_eer": _aggregate(rows, "macro_eer"),
            "cross_text_macro_eer": _aggregate(rows, "cross_text_macro_eer"),
            "short_macro_eer": _aggregate(rows, "short_macro_eer"),
            "vowel_eer": {
                vowel: {
                    "mean": statistics.mean(row["vowel_eer"][vowel] for row in rows),
                    "std_population": statistics.pstdev(
                        row["vowel_eer"][vowel] for row in rows
                    ),
                }
                for vowel in VOWELS
            },
        }
    return {
        "schema_version": 1,
        "selected_config_id": CONFIG_ID,
        "seeds": seeds,
        "cohorts": cohorts,
        "runs": results,
        "test_used": False,
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Phase 3採用encoderの学習話者数別validation結果",
        "",
        f"採用設定: `{CONFIG_ID}`。10/25/50話者の各3 seedを追加学習し、70話者の3 seedを再利用した。testは未使用。",
        "",
        "| 学習話者数 | macro EER 平均 | seed間標準偏差 | cross-text | 30–49 ms |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for cohort in COHORTS:
        row = result["cohorts"][str(cohort)]
        lines.append(
            f"| {cohort} | {row['macro_eer']['mean']:.3%} | "
            f"{row['macro_eer']['std_population']:.3%} | "
            f"{row['cross_text_macro_eer']['mean']:.3%} | "
            f"{row['short_macro_eer']['mean']:.3%} |"
        )
    lines.extend(
        [
            "",
            "3 seedは同じvalidation話者を使うため、独立した評価話者集合ではない。",
            "70話者で選択済みの構成に条件付けた診断であり、結果で候補や閾値を選び直さない。",
            "",
            "詳細なseed別・母音別値は同じディレクトリの`learning-curve-results.json`に保存した。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build()
    (CURVE_ROOT / "learning-curve-results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (CURVE_ROOT / "learning-curve-results.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(_markdown(result))


if __name__ == "__main__":
    main()
