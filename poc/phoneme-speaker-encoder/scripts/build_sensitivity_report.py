"""Summarize the selected encoder's three-seed validation sensitivity runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2"
INPUT = OUTPUT / "sensitivity"
SEEDS = (20260926, 20260927, 20260928)
VOWELS = ("a", "i", "u", "e", "o")
CONDITIONS = (
    "baseline-full",
    "baseline-common",
    "boundary--240-samples",
    "boundary--120-samples",
    "boundary-+120-samples",
    "boundary-+240-samples",
    "gain--6-db",
    "gain-+6-db",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _macro_rate(metrics: dict, name: str, rate: str) -> float:
    values = [
        metrics["vowels"][vowel]["fixed_threshold_rates"][name][rate]
        for vowel in VOWELS
    ]
    if any(value is None for value in values):
        raise ValueError(f"undefined {rate} for {name}")
    return statistics.mean(values)


def build() -> dict:
    per_seed = []
    for seed in SEEDS:
        summary = _read(INPUT / str(seed) / "summary.json")
        if summary["seed"] != seed or summary["test_used"]:
            raise ValueError(f"invalid sensitivity summary for seed {seed}")
        conditions = {}
        for condition in CONDITIONS:
            metrics = _read(INPUT / str(seed) / condition / "metrics.json")
            if (
                metrics["score_sha256"]
                != summary["conditions"][condition]["score_sha256"]
            ):
                raise ValueError(f"score checksum mismatch: {seed}, {condition}")
            conditions[condition] = {
                "macro_eer": metrics["macro_eer"],
                "far_1pct_macro_far": _macro_rate(metrics, "far_1pct", "far"),
                "far_1pct_macro_frr": _macro_rate(metrics, "far_1pct", "frr"),
                "far_0_1pct_macro_far": _macro_rate(metrics, "far_0_1pct", "far"),
                "far_0_1pct_macro_frr": _macro_rate(metrics, "far_0_1pct", "frr"),
                "trial_count": metrics["trial_count"],
            }
        per_seed.append(
            {
                "seed": seed,
                "common_query_count": summary["common_query_count"],
                "excluded_query_count": summary["excluded_query_count"],
                "conditions": conditions,
            }
        )
    aggregate = {}
    for condition in CONDITIONS:
        baseline = (
            "baseline-common"
            if condition.startswith("boundary-") or condition == "baseline-common"
            else "baseline-full"
        )
        values = [row["conditions"][condition]["macro_eer"] for row in per_seed]
        aggregate[condition] = {
            "macro_eer_mean": statistics.mean(values),
            "macro_eer_std_population": statistics.pstdev(values),
            "macro_eer_difference_mean": statistics.mean(
                row["conditions"][condition]["macro_eer"]
                - row["conditions"][baseline]["macro_eer"]
                for row in per_seed
            ),
            "far_1pct_macro_far_mean": statistics.mean(
                row["conditions"][condition]["far_1pct_macro_far"] for row in per_seed
            ),
            "far_1pct_macro_frr_mean": statistics.mean(
                row["conditions"][condition]["far_1pct_macro_frr"] for row in per_seed
            ),
            "far_0_1pct_macro_far_mean": statistics.mean(
                row["conditions"][condition]["far_0_1pct_macro_far"] for row in per_seed
            ),
            "far_0_1pct_macro_frr_mean": statistics.mean(
                row["conditions"][condition]["far_0_1pct_macro_frr"] for row in per_seed
            ),
        }
    return {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "conditions": aggregate,
        "per_seed": per_seed,
        "test_used": False,
    }


def _markdown(result: dict) -> str:
    lines = [
        "# Phase 3採用encoderのvalidation感度診断",
        "",
        "採用済み70話者モデルの3 seedで、登録profileと閾値を固定し、queryだけを変換した。testは未使用。",
        "",
        "| 条件 | macro EER | 元条件との差 | FAR 1%閾値でのFAR | 同FRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CONDITIONS:
        row = result["conditions"][name]
        lines.append(
            f"| `{name}` | {row['macro_eer_mean']:.3%} | "
            f"{row['macro_eer_difference_mean']:+.3%} | "
            f"{row['far_1pct_macro_far_mean']:.3%} | "
            f"{row['far_1pct_macro_frr_mean']:.3%} |"
        )
    lines.extend(
        [
            "",
            "境界条件は元WAVの区間を±120/240 sample（±5/10 ms）平行移動し、全条件で有効なquery集合の元条件と比較した。",
            "音量条件はcrop後・DC除去前の波形を±6 dB変換し、全queryの元条件と比較した。clipは行っていない。",
            "FAR/FRRはvalidationで固定済みの母音別閾値をそのまま適用した値。条件ごとに再較正していない。",
            "これは3 seed・同一validation話者に条件付けた感度であり、候補や閾値の再選択には使わない。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = build()
    (OUTPUT / "sensitivity-results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "sensitivity-results.md").write_text(_markdown(result), encoding="utf-8")
    print(_markdown(result))


if __name__ == "__main__":
    main()
