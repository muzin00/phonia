"""Summarize paired acoustic seeds; never mix them with fusion-seed averages."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def main():
    config = json.loads((BASE / "config.json").read_text())
    run = ROOT / config["run_directory"]
    assert (run / "complete-audit.json").exists()
    results = json.loads((run / "results.json").read_text())["conditions"]
    seeds = config["encoder_seeds"]
    variants = [
        "equal",
        *[f"u{n}-{m}" for n in config["fusion_budgets"] for m in ("selected", "final")],
    ]
    summary = {}
    lines = [
        "# データ量・学習更新数・encoder seedの比較",
        "",
        "音響encoder 3 seeds、Transformerは固定1 seed。EERは採点可能分、FAR/FRRは未採点を拒否に含む。全test既観測。±は音響学習seed間の標準偏差であり信頼区間ではない。",
        "",
        "| corpus | labels | encoder updates | fusion | EER mean ± SD | FAR | FRR |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for corpus, values in results.items():
        summary[corpus] = {}
        for arm in config["arms"]:
            for count in config["encoder_budgets"]:
                for variant in variants:
                    rows = [
                        values[f"{arm}-seed{seed}-e{count}-{variant}"]["verification"]
                        for seed in seeds
                    ]
                    errors = np.array([v["eer"] * 100 for v in rows])
                    item = {
                        "eer_by_acoustic_seed_percent": dict(
                            zip(map(str, seeds), errors.tolist(), strict=True)
                        ),
                        "mean_eer_percent": float(errors.mean()),
                        "sd_eer_percent": float(errors.std(ddof=1)),
                        "mean_far_percent": float(
                            np.mean([v["far"] for v in rows]) * 100
                        ),
                        "mean_frr_percent": float(
                            np.mean([v["frr"] for v in rows]) * 100
                        ),
                        "coverage_percent": float(rows[0]["coverage"] * 100),
                    }
                    summary[corpus][f"{arm}-e{count}-{variant}"] = item
                    lines.append(
                        f"| {corpus} | {arm} | {count} | {variant} | {item['mean_eer_percent']:.3f} ± {item['sd_eer_percent']:.3f}% | {item['mean_far_percent']:.3f}% | {item['mean_frr_percent']:.3f}% |"
                    )
    contrasts = {}
    for corpus, values in summary.items():
        contrasts[corpus] = {}
        pairs = []
        for count in config["encoder_budgets"]:
            for variant in variants:
                pairs.append(
                    (
                        f"data/e{count}/{variant}",
                        f"baseline280-e{count}-{variant}",
                        f"combined420-e{count}-{variant}",
                    )
                )
        for arm in config["arms"]:
            for variant in variants:
                pairs.append(
                    (
                        f"encoder/{arm}/{variant}",
                        f"{arm}-e216000-{variant}",
                        f"{arm}-e432000-{variant}",
                    )
                )
            for count in config["encoder_budgets"]:
                for mode in ("selected", "final"):
                    pairs.append(
                        (
                            f"fusion/{arm}/e{count}/{mode}",
                            f"{arm}-e{count}-u16000-{mode}",
                            f"{arm}-e{count}-u64000-{mode}",
                        )
                    )
        for label, before, after in pairs:
            delta = np.array(
                list(values[after]["eer_by_acoustic_seed_percent"].values())
            ) - np.array(list(values[before]["eer_by_acoustic_seed_percent"].values()))
            contrasts[corpus][label] = {
                "before": before,
                "after": after,
                "mean_delta_percentage_points": float(delta.mean()),
                "paired_deltas": delta.tolist(),
                "improved_seeds": int((delta < -1e-10).sum()),
            }
    (run / "comparison-summary.json").write_text(
        json.dumps(
            {"summary": summary, "paired_contrasts": contrasts},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (run / "results.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    points = [
        (count, variant)
        for count in config["encoder_budgets"]
        for variant in ("equal", "u16000-selected", "u64000-selected")
    ]
    for ax, (corpus, values) in zip(axes, summary.items(), strict=True):
        for arm in config["arms"]:
            items = [values[f"{arm}-e{n}-{v}"] for n, v in points]
            ax.errorbar(
                np.arange(len(points)),
                [v["mean_eer_percent"] for v in items],
                yerr=[v["sd_eer_percent"] for v in items],
                marker="o",
                capsize=3,
                label=arm,
            )
        ax.set_xticks(
            np.arange(len(points)),
            ["E216k", "+T16k", "+T64k", "E432k", "+T16k", "+T64k"],
            fontsize=8,
        )
        ax.axvline(2.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)
        ax.set_title(corpus)
        ax.set_ylabel("Scored-query EER (%)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Data and training budgets across 3 acoustic seeds")
    fig.supxlabel(
        "E: encoder updates; T: Transformer updates (validation-selected). Bars: seed SD. Fixed fusion seed; observed tests.",
        fontsize=9,
    )
    fig.savefig(run / "data-budget-eer.png", dpi=180)
    plt.close(fig)
    print("Report saved:", run / "results.md")


if __name__ == "__main__":
    main()
