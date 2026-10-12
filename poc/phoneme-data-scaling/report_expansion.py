"""Render audited data-scaling results with uncertainty across fusion seeds."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    config = read(BASE / "study-config.json")
    run = ROOT / config["run_directory"]
    assert (run / "complete-audit.json").exists(), "Wait for independent result audit"
    results = read(run / "results.json")["conditions"]
    arms = list(config["arms"])
    seeds = config["fusion_seeds"]
    names = [
        "Baseline\n280 labels",
        "More utterances\n280 labels",
        "More labels\n420 labels",
        "Both\n420 labels",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    summary = {}
    for ax, (corpus, values) in zip(axes, results.items(), strict=True):
        encoder = (
            np.array([values[f"{a}-equal"]["verification"]["eer"] for a in arms]) * 100
        )
        selected = (
            np.array(
                [
                    [values[f"{a}-selected-{s}"]["verification"]["eer"] for s in seeds]
                    for a in arms
                ]
            )
            * 100
        )
        final = (
            np.array(
                [
                    [values[f"{a}-final-{s}"]["verification"]["eer"] for s in seeds]
                    for a in arms
                ]
            )
            * 100
        )
        x = np.arange(len(arms))
        ax.bar(x - 0.19, encoder, 0.36, label="Encoder only", color="#94a3b8")
        ax.bar(
            x + 0.19,
            selected.mean(axis=1),
            0.36,
            yerr=selected.std(axis=1, ddof=1),
            capsize=3,
            label="Transformer 16k (selected)",
            color="#167da0",
        )
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel("Scored-query EER (%)")
        ax.set_title(
            {"jvs": "JVS", "common_voice": "Common Voice", "jtube": "JTube"}[corpus]
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        summary[corpus] = {
            a: {
                "encoder_eer_percent": float(encoder[i]),
                "selected_mean_eer_percent": float(selected[i].mean()),
                "selected_sd_eer_percent": float(selected[i].std(ddof=1)),
                "final_mean_eer_percent": float(final[i].mean()),
                "paired_seed_improvements_vs_baseline": int(
                    (selected[i] < selected[0] - 1e-10).sum()
                ),
            }
            for i, a in enumerate(arms)
        }
    axes[0].legend(fontsize=8)
    fig.suptitle("Data growth at matched training budgets", fontsize=15)
    fig.supxlabel(
        "One acoustic seed per condition; error bars = SD over 3 fusion seeds. Previously observed test sets.",
        fontsize=9,
    )
    fig.savefig(run / "data-scaling-eer.png", dpi=180)
    plt.close(fig)
    (run / "comparison-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
