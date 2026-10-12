"""Summarize paired seed changes and plot the completed learning-budget sweep."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phonia-budget-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run as b

s = b.s


def main():
    config = s.read(b.CONFIG)
    run = s.ROOT / config["run_directory"]
    results = s.read(run / "results.json")
    audit = s.read(run / "complete-audit.json")
    b.r.checked_files(audit["files"])
    records = {}
    for corpus, values in results["aggregate"].items():
        records[corpus] = {}
        for mode in ("selected", "final"):
            reference = values[f"u4000-{mode}"]
            for budget in config["budgets"]:
                record = {}
                for metric in ("eer", "far", "frr"):
                    v = s.np.array(values[f"u{budget}-{mode}"][metric]["values"])
                    delta = v - reference[metric]["values"]
                    record[metric] = {
                        "mean": float(v.mean()),
                        "sd": float(v.std(ddof=1)),
                        "paired_delta_mean": float(delta.mean()),
                        "paired_deltas": delta.tolist(),
                        "improved_seeds": int((delta < -1e-12).sum()),
                        "worse_seeds": int((delta > 1e-12).sum()),
                    }
                records[corpus][f"u{budget}-{mode}"] = record
    s.fixed(
        run / "paired-summary.json",
        {"results_sha256": s.sha(run / "results.json"), "metrics": records},
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, corpus, title in zip(
        axes, s.CORPORA, ("JVS", "Common Voice", "JTube"), strict=True
    ):
        for mode, label, style in (
            ("selected", "Validation-selected", "-o"),
            ("final", "Final checkpoint", "--s"),
        ):
            values = [records[corpus][f"u{v}-{mode}"]["eer"] for v in config["budgets"]]
            ax.errorbar(
                config["budgets"],
                [v["mean"] * 100 for v in values],
                yerr=[v["sd"] * 100 for v in values],
                fmt=style,
                capsize=4,
                label=label,
            )
        equal = results["conditions"][corpus]["equal"]["verification"]["eer"] * 100
        ax.axhline(equal, color="gray", linestyle=":", label="Equal-weight baseline")
        ax.set_xscale("log", base=2)
        ax.set_xticks(config["budgets"], ["4k", "16k", "64k"])
        ax.set(title=title, xlabel="Total updates", ylabel="Test EER (%)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Fixed encoder and data; mean +/- sample SD over 3 fusion seeds", fontsize=11
    )
    fig.savefig(run / "test-eer-by-budget.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for seed in config["seeds"]:
        history = s.read(run / f"seed-{seed}/history.json")
        updates = [v["update"] for v in history]
        values = s.np.array([v["macro_eer"] for v in history]) * 100
        line = ax.plot(updates, values, alpha=0.25)[0]
        ax.plot(
            updates,
            s.np.minimum.accumulate(values),
            label=f"seed {seed}: best so far",
            color=line.get_color(),
        )
    for budget in config["budgets"]:
        ax.axvline(budget, color="gray", linestyle=":", alpha=0.5)
    ax.set(
        xlabel="Total updates",
        ylabel="Validation macro EER (%)",
        title="Validation trajectory and checkpoint selection",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(run / "validation-learning-curves.png", dpi=180)
    plt.close(fig)
    print(records)


if __name__ == "__main__":
    main()
