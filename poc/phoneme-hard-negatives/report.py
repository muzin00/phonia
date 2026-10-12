"""Paired acoustic-seed summary for hard-negative continuation."""

import hard_negative as h
import numpy as np


def main():
    config = h.d.read(h.CONFIG)
    run = h.d.ROOT / config["run_directory"]
    assert (run / "complete-audit.json").exists()
    results = h.d.read(run / "results.json")["conditions"]
    summary = {}
    lines = [
        "# Train-only hard-negative continuation",
        "",
        "Three acoustic seeds; one fixed Transformer seed; same reviewed 420-label dataset and frozen 216k encoder. EER on scored trials; FAR/FRR include unscored rejects. All tests previously observed.",
        "",
        "| corpus | budget | mode | random EER ± SD | mixed EER ± SD | paired delta pp | improved seeds | mixed FAR | mixed FRR |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, values in results.items():
        summary[corpus] = {}
        for count in config["budgets"][1:]:
            for mode in ("selected", "final"):
                arms = {}
                for arm in ("random", "mixed"):
                    metrics = [
                        values[f"{arm}-seed{seed}-u{count}-{mode}"]["verification"]
                        for seed in config["encoder_seeds"]
                    ]
                    arms[arm] = {
                        "eer_percent": [v["eer"] * 100 for v in metrics],
                        "far_percent": [v["far"] * 100 for v in metrics],
                        "frr_percent": [v["frr"] * 100 for v in metrics],
                        "coverage_percent": [v["coverage"] * 100 for v in metrics],
                    }
                delta = (
                    np.array(arms["mixed"]["eer_percent"])
                    - arms["random"]["eer_percent"]
                )
                row = {
                    "arms": arms,
                    "paired_delta_pp": delta.tolist(),
                    "mean_delta_pp": float(delta.mean()),
                    "improved_seeds": int((delta < -1e-10).sum()),
                }
                summary[corpus][f"u{count}-{mode}"] = row
                random, mixed = [arms[a]["eer_percent"] for a in ("random", "mixed")]
                lines.append(
                    f"| {corpus} | {count} | {mode} | {np.mean(random):.3f} ± {np.std(random, ddof=1):.3f}% | {np.mean(mixed):.3f} ± {np.std(mixed, ddof=1):.3f}% | {delta.mean():+.3f} | {row['improved_seeds']}/3 | {np.mean(arms['mixed']['far_percent']):.3f}% | {np.mean(arms['mixed']['frr_percent']):.3f}% |"
                )
    h.d.fixed(run / "summary.json", summary)
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("Report saved", run / "results.md")


if __name__ == "__main__":
    main()
