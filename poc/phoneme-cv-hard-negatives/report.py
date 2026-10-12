"""Three-arm paired-seed comparison for CV-only hard-negative training."""

import cv_only as h
import numpy as np


def main():
    config = h.d.read(h.CONFIG)
    run = h.d.ROOT / config["run_directory"]
    assert (run / "complete-audit.json").exists()
    results = h.d.read(run / "results.json")["conditions"]
    summary = {}
    lines = [
        "# CV-only hard negatives",
        "",
        "Three paired acoustic seeds, frozen 216k encoder, fixed Transformer seed, existing 420-label data. EER on scored trials; FAR/FRR include unscored rejects. All tests previously observed. ± is seed SD, not confidence interval.",
        "",
        "| corpus | budget | mode | random EER | all-hard EER | CV-only EER ± SD | CV-only vs random pp | CV-only vs all-hard pp | CV-only FAR | CV-only FRR |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, values in results.items():
        summary[corpus] = {}
        for count in config["budgets"][1:]:
            for mode in ("selected", "final"):
                arms = {}
                for arm in ("random", "mixed", "cv_only"):
                    metrics = [
                        values[f"{arm}-seed{seed}-u{count}-{mode}"]["verification"]
                        for seed in config["encoder_seeds"]
                    ]
                    arms[arm] = {
                        f"{metric}_percent": [v[metric] * 100 for v in metrics]
                        for metric in ("eer", "far", "frr", "coverage")
                    }
                contrasts = {}
                for control in ("random", "mixed"):
                    delta = (
                        np.array(arms["cv_only"]["eer_percent"])
                        - arms[control]["eer_percent"]
                    )
                    contrasts[control] = {
                        "paired_delta_pp": delta.tolist(),
                        "mean_delta_pp": float(delta.mean()),
                        "improved_seeds": int((delta < -1e-10).sum()),
                    }
                summary[corpus][f"u{count}-{mode}"] = {
                    "arms": arms,
                    "contrasts": contrasts,
                }
                means = {a: float(np.mean(v["eer_percent"])) for a, v in arms.items()}
                cv = arms["cv_only"]
                lines.append(
                    f"| {corpus} | {count} | {mode} | {means['random']:.3f}% | {means['mixed']:.3f}% | {means['cv_only']:.3f} ± {np.std(cv['eer_percent'], ddof=1):.3f}% | {contrasts['random']['mean_delta_pp']:+.3f} | {contrasts['mixed']['mean_delta_pp']:+.3f} | {np.mean(cv['far_percent']):.3f}% | {np.mean(cv['frr_percent']):.3f}% |"
                )
    h.d.fixed(run / "summary.json", summary)
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("Report saved", run / "results.md")


if __name__ == "__main__":
    main()
