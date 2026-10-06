"""Render readable primary ROC/DET figures from immutable Phase 7 curves."""

import argparse
import json
import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_report import CAPS, COLORS, plt, save_figure
from pilot import METHODS
from smoke import sha256_file, write_json
from validation import iter_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.run_dir / "test-report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("requires completed fixed-threshold test report")
    curves = {
        c["condition"]: c for c in iter_rows(args.run_dir / "test-curves.jsonl.gz")
    }
    nd = NormalDist()
    ticks = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
    for role in ("verification", "cross_text_verification"):
        fig, axes = plt.subplots(2, 5, figsize=(19, 7), constrained_layout=True)
        for j, cap in enumerate(CAPS):
            for method in METHODS:
                curve = curves[f"native/{method}/n10/{cap}/{role}"]
                if not curve["far"]:
                    axes[0, j].plot(
                        [], [], color=COLORS[method], label=f"{method} (NE)"
                    )
                    continue
                # Retain every ROC point in the primary publication figures.
                far, frr = curve["far"], curve["frr"]
                axes[0, j].plot(
                    far, [1 - v for v in frr], color=COLORS[method], label=method
                )
                axes[1, j].plot(
                    [nd.inv_cdf(min(max(v, 1e-5), 1 - 1e-5)) for v in far],
                    [nd.inv_cdf(min(max(v, 1e-5), 1 - 1e-5)) for v in frr],
                    color=COLORS[method],
                )
            axes[0, j].set(
                title=cap,
                xlabel="FAR (scored only)",
                ylabel="TPR",
                xlim=(0, 0.1),
                ylim=(0.5, 1.005),
            )
            axes[1, j].set(
                xticks=[nd.inv_cdf(t) for t in ticks],
                xticklabels=["0.01", "0.1", "1", "10", "50", "90"],
                yticks=[nd.inv_cdf(t) for t in ticks],
                yticklabels=["0.01", "0.1", "1", "10", "50", "90"],
                xlabel="FAR (%)",
                ylabel="FRR (%)",
            )
            axes[1, j].tick_params(axis="x", labelrotation=35, labelsize=8)
            axes[1, j].tick_params(axis="y", labelsize=8)
            for ax in axes[:, j]:
                ax.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"test / native / enrollment 10 / {role}; NE retains missing speakers"
        )
        save_figure(fig, args.output_dir, f"roc-det-test-native-n10-{role}")
    write_json(
        args.output_dir / "manifest.json",
        {
            "status": "completed",
            "source_test_report_sha256": sha256_file(args.run_dir / "test-report.json"),
            "source_curves_sha256": sha256_file(args.run_dir / "test-curves.jsonl.gz"),
            "renderer_sha256": sha256_file(Path(__file__)),
            "changes": "primary ROC/DET retain all points; readable rotated DET labels; no metric/threshold/inference change",
            "files": {
                p.name: sha256_file(p)
                for p in sorted(args.output_dir.iterdir())
                if p.is_file()
            },
        },
    )


if __name__ == "__main__":
    main()
