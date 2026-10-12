"""Report new initialization seeds separately from the exploratory reference."""

import numpy as np
import replication as x


def main():
    config = x.d.read(x.CONFIG)
    run = x.d.ROOT / config["run_directory"]
    assert (run / "complete-audit.json").exists()
    results = x.d.read(run / "results.json")["conditions"]
    summary = {}
    lines = [
        "# Transformer initialization replication",
        "",
        "Fixed training schedules, three fixed acoustic encoders. Two new Transformer initialization/dropout seeds. Reference seed previously observed. Paired runs reuse the same evaluation speakers; not independent population replications.",
        "",
        "| seed group | corpus | budget | mode | random EER | CV-only EER | delta pp | improved pairs |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    groups = {
        str(f): [f]
        for f in [config["reference_fusion_seed"], *config["new_fusion_seeds"]]
    }
    groups["new_seeds_only"] = config["new_fusion_seeds"]
    for group, seeds in groups.items():
        summary[group] = {}
        for corpus in [*x.s.CORPORA, "macro"]:
            summary[group][corpus] = {}
            for budget in config["budgets"][1:]:
                for mode in ("selected", "final"):
                    values = {
                        arm: {
                            f"{m}_percent": []
                            for m in ("eer", "far", "frr", "coverage")
                        }
                        for arm in config["arms"]
                    }
                    pairs = []
                    for acoustic in config["encoder_seeds"]:
                        for fusion in seeds:
                            pairs.append({"acoustic": acoustic, "fusion": fusion})
                            for arm in config["arms"]:
                                label = f"{arm}-seed{acoustic}-fusion{fusion}-u{budget}-{mode}"
                                corpora = x.s.CORPORA if corpus == "macro" else [corpus]
                                for metric in ("eer", "far", "frr", "coverage"):
                                    values[arm][f"{metric}_percent"].append(
                                        float(
                                            np.mean(
                                                [
                                                    results[c][label]["verification"][
                                                        metric
                                                    ]
                                                    * 100
                                                    for c in corpora
                                                ]
                                            )
                                        )
                                    )
                    delta = (
                        np.array(values["cv_only"]["eer_percent"])
                        - values["random"]["eer_percent"]
                    )
                    condition = {
                        "pairs": pairs,
                        "arms": values,
                        "paired_delta_pp": delta.tolist(),
                        "mean_delta_pp": float(delta.mean()),
                        "improved_pairs": int((delta < -1e-10).sum()),
                        "worsened_pairs": int((delta > 1e-10).sum()),
                    }
                    summary[group][corpus][f"u{budget}-{mode}"] = condition
                    lines.append(
                        f"| {group} | {corpus} | {budget} | {mode} | {np.mean(values['random']['eer_percent']):.3f}% | {np.mean(values['cv_only']['eer_percent']):.3f}% | {delta.mean():+.3f} | {(delta < -1e-10).sum()}/{len(delta)} |"
                    )
    x.d.fixed(run / "summary.json", summary)
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("Report saved", run / "results.md")


if __name__ == "__main__":
    main()
