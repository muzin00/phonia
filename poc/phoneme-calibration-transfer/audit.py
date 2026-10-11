"""Independently recount all operating points and verify each empirical FAR cap."""

import math
import statistics

import diagnose as d

r, s = d.r, d.s


def check(records, metrics):
    positive = s.np.array([v["is_genuine"] for v in records], dtype=bool)
    values = s.np.array(
        [s.np.nan if v["score"] is None else v["score"] for v in records]
    )
    accepted = values >= float(metrics["threshold"])
    fa = int((accepted & ~positive).sum())
    fr = int((~accepted & positive).sum())
    assert metrics["false_accepts"] == fa and metrics["false_rejects"] == fr
    assert metrics["all_genuine"] == int(positive.sum())
    assert metrics["all_impostor"] == int((~positive).sum())
    assert metrics["far"] == fa / int((~positive).sum())
    assert metrics["frr"] == fr / int(positive.sum())
    assert metrics["coverage"] == int((positive & s.np.isfinite(values)).sum()) / int(
        positive.sum()
    )


def main():
    s.torch.set_num_threads(1)
    config = s.read(d.CONFIG)
    run = s.ROOT / config["run_directory"]
    r.checked_files(s.read(run / "design-freeze.json")["files"])
    result = s.read(run / "results.json")
    assert result["validation_freeze_sha256"] == s.sha(run / "validation-freeze.json")
    assert result["per_seed_sha256"] == s.sha(run / "per-seed-results.json")
    validation = s.read(run / "validation-freeze.json")
    assert not validation["test_used_to_choose_thresholds"]
    assert validation["design_sha256"] == s.sha(run / "design-freeze.json")
    rows = s.read(run / "per-seed-results.json")
    checked_points = 0
    for seed in config["seeds"]:
        for corpus in s.CORPORA:
            for arm in config["arms"]:
                val = d.load(config, seed, corpus, arm, "validation")
                test = d.load(config, seed, corpus, arm, "test")
                entry = rows[str(seed)][corpus][arm]
                for target in config["validation_scored_far_targets"]:
                    point = entry["validation_locked"][d.key(target)]
                    assert (
                        point["test"]["threshold"]
                        == point["validation"]["threshold"]
                        == validation["thresholds"][str(seed)][corpus][arm][
                            d.key(target)
                        ]
                    )
                    check(val, point["validation"])
                    check(test, point["test"])
                    assert (
                        point["validation"]["scored_only_error_rates"]["far"] <= target
                    )
                    assert (
                        point["scored_far_transfer_gap"]
                        == point["test"]["scored_only_error_rates"]["far"]
                        - point["validation"]["scored_only_error_rates"]["far"]
                    )
                    assert (
                        point["all_input_far_transfer_gap"]
                        == point["test"]["far"] - point["validation"]["far"]
                    )
                    checked_points += 2
                for target in config["diagnostic_test_all_input_far_caps"]:
                    diagnostic = entry["retrospective_roc"][d.key(target)]
                    assert (
                        diagnostic["diagnostic_only"]
                        and diagnostic["not_a_deployment_threshold"]
                    )
                    metrics = diagnostic["operating"]
                    check(test, metrics)
                    assert metrics["far"] <= target
                    threshold = float(metrics["threshold"])
                    if math.isfinite(threshold):
                        lower = s.np.nextafter(threshold, -s.np.inf)
                        impostors = [v for v in test if not v["is_genuine"]]
                        next_fa = sum(
                            v["score"] is not None and v["score"] >= lower
                            for v in impostors
                        )
                        assert next_fa / len(impostors) > target, (
                            "A lower ROC threshold still fits the cap"
                        )
                    checked_points += 1
    for mode, targets in result["aggregate"].items():
        for target, corpora in targets.items():
            for corpus, arms in corpora.items():
                for arm, metrics in arms.items():
                    for metric in ("far", "frr", "coverage"):
                        report = metrics[metric]
                        field = "test" if mode == "validation_locked" else "operating"
                        values = [
                            rows[str(seed)][corpus][arm][mode][target][field][metric]
                            for seed in config["seeds"]
                        ]
                        delta = [
                            value
                            - rows[str(seed)][corpus]["expanded280"][mode][target][
                                field
                            ][metric]
                            for seed, value in zip(config["seeds"], values, strict=True)
                        ]
                        assert (
                            report["values"] == values
                            and report["paired_delta_vs_expanded280"] == delta
                        )
                        assert math.isclose(
                            report["mean"], statistics.mean(values), abs_tol=1e-12
                        )
                        assert math.isclose(
                            report["sd"], statistics.stdev(values), abs_tol=1e-12
                        )
                        assert math.isclose(
                            report["paired_delta_mean"],
                            statistics.mean(delta),
                            abs_tol=1e-12,
                        )
                        assert report["decreased_seeds"] == sum(v < 0 for v in delta)
                        assert report["increased_seeds"] == sum(v > 0 for v in delta)
    report = {
        "status": "passed",
        "operating_points_recounted": checked_points,
        "validation_thresholds": 300,
        "retrospective_caps": 300,
        "source_1pct_results_reproduced_during_analysis": 75,
        "files": r.pin(
            [
                run / "results.json",
                run / "results.md",
                run / "per-seed-results.json",
                d.BASE / "audit.py",
            ]
        ),
    }
    s.fixed(run / "completion-audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
