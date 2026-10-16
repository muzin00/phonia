"""Lock every model and threshold before paired evaluation on unchanged test."""

from shared_quality import (
    ARMS,
    evaluate,
    evaluation_prepared,
    load_evaluations,
    prepared,
    rt,
    target,
)

# isort: split
from results import bootstrap, load_model, summarize


def finalize(config, run):
    audit = rt.read(prepared(config) / "audit.json")
    products, costs, coverage = [], {}, {}
    for arm in ARMS:
        for seed in config["evaluation_seeds"]:
            directory = target(run, arm, seed)
            rt.check(rt.read(directory / "complete.json")["files"])
            products.extend(
                [directory / "acoustic/encoder.pt", directory / "fusion/selected.pt"]
            )
            costs[f"{arm}/{seed}"] = {
                name: rt.read(directory / name / "complete.json")
                for name in ("acoustic", "fusion")
            }
            coverage[f"{arm}/{seed}"] = rt.read(directory / "coverage-by-corpus.json")
    rt.write(
        run / "models-freeze.json",
        {"files": rt.pin(products), "test_not_used_for_choices": True},
        immutable=True,
    )
    selection = load_evaluations(config, evaluation_prepared(config), "selection")
    thresholds, records, selections = {}, [], {}
    for arm in ARMS:
        for seed in config["evaluation_seeds"]:
            model, chosen = load_model(
                config, target(run, arm, seed), seed, len(audit[arm]["speakers"])
            )
            eers, scores = evaluate(model, selection, save_scores=True)
            selections[f"{arm}/{seed}"] = chosen["update"]
            for c, values in scores.items():
                if abs(eers[c] - chosen["selection_eer"][c]) > 1e-12:
                    raise AssertionError("Selection checkpoint metric not reproduced")
                threshold, _ = rt.metrics.far_target_threshold(
                    values, selection[c].labels, 0.01
                )
                thresholds[f"{arm}/{seed}/{c}"] = rt.metrics.json_threshold(threshold)
                records.append(
                    {
                        "split": "selection",
                        "arm": arm,
                        "seed": seed,
                        "corpus": c,
                        **summarize(values, selection[c], threshold),
                    }
                )
    rt.write(
        run / "selection-freeze.json",
        {
            "thresholds": thresholds,
            "selected_updates": selections,
            "models_sha256": rt.sha(run / "models-freeze.json"),
        },
        immutable=True,
    )
    comparisons, test_means = {}, {}
    if not config.get("smoke_test_only", False):
        test = load_evaluations(config, evaluation_prepared(config), "test")
        draws = {
            c: bootstrap.speaker_draws(
                len(data.bank.speakers),
                config["bootstrap_replicates"],
                rt.seed_for(config["bootstrap_seed"], c),
            )[1]
            for c, data in test.items()
        }
        boot = {arm: {c: [] for c in config["corpora"]} for arm in ARMS}
        for arm in ARMS:
            for seed in config["evaluation_seeds"]:
                model, _ = load_model(
                    config, target(run, arm, seed), seed, len(audit[arm]["speakers"])
                )
                _, scores = evaluate(model, test, save_scores=True)
                for c, values in scores.items():
                    records.append(
                        {
                            "split": "test",
                            "arm": arm,
                            "seed": seed,
                            "corpus": c,
                            **summarize(
                                values, test[c], float(thresholds[f"{arm}/{seed}/{c}"])
                            ),
                        }
                    )
                    path = run / f"scores/{arm}-{seed}-{c}.npz"
                    path.parent.mkdir(exist_ok=True)
                    rt.np.savez_compressed(
                        path,
                        scores=values,
                        labels=test[c].labels,
                        query_speakers=test[c].query_speakers,
                        claims=test[c].claims,
                    )
                    boot[arm][c].append(
                        bootstrap.weighted_eers(
                            values, test[c].query_speakers, test[c].claims, draws[c]
                        )
                    )
        rng = rt.np.random.default_rng(config["bootstrap_seed"])
        n, replicates = len(config["evaluation_seeds"]), config["bootstrap_replicates"]
        seeds = rng.integers(n, size=(replicates, n))
        samples = {}
        for arm in ARMS:
            samples[arm] = {
                c: rt.np.asarray(values)[seeds, rt.np.arange(replicates)[:, None]].mean(
                    1
                )
                for c, values in boot[arm].items()
            }
            samples[arm]["macro"] = rt.np.mean(list(samples[arm].values()), axis=0)
            test_means[arm] = {
                c: float(
                    rt.np.mean(
                        [
                            r["eer"]
                            for r in records
                            if r["arm"] == arm
                            and r["split"] == "test"
                            and r["corpus"] == c
                        ]
                    )
                )
                for c in config["corpora"]
            }
            test_means[arm]["macro"] = float(rt.np.mean(list(test_means[arm].values())))
        for treatment, control in (
            ("screened", "original"),
            ("random_matched", "original"),
            ("screened", "random_matched"),
        ):
            label = f"{treatment}_minus_{control}"
            comparisons[label] = {}
            for c in [*config["corpora"], "macro"]:
                comparisons[label][c] = {
                    "eer_difference": test_means[treatment][c] - test_means[control][c],
                    "paired_95pct_interval": rt.np.quantile(
                        samples[treatment][c] - samples[control][c], [0.025, 0.975]
                    ).tolist(),
                }
    rt.write(
        run / "results.json",
        {
            "status": "smoke_complete_no_efficacy_conclusion"
            if config.get("smoke_test_only")
            else "complete",
            "config": config,
            "records": records,
            "test_eer": test_means,
            "comparisons": comparisons,
            "selected_updates": selections,
            "compute": costs,
            "coverage": coverage,
            "test_previously_observed": True,
            "automatic_promotion": False,
            "uncertainty": "Paired evaluation-speaker and student-seed bootstrap; only three paired training seeds/random-filter draws and previously observed test sets.",
            "quality_causality_established": False,
            "data_comparison": audit,
        },
        immutable=True,
    )
