"""Freeze choices, then compare selected and final models on identical trials."""

from scoring import FrozenModel
from shared import (
    ARMS,
    bootstrap,
    cached_evaluations,
    encoder_path,
    evaluate,
    rt,
    summarize,
    target,
)

MODES = ("selected", "final")


def model_at(config, run, arm, seed, mode):
    model = FrozenModel(config, arm, seed)
    saved = rt.torch.load(target(run, arm, seed) / f"{mode}.pt", weights_only=True)
    model.fusion.load_state_dict(saved["fusion"])
    return model, saved


def finalize(config, run):
    products, compute = [], {}
    for seed in config["evaluation_seeds"]:
        products.append(encoder_path(config, seed))
        for arm in ARMS:
            directory = target(run, arm, seed)
            done = rt.read(directory / "worker-complete.json")
            rt.check(done["files"])
            compute[f"{arm}/{seed}"] = rt.read(directory / "complete.json")
            products.extend(directory / f"{mode}.pt" for mode in MODES)
    rt.write(run / "models-freeze.json", {"files": rt.pin(products)}, immutable=True)
    thresholds, records, selected_updates = {}, [], {}
    for seed in config["evaluation_seeds"]:
        reference = FrozenModel(config, "weighted", seed)
        selection = cached_evaluations(config, reference.encoder, "selection")
        for arm in ARMS:
            for mode in MODES:
                model, saved = model_at(config, run, arm, seed, mode)
                eers, scores = evaluate(model, selection, save_scores=True)
                label = f"{arm}/{seed}/{mode}"
                selected_updates[label] = saved["update"]
                for c, values in scores.items():
                    if (
                        mode == "selected"
                        and abs(eers[c] - saved["selection_eer"][c]) > 1e-12
                    ):
                        raise AssertionError("Selected metric not reproduced")
                    threshold, _ = rt.metrics.far_target_threshold(
                        values, selection[c].labels, 0.01
                    )
                    thresholds[f"{label}/{c}"] = rt.metrics.json_threshold(threshold)
                    records.append(
                        {
                            "split": "selection",
                            "arm": arm,
                            "seed": seed,
                            "mode": mode,
                            "corpus": c,
                            **summarize(values, selection[c], threshold),
                        }
                    )
    rt.write(
        run / "selection-freeze.json",
        {
            "thresholds": thresholds,
            "selected_updates": selected_updates,
            "models_sha256": rt.sha(run / "models-freeze.json"),
        },
        immutable=True,
    )
    test_means, comparisons = {}, {}
    boot = {
        mode: {arm: {c: [] for c in config["corpora"]} for arm in ARMS}
        for mode in MODES
    }
    if not config.get("smoke_test_only"):
        for seed in config["evaluation_seeds"]:
            reference = FrozenModel(config, "weighted", seed)
            test = cached_evaluations(config, reference.encoder, "test")
            draws = {
                c: bootstrap.speaker_draws(
                    len(data.bank.speakers),
                    config["bootstrap_replicates"],
                    rt.seed_for(config["bootstrap_seed"], c),
                )[1]
                for c, data in test.items()
            }
            for arm in ARMS:
                for mode in MODES:
                    model, _ = model_at(config, run, arm, seed, mode)
                    _, scores = evaluate(model, test, save_scores=True)
                    for c, values in scores.items():
                        records.append(
                            {
                                "split": "test",
                                "arm": arm,
                                "seed": seed,
                                "mode": mode,
                                "corpus": c,
                                **summarize(
                                    values,
                                    test[c],
                                    float(thresholds[f"{arm}/{seed}/{mode}/{c}"]),
                                ),
                            }
                        )
                        path = run / "scores" / f"{arm}-{seed}-{mode}-{c}.npz"
                        path.parent.mkdir(exist_ok=True)
                        rt.np.savez_compressed(
                            path,
                            scores=values,
                            labels=test[c].labels,
                            queries=test[c].queries,
                            query_speakers=test[c].query_speakers,
                            claims=test[c].claims,
                        )
                        boot[mode][arm][c].append(
                            bootstrap.weighted_eers(
                                values, test[c].query_speakers, test[c].claims, draws[c]
                            )
                        )
        rng = rt.np.random.default_rng(config["bootstrap_seed"])
        n, replicates = len(config["evaluation_seeds"]), config["bootstrap_replicates"]
        seeds = rng.integers(n, size=(replicates, n))
        for mode in MODES:
            samples, test_means[mode], comparisons[mode] = {}, {}, {}
            for arm in ARMS:
                samples[arm] = {
                    c: rt.np.asarray(values)[
                        seeds, rt.np.arange(replicates)[:, None]
                    ].mean(1)
                    for c, values in boot[mode][arm].items()
                }
                samples[arm]["macro"] = rt.np.mean(list(samples[arm].values()), axis=0)
                means = {
                    c: float(
                        rt.np.mean(
                            [
                                r["eer"]
                                for r in records
                                if r["split"] == "test"
                                and r["arm"] == arm
                                and r["mode"] == mode
                                and r["corpus"] == c
                            ]
                        )
                    )
                    for c in config["corpora"]
                }
                means["macro"] = float(rt.np.mean(list(means.values())))
                test_means[mode][arm] = means
            for treatment, control in (
                ("direct", "weighted"),
                ("direct", "weighted_no_kl"),
                ("weighted_no_kl", "weighted"),
            ):
                comparisons[mode][f"{treatment}_minus_{control}"] = {
                    c: {
                        "eer_difference": test_means[mode][treatment][c]
                        - test_means[mode][control][c],
                        "paired_95pct_interval": rt.np.quantile(
                            samples[treatment][c] - samples[control][c], [0.025, 0.975]
                        ).tolist(),
                    }
                    for c in [*config["corpora"], "macro"]
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
            "selected_updates": selected_updates,
            "compute": compute,
            "data_audit": rt.read(run / "prepared/audit.json"),
            "primary_comparison": "selected/direct_minus_weighted/macro",
            "test_previously_observed": True,
            "automatic_promotion": False,
            "uncertainty": "Paired evaluation-speaker and acoustic/fusion-seed bootstrap conditional on fixed data/protocol; three paired seeds, previously observed tests. Secondary comparisons exploratory, no multiplicity adjustment.",
        },
        immutable=True,
    )
