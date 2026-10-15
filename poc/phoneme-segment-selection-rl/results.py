"""Freeze choices before test; report coverage, compute and paired uncertainty."""

from __future__ import annotations

from runtime import (
    ARMS,
    BASE,
    check,
    metrics,
    module,
    np,
    pin,
    read,
    seed_for,
    sha,
    torch,
    write,
)
from student import Student
from verification import evaluate, load_evaluations

bootstrap = module(
    "segment_bootstrap",
    BASE.parent / "phoneme-speaker-encoder/scripts/weighted_bootstrap.py",
)


def independent_eer(scores, labels):
    positive, negative = np.sort(scores[labels]), np.sort(scores[~labels])
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    far = 1 - np.searchsorted(negative, thresholds, side="left") / len(negative)
    frr = np.searchsorted(positive, thresholds, side="left") / len(positive)
    difference = far - frr
    right = np.flatnonzero(difference >= 0)[0]
    if difference[right] == 0:
        return float(far[right])
    left = right - 1
    fraction = -difference[left] / (difference[right] - difference[left])
    return float(far[left] + fraction * (far[right] - far[left]))


def load_model(config, target, seed, speakers):
    model = Student(config, speakers, seed)
    model.load_state_dict(torch.load(target / "acoustic/encoder.pt", weights_only=True))
    chosen = torch.load(target / "fusion/selected.pt", weights_only=True)
    model.fusion.load_state_dict(chosen["fusion"])
    return model, chosen


def summarize(values, data, threshold):
    labels = data.labels
    eer = float(metrics.roc_eer(values, labels)["eer"])
    if abs(eer - independent_eer(values, labels)) > 1e-12:
        raise AssertionError("Independent EER calculation disagrees")
    accepted = values >= threshold
    return {
        "eer": eer,
        "far": int((accepted & ~labels).sum()) / data.total_impostor,
        "frr": (
            int((~accepted & labels).sum()) + data.total_genuine - int(labels.sum())
        )
        / data.total_genuine,
        "scorable_trials": len(values),
        "total_trials": data.total_genuine + data.total_impostor,
        "score_coverage": len(values) / (data.total_genuine + data.total_impostor),
    }


def finalize(config, run):
    from study import prepared_path

    prepared = prepared_path(config)
    speaker_count = len(read(prepared / "pool.json")["speakers"])
    budget = config["teacher_checkpoints"][-1]
    paths, curves, compute, coverage = {}, [], {}, {}
    for arm in ARMS:
        check(read(run / arm / "complete.json")["files"])
        values = []
        for b in [0] if arm == "uniform" else config["teacher_checkpoints"]:
            per_seed = []
            for seed in config["evaluation_seeds"]:
                target = run / arm / f"deployment/budget-{b}/seed-{seed}"
                acoustic = read(target / "acoustic/complete.json")
                fusion = read(target / "fusion/complete.json")
                values.append(
                    {
                        "budget": b,
                        "seed": seed,
                        "acoustic_cpu_seconds": acoustic["cpu_seconds"],
                        "fusion_cpu_seconds": fusion["cpu_seconds"],
                        "student_updates": acoustic["actual_updates"],
                    }
                )
                per_seed.append(fusion["selected_eer"])
                if b == (0 if arm == "uniform" else budget):
                    paths[arm, seed] = target
                    coverage[f"{arm}/{seed}"] = read(target / "acoustic/coverage.json")
            curves.append(
                {
                    "arm": arm,
                    "teacher_episodes": b,
                    "selection_eer_per_seed": per_seed,
                    "mean_selection_eer": float(np.mean(per_seed)),
                }
            )
        teacher_cost = [
            read(p)
            for p in sorted((run / arm / "training").glob("episode-*/complete.json"))
        ]
        shared = []
        if arm == "hybrid":
            shared = [
                read(run / f"proxy/training/episode-{i + 1}/complete.json")
                for i in range(config["hybrid_proxy_episodes"])
            ]
        compute[arm] = {
            "deployment": values,
            "teacher_actual_updates": sum(
                v["actual_updates"] for v in teacher_cost + shared
            ),
            "teacher_updates": sum(v["teacher_updates"] for v in teacher_cost + shared),
            "teacher_cpu_seconds_including_shared_warm_start": sum(
                v["cpu_seconds"] for v in teacher_cost + shared
            ),
            "shared_proxy_episodes": len(shared),
        }
    model_files = [
        p
        for target in paths.values()
        for p in (target / "acoustic/encoder.pt", target / "fusion/selected.pt")
    ]
    write(
        run / "models-freeze.json",
        {
            "files": pin(model_files),
            "primary_teacher_episodes": budget,
            "test_not_used_for_choices": True,
        },
        immutable=True,
    )
    selection = load_evaluations(config, prepared, "selection")
    thresholds, records = {}, []
    for (arm, seed), target in paths.items():
        model, chosen = load_model(config, target, seed, speaker_count)
        eers, scores = evaluate(model, selection, save_scores=True)
        for c, values in scores.items():
            if abs(eers[c] - chosen["selection_eer"][c]) > 1e-12:
                raise AssertionError(
                    "Selected checkpoint cannot reproduce selection metric"
                )
            threshold, _ = metrics.far_target_threshold(
                values, selection[c].labels, 0.01
            )
            thresholds[f"{arm}/{seed}/{c}"] = metrics.json_threshold(threshold)
            records.append(
                {
                    "split": "selection",
                    "arm": arm,
                    "seed": seed,
                    "corpus": c,
                    **summarize(values, selection[c], threshold),
                }
            )
    write(
        run / "selection-freeze.json",
        {
            "thresholds": thresholds,
            "models_sha256": sha(run / "models-freeze.json"),
            "learning_curve": curves,
        },
        immutable=True,
    )
    intervals, means = {}, {}
    if not config.get("smoke_test_only"):
        test = load_evaluations(config, prepared, "test")
        draws = {
            c: bootstrap.speaker_draws(
                len(data.bank.speakers),
                config["bootstrap_replicates"],
                seed_for(config["bootstrap_seed"], c),
            )[1]
            for c, data in test.items()
        }
        boot = {arm: [] for arm in ARMS}
        for (arm, seed), target in paths.items():
            model, _ = load_model(config, target, seed, speaker_count)
            _, scores = evaluate(model, test, save_scores=True)
            seed_draws = []
            for c, values in scores.items():
                saved_threshold = thresholds[f"{arm}/{seed}/{c}"]
                threshold = float(saved_threshold)
                record = {
                    "split": "test",
                    "arm": arm,
                    "seed": seed,
                    "corpus": c,
                    **summarize(values, test[c], threshold),
                }
                records.append(record)
                destination = run / f"scores/{arm}-{seed}-{c}.npz"
                destination.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    destination,
                    scores=values,
                    labels=test[c].labels,
                    query_speakers=test[c].query_speakers,
                    claims=test[c].claims,
                )
                seed_draws.append(
                    bootstrap.weighted_eers(
                        values, test[c].query_speakers, test[c].claims, draws[c]
                    )
                )
            boot[arm].append(np.mean(seed_draws, axis=0))
        rng = np.random.default_rng(config["bootstrap_seed"])
        n = len(config["evaluation_seeds"])
        chosen_seeds = rng.integers(n, size=(config["bootstrap_replicates"], n))
        sampled = {
            arm: np.asarray(values)[
                chosen_seeds, np.arange(config["bootstrap_replicates"])[:, None]
            ].mean(1)
            for arm, values in boot.items()
        }
        for arm in ARMS:
            means[arm] = float(
                np.mean(
                    [
                        v["eer"]
                        for v in records
                        if v["arm"] == arm and v["split"] == "test"
                    ]
                )
            )
        for a, b in (
            ("proxy", "uniform"),
            ("eer", "uniform"),
            ("hybrid", "uniform"),
            ("hybrid", "proxy"),
            ("hybrid", "eer"),
        ):
            intervals[f"{a}-minus-{b}"] = {
                "difference": means[a] - means[b],
                "paired_95pct_interval": np.quantile(
                    sampled[a] - sampled[b], [0.025, 0.975]
                ).tolist(),
            }
    result = {
        "status": "smoke_complete_no_efficacy_conclusion"
        if config.get("smoke_test_only")
        else "complete",
        "config": config,
        "records": records,
        "learning_curve": curves,
        "test_macro_eer": means,
        "comparisons": intervals,
        "compute": compute,
        "coverage": coverage,
        "uncertainty": "Paired resampling of evaluation speakers and fresh student seeds, conditional on one learned teacher per arm; does not include teacher-seed variability.",
        "compute_matched_superiority_tested": False,
        "test_previously_observed": True,
        "automatic_promotion": False,
    }
    write(run / "results.json", result, immutable=True)
