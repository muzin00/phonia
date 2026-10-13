"""Three fixed scoring conditions, selected/final checkpoints and paired CIs."""

from common import (
    ROOT,
    create_model,
    np,
    old_model_path,
    pin,
    prior,
    read,
    sha,
    state,
    torch,
    write,
)

ARMS = ("train36_score36", "train36_score34", "train34_score34")
MODES = ("selected", "final")


def model_paths(config, run):
    return {
        (seed, trained, mode): old_model_path(config, seed, mode)
        if trained == 36
        else run / f"acoustic-{seed}/{mode}.pt"
        for seed in config["encoder_seeds"]
        for trained in (36, 34)
        for mode in MODES
    }


def evaluate(config, run, frozen):
    paths = model_paths(config, run)
    write(run / "models-freeze.json", {"files": pin(paths.values())}, immutable=True)
    points, records, checks = {}, [], []
    for split in ("validation", "test"):
        if split == "test":
            locked = read(run / "selection-freeze.json")
            assert locked["thresholds"] == points
            assert locked["models_sha256"] == sha(run / "models-freeze.json")
        for seed in config["encoder_seeds"]:
            models = {}
            for trained in (36, 34):
                for mode in MODES:
                    model = create_model(config, seed)
                    bundle = torch.load(
                        paths[seed, trained, mode],
                        weights_only=True,
                        map_location="cpu",
                    )
                    model.load_state_dict(bundle["state_dict"])
                    models[trained, mode] = (model.eval(), bundle["update"])
            for corpus in config["corpora"]:
                state(run, "evaluating", split=split, corpus=corpus, seed=seed)
                arrays, data, rows = prior.load_inputs(config, seed, corpus, split)
                ids = prior.positions(rows, normal_only=False)
                labels = np.array([rows[i]["is_genuine"] for i in ids], bool)
                roles = np.array([rows[i]["role"] for i in ids])
                normal = roles == "verification"
                scores = {
                    "ids": ids,
                    "labels": labels,
                    "roles": roles,
                    "query_speakers": np.array(
                        [data["speakers"].index(rows[i]["speaker_id"]) for i in ids]
                    ),
                    "claims": ids % len(data["speakers"]),
                }
                for mode in MODES:
                    for arm in ARMS:
                        trained = 34 if arm == "train34_score34" else 36
                        subset = (
                            tuple(range(36))
                            if arm == "train36_score36"
                            else tuple(frozen["subset"])
                        )
                        model, update = models[trained, mode]
                        scorer = prior.Scorer(model, arrays, ids)
                        values = scorer.score(subset)
                        direct = scorer.score(subset, direct=True)
                        difference = float(np.max(np.abs(values - direct)))
                        assert difference < 1e-12, difference
                        key = f"{arm}-{mode}"
                        scores[key] = values
                        if arm == "train36_score36":
                            previous = read(
                                ROOT
                                / config["source_run"]
                                / "scores"
                                / f"{corpus}-src490-seed{seed}-u{config['updates']}-{mode}-{split}.json"
                            )
                            np.testing.assert_array_equal(
                                values, [previous[i]["score"] for i in ids]
                            )
                        if mode == "selected" and trained == 36:
                            with np.load(
                                ROOT
                                / config["subset_run"]
                                / "scores"
                                / f"{corpus}-{seed}-{split}.npz",
                                allow_pickle=False,
                            ) as previous:
                                np.testing.assert_array_equal(ids, previous["ids"])
                                np.testing.assert_array_equal(
                                    values,
                                    previous[
                                        "all36"
                                        if arm == "train36_score36"
                                        else "selected"
                                    ],
                                )
                        threshold_key = f"{corpus}/{seed}/{key}"
                        if split == "validation":
                            point, _ = prior.metrics.far_target_threshold(
                                values[normal], labels[normal], 0.01
                            )
                            points[threshold_key] = prior.metrics.json_threshold(point)
                        point = points[threshold_key]
                        numeric = (
                            np.inf
                            if point == "+inf"
                            else -np.inf
                            if point == "-inf"
                            else point
                        )
                        for role in sorted(set(roles)):
                            keep = roles == role
                            original = [r for r in rows if r["role"] == role]
                            positive = sum(r["is_genuine"] for r in original)
                            result = prior.summarize(
                                values[keep],
                                labels[keep],
                                numeric,
                                positive,
                                len(original) - positive,
                            )
                            assert (
                                abs(
                                    result["eer"]
                                    - prior.independent_eer(values[keep], labels[keep])
                                )
                                < 1e-12
                            )
                            if (
                                split == "validation"
                                and arm == "train34_score34"
                                and role == "verification"
                            ):
                                history = read(run / f"acoustic-{seed}/history.json")
                                recorded = next(
                                    v for v in history if v["update"] == update
                                )
                                assert (
                                    result["eer"] == recorded["validation_eer"][corpus]
                                )
                            records.append(
                                {
                                    "split": split,
                                    "corpus": corpus,
                                    "seed": seed,
                                    "arm": arm,
                                    "mode": mode,
                                    "update": update,
                                    "role": role,
                                    **result,
                                }
                            )
                        checks.append(
                            {
                                "split": split,
                                "corpus": corpus,
                                "seed": seed,
                                "arm": arm,
                                "mode": mode,
                                "scored_trials": len(ids),
                                "maximum_direct_forward_difference": difference,
                            }
                        )
                path = run / "scores" / f"{corpus}-{seed}-{split}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        for key, value in scores.items():
                            np.testing.assert_array_equal(saved[key], value)
                else:
                    temp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp, **scores)
                    temp.replace(path)
                del arrays, scorer
        if split == "validation":
            write(run / "validation-metrics.json", records.copy(), immutable=True)
            write(
                run / "selection-freeze.json",
                {
                    "thresholds": points,
                    "models_sha256": sha(run / "models-freeze.json"),
                    "validation_files": pin(
                        [
                            run / "validation-metrics.json",
                            *run.glob("scores/*-validation.npz"),
                        ]
                    ),
                },
                immutable=True,
            )
    write(run / "metrics.json", records, immutable=True)
    write(run / "score-audit.json", checks, immutable=True)


def bootstrap_comparisons(config, run):
    distributions, result = {}, {}
    for corpus_index, corpus in enumerate(config["corpora"]):
        per_seed = {}
        for seed in config["encoder_seeds"]:
            with np.load(
                run / "scores" / f"{corpus}-{seed}-test.npz", allow_pickle=False
            ) as scores:
                keep = scores["roles"] == "verification"
                q, c = scores["query_speakers"][keep], scores["claims"][keep]
                _, counts = prior.bootstrap.speaker_draws(
                    int(max(q.max(), c.max())) + 1,
                    config["bootstrap_replicates"],
                    config["bootstrap_seed"] + corpus_index,
                )
                for mode in MODES:
                    estimates = {
                        arm: prior.bootstrap.weighted_eers(
                            scores[f"{arm}-{mode}"][keep], q, c, counts
                        )
                        for arm in ARMS
                    }
                    for control in ARMS[:2]:
                        key = f"{mode}-versus-{control}"
                        per_seed.setdefault(key, []).append(
                            estimates["train34_score34"] - estimates[control]
                        )
        for key, values in per_seed.items():
            average = np.mean(values, axis=0)
            distributions[f"{corpus}/{key}"] = average
            result[f"{corpus}/{key}"] = (
                100 * np.quantile(average, [0.025, 0.975])
            ).tolist()
    for mode in MODES:
        for control in ARMS[:2]:
            key = f"{mode}-versus-{control}"
            average = np.mean(
                [distributions[f"{c}/{key}"] for c in config["corpora"]], axis=0
            )
            distributions[f"macro/{key}"] = average
            result[f"macro/{key}"] = (
                100 * np.quantile(average, [0.025, 0.975])
            ).tolist()
    path = run / "bootstrap.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            for key, value in distributions.items():
                np.testing.assert_array_equal(previous[key], value)
    else:
        np.savez_compressed(path, **distributions)
    write(
        run / "bootstrap.json",
        {
            "method": "Paired query/claim speaker bootstrap, same draws across acoustic seeds and arms; independent corpus draws. Fixed subset/models, excludes selection and training uncertainty.",
            "replicates": config["bootstrap_replicates"],
            "eer_delta_ci95_pp": result,
        },
        immutable=True,
    )
