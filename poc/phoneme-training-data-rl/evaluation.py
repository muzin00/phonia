"""Freeze all selection choices before test, and independently reconstruct scores."""

from common import (
    ALL_PHONES,
    VARIANTS,
    comparisons,
    create_model,
    evaluation_inputs,
    model_path,
    np,
    pin,
    positions,
    prior,
    read,
    sha,
    state,
    torch,
    write,
)


def numeric_threshold(value):
    return np.inf if value == "+inf" else -np.inf if value == "-inf" else value


def evaluate(config, run):
    paths = {
        (seed, variant): model_path(config, run, seed, variant)
        for seed in config["encoder_seeds"]
        for variant in VARIANTS
    }
    write(run / "models-freeze.json", {"files": pin(paths.values())}, immutable=True)
    points, records, checks = {}, [], []
    for split in ("selection", "test"):
        if split == "test":
            locked = read(run / "selection-freeze.json")
            assert locked["thresholds"] == points
            assert locked["models_sha256"] == sha(run / "models-freeze.json")
        for seed in config["encoder_seeds"]:
            models = {}
            for variant in VARIANTS:
                model = create_model(config, seed)
                bundle = torch.load(
                    paths[seed, variant], weights_only=True, map_location="cpu"
                )
                model.load_state_dict(bundle["state_dict"])
                models[variant] = (
                    model.eval(),
                    0 if variant == "initial" else bundle["update"],
                )
            for corpus in config["corpora"]:
                state(run, "evaluating", split=split, corpus=corpus, seed=seed)
                arrays, data, rows = evaluation_inputs(config, run, seed, corpus, split)
                ids = positions(rows)
                lookup = {r["_source_index"]: r for r in rows}
                labels = np.array([lookup[i]["is_genuine"] for i in ids], bool)
                roles = np.array([lookup[i]["role"] for i in ids])
                normal = roles == "verification"
                scores = {
                    "ids": ids,
                    "labels": labels,
                    "roles": roles,
                    "query_speakers": np.array(
                        [data["speakers"].index(lookup[i]["speaker_id"]) for i in ids]
                    ),
                    "claims": ids % len(data["speakers"]),
                }
                for variant in VARIANTS:
                    model, update = models[variant]
                    scorer = prior.Scorer(model, arrays, ids)
                    values = scorer.score(ALL_PHONES)
                    direct = scorer.score(ALL_PHONES, direct=True)
                    difference = float(np.max(np.abs(values - direct)))
                    assert difference < 1e-12
                    scores[variant] = values
                    if variant == "initial":
                        mask = arrays["masks"][ids]
                        expected = np.where(
                            mask, arrays["components"][ids].astype(np.float64), 0
                        ).sum(axis=1) / mask.sum(axis=1)
                        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
                    key = f"{corpus}/{seed}/{variant}"
                    if split == "selection":
                        point, _ = prior.metrics.far_target_threshold(
                            values[normal], labels[normal], 0.01
                        )
                        points[key] = prior.metrics.json_threshold(point)
                    for role in sorted(set(roles)):
                        keep = roles == role
                        original = [r for r in rows if r["role"] == role]
                        positive = sum(r["is_genuine"] for r in original)
                        result = prior.summarize(
                            values[keep],
                            labels[keep],
                            numeric_threshold(points[key]),
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
                            split == "selection"
                            and variant != "initial"
                            and role == "verification"
                        ):
                            arm = variant.split("-")[0]
                            history = read(run / f"acoustic-{seed}/{arm}/history.json")
                            recorded = next(v for v in history if v["update"] == update)
                            assert result["eer"] == recorded["selection_eer"][corpus]
                        records.append(
                            {
                                "split": split,
                                "corpus": corpus,
                                "seed": seed,
                                "variant": variant,
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
                            "variant": variant,
                            "scored_trials": len(ids),
                            "maximum_direct_forward_difference": difference,
                        }
                    )
                path = run / "scores" / f"{corpus}-{seed}-{split}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        assert set(saved.files) == set(scores)
                        for key, value in scores.items():
                            np.testing.assert_array_equal(saved[key], value)
                else:
                    temp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp, **scores)
                    temp.replace(path)
                del arrays, scorer
        if split == "selection":
            write(run / "selection-metrics.json", records.copy(), immutable=True)
            write(
                run / "selection-freeze.json",
                {
                    "thresholds": points,
                    "models_sha256": sha(run / "models-freeze.json"),
                    "selection_files": pin(
                        [
                            run / "selection-metrics.json",
                            *run.glob("scores/*-selection.npz"),
                        ]
                    ),
                },
                immutable=True,
            )
    write(run / "metrics.json", records, immutable=True)
    write(run / "score-audit.json", checks, immutable=True)


def bootstrap_comparisons(config, run):
    distributions, result = {}, {}
    pairs = comparisons()
    for index, corpus in enumerate(config["corpora"]):
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
                    config["bootstrap_seed"] + index,
                )
                estimates = {
                    v: prior.bootstrap.weighted_eers(scores[v][keep], q, c, counts)
                    for v in VARIANTS
                }
                for treatment, control in pairs:
                    key = f"{treatment}-versus-{control}"
                    per_seed.setdefault(key, []).append(
                        estimates[treatment] - estimates[control]
                    )
        for key, values in per_seed.items():
            average = np.mean(values, axis=0)
            distributions[f"{corpus}/{key}"] = average
            result[f"{corpus}/{key}"] = (
                100 * np.quantile(average, [0.025, 0.975])
            ).tolist()
    for treatment, control in pairs:
        key = f"{treatment}-versus-{control}"
        average = np.mean(
            [distributions[f"{c}/{key}"] for c in config["corpora"]], axis=0
        )
        distributions[f"macro/{key}"] = average
        result[f"macro/{key}"] = (100 * np.quantile(average, [0.025, 0.975])).tolist()
    path = run / "bootstrap.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            for k, v in distributions.items():
                np.testing.assert_array_equal(old[k], v)
    else:
        np.savez_compressed(path, **distributions)
    write(
        run / "bootstrap.json",
        {
            "method": "Paired query/claim speaker bootstrap, shared draws across seeds and variants, independent corpora. Conditional on fixed models; excludes training/selection uncertainty.",
            "replicates": config["bootstrap_replicates"],
            "eer_delta_ci95_pp": result,
        },
        immutable=True,
    )
