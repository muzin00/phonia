"""Identical pair availability, equal-type fusion and validation-only thresholds."""

from pair_common import (
    ARMS,
    bootstrap,
    check,
    metrics,
    np,
    pin,
    read,
    sha,
    state,
    torch,
    write,
)
from pair_model import PairEncoder


def embeddings(model, matrix):
    with torch.inference_mode():
        return np.concatenate(
            [
                model(torch.from_numpy(np.array(matrix[start : start + 1024]))).numpy()
                for start in range(0, len(matrix), 1024)
            ]
        ).astype(np.float64)


def normalized_average(values):
    mean = np.mean(values, axis=0)
    return mean / max(float(np.linalg.norm(mean)), 1e-12)


def score(data, vectors, pairs):
    speakers = data["speakers"]
    enrolled = np.zeros((len(speakers), len(pairs), 128), dtype=np.float64)
    queries = np.zeros((len(data["queries"]), len(pairs), 128), dtype=np.float64)
    mask = np.zeros(queries.shape[:2], dtype=bool)
    for p, pair in enumerate(pairs):
        if pair not in data["universally_registered"]:
            continue
        for s, speaker in enumerate(speakers):
            enrolled[s, p] = normalized_average(
                vectors[data["profiles"][speaker][pair]]
            )
        for q, query in enumerate(data["queries"]):
            ids = query["groups"][pair]
            if query["old_eligible"] and ids:
                queries[q, p] = normalized_average(vectors[ids])
                mask[q, p] = True
    components = np.einsum("qpd,spd->qsp", queries, enrolled)
    available = mask.any(axis=1)
    assert available.tolist() == [q["scorable"] for q in data["queries"]]
    scores = np.full(components.shape[:2], np.nan)
    scores[available] = (components[available] * mask[available, None, :]).sum(
        -1
    ) / mask[available].sum(-1)[:, None]
    # A separately assembled mean of scalar per-type dot products.
    error = 0.0
    for q in np.flatnonzero(available):
        for s in range(len(speakers)):
            direct = sum(
                float(queries[q, p] @ enrolled[s, p]) for p in np.flatnonzero(mask[q])
            ) / int(mask[q].sum())
            error = max(error, abs(direct - scores[q, s]))
    assert error < 1e-12
    return scores, components, mask, error


def summarize(values, labels, threshold, total_positive, total_negative):
    keep = np.isfinite(values)
    x, y = values[keep], labels[keep]
    accepted = x >= threshold
    result = {
        "eer": metrics.roc_eer(x, y)["eer"],
        "far": int(np.sum(accepted & ~y)) / total_negative,
        "frr": (total_positive - int(np.sum(accepted & y))) / total_positive,
        "coverage": int(y.sum()) / total_positive,
        "scored_genuine": int(y.sum()),
        "total_genuine": total_positive,
        "scored_impostor": int((~y).sum()),
        "total_impostor": total_negative,
    }
    return result


def numeric(value):
    return np.inf if value == "+inf" else -np.inf if value == "-inf" else float(value)


def evaluate_split(config, run, split):
    pairs = read(run / "pair-selection.json")["selected"]
    records, details, points, audits = [], [], {}, []
    if split == "test":
        locked = read(run / "thresholds-freeze.json")
        check(locked["files"])
        points = locked["thresholds"]
    for c in config["corpora"]:
        directory = run / "data" / f"{c}-{split}"
        data = read(directory / "inputs.json")
        matrix = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        n = len(data["speakers"])
        owners = np.array(
            [data["speakers"].index(q["speaker_id"]) for q in data["queries"]]
        )
        roles = np.array([q["role"] for q in data["queries"]])
        labels = owners[:, None] == np.arange(n)[None]
        normal = roles == "verification"
        for seed in config["seeds"]:
            state(run, "evaluating", split=split, corpus=c, seed=seed)
            saved = {
                "query_speakers": np.repeat(owners, n),
                "claims": np.tile(np.arange(n), len(owners)),
                "roles": np.repeat(roles, n),
                "labels": labels.ravel(),
            }
            shared = None
            for arm in ARMS:
                path = run / f"seed-{seed}/{arm}/encoder.pt"
                bundle = torch.load(path, weights_only=True, map_location="cpu")
                assert bundle["arm"] == arm
                model = PairEncoder(arm).eval()
                model.load_state_dict(bundle["model"])
                vectors = embeddings(model, matrix)
                values, components, mask, error = score(data, vectors, pairs)
                if shared is not None:
                    np.testing.assert_array_equal(mask, shared)
                shared = mask
                saved[arm] = values.ravel()
                saved[f"{arm}_components"] = components
                saved["mask"] = mask
                key = f"{c}/{seed}/{arm}"
                if split == "validation":
                    keep = normal[:, None] & np.isfinite(values)
                    point, _ = metrics.far_target_threshold(
                        values[keep], labels[keep], 0.01
                    )
                    points[key] = metrics.json_threshold(point)
                for role in sorted(set(roles)):
                    keep = roles == role
                    result = summarize(
                        values[keep].ravel(),
                        labels[keep].ravel(),
                        numeric(points[key]),
                        int(keep.sum()),
                        int(keep.sum()) * (n - 1),
                    )
                    records.append(
                        {
                            "split": split,
                            "corpus": c,
                            "seed": seed,
                            "arm": arm,
                            "role": role,
                            "threshold": points[key],
                            **result,
                        }
                    )
                for p, pair in enumerate(pairs):
                    keep = normal & mask[:, p]
                    x, y = components[keep, :, p].ravel(), labels[keep].ravel()
                    details.append(
                        {
                            "split": split,
                            "corpus": c,
                            "seed": seed,
                            "arm": arm,
                            "pair": pair,
                            "eer": metrics.roc_eer(x, y)["eer"],
                            "queries": int(keep.sum()),
                            "total_queries": int(normal.sum()),
                        }
                    )
                audits.append(
                    {
                        "split": split,
                        "corpus": c,
                        "seed": seed,
                        "arm": arm,
                        "scalar_score_maximum_difference": error,
                    }
                )
            path = run / "scores" / f"{c}-{seed}-{split}.npz"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as old:
                    assert set(old.files) == set(saved)
                    for k, v in saved.items():
                        np.testing.assert_array_equal(old[k], v)
            else:
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **saved)
                temp.replace(path)
    write(run / f"{split}-metrics.json", records, immutable=True)
    write(run / f"{split}-pairs.json", details, immutable=True)
    write(run / f"{split}-score-audit.json", audits, immutable=True)
    if split == "validation":
        paths = [
            run / "models-freeze.json",
            run / "validation-metrics.json",
            run / "validation-pairs.json",
            *run.glob("scores/*-validation.npz"),
        ]
        write(
            run / "thresholds-freeze.json",
            {
                "thresholds": points,
                "models_freeze_sha256": sha(run / "models-freeze.json"),
                "files": {**read(run / "models-freeze.json")["files"], **pin(paths)},
            },
            immutable=True,
        )


def confidence(config, run):
    distributions, result = {}, {}
    for ci, c in enumerate(config["corpora"]):
        delta = []
        for seed in config["seeds"]:
            with np.load(
                run / "scores" / f"{c}-{seed}-test.npz", allow_pickle=False
            ) as data:
                keep = (data["roles"] == "verification") & np.isfinite(data["split"])
                assert np.array_equal(
                    np.isfinite(data["split"]), np.isfinite(data["joined"])
                )
                q, cl = data["query_speakers"][keep], data["claims"][keep]
                assert np.array_equal(data["labels"][keep], q == cl)
                _, counts = bootstrap.speaker_draws(
                    int(cl.max()) + 1,
                    config["bootstrap_replicates"],
                    config["bootstrap_seed"] + ci,
                )
                estimates = {
                    a: bootstrap.weighted_eers(data[a][keep], q, cl, counts)
                    for a in ARMS
                }
                delta.append(estimates["joined"] - estimates["split"])
        distributions[c] = np.mean(delta, axis=0)
    distributions["macro"] = np.mean(list(distributions.values()), axis=0)
    for c, values in distributions.items():
        assert np.isfinite(values).all()
        result[c] = (np.quantile(values, [0.025, 0.975]) * 100).tolist()
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
            "eer_delta_ci95_pp": result,
            "replicates": config["bootstrap_replicates"],
            "method": "Paired query/claim speaker bootstrap, shared draws across arms and seeds, independent corpora. Fixed models/pair selection; excludes training/selection uncertainty.",
        },
        immutable=True,
    )
