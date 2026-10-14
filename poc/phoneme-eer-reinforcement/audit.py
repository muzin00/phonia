"""Independently verify checkpoint selection, paired starts and published metrics."""

from common import (
    ARMS,
    VARIANTS,
    assert_nested_equal,
    check,
    family,
    model_hash,
    np,
    pin,
    prior,
    read,
    sha,
    starting_path,
    torch,
    write,
)
from evaluation import numeric_threshold
from policies import build


def audit(config, run):
    check(read(run / "design-freeze.json")["files"])
    check(read(run / "models-freeze.json")["files"])
    selection = read(run / "selection-freeze.json")
    check(selection["validation_files"])
    assert selection["models_sha256"] == sha(run / "models-freeze.json")
    training = {}
    for seed in config["encoder_seeds"]:
        initial = torch.load(starting_path(config, seed), weights_only=True)
        results = {}
        with np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as actual:
            source = (
                family(config, seed)
                / "continued"
                / f"schedule-{config['fusion_seed']}.npz"
            )
            with np.load(source, allow_pickle=False) as old:
                for key in old.files:
                    first = config["source_schedule_start"]
                    stop = (
                        first + config["source_batches_per_update"] * config["updates"]
                    )
                    np.testing.assert_array_equal(
                        actual[key], old[key][first:stop].reshape(config["updates"], -1)
                    )
        for arm in ARMS:
            directory = run / f"acoustic-{seed}/{arm}"
            check(read(directory / "complete.json")["files"])
            saved = torch.load(
                directory / "last.pt", weights_only=False, map_location="cpu"
            )
            history = read(directory / "history.json")
            assert saved["history"] == history
            initial_policy = build(
                config, seed, arm, read(run / f"calibration-{seed}.json")["sigma"]
            )
            assert saved["initial_model_sha256"] == model_hash(
                initial_policy.state_dict()
            )
            for name, value in initial["state_dict"].items():
                torch.testing.assert_close(
                    saved["model"][f"reference.{name}"], value, rtol=0, atol=0
                )
            assert any(
                not torch.equal(saved["model"][k], v)
                for k, v in initial_policy.state_dict().items()
                if k.startswith("actor.")
            )
            assert saved["design_sha256"] == sha(run / "design-freeze.json")
            assert saved["update"] == config["updates"]
            assert [h["update"] for h in history] == list(
                range(0, config["updates"] + 1, config["validation_every"])
            )
            best = history[0]
            for row in history:
                assert row["macro_eer"] == sum(row["validation_eer"].values()) / len(
                    config["corpora"]
                )
                if row["macro_eer"] < best["macro_eer"] - 1e-12:
                    best = row
            assert saved["best_bundle"]["update"] == best["update"]
            assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {
                config["updates"]
            }
            assert {v["lr"] for v in saved["optimizer"]["param_groups"]} == {
                config["learning_rate"]
            }
            for mode in ("selected", "final"):
                bundle = torch.load(directory / f"{mode}.pt", weights_only=True)
                expected = (
                    saved["best_bundle"]["state_dict"]
                    if mode == "selected"
                    else saved["model"]
                )
                assert_nested_equal(bundle["state_dict"], expected)
                assert bundle["update"] == (
                    best["update"] if mode == "selected" else config["updates"]
                )
            results[arm] = saved
            training[f"{seed}/{arm}"] = {
                "selected_update": best["update"],
                "selected_validation_macro_eer": best["macro_eer"],
            }
        for arm in ARMS:
            assert results["bce"]["history"][0] == results[arm]["history"][0]
    records = read(run / "metrics.json")
    max_error, count = 0.0, 0
    for seed in config["encoder_seeds"]:
        for split in ("validation", "test"):
            for corpus in config["corpora"]:
                arrays, data, rows = prior.load_inputs(config, seed, corpus, split)
                del arrays
                with np.load(
                    run / "scores" / f"{corpus}-{seed}-{split}.npz", allow_pickle=False
                ) as scores:
                    ids, labels = scores["ids"], scores["labels"]
                    np.testing.assert_array_equal(
                        ids, prior.positions(rows, normal_only=False)
                    )
                    np.testing.assert_array_equal(
                        labels, [rows[i]["is_genuine"] for i in ids]
                    )
                    np.testing.assert_array_equal(
                        scores["roles"], [rows[i]["role"] for i in ids]
                    )
                    np.testing.assert_array_equal(
                        scores["claims"], ids % len(data["speakers"])
                    )
                    np.testing.assert_array_equal(
                        scores["query_speakers"],
                        [data["speakers"].index(rows[i]["speaker_id"]) for i in ids],
                    )
                    for variant in VARIANTS:
                        values = scores[variant]
                        point = selection["thresholds"][f"{corpus}/{seed}/{variant}"]
                        numeric = numeric_threshold(point)
                        if split == "validation":
                            keep = scores["roles"] == "verification"
                            expected, _ = prior.metrics.far_target_threshold(
                                values[keep], labels[keep], 0.01
                            )
                            assert prior.metrics.json_threshold(expected) == point
                        for role in sorted(set(scores["roles"])):
                            keep = scores["roles"] == role
                            x, y = values[keep], labels[keep]
                            original = [r for r in rows if r["role"] == role]
                            positive = sum(r["is_genuine"] for r in original)
                            negative = len(original) - positive
                            accepted = x >= numeric
                            expected = {
                                "eer": prior.independent_eer(x, y),
                                "far": int(np.sum(accepted & ~y)) / negative,
                                "frr": (positive - int(np.sum(accepted & y)))
                                / positive,
                                "coverage": int(y.sum()) / positive,
                            }
                            matches = [
                                r
                                for r in records
                                if all(
                                    r[k] == v
                                    for k, v in {
                                        "split": split,
                                        "corpus": corpus,
                                        "seed": seed,
                                        "variant": variant,
                                        "role": role,
                                    }.items()
                                )
                            ]
                            assert len(matches) == 1
                            for key, value in expected.items():
                                error = abs(value - matches[0][key])
                                assert error < 1e-12
                                max_error = max(max_error, error)
                        count += len(values)
    direct = read(run / "score-audit.json")
    assert len(direct) == len(config["encoder_seeds"]) * len(
        config["corpora"]
    ) * 2 * len(VARIANTS)
    write(
        run / "independent-audit.json",
        {
            "training": training,
            "saved_score_values_checked": count,
            "maximum_metric_difference": max_error,
            "maximum_direct_forward_difference": max(
                r["maximum_direct_forward_difference"] for r in direct
            ),
            "initial_scores_frozen_reference_and_pair_schedules_verified": True,
            "files": pin(
                [
                    run / "metrics.json",
                    run / "selection-freeze.json",
                    run / "models-freeze.json",
                ]
            ),
        },
        immutable=True,
    )
