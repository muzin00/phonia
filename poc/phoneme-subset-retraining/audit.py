"""Audit selection, immutable training inputs and independent saved-score metrics."""

from common import (
    assert_nested_equal,
    check,
    family,
    model_hash,
    np,
    old_model_path,
    pin,
    prior,
    read,
    sha,
    torch,
    write,
)
from evaluation import ARMS, MODES


def audit(config, run):
    check(read(run / "design-freeze.json")["files"])
    check(read(run / "models-freeze.json")["files"])
    selection = read(run / "selection-freeze.json")
    check(selection["validation_files"])
    assert selection["models_sha256"] == sha(run / "models-freeze.json")
    reference = read(run / "reference-prefix/training-summary.json")
    assert reference["original_prefix_model_optimizer_rng_history_exact_match"]
    training = {}
    for seed in config["encoder_seeds"]:
        directory = run / f"acoustic-{seed}"
        check(read(directory / "complete.json")["files"])
        saved = torch.load(
            directory / "last.pt", weights_only=False, map_location="cpu"
        )
        history = read(directory / "history.json")
        summary = read(directory / "training-summary.json")
        assert saved["history"] == history
        assert [v["update"] for v in history] == list(
            range(0, config["updates"] + 1, config["validation_every"])
        )
        best = history[0]
        for row in history:
            assert row["macro_eer"] == sum(row["validation_eer"].values()) / len(
                config["corpora"]
            )
            if row["macro_eer"] < best["macro_eer"] - 1e-12:
                best = row
        assert summary["selected_update"] == best["update"]
        assert summary["selected_validation_macro_eer"] == best["macro_eer"]
        assert summary["initial_model_sha256"] == reference["initial_model_sha256"]
        assert summary["initial_rng_sha256"] == reference["initial_rng_sha256"]
        assert summary["final_model_sha256"] == model_hash(saved["model"])
        assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {
            config["updates"]
        }
        assert {v["lr"] for v in saved["optimizer"]["param_groups"]} == {
            config["continuation_learning_rate"]
        }
        assert summary["schedule_sha256"] == sha(
            family(config, seed) / "continued" / f"schedule-{config['fusion_seed']}.npz"
        )
        assert summary["encoder_sha256"] == sha(
            family(config, seed) / "trial/encoder.pt"
        )
        for mode in MODES:
            bundle = torch.load(
                directory / f"{mode}.pt", weights_only=True, map_location="cpu"
            )
            assert bundle["update"] == (
                best["update"] if mode == "selected" else config["updates"]
            )
            assert_nested_equal(
                bundle["state_dict"],
                saved["best_bundle"]["state_dict"]
                if mode == "selected"
                else saved["model"],
            )
        training[str(seed)] = {
            "selected_update": best["update"],
            "selected_validation_macro_eer": best["macro_eer"],
            "fixed_final_update": saved["update"],
        }
    records = read(run / "metrics.json")
    max_error, count = 0.0, 0
    for seed in config["encoder_seeds"]:
        for split in ("validation", "test"):
            for corpus in config["corpora"]:
                arrays, data, rows = prior.load_inputs(config, seed, corpus, split)
                del arrays
                expected_ids = prior.positions(rows, normal_only=False)
                with np.load(
                    run / "scores" / f"{corpus}-{seed}-{split}.npz", allow_pickle=False
                ) as scores:
                    ids = scores["ids"]
                    np.testing.assert_array_equal(ids, expected_ids)
                    labels = scores["labels"]
                    np.testing.assert_array_equal(
                        labels, [rows[i]["is_genuine"] for i in ids]
                    )
                    np.testing.assert_array_equal(
                        scores["claims"], ids % len(data["speakers"])
                    )
                    for arm in ARMS:
                        for mode in MODES:
                            values = scores[f"{arm}-{mode}"]
                            point = selection["thresholds"][
                                f"{corpus}/{seed}/{arm}-{mode}"
                            ]
                            numeric = (
                                np.inf
                                if point == "+inf"
                                else -np.inf
                                if point == "-inf"
                                else point
                            )
                            if split == "validation":
                                normal = scores["roles"] == "verification"
                                expected, _ = prior.metrics.far_target_threshold(
                                    values[normal], labels[normal], 0.01
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
                                    if r["split"] == split
                                    and r["corpus"] == corpus
                                    and r["seed"] == seed
                                    and r["arm"] == arm
                                    and r["mode"] == mode
                                    and r["role"] == role
                                ]
                                assert len(matches) == 1
                                row = matches[0]
                                for name, value in expected.items():
                                    difference = abs(value - row[name])
                                    assert difference < 1e-12
                                    max_error = max(max_error, difference)
                            count += len(values)
    direct = read(run / "score-audit.json")
    assert len(direct) == len(config["encoder_seeds"]) * len(config["corpora"]) * 2 * 6
    write(
        run / "independent-audit.json",
        {
            "training": training,
            "saved_score_values_checked": count,
            "maximum_metric_difference": max_error,
            "maximum_direct_forward_difference": max(
                r["maximum_direct_forward_difference"] for r in direct
            ),
            "checkpoint_rules_and_updates_verified": True,
            "files": pin(
                [
                    run / "metrics.json",
                    run / "selection-freeze.json",
                    run / "models-freeze.json",
                    *[
                        old_model_path(config, seed, mode)
                        for seed in config["encoder_seeds"]
                        for mode in MODES
                    ],
                ]
            ),
        },
        immutable=True,
    )
