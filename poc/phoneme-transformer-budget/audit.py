"""Audit continuation, selected/final checkpoints and every saved score row."""

import importlib.util

import run as b

s, r, e = b.s, b.r, b.e
spec = importlib.util.spec_from_file_location(
    "source_fusion_audit", e.BASE / "audit.py"
)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def main():
    s.torch.set_num_threads(1)
    config = s.read(b.CONFIG)
    run = s.ROOT / config["run_directory"]
    ref = b.source(config)
    assert s.read(run / "state.json")["status"] == "complete"
    design = s.read(run / "design-freeze.json")
    selection = s.read(run / "selection-freeze.json")
    for files in (
        design["files"],
        s.read(run / "schedule-freeze.json")["files"],
        selection["models"],
        selection["validation_files"],
        s.read(run / "all-models-freeze.json")["files"],
    ):
        r.checked_files(files)
    results = s.read(run / "results.json")
    assert results["selection_sha256"] == s.sha(run / "selection-freeze.json")
    data = s.read(ref / "train-inputs.json")
    original = b.original_config(config)
    histories, selected_updates = {}, {}
    for seed in config["seeds"]:
        directory = run / f"seed-{seed}"
        r.checked_files(s.read(directory / "complete.json")["files"])
        assert s.read(directory / "complete.json")["updates"] == 64000
        expected = b.build_schedule(config, original, data, seed)
        with s.np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as saved:
            for k in expected:
                s.np.testing.assert_array_equal(saved[k], expected[k])
        history = s.read(directory / "history.json")
        histories[seed] = history
        reference_history = s.read(ref / f"transformer-{seed}/history.json")
        assert history[: len(reference_history)] == reference_history
        assert [item["update"] for item in history] == list(range(0, 64001, 500))
        for item in history:
            assert item["macro_eer"] == sum(item["validation_eer"].values()) / 3
        last = s.torch.load(directory / "last.pt", weights_only=False)
        assert last["update"] == 64000
        for group in last["optimizer"]["param_groups"]:
            assert group["lr"] == config["continuation_learning_rate"]
        assert all(
            int(state["step"]) == 64000 for state in last["optimizer"]["state"].values()
        )
        for budget in config["budgets"]:
            summary = s.read(directory / f"budget-{budget}/summary.json")
            r.checked_files(summary["models"])
            best = b.select_best([v for v in history if v["update"] <= budget])
            assert summary["selected_update"] == best["update"]
            assert summary["validation_macro_eer"] == best["macro_eer"]
            assert summary["completed_updates"] == budget
            assert summary["encoder_updated"] is False
            assert summary["encoder_sha256"] == original["encoder_sha256"]
            selected_updates[f"u{budget}-{seed}"] = best["update"]
            for mode in ("selected", "final"):
                bundle = s.torch.load(
                    directory / f"budget-{budget}/{mode}.pt", weights_only=True
                )
                assert bundle["update"] == (
                    budget if mode == "final" else best["update"]
                )
                if budget == 4000:
                    original_bundle = s.torch.load(
                        ref
                        / f"transformer-{seed}"
                        / ("last.pt" if mode == "final" else "fusion.pt"),
                        weights_only=False,
                    )
                    weights = original_bundle[
                        "model" if mode == "final" else "state_dict"
                    ]
                    for name, tensor in weights.items():
                        s.torch.testing.assert_close(
                            tensor, bundle["state_dict"][name], rtol=0, atol=0
                        )
                if budget == 64000 and mode == "final":
                    for name, tensor in last["model"].items():
                        s.torch.testing.assert_close(
                            tensor, bundle["state_dict"][name], rtol=0, atol=0
                        )
    paths = b.variants(config, run)
    models = {name: b.load_model(config, path) for name, path in paths.items()}
    count, difference, files = 0, 0.0, []
    original_results = s.read(ref / "results.json")
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            arrays = e.load_arrays(ref / f"{corpus}-{split}.npz")
            data = s.read(ref / f"{corpus}-{split}-inputs.json")
            baseline = s.read(ref / f"{corpus}-{split}-baseline.json")
            for name in ("equal", *models):
                path = run / "scores" / f"{corpus}-{name}-{split}.json"
                records = s.read(path)
                assert len(records) == len(baseline)
                for a, z in zip(records, baseline, strict=True):
                    assert {k: v for k, v in a.items() if k != "score"} == {
                        k: v for k, v in z.items() if k != "score"
                    }
                    assert (a["score"] is None) == (z["score"] is None)
                if name == "equal":
                    assert records == baseline
                else:
                    difference = max(
                        difference,
                        previous.independently_score(
                            models[name], arrays, data, records
                        ),
                    )
                    if name.startswith("u4000-selected-"):
                        seed = int(name.rsplit("-", 1)[1])
                        assert records == s.read(
                            ref / "scores" / f"{corpus}-transformer-{seed}-{split}.json"
                        )
                threshold = selection["thresholds"][corpus][name]
                if split == "validation":
                    normal = [
                        v
                        for v in records
                        if v["role"] == "verification" and v["score"] is not None
                    ]
                    point, _ = s.far_target_threshold(
                        [v["score"] for v in normal],
                        [v["is_genuine"] for v in normal],
                        0.01,
                    )
                    assert s.json_threshold(point) == threshold
                    if name != "equal":
                        budget, mode, seed = name.split("-")
                        budget, seed = int(budget[1:]), int(seed)
                        history = histories[seed]
                        expected = (
                            next(v for v in history if v["update"] == budget)
                            if mode == "final"
                            else b.select_best(
                                [v for v in history if v["update"] <= budget]
                            )
                        )
                        assert e.eer(records) == expected["validation_eer"][corpus]
                else:
                    for role, metrics in results["conditions"][corpus][name].items():
                        assert metrics == s.operating(
                            [v for v in records if v["role"] == role], threshold
                        )
                    if name == "equal":
                        assert (
                            results["conditions"][corpus][name]
                            == original_results["conditions"][corpus][name]
                        )
                    elif name.startswith("u4000-selected-"):
                        assert (
                            results["conditions"][corpus][name]
                            == original_results["conditions"][corpus][
                                f"transformer-{seed}"
                            ]
                        )
                files.append(path)
                count += len(records)
        for name, metrics in results["aggregate"][corpus].items():
            for metric, value in metrics.items():
                expected = [
                    results["conditions"][corpus][f"{name}-{seed}"]["verification"][
                        metric
                    ]
                    for seed in config["seeds"]
                ]
                assert value["values"] == expected
                assert value["mean"] == float(s.np.mean(expected))
    report = {
        "verified_score_rows": count,
        "maximum_recomputed_score_difference": difference,
        "selected_updates": selected_updates,
        "continuation_updates_per_seed": 60000,
        "source_4000_selected_scores_identical": True,
        "files": r.pin(
            [
                *files,
                run / "results.json",
                run / "selection-freeze.json",
                b.BASE / "audit.py",
            ]
        ),
    }
    s.fixed(run / "complete-audit.json", report)
    print({k: v for k, v in report.items() if k != "files"})


if __name__ == "__main__":
    main()
