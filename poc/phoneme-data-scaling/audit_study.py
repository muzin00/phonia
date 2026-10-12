"""Recompute thresholds, checkpoint choices and every saved evaluation score."""

import importlib.util
from pathlib import Path

from scaling_study import (
    CONFIG,
    ROOT,
    budget,
    check_files,
    e,
    fixed,
    model_sources,
    pin,
    read,
    s,
    sha,
    verify,
)


def main():
    s.torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    frozen = verify(run)
    assert read(run / "state.json")["status"] == "complete"
    selection = read(run / "selection-freeze.json")
    check_files(selection["models"])
    check_files(selection["validation_files"])
    results = read(run / "results.json")
    assert results["selection_sha256"] == sha(run / "selection-freeze.json")
    spec = importlib.util.spec_from_file_location(
        "independent_fusion_audit", ROOT / "poc/phoneme-corpus-transformer/audit.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    sources = model_sources(config, run)
    maximum = 0.0
    count = 0
    files = []
    selected = {}
    for arm, paths in sources.items():
        original = read(paths["fusion"] / "design-freeze.json")["config"]
        check_files(read(paths["fusion"] / "prepared-freeze.json")["files"])
        train = read(paths["fusion"] / "train-inputs.json")
        assert not set(train["enrollment_wav_hashes"]) & set(train["query_wav_hashes"])
        models = {}
        for seed in config["fusion_seeds"]:
            history = read(paths["budget"] / f"seed-{seed}/history.json")
            within = [v for v in history if v["update"] <= 16000]
            assert [v["update"] for v in within] == list(range(0, 16001, 500))
            best = budget.select_best(within)
            selected[f"{arm}-{seed}"] = best["update"]
            for mode in ("selected", "final"):
                path = paths["budget"] / f"seed-{seed}/budget-16000/{mode}.pt"
                bundle = s.torch.load(path, weights_only=True)
                assert bundle["update"] == (
                    best["update"] if mode == "selected" else 16000
                )
                model = e.create_model(original, "transformer")
                model.load_state_dict(bundle["state_dict"])
                models[f"{mode}-{seed}"] = model.eval()
            # Check the exact shared 4k prefix and continuation blocks through 16k.
            schedule_config = read(paths["budget"] / "design-freeze.json")["config"]
            schedule_config = {**schedule_config, "budgets": [4000, 16000]}
            regenerated = budget.build_schedule(schedule_config, original, train, seed)
            with s.np.load(
                paths["budget"] / f"schedule-{seed}.npz", allow_pickle=False
            ) as saved:
                for key, value in regenerated.items():
                    assert s.np.array_equal(saved[key][:16000], value)
        for corpus in s.CORPORA:
            for split in ("validation", "test"):
                data = read(paths["fusion"] / f"{corpus}-{split}-inputs.json")
                assert not set(train["speakers"]) & set(data["speakers"])
                # All arms must retain exactly the original evaluation queries and profiles.
                assert data == read(
                    ROOT
                    / config["reference_run"]
                    / "evaluation-inputs"
                    / f"{corpus}-{split}.json"
                )
                arrays = e.load_arrays(paths["fusion"] / f"{corpus}-{split}.npz")
                baseline = read(paths["fusion"] / f"{corpus}-{split}-baseline.json")
                for variant in ["equal", *models]:
                    key = f"{arm}-{variant}"
                    path = run / "scores" / f"{corpus}-{key}-{split}.json"
                    records = read(path)
                    files.append(path)
                    assert len(records) == len(baseline)
                    for row, expected in zip(records, baseline, strict=True):
                        assert {k: v for k, v in row.items() if k != "score"} == {
                            k: v for k, v in expected.items() if k != "score"
                        }
                        assert (row["score"] is None) == (expected["score"] is None)
                    if variant == "equal":
                        assert records == baseline
                    else:
                        maximum = max(
                            maximum,
                            auditor.independently_score(
                                models[variant], arrays, data, records
                            ),
                        )
                    threshold = selection["thresholds"][corpus][key]
                    if split == "validation":
                        normal = [
                            v
                            for v in records
                            if v["role"] == "verification" and v["score"] is not None
                        ]
                        expected, _ = s.far_target_threshold(
                            [v["score"] for v in normal],
                            [v["is_genuine"] for v in normal],
                            0.01,
                        )
                        assert s.json_threshold(expected) == threshold
                    else:
                        for role, metrics in results["conditions"][corpus][key].items():
                            assert metrics == s.operating(
                                [v for v in records if v["role"] == role], threshold
                            )
                    count += len(records)
        if arm != "baseline280":
            summary = read(paths["trial"] / "training-summary.json")
            assert summary["completed_updates"] == 216000
            check_files(read(run / arm / "design-freeze.json")["files"])
    baseline_results = read(ROOT / config["reference_budget"] / "results.json")
    for corpus in s.CORPORA:
        for mode in ("selected", "final"):
            for seed in config["fusion_seeds"]:
                assert (
                    results["conditions"][corpus][f"baseline280-{mode}-{seed}"]
                    == baseline_results["conditions"][corpus][f"u16000-{mode}-{seed}"]
                )
    files += [run / "results.json", run / "selection-freeze.json", Path(__file__)]
    fixed(
        run / "complete-audit.json",
        {
            "records_verified": count,
            "maximum_score_difference": maximum,
            "selected_updates": selected,
            "all_evaluation_inputs_unchanged": True,
            "baseline16k_reproduced": True,
            "files": pin(files),
            "interpretation": frozen["config"]["interpretation"],
        },
    )
    print(
        {
            "records_verified": count,
            "maximum_score_difference": maximum,
            "baseline16k_reproduced": True,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
