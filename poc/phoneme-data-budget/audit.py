"""Independently recompute scores and verify both continuation trajectories."""

import gc
import importlib.util
from pathlib import Path

import data_budget as d

s, e = d.s, d.e


def main():
    s.torch.set_num_threads(1)
    config = d.read(d.CONFIG)
    run = d.ROOT / config["run_directory"]
    frozen = d.verify(run)
    selection = d.read(run / "selection-freeze.json")
    d.check_files(selection["models"])
    d.check_files(selection["validation_files"])
    results = d.read(run / "results.json")
    assert results["selection_sha256"] == d.sha(run / "selection-freeze.json")
    spec = importlib.util.spec_from_file_location(
        "independent_fusion_audit", d.ROOT / "poc/phoneme-corpus-transformer/audit.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    count, maximum = 0, 0.0
    files, selected = [], {}
    initial = {}
    seed = config["fusion_seed"]
    oldresults = d.read(d.ROOT / config["data_run"] / "results.json")
    for arm, acoustic_seed in d.jobs(config):
        family = run / d.name(arm, acoustic_seed)
        design = d.read(family / "design-freeze.json")
        d.check_files(design["files"])
        short = family / f"encoder-{config['encoder_budgets'][0]}/trial"
        long = family / f"encoder-{config['encoder_budgets'][1]}/trial"
        initial[arm, acoustic_seed] = d.read(short / "training-summary.json")[
            "initial_encoder_sha256"
        ]
        pool = s.TrainingPool(design["config"], family)
        phones = d.acoustic.s.ordered(frozen["phones"])
        original_plan = d.acoustic.s.schedule(
            pool.groups,
            phones,
            config["encoder_budgets"][0],
            acoustic_seed,
            pool.speaker_to_class,
        )
        digest = d.acoustic.s.json_sha256(
            dict(
                zip(
                    ("planned", "labels", "valid"),
                    d.acoustic.schedule_digest(
                        dict(
                            zip(
                                ("planned", "classes", "valid"),
                                original_plan,
                                strict=True,
                            )
                        )
                    ).values(),
                    strict=True,
                )
            )
        )
        assert digest == d.read(short / "training-summary.json")["schedule_sha256"]
        del original_plan
        plan = d.acoustic.make_schedule(config, pool, phones, acoustic_seed)
        longfreeze = d.read(long / "training-freeze.json")
        assert d.acoustic.schedule_digest(plan) == longfreeze["schedule_digests"]
        d.check_files(longfreeze["files"])
        d.check_files(d.read(long / "complete.json")["files"])
        assert longfreeze["manifest_sha256"] == pool.manifest_sha256
        assert longfreeze["feature_sha256"] == pool.feature_sha256
        saved = s.torch.load(long / "last.pt", weights_only=False)
        assert saved["update"] == config["encoder_budgets"][1]
        assert saved["training_freeze_sha256"] == d.sha(long / "training-freeze.json")
        assert all(
            int(v["step"]) == saved["update"]
            for v in saved["optimizer"]["state"].values()
        )
        assert all(
            g["lr"] == config["encoder_continuation_learning_rate"]
            for g in saved["optimizer"]["param_groups"]
        )
        exported = s.torch.load(long / "encoder.pt", weights_only=True)["model"]
        assert all(s.torch.equal(v, exported[k]) for k, v in saved["model"].items())
        files.extend(
            long / n
            for n in (
                "encoder.pt",
                "last.pt",
                "training-summary.json",
                "training-freeze.json",
                "complete.json",
            )
        )
        del pool, plan, saved
        gc.collect()
    for acoustic_seed in config["encoder_seeds"]:
        assert len({initial[arm, acoustic_seed] for arm in config["arms"]}) == 1
    assert len(
        {initial[config["arms"][0], seed] for seed in config["encoder_seeds"]}
    ) == len(config["encoder_seeds"])
    for key, paths in d.sources(config, run).items():
        fusion = d.ROOT / paths["fusion"]
        continued = d.ROOT / paths["budget"]
        original = d.read(fusion / "design-freeze.json")["config"]
        train = d.read(fusion / "train-inputs.json")
        prior = d.ROOT / frozen["prior_sources"][paths["arm"]]["fusion"]
        assert train == d.read(prior / "train-inputs.json")
        assert not set(train["enrollment_wav_hashes"]) & set(train["query_wav_hashes"])
        d.check_files(d.read(fusion / "prepared-freeze.json")["files"])
        d.check_files(d.read(continued / f"seed-{seed}/complete.json")["files"])
        schedule_config = d.read(continued / "design-freeze.json")["config"]
        expected = d.budget.build_schedule(schedule_config, original, train, seed)
        with s.np.load(continued / f"schedule-{seed}.npz", allow_pickle=False) as saved:
            for k, value in expected.items():
                assert s.np.array_equal(value, saved[k])
        history = d.read(continued / f"seed-{seed}/history.json")
        assert [v["update"] for v in history] == list(
            range(0, max(config["fusion_budgets"]) + 1, 500)
        )
        for updates in config["fusion_budgets"]:
            chosen = d.budget.select_best(
                [v for v in history if v["update"] <= updates]
            )["update"]
            selected[f"{key}-u{updates}"] = chosen
            for mode in ("selected", "final"):
                bundle = s.torch.load(
                    d.model_paths(config, paths)[f"u{updates}-{mode}"],
                    weights_only=True,
                )
                assert bundle["update"] == (chosen if mode == "selected" else updates)
        encoder, _ = s.load_model(d.ROOT / paths["trial"])
        for corpus in s.CORPORA:
            for split in ("validation", "test"):
                data, features, _ = e.evaluation_source(
                    config, frozen["reference_design"], corpus, split
                )
                assert data == d.read(fusion / f"{corpus}-{split}-inputs.json")
                assert not set(data["speakers"]) & set(train["speakers"])
                arrays = e.load_arrays(fusion / f"{corpus}-{split}.npz")
                baseline = d.read(fusion / f"{corpus}-{split}-baseline.json")
                recomputed = s.score(
                    data, s.model_embeddings(encoder, features), frozen["phones"]
                )
                assert recomputed == baseline
                for variant, model in {
                    "equal": None,
                    **d.models(config, paths),
                }.items():
                    label = f"{key}-{variant}"
                    path = run / "scores" / f"{corpus}-{label}-{split}.json"
                    records = d.read(path)
                    files.append(path)
                    assert len(records) == len(baseline)
                    for row, ref in zip(records, baseline, strict=True):
                        assert {k: v for k, v in row.items() if k != "score"} == {
                            k: v for k, v in ref.items() if k != "score"
                        }
                        assert (row["score"] is None) == (ref["score"] is None)
                    if model is None:
                        assert records == baseline
                    else:
                        maximum = max(
                            maximum,
                            auditor.independently_score(model, arrays, data, records),
                        )
                    threshold = selection["thresholds"][corpus][label]
                    if split == "validation":
                        normal = [
                            v
                            for v in records
                            if v["role"] == "verification" and v["score"] is not None
                        ]
                        expected_threshold, _ = s.far_target_threshold(
                            [v["score"] for v in normal],
                            [v["is_genuine"] for v in normal],
                            0.01,
                        )
                        assert s.json_threshold(expected_threshold) == threshold
                    else:
                        for role, metrics in results["conditions"][corpus][
                            label
                        ].items():
                            assert metrics == s.operating(
                                [v for v in records if v["role"] == role], threshold
                            )
                        if (
                            paths["encoder_seed"] == config["encoder_seeds"][0]
                            and paths["encoder_updates"] == config["encoder_budgets"][0]
                            and variant in ("equal", "u16000-selected", "u16000-final")
                        ):
                            oldkey = paths["arm"] + (
                                "-equal"
                                if variant == "equal"
                                else f"-{variant.split('-')[1]}-{seed}"
                            )
                            assert (
                                results["conditions"][corpus][label]
                                == oldresults["conditions"][corpus][oldkey]
                            )
                    count += len(records)
        print({"audited": key, "records": count}, flush=True)
    files.extend([run / "results.json", run / "selection-freeze.json", Path(__file__)])
    d.fixed(
        run / "complete-audit.json",
        {
            "files": d.pin(files),
            "records_verified": count,
            "maximum_score_difference": maximum,
            "selected_updates": selected,
            "original216k_16k_reproduced": True,
            "paired_initial_encoders_verified": True,
            "encoder_scores_recomputed": True,
            "all_evaluation_inputs_unchanged": True,
        },
    )
    print({"records_verified": count, "maximum_score_difference": maximum}, flush=True)


if __name__ == "__main__":
    main()
