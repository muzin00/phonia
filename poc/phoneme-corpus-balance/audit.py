"""Audit actual sampling budgets, frozen models, thresholds and all score rows."""

import hashlib
import math

import run as experiment

r, s = experiment.r, experiment.s


def main():
    s.torch.set_num_threads(1)
    config = s.read(experiment.CONFIG)
    run = s.ROOT / config["run_directory"]
    frozen = s.read(run / "design-freeze.json")
    r.checked_files(frozen["files"])
    r.checked_files(s.read(run / "all-models-freeze.json")["files"])
    evaluation = s.read(run / "evaluation-freeze.json")
    r.checked_files(evaluation["validation_files"])
    aggregate = s.read(run / "results.json")
    r.checked_files(aggregate["source_results"])
    verified, results = 0, []
    for seed in config["seeds"]:
        directory = run / f"seed-{seed}"
        source = experiment.reference(config, seed)
        result = s.read(directory / "results.json")
        results.append(result)
        assert result["seed"] == seed
        assert result["evaluation_freeze_sha256"] == s.sha(
            run / "evaluation-freeze.json"
        )
        original = s.read(source / "results.json")
        initial = s.read(source / "expanded280/trial/training-summary.json")[
            "initial_encoder_sha256"
        ]
        control = s.read(directory / "original-exposure.json")
        for arm in config["conditions"]:
            current = s.read(directory / "design-freeze.json")["config"]
            summary = r.validate_trial(current, directory / arm, initial)
            assert summary["training_examples"] == control["valid_examples"]
            exposure = s.read(directory / arm / "sampling-exposure.json")
            assert exposure["valid_examples"] == control["valid_examples"]
            for phone, counts in exposure["phone_corpus_examples"].items():
                assert sum(counts.values()) == sum(
                    control["phone_corpus_examples"][phone].values()
                )
            for corpus in s.CORPORA:
                total = sum(
                    v[corpus] for v in exposure["phone_corpus_examples"].values()
                )
                assert exposure["corpus_examples"][corpus] == total
                assert (
                    exposure["corpus_fractions"][corpus]
                    == total / exposure["valid_examples"]
                )
        for arm in (*experiment.CONTEXT, *config["conditions"]):
            trial = (source if arm in experiment.CONTEXT else directory) / arm / "trial"
            model, bundle = s.load_model(trial)
            del model
            assert bundle["run_seed"] == seed and bundle["checkpoint_update"] == 216000
            assert evaluation["seeds"][str(seed)]["models"][arm] == s.sha(
                trial / "encoder.pt"
            )
        for corpus in s.CORPORA:
            for arm in experiment.CONTEXT:
                assert (
                    result["conditions"][corpus][arm]
                    == original["conditions"][corpus][arm]
                )
            for split in ("validation", "test"):
                name = f"{corpus}-{split}.json"
                assert s.sha(directory / "evaluation-inputs" / name) == s.sha(
                    source / "evaluation-inputs" / name
                )
            for arm in (*experiment.CONTEXT, *config["conditions"]):
                point = evaluation["seeds"][str(seed)]["thresholds"][corpus][arm]
                validation = s.read(
                    directory / "scores" / f"{corpus}-{arm}-validation.json"
                )
                scored = [
                    v
                    for v in validation
                    if v["role"] == "verification" and v["score"] is not None
                ]
                recalculated, _ = s.far_target_threshold(
                    s.np.array([v["score"] for v in scored]),
                    s.np.array([v["is_genuine"] for v in scored]),
                    0.01,
                )
                assert s.json_threshold(recalculated) == point
                records = s.read(directory / "scores" / f"{corpus}-{arm}-test.json")
                verified += len(records)
                for role, metrics in result["conditions"][corpus][arm].items():
                    rows = [v for v in records if v["role"] == role]
                    genuine = [v for v in rows if v["is_genuine"]]
                    impostor = [v for v in rows if not v["is_genuine"]]
                    threshold = float(point)
                    fa = sum(
                        v["score"] is not None and v["score"] >= threshold
                        for v in impostor
                    )
                    fr = sum(
                        v["score"] is None or v["score"] < threshold for v in genuine
                    )
                    assert (
                        metrics["false_accepts"] == fa
                        and metrics["false_rejects"] == fr
                    )
                    assert metrics["all_genuine"] == len(genuine) and metrics[
                        "all_impostor"
                    ] == len(impostor)
                    assert metrics["far"] == fa / len(impostor) and metrics[
                        "frr"
                    ] == fr / len(genuine)
                    assert metrics["coverage"] == sum(
                        v["score"] is not None for v in genuine
                    ) / len(genuine)
                    assert metrics["threshold"] == point
    for corpus, arms in aggregate["aggregate"].items():
        for arm, metrics in arms.items():
            for metric, reported in metrics.items():
                values = s.np.array(
                    [
                        v["conditions"][corpus][arm]["verification"][metric]
                        for v in results
                    ]
                )
                baseline = s.np.array(
                    [
                        v["conditions"][corpus]["expanded280"]["verification"][metric]
                        for v in results
                    ]
                )
                delta = values - baseline
                assert reported["values"] == values.tolist()
                assert reported["paired_delta_vs_expanded280"] == delta.tolist()
                for key, value in (
                    ("mean", values.mean()),
                    ("sd", values.std(ddof=1)),
                    ("paired_delta_mean", delta.mean()),
                ):
                    assert math.isclose(reported[key], value, abs_tol=1e-12)
                assert reported["decreased_seeds"] == int((delta < 0).sum())
                assert reported["increased_seeds"] == int((delta > 0).sum())
    # Rebuild both full schedules for the first seed and match the actual training hashes.
    seed = config["seeds"][0]
    directory = run / f"seed-{seed}"
    current = s.read(directory / "design-freeze.json")["config"]
    phones = frozen["reference_design"]["phonemes"]
    pool = s.TrainingPool(current, directory / "balanced280")
    for arm, cycle in config["conditions"].items():
        planned, labels, valid = experiment.sampler.schedule(
            pool.groups,
            phones,
            216000,
            seed,
            pool.speaker_to_class,
            cycle=cycle,
            rank=s.rank,
        )
        schedule_sha = experiment.learner.json_sha256(
            {
                "planned": hashlib.sha256(planned.tobytes()).hexdigest(),
                "labels": hashlib.sha256(labels.tobytes()).hexdigest(),
                "valid": hashlib.sha256(valid.tobytes()).hexdigest(),
            }
        )
        assert (
            schedule_sha
            == s.read(directory / arm / "trial/training-summary.json")[
                "schedule_sha256"
            ]
        )
        assert experiment.sampler.exposure(
            labels, valid, phones, pool.speakers
        ) == s.read(directory / arm / "sampling-exposure.json")
        del planned, labels, valid
    report = {
        "status": "passed",
        "seeds": 5,
        "models": 25,
        "new_models": 10,
        "test_records": verified,
        "calibration_thresholds": 75,
        "full_sampling_schedules_rebuilt": 2,
        "per_phone_and_total_training_budgets_match": True,
        "files": r.pin(
            [run / "results.json", run / "results.md", experiment.BASE / "audit.py"]
        ),
    }
    s.fixed(run / "completion-audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
