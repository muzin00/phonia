"""Verify saved models, all score denominators, calibration and seed aggregation."""

import math
import statistics

import replicate as r


def numeric(value):
    return float(value) if isinstance(value, str) else value


def main():
    s = r.s
    s.torch.set_num_threads(1)
    config = s.read(r.CONFIG)
    run = s.ROOT / config["run_directory"]
    frozen = s.read(run / "design-freeze.json")
    r.checked_files(frozen["files"])
    assert frozen["runtime"] == s.expanded.runtime()
    r.checked_files(s.read(run / "all-models-freeze.json")["files"])
    reference = s.ROOT / config["reference_run"]
    sources = [
        reference,
        *(run / f"seed-{seed}" for seed in config["additional_seeds"]),
    ]
    results = [s.read(path / "results.json") for path in sources]
    aggregate = s.read(run / "results.json")
    r.checked_files(aggregate["source_results"])
    initial_hashes, verified_records = set(), 0
    for seed, source, result in zip(r.seeds(config), sources, results, strict=True):
        evaluation = s.read(source / "evaluation-freeze.json")
        assert result["evaluation_freeze_sha256"] == s.sha(
            source / "evaluation-freeze.json"
        )
        initials = set()
        for arm in s.ARMS:
            trial = source / arm / "trial"
            summary = s.read(trial / "training-summary.json")
            assert summary["completed_updates"] == 216000
            assert (
                summary["encoder_sha256"]
                == evaluation["models"][arm]
                == s.sha(trial / "encoder.pt")
            )
            _, bundle = s.load_model(trial)
            assert bundle["run_seed"] == seed
            initials.add(summary["initial_encoder_sha256"])
        assert len(initials) == 1 and not initial_hashes & initials
        initial_hashes.update(initials)
        for corpus in s.CORPORA:
            for split in ("validation", "test"):
                name = f"{corpus}-{split}.json"
                assert s.sha(source / "evaluation-inputs" / name) == s.sha(
                    reference / "evaluation-inputs" / name
                )
            for arm in s.ARMS:
                threshold = evaluation["thresholds"][corpus][arm]
                validation = s.read(
                    source / "scores" / f"{corpus}-{arm}-validation.json"
                )
                validation = [
                    v
                    for v in validation
                    if v["role"] == "verification" and v["score"] is not None
                ]
                point, _ = s.far_target_threshold(
                    s.np.array([v["score"] for v in validation]),
                    s.np.array([v["is_genuine"] for v in validation]),
                    0.01,
                )
                assert threshold == s.json_threshold(point)
                scores = s.read(source / "scores" / f"{corpus}-{arm}-test.json")
                verified_records += len(scores)
                for role, reported in result["conditions"][corpus][arm].items():
                    rows = [v for v in scores if v["role"] == role]
                    genuine = [v for v in rows if v["is_genuine"]]
                    impostor = [v for v in rows if not v["is_genuine"]]
                    fa = sum(
                        v["score"] is not None and v["score"] >= numeric(threshold)
                        for v in impostor
                    )
                    fr = sum(
                        v["score"] is None or v["score"] < numeric(threshold)
                        for v in genuine
                    )
                    assert (
                        reported["false_accepts"] == fa
                        and reported["false_rejects"] == fr
                    )
                    assert reported["all_genuine"] == len(genuine) and reported[
                        "all_impostor"
                    ] == len(impostor)
                    assert reported["far"] == fa / len(impostor) and reported[
                        "frr"
                    ] == fr / len(genuine)
                    assert reported["coverage"] == sum(
                        v["score"] is not None for v in genuine
                    ) / len(genuine)
                    assert reported["threshold"] == threshold
    for corpus in s.CORPORA:
        for arm in s.ARMS:
            for metric in ("eer", "far", "frr", "coverage"):
                values = [
                    v["conditions"][corpus][arm]["verification"][metric]
                    for v in results
                ]
                deltas = [
                    value
                    - v["conditions"][corpus]["baseline140"]["verification"][metric]
                    for value, v in zip(values, results, strict=True)
                ]
                report = aggregate["aggregate"][corpus][arm][metric]
                assert (
                    report["values"] == values
                    and report["paired_delta_values"] == deltas
                )
                assert math.isclose(
                    report["mean"], statistics.mean(values), abs_tol=1e-12
                )
                assert math.isclose(
                    report["sd"], statistics.stdev(values), abs_tol=1e-12
                )
                assert math.isclose(
                    report["paired_delta_mean"], statistics.mean(deltas), abs_tol=1e-12
                )
                assert report["min"] == min(values) and report["max"] == max(values)
                assert report["paired_delta_min"] == min(deltas) and report[
                    "paired_delta_max"
                ] == max(deltas)
                assert report["decreased_seeds"] == sum(d < 0 for d in deltas)
                assert report["increased_seeds"] == sum(d > 0 for d in deltas)
                assert report["unchanged_seeds"] == sum(d == 0 for d in deltas)
    report = {
        "status": "passed",
        "seeds_verified": len(sources),
        "models_verified": len(sources) * len(s.ARMS),
        "test_records_verified": verified_records,
        "calibration_thresholds_verified": len(sources) * len(s.ARMS) * len(s.CORPORA),
        "checks": [
            "frozen code and input hashes",
            "matched seed and initial encoder",
            "identical evaluation inputs",
            "validation-only threshold recomputation",
            "test FAR/FRR and coverage denominators",
            "independent mean/SD/paired delta aggregation",
        ],
        "files": r.pin([run / "results.json", run / "results.md", r.BASE / "audit.py"]),
    }
    s.fixed(run / "completion-audit.json", report)
    print(report)


if __name__ == "__main__":
    main()
