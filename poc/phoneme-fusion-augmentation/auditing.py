"""Independent reconstruction, paired uncertainty and immutable output provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import shared as s
from augmentation import make_schedule
from experiment import load_npz


def check_training(config, run, design, selection):
    protocol = design["training_protocol"]
    previous = s.source(config)
    data = s.read_json(previous / "train-inputs.json")
    clean = s.frozen.load_arrays(previous, "train")
    noisy = load_npz(run / "train-noisy-aggregates.npz")
    for key in ("enrollment", "emeta", "emask", "qmask"):
        if not s.torch.equal(clean[key], noisy[key]):
            raise ValueError(f"training enrollment or coverage changed: {key}")
    evaluation = [
        s.read_json(previous / f"{split}-inputs.json")
        for split in ("validation", "test")
    ]
    if set(data["speakers"]) & {p for d in evaluation for p in d["speakers"]}:
        raise ValueError("train/evaluation speaker overlap")
    source = s.ROOT / protocol["source_run"]
    evaluation_hashes = {
        row["source_sha256"]
        for split in ("validation", "test")
        for row in s.frozen.rows(source / f"{split}-segments.jsonl")
    }
    enrollment_sources = set(data["enrollment_sources"])
    enrollment_hashes = {
        item["sha256"]
        for catalog in data["catalog"].values()
        for name, item in catalog.items()
        if name in enrollment_sources
    }
    query_hashes = {q["source_sha256"] for q in data["queries"]}
    if (enrollment_hashes & query_hashes) or (
        evaluation_hashes & (enrollment_hashes | query_hashes)
    ):
        raise ValueError("audio hash overlap")
    needed = {
        i for q in data["queries"] for indices in q["groups"].values() for i in indices
    }
    provenance = s.read_json(run / "training-noise-provenance.json")
    expected = []
    for row in s.frozen.rows(source / "train-segments.jsonl"):
        i = row["cache_index"]
        if i not in needed:
            continue
        digest = s.primitives.rank(
            config["augmentation"]["training_noise_seed"], row["segment_id"]
        )
        if (
            int(digest[:8], 16) / 2**32
            < config["augmentation"]["noisy_query_interval_probability"]
        ):
            if row["source_file"] in enrollment_sources:
                raise ValueError("noise applied to registration")
            expected.append(i)
    if sorted(expected) != sorted(provenance["corrupted_cache_indices"]):
        raise ValueError("training noise selection differs from frozen policy")
    if provenance["clean_replay_max_embedding_error"] > 2e-6:
        raise ValueError("clean encoder waveform replay differs")
    lookup = {p: i for i, p in enumerate(data["speakers"])}
    owners = s.np.array([lookup[q["speaker_id"]] for q in data["queries"]])
    corpora = s.np.array(
        [next(iter(data["catalog"][p].values()))["corpus"] for p in data["speakers"]]
    )
    summaries = {}
    for seed in protocol["seeds"]:
        with s.np.load(previous / f"schedule-{seed}.npz", allow_pickle=False) as pairs:
            positive, negative = owners[pairs["queries"]], pairs["negative_claims"]
            if s.np.any(positive == negative) or s.np.any(
                corpora[positive] != corpora[negative]
            ):
                raise ValueError("invalid or cross-corpus negative training pair")
        regenerated = make_schedule(
            protocol, config["augmentation"], seed, len(design["phones"])
        )
        with s.np.load(
            run / f"augmentation-{seed}.npz", allow_pickle=False
        ) as schedule:
            for key, values in regenerated.items():
                if not s.np.array_equal(values, schedule[key]):
                    raise ValueError("augmentation schedule not reproducible")
            counts = s.np.bincount(schedule["modes"].ravel(), minlength=4).tolist()
            if (
                not schedule["keep"][..., :5].all()
                or not schedule["keep"][schedule["modes"] < 2].all()
            ):
                raise ValueError("clean inputs or vowels masked")
        for kind in protocol["architectures"]:
            name = s.fusion.variant(kind, seed)
            summary = s.read_json(run / name / "training-summary.json")
            if summary["completed_updates"] != protocol["updates"]:
                raise ValueError("training incomplete")
            best = min(summary["history"], key=lambda h: (h["normal_eer"], h["update"]))
            if best["update"] != summary["selected_update"]:
                raise ValueError("checkpoint selection changed")
            if [h["update"] for h in summary["history"]] != list(
                range(0, protocol["updates"] + 1, protocol["validation_every"])
            ):
                raise ValueError("validation schedule changed")
            s.frozen.expanded.seed_everything(seed)
            initial = s.fusion.create_model(protocol, kind)
            if (
                s.frozen.expanded.model_sha256(initial)
                != summary["initial_model_state_sha256"]
            ):
                raise ValueError("initialization changed")
            if summary["pairs_schedule_sha256"] != s.sha256_file(
                previous / f"schedule-{seed}.npz"
            ) or summary["augmentation_schedule_sha256"] != s.sha256_file(
                run / f"augmentation-{seed}.npz"
            ):
                raise ValueError("schedules changed during training")
            if (
                summary["presented_queries_by_mode"] != counts
                or summary["encoder_updated"]
            ):
                raise ValueError("augmentation count or encoder policy changed")
            checkpoint = s.torch.load(
                run / name / "fusion.pt", map_location="cpu", weights_only=True
            )
            if (
                checkpoint["update"] != summary["selected_update"]
                or name not in selection["models"]
            ):
                raise ValueError("selected checkpoint inconsistent")
            summaries[name] = summary
    return summaries


def audit(config, run):
    design = s.verify(config, run, full=True)
    selection = s.selected_models(config, run)
    training = check_training(config, run, design, selection)
    checks, scenarios = {}, {}
    condition = s.original_case(config, "clean")
    baseline = list(s.frozen.rows(condition / "validation-scores.jsonl"))
    report = s.read_json(condition / "validation-metrics.json")
    checks["equal/validation"] = s.frozen.audit_trial(
        condition, condition, "validation", report
    )
    validation = {"equal": report["metrics"]}
    for name in selection["models"]:
        values = list(s.frozen.rows(run / name / "validation-scores.jsonl"))
        report = s.read_json(run / name / "validation-metrics.json")
        checks[f"augmented-{name}/validation"] = s.checker.check_values(
            values, baseline, report
        )
        validation[f"augmented-{name}"] = report["metrics"]
        if (
            abs(
                report["metrics"][s.ROLES[0]]["eer"]
                - min(h["normal_eer"] for h in training[name]["history"])
            )
            > 1e-12
        ):
            raise ValueError("saved validation score does not match selection")
    for case in config["test_scenarios"]:
        original = s.original_case(config, case)
        baseline = list(s.frozen.rows(original / "test-scores.jsonl"))
        report = s.read_json(original / "test-metrics.json")
        checks[f"equal/{case}"] = s.frozen.audit_trial(
            original, original, "test", report
        )
        variants = {"equal": report["metrics"]}
        for name in selection["models"]:
            old = s.read_json(s.original_model_file(config, case, name, "metrics.json"))
            variants[f"previous-{name}"] = old["metrics"]
            path = run / name / case
            values = list(s.frozen.rows(path / "scores.jsonl"))
            report = s.read_json(path / "metrics.json")
            checks[f"augmented-{name}/{case}"] = s.checker.check_values(
                values, baseline, report
            )
            variants[f"augmented-{name}"] = report["metrics"]
        scenarios[case] = variants
        print(f"Independent score reconstruction passed: {case}", flush=True)
    speakers = s.read_json(s.source(config) / "test-inputs.json")["speakers"]
    lookup = {p: i for i, p in enumerate(speakers)}
    indices, counts = s.frozen.speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    estimates, bootstrap_checks, differences = {}, 0, {}
    names = [
        "equal",
        *[
            f"{group}-transformer-{seed}"
            for group in ("previous", "augmented")
            for seed in design["training_protocol"]["seeds"]
        ],
    ]
    for case in config["test_scenarios"]:
        for name in names:
            if name == "equal":
                path = s.original_case(config, case) / "test-scores.jsonl"
            elif name.startswith("previous-"):
                path = s.original_model_file(
                    config, case, name.removeprefix("previous-"), "scores.jsonl"
                )
            else:
                path = run / name.removeprefix("augmented-") / case / "scores.jsonl"
            values = list(s.frozen.rows(path))
            for role in s.ROLES:
                scored = [
                    r for r in values if r["role"] == role and r["status"] == "scored"
                ]
                scores = s.np.array([r["score"] for r in scored])
                q = s.np.array([lookup[r["speaker_id"]] for r in scored])
                c = s.np.array([lookup[r["claimed_speaker_id"]] for r in scored])
                draws = s.frozen.weighted_eers(scores, q, c, counts)
                for sample in range(0, len(counts), max(1, len(counts) // 40)):
                    weights = s.np.where(
                        q == c, counts[sample, q], counts[sample, q] * counts[sample, c]
                    )
                    if (
                        abs(
                            s.frozen.independent_eer(scores, q == c, weights)
                            - draws[sample]
                        )
                        > 1e-12
                    ):
                        raise ValueError("independent speaker bootstrap differs")
                    bootstrap_checks += 1
                estimates[f"{case}/{name}/{role}"] = draws
        for role in s.ROLES:
            means, draw_means = {}, {}
            for group in ("previous", "augmented"):
                variants = [
                    f"{group}-transformer-{seed}"
                    for seed in design["training_protocol"]["seeds"]
                ]
                means[group] = float(
                    s.np.mean([scenarios[case][n][role]["eer"] for n in variants])
                )
                draw_means[group] = s.np.mean(
                    [estimates[f"{case}/{n}/{role}"] for n in variants], axis=0
                )
            for control in ("previous", "equal"):
                value = (
                    means["previous"]
                    if control == "previous"
                    else scenarios[case]["equal"][role]["eer"]
                )
                draws = (
                    draw_means["previous"]
                    if control == "previous"
                    else estimates[f"{case}/equal/{role}"]
                )
                differences[
                    f"{case}/augmented-transformer-mean-minus-{control}/{role}"
                ] = {
                    "difference": means["augmented"] - value,
                    "ci95": s.np.percentile(
                        draw_means["augmented"] - draws, [2.5, 97.5]
                    ).tolist(),
                }
        print(f"Paired 2,000-speaker-draw bootstrap passed: {case}", flush=True)
    s.np.savez_compressed(
        run / "bootstrap.npz",
        speaker_indices=indices,
        speaker_counts=counts,
        **estimates,
    )
    s.write_json(
        run / "bootstrap-results.json",
        {
            "replicates": config["bootstrap_replicates"],
            "seed": config["bootstrap_seed"],
            "speakers": len(speakers),
            "differences": differences,
            "scope": "paired test-speaker uncertainty conditional on these fixed trained models; mean EER is not ensemble scoring; training-seed population uncertainty not estimated",
        },
    )
    s.write_json(
        run / "results.json",
        {
            "config": config,
            "phones": design["phones"],
            "training": training,
            "validation": validation,
            "scenarios": scenarios,
            "training_noise": {
                k: v
                for k, v in s.read_json(run / "training-noise-provenance.json").items()
                if k != "corrupted_cache_indices"
            },
        },
    )
    s.verify(config, run, full=True)
    s.selected_models(config, run)
    totals = {
        "status": "passed",
        "checks": checks,
        "score_rows": sum(c["scores"] for c in checks.values()),
        "eer_checks": sum(c["eer_checks"] for c in checks.values()),
        "rate_checks": sum(c["rate_checks"] for c in checks.values()),
        "independent_bootstrap_checks": bootstrap_checks,
        "same_encoder_initialization_pairs_and_training_budget": True,
        "same_test_identities_components_and_support": True,
        "models_and_clean_validation_thresholds_frozen_before_new_test": True,
        "original_test_previously_observed": True,
        "training_audio_disjoint_from_registration_and_evaluation": True,
        "source_hashes_verified_after_training": True,
    }
    s.write_json(run / "independent-audit.json", totals)
    outputs = {
        str(p.relative_to(s.ROOT)): s.sha256_file(p)
        for p in run.rglob("*")
        if p.is_file()
        and p.name not in ("completion-verification.json", "publication-freeze.json")
    }
    s.write_json(
        run / "completion-verification.json", {**totals, "output_sha256": outputs}
    )
    print(json.dumps({k: v for k, v in totals.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    settings = s.read_json(s.CONFIG)
    audit(settings, args.run or s.ROOT / settings["run_directory"])
