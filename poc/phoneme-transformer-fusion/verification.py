"""Independent score/rate reconstruction and shared-speaker bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import (
    CONFIG,
    ROLES,
    ROOT,
    audit_trial,
    checked,
    independent_eer,
    np,
    read_json,
    rows,
    sha256_file,
    speaker_draws,
    verify,
    weighted_eers,
    write_json,
)


def check_values(values, baseline, report):
    if len(values) != len(baseline):
        raise ValueError("coverage changed")
    max_error = 0.0
    for row, original in zip(values, baseline):
        for key in (
            "query_id",
            "speaker_id",
            "claimed_speaker_id",
            "is_genuine",
            "status",
            "used_phones",
            "phone_scores",
            "role",
            "split",
        ):
            if row[key] != original[key]:
                raise ValueError(f"trial identity or components changed: {key}")
        if row["status"] == "no_score":
            if row["score"] is not None or row.get("phone_weights"):
                raise ValueError("missing vowels no longer rejected")
            continue
        weights = row["phone_weights"]
        if (
            set(weights) != set(row["phone_scores"])
            or min(weights.values()) < 0
            or abs(sum(weights.values()) - 1) > 1e-12
        ):
            raise ValueError("invalid learned weights")
        reconstructed = sum(weights[p] * row["phone_scores"][p] for p in weights)
        max_error = max(max_error, abs(reconstructed - row["score"]))
    if max_error > 1e-12:
        raise ValueError("weighted score cannot be reconstructed")
    rate_checks = 0
    for role, metric in report["metrics"].items():
        all_values = [r for r in values if r["role"] == role]
        selected = [r for r in all_values if r["status"] == "scored"]
        scores = np.array([r["score"] for r in selected])
        genuine = np.array([r["is_genuine"] for r in selected])
        if abs(independent_eer(scores, genuine) - metric["eer"]) > 1e-12:
            raise ValueError("independent EER differs")
        all_genuine = sum(r["is_genuine"] for r in all_values)
        if metric["all_queries"] != all_genuine or metric["scored_queries"] != int(
            genuine.sum()
        ):
            raise ValueError("coverage count differs")
        for point in metric["operating_points"].values():
            threshold = float(point["threshold"])
            fa = int(((scores >= threshold) & ~genuine).sum())
            fr = int(((scores < threshold) & genuine).sum())
            rates = {
                "far": fa / int((~genuine).sum()),
                "frr": fr / int(genuine.sum()),
                "all_input_far": fa / (len(all_values) - all_genuine),
                "all_input_frr": (fr + all_genuine - int(genuine.sum())) / all_genuine,
            }
            if (
                point["false_accepts"] != fa
                or point["false_rejects"] != fr
                or any(abs(point[k] - v) > 1e-12 for k, v in rates.items())
            ):
                raise ValueError("independent FAR/FRR differs")
            rate_checks += len(rates)
    return {
        "scores": len(values),
        "max_score_error": max_error,
        "eer_checks": 2,
        "rate_checks": rate_checks,
    }


def audit(config, run):
    frozen = verify(run, full=True)
    selection = read_json(run / "selection-freeze.json")
    checked(run / "design-freeze.json", selection["design_sha256"])
    condition = ROOT / config["evaluation_run"] / "enrollment-30"
    checks, summaries, output_hashes = {}, {}, {}
    for split in ("validation", "test"):
        baseline = list(rows(condition / f"{split}-scores.jsonl"))
        report = read_json(condition / f"{split}-metrics.json")
        checks[f"equal/{split}"] = audit_trial(condition, condition, split, report)
        summaries.setdefault("equal", {})[split] = report["metrics"]
        for name, files in selection["models"].items():
            for file, checksum in files.items():
                checked(run / name / file, checksum)
            values = list(rows(run / name / f"{split}-scores.jsonl"))
            report = read_json(run / name / f"{split}-metrics.json")
            checks[f"{name}/{split}"] = check_values(values, baseline, report)
            summaries.setdefault(name, {})[split] = report["metrics"]
    speakers = read_json(run / "test-inputs.json")["speakers"]
    lookup = {s: i for i, s in enumerate(speakers)}
    indices, counts = speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    estimates, bootstrap_checks = {}, 0
    for name in summaries:
        path = (
            condition / "test-scores.jsonl"
            if name == "equal"
            else run / name / "test-scores.jsonl"
        )
        values = list(rows(path))
        for role in ROLES:
            selected = [
                r for r in values if r["role"] == role and r["status"] == "scored"
            ]
            scores = np.array([r["score"] for r in selected])
            q = np.array([lookup[r["speaker_id"]] for r in selected])
            c = np.array([lookup[r["claimed_speaker_id"]] for r in selected])
            result = weighted_eers(scores, q, c, counts)
            for sample in range(0, len(counts), max(1, len(counts) // 40)):
                weights = np.where(
                    q == c, counts[sample, q], counts[sample, q] * counts[sample, c]
                )
                if (
                    abs(independent_eer(scores, q == c, weights) - result[sample])
                    > 1e-12
                ):
                    raise ValueError("independent paired bootstrap differs")
                bootstrap_checks += 1
            estimates[f"{name}/{role}"] = result
    differences = {}
    for kind in config["architectures"]:
        names = [f"{kind}-{s}" for s in config["seeds"]]
        for role in ROLES:
            mean_eer = np.mean([summaries[n]["test"][role]["eer"] for n in names])
            draw_mean = np.mean([estimates[f"{n}/{role}"] for n in names], axis=0)
            delta = draw_mean - estimates[f"equal/{role}"]
            differences[f"{kind}-mean-minus-equal/{role}"] = {
                "difference": float(mean_eer - summaries["equal"]["test"][role]["eer"]),
                "ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
            }
    for role in ROLES:
        delta = np.mean(
            [
                estimates[f"transformer-{s}/{role}"] - estimates[f"mlp-{s}/{role}"]
                for s in config["seeds"]
            ],
            axis=0,
        )
        difference = np.mean(
            [
                summaries[f"transformer-{s}"]["test"][role]["eer"]
                - summaries[f"mlp-{s}"]["test"][role]["eer"]
                for s in config["seeds"]
            ]
        )
        differences[f"transformer-mean-minus-mlp-mean/{role}"] = {
            "difference": float(difference),
            "ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
        }
    np.savez_compressed(
        run / "bootstrap.npz",
        speaker_indices=indices,
        speaker_counts=counts,
        **estimates,
    )
    write_json(
        run / "bootstrap-results.json",
        {
            "replicates": config["bootstrap_replicates"],
            "differences": differences,
            "scope": "paired test-speaker uncertainty conditional on these six fixed trained models; means are mean EER, not ensemble scores; no population claim about training seeds",
        },
    )
    write_json(
        run / "results.json",
        {"config": config, "phones": frozen["phones"], "variants": summaries},
    )
    for directory in (run, *(run / n for n in selection["models"])):
        for path in directory.glob("*"):
            if path.is_file() and path.name not in (
                "completion-verification.json",
                "independent-audit.json",
            ):
                output_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    total = {
        "status": "passed",
        "checks": checks,
        "score_rows": sum(c["scores"] for c in checks.values()),
        "eer_checks": sum(c["eer_checks"] for c in checks.values()),
        "rate_checks": sum(c["rate_checks"] for c in checks.values()),
        "independent_bootstrap_checks": bootstrap_checks,
        "equal_enrollment30_exactly_reproduced": True,
        "full_source_hashes_verified_after_training": True,
        "all_models_thresholds_frozen_before_new_test": True,
        "output_sha256": output_hashes,
    }
    write_json(run / "independent-audit.json", total)
    write_json(run / "completion-verification.json", total)
    print(
        {k: v for k, v in total.items() if k not in ("checks", "output_sha256")},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1",
    )
    args = parser.parse_args()
    audit(read_json(CONFIG), args.run)
