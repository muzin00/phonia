"""Independently reconstruct new scores and measure paired speaker uncertainty."""

from __future__ import annotations

import argparse
from pathlib import Path

import shared as s


def check_metrics(values, report, points):
    checks = 0
    for role in s.ROLES:
        chosen = [r for r in values if r["role"] == role]
        scored = [r for r in chosen if r["status"] == "scored"]
        scores = s.np.array([r["score"] for r in scored])
        genuine = s.np.array([r["is_genuine"] for r in scored])
        total_g = sum(r["is_genuine"] for r in chosen)
        total_i = len(chosen) - total_g
        metric = report[role]
        eer = s.original.independent_eer(scores, genuine)
        if abs(eer - metric["eer"]) > 1e-12 or metric["scored_queries"] != int(
            genuine.sum()
        ):
            raise ValueError("independent EER or coverage differs")
        if (
            metric["all_queries"] != total_g
            or abs(metric["coverage"] - genuine.sum() / total_g) > 1e-12
        ):
            raise ValueError("full-input denominator differs")
        for name, threshold in points.items():
            accepted = scores >= s.numeric(threshold)
            fa = int((accepted & ~genuine).sum())
            fr = int((~accepted & genuine).sum())
            expected = {
                "false_accepts": fa,
                "false_rejects": fr,
                "far": fa / (~genuine).sum(),
                "frr": fr / genuine.sum(),
                "all_input_far": fa / total_i,
                "all_input_frr": (fr + total_g - genuine.sum()) / total_g,
            }
            if any(
                abs(metric["operating_points"][name][key] - value) > 1e-12
                for key, value in expected.items()
            ):
                raise ValueError("independent full-input FAR/FRR differs")
            checks += len(expected)
    return checks


def component_reference(config, split, phones):
    data = s.read_json(s.source(config) / f"{split}-inputs.json")
    vectors = s.np.load(
        s.control(config) / f"{split}-embeddings.npy", allow_pickle=False
    ).astype(s.np.float64)
    use = sorted(set(phones) & set(data["universally_registered"]))
    profiles = {}
    for p in use:
        means = s.np.stack(
            [
                s.np.add.reduce(vectors[data["profiles"][speaker][p]], axis=0)
                / len(data["profiles"][speaker][p])
                for speaker in data["speakers"]
            ]
        )
        profiles[p] = means / s.np.sqrt(s.np.einsum("ij,ij->i", means, means))[:, None]
    references = []
    for query in data["queries"]:
        components = {}
        for p in use:
            indices = query["groups"][p]
            if indices:
                mean = s.np.add.reduce(vectors[indices], axis=0) / len(indices)
                components[p] = profiles[p] @ mean
        for c, speaker in enumerate(data["speakers"]):
            references.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": speaker,
                    "role": query["role"],
                    "is_genuine": speaker == query["speaker_id"],
                    "split": split,
                    "components": {p: float(v[c]) for p, v in components.items()},
                }
            )
    return references


def paired_draws(values, points, counts, lookup):
    """Rates use all trials, EER uses only scored trials, on shared speaker draws."""
    q = s.np.array([lookup[r["speaker_id"]] for r in values])
    c = s.np.array([lookup[r["claimed_speaker_id"]] for r in values])
    genuine = q == c
    scored = s.np.array([r["status"] == "scored" for r in values])
    scores = s.np.array(
        [r["score"] if r["score"] is not None else -s.np.inf for r in values]
    )
    n = len(lookup)
    weights = (counts[:, :, None] * counts[:, None, :]).reshape(len(counts), -1)
    weights[:, s.np.arange(n) * (n + 1)] = counts

    def totals(mask):
        bins = s.np.bincount(q[mask] * n + c[mask], minlength=n * n)
        return weights @ bins

    total_g, total_i = totals(genuine), totals(~genuine)
    result = {
        "coverage": totals(genuine & scored) / total_g,
        "eer": s.original.weighted_eers(scores[scored], q[scored], c[scored], counts),
    }
    for name, threshold in points.items():
        accepted = scored & (scores >= s.numeric(threshold))
        result[f"{name}/far"] = totals(~genuine & accepted) / total_i
        result[f"{name}/frr"] = totals(genuine & ~accepted) / total_g
    # Direct weighted counting/ROC independently checks a subset of draws.
    checks = 0
    for i in range(0, len(counts), max(1, len(counts) // 20)):
        w = s.np.where(genuine, counts[i, q], counts[i, q] * counts[i, c])
        expected = {
            "coverage": float(w[genuine & scored].sum() / w[genuine].sum()),
            "eer": s.original.independent_eer(
                scores[scored], genuine[scored], w[scored]
            ),
        }
        for name, threshold in points.items():
            accepted = scored & (scores >= s.numeric(threshold))
            expected[f"{name}/far"] = float(
                w[~genuine & accepted].sum() / w[~genuine].sum()
            )
            expected[f"{name}/frr"] = float(
                w[genuine & ~accepted].sum() / w[genuine].sum()
            )
        for key, value in expected.items():
            if abs(result[key][i] - value) > 1e-12:
                raise ValueError("independent bootstrap differs")
            checks += 1
    return result, checks


def estimate(report, metric):
    if metric in ("eer", "coverage"):
        return report[metric]
    point, kind = metric.split("/")
    return report["operating_points"][point][f"all_input_{kind}"]


def audit(config, run):
    design = s.verify(config, run)
    s.selected(config, run)
    results = {
        split: s.read_json(run / split / "results.json")
        for split in ("validation", "test")
    }
    total_rows = metric_checks = replay_checks = 0
    maximum_error = 0.0
    for split in ("validation", "test"):
        references = component_reference(config, split, design["phones"])
        for model in config["models"]:
            candidates = list(s.rows(run / split / f"{model}-candidates.jsonl"))
            baseline = list(
                s.rows(s.original_path(config, model, split, "scores.jsonl"))
            )
            if len(candidates) != len(references) or len(candidates) != len(baseline):
                raise ValueError("trial count changed")
            for r, ref, old in zip(candidates, references, baseline):
                if any(
                    r[key] != value for key, value in ref.items() if key != "components"
                ):
                    raise ValueError("trial identity or label differs")
                components = ref["components"]
                if set(r["phone_scores"]) != set(components) or r[
                    "used_phones"
                ] != sorted(components):
                    raise ValueError("phone support differs from input provenance")
                weights = r.get(
                    "phone_weights", {p: 1 / len(components) for p in components}
                )
                if components:
                    if (
                        set(weights) != set(components)
                        or min(weights.values()) < 0
                        or abs(sum(weights.values()) - 1) > 1e-12
                    ):
                        raise ValueError("invalid fusion weights")
                    error = max(
                        abs(r["phone_scores"][p] - value)
                        for p, value in components.items()
                    )
                    score_error = abs(
                        r["score"]
                        - sum(weights[p] * value for p, value in components.items())
                    )
                    maximum_error = max(maximum_error, error, score_error)
                if old["status"] == "scored" and r != old:
                    raise ValueError("existing scored row not exactly preserved")
            if maximum_error > 2e-6:
                raise ValueError("float64 embedding reconstruction differs")
            # Recompute model weights across all trials, with different batch grouping.
            if model != "equal":
                bundle = s.torch.load(
                    s.source(config) / model / "fusion.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                network = s.fusion.create_model(
                    design["training_protocol"], bundle["kind"]
                )
                network.load_state_dict(bundle["state_dict"])
                arrays = s.original.load_arrays(s.source(config), split)
                replay = s.fusion.score(network, arrays, candidates, design["phones"])
                for a, b in zip(replay, candidates):
                    if a["status"] == "scored":
                        if abs(a["score"] - b["score"]) > 2e-6 or any(
                            abs(a["phone_weights"][p] - b["phone_weights"][p]) > 2e-6
                            for p in a["used_phones"]
                        ):
                            raise ValueError("frozen model replay differs")
                        replay_checks += 1
            for policy in config["policies"]:
                name = policy["name"]
                values = s.apply_policy(candidates, policy)
                report = results[split][model][name]
                for r in values:
                    n_vowels = sum(p in s.VOWELS for p in r["used_phones"])
                    eligible = (
                        n_vowels >= policy["minimum_vowels"]
                        and len(r["used_phones"]) >= policy["minimum_phones"]
                    )
                    if (r["status"] == "scored") != eligible:
                        raise ValueError("policy support differs")
                for mode, points in (
                    ("recalibrated", report["thresholds"]),
                    ("original", report["original_thresholds"]),
                ):
                    metric_checks += check_metrics(
                        values, report["metrics"][mode], points
                    )
                    for role in s.ROLES:
                        for point in points:
                            diagnostic = report["diagnostics"][role][mode][point]
                            patterns = diagnostic["patterns"].values()
                            metric = report["metrics"][mode][role]
                            operating = metric["operating_points"][point]
                            if (
                                sum(p["false_accepts"] for p in patterns)
                                != operating["false_accepts"]
                                or sum(p["all_false_rejects"] for p in patterns)
                                != operating["false_rejects"]
                                + metric["all_queries"]
                                - metric["scored_queries"]
                            ):
                                raise ValueError("pattern count totals differ")
                            if mode == "original" and any(
                                diagnostic["changes"]["previously_scored"].values()
                            ):
                                raise ValueError(
                                    "original-threshold decisions on existing inputs changed"
                                )
                total_rows += len(values)
            print(f"Independent scores and metrics passed: {split}/{model}", flush=True)
    speakers = s.read_json(s.source(config) / "test-inputs.json")["speakers"]
    lookup = {p: i for i, p in enumerate(speakers)}
    indices, counts = s.original.speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    draws, differences, bootstrap_checks = {}, {}, 0
    for model in config["models"]:
        candidates = list(s.rows(run / "test" / f"{model}-candidates.jsonl"))
        for policy in config["policies"]:
            name = policy["name"]
            values = s.apply_policy(candidates, policy)
            report = results["test"][model][name]
            for role in s.ROLES:
                chosen = [r for r in values if r["role"] == role]
                for mode, points in (
                    ("recalibrated", report["thresholds"]),
                    ("original", report["original_thresholds"]),
                ):
                    estimates, checks = paired_draws(chosen, points, counts, lookup)
                    bootstrap_checks += checks
                    for metric, array in estimates.items():
                        key = f"{model}/{name}/{role}/{mode}/{metric}"
                        draws[key] = array
                        if name != "vowels5":
                            control_key = f"{model}/vowels5/{role}/{mode}/{metric}"
                            value = estimate(report["metrics"][mode][role], metric)
                            before = estimate(
                                results["test"][model]["vowels5"]["metrics"][mode][
                                    role
                                ],
                                metric,
                            )
                            differences[key] = {
                                "difference": value - before,
                                "ci95": s.np.percentile(
                                    array - draws[control_key], [2.5, 97.5]
                                ).tolist(),
                            }
        print(f"Paired bootstrap passed: {model}", flush=True)
    # Descriptive three-seed means, using the same speaker draw in every seed.
    learned = [m for m in config["models"] if m != "equal"]
    for key in list(differences):
        if not key.startswith(learned[0] + "/"):
            continue
        tail = key.split("/", 1)[1]
        policy, role, mode, *metric_parts = tail.split("/")
        metric = "/".join(metric_parts)
        delta = s.np.mean(
            [
                draws[f"{m}/{tail}"] - draws[f"{m}/vowels5/{role}/{mode}/{metric}"]
                for m in learned
            ],
            axis=0,
        )
        differences[f"transformer-mean/{tail}"] = {
            "difference": float(
                s.np.mean([differences[f"{m}/{tail}"]["difference"] for m in learned])
            ),
            "ci95": s.np.percentile(delta, [2.5, 97.5]).tolist(),
        }
    s.np.savez_compressed(
        run / "bootstrap.npz", speaker_indices=indices, speaker_counts=counts, **draws
    )
    s.write_json(
        run / "bootstrap-results.json",
        {
            "replicates": len(counts),
            "differences": differences,
            "scope": "paired test-speaker uncertainty for fixed models and thresholds; excludes training, calibration and text resampling; no multiple-comparison adjustment",
        },
    )
    s.verify(config, run)
    s.selected(config, run)
    audit_result = {
        "status": "passed",
        "score_policy_rows": total_rows,
        "metric_checks": metric_checks,
        "model_replay_score_checks": replay_checks,
        "max_float64_score_error": maximum_error,
        "independent_bootstrap_checks": bootstrap_checks,
        "existing_scores_and_fixed_threshold_decisions_unchanged": True,
        "all_source_hashes_unchanged": True,
    }
    s.write_json(run / "independent-audit.json", audit_result)
    files = {
        str(p.relative_to(s.ROOT)): s.sha256_file(p)
        for p in run.rglob("*")
        if p.is_file()
        and p.name != "completion-verification.json"
        and not p.name.endswith(".log")
    }
    s.write_json(
        run / "completion-verification.json", {**audit_result, "output_sha256": files}
    )
    print(audit_result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    settings = s.read_json(s.CONFIG)
    audit(settings, args.run or s.ROOT / settings["run_directory"])
