"""Frozen evaluation of naturally missing vowels, using unchanged enrollment."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-enrollment-scaling"))
import numpy as np
import scaling as previous
import torch
from final_metrics import bootstrap_cell, interval
from partial_scoring import fuse_available, prepare_available, verify_available
from pilot import EmbeddingCache, digest, vowel_scores
from registration import EnrollmentSegment, FrozenEncoder, load_profile
from validation import iter_rows, writer
from validation_metrics import (
    POINTS,
    calibrate_cell,
    evaluate_cell,
    numeric_threshold,
    rates,
)
from weighted_bootstrap import speaker_draws

ROOT = previous.ROOT
VOWELS = previous.VOWELS
METHODS = {"strict5": 5, "available4": 4, "available3": 3}
ROLES = ("verification", "cross_text_verification")
METRICS = ("far", "frr", "all_input_far", "all_input_frr")
checked, sha256_file, write_json = (
    previous.checked,
    previous.sha256_file,
    previous.write_json,
)


def pin(files, path):
    files[str(path.relative_to(ROOT))] = sha256_file(path)


def prepare(run):
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused output directory inside artifacts required")
    config_path = BASE / "config/protocol.json"
    config = json.loads(config_path.read_text())
    if (
        config["methods"] != METHODS
        or config["enrollment_count_per_vowel"] != 10
        or config["operating_points"] != [p for p, _ in POINTS]
        or config["query_window"] != "full"
        or config["retrain_encoder"]
        or config["change_registration"]
        or config["test_threshold_recalibration"]
        or config["missing_mask_specific_thresholds"]
        or config["synthetic_deletion"]
        or config["bootstrap_replicates"] != 10000
        or config["bootstrap_seed"] != 20260929
        or config["repeats_per_unique_embedding"] != 3
        or config["calibration"]
        != "one_pooled_validation_verification_threshold_per_method_and_operating_point"
        or config["fusion"]
        != "mean_segment_cosine_within_vowel_then_equal_mean_over_present_vowels"
        or config["no_score"] != "reject_and_keep_in_all_input_denominators"
    ):
        raise ValueError("unsupported missing-vowel protocol")
    source = ROOT / config["prior_run"]
    checked(source / "study-report.json", config["prior_report_sha256"])
    report = json.loads((source / "study-report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("prior study incomplete")
    freeze = json.loads((source / "design-freeze.json").read_text())
    files = dict(freeze["files"])
    for name, checksum in report["outputs_sha256"].items():
        checked(source / name, checksum)
        files[str((source / name).relative_to(ROOT))] = checksum
    pin(files, source / "study-report.json")
    pin(files, config_path)
    prepared = {}
    for split in ("validation", "test"):
        old = previous.frozen_inputs(source, split)
        inputs = {
            k: old[k] for k in ("split", "speakers", "queries", "sources", "model")
        }
        inputs["enrollment"] = [
            e for e in old["enrollment"] if e["enrollment_count"] == 10
        ]
        inputs["phase6_run"] = old["config"]["phase6_run"]
        inputs["config"] = config
        inputs["trials"] = [
            {
                **trial,
                "method_id": method,
                "score_id": digest(
                    [config["protocol_version"], method, trial["trial_id"]]
                ),
            }
            for trial in previous.build_trials(
                inputs["queries"], inputs["speakers"], [10], config["protocol_version"]
            )
            for method in METHODS
        ]
        if (
            len(inputs["enrollment"]) != 15
            or len(inputs["queries"]) != 1200
            or len(inputs["trials"]) != 54000
        ):
            raise ValueError("unexpected population")
        prepared[split] = inputs
    if set(prepared["validation"]["speakers"]) & set(prepared["test"]["speakers"]):
        raise ValueError("validation/test speaker leakage")
    for path in BASE.rglob("*.py"):
        pin(files, path)
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    run.mkdir(parents=True)
    for split, inputs in prepared.items():
        write_json(run / f"{split}-inputs.json", inputs)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "design_frozen_before_new_inference",
            "config": config,
            "files": files,
            "runtime": previous.runtime(),
        },
    )


def frozen_inputs(run, split):
    if split not in ("validation", "test"):
        raise ValueError("validation/test only")
    freeze = json.loads((run / "design-freeze.json").read_text())
    if (
        freeze["status"] != "design_frozen_before_new_inference"
        or freeze["runtime"] != previous.runtime()
    ):
        raise ValueError("frozen design or runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    inputs = json.loads((run / f"{split}-inputs.json").read_text())
    if inputs["split"] != split:
        raise ValueError("input split mismatch")
    if split == "test":
        evaluation = json.loads((run / "evaluation-freeze.json").read_text())
        if evaluation["status"] != "thresholds_frozen_before_new_test_inference":
            raise ValueError("fixed validation thresholds required")
        for name, checksum in evaluation["files"].items():
            checked(run / name, checksum)
    return inputs


def infer(run, split):
    inputs = frozen_inputs(run, split)
    output = run / split
    output.mkdir(exist_ok=False)
    started = perf_counter()
    encoder = FrozenEncoder.from_bundle(ROOT / inputs["model"]["bundle"])
    snapshot = previous.model_snapshot(encoder)
    feature_tensors = [encoder.pipeline.mean.clone(), encoder.pipeline.std.clone()]
    phase6 = ROOT / inputs["phase6_run"]
    profiles = {
        speaker: load_profile(
            phase6 / f"profiles/{split}/10/{speaker}.json", encoder=encoder
        )
        for speaker in inputs["speakers"]
    }
    checksums = {s: p.checksum for s, p in profiles.items()}
    references = {s: {v: p.vector(v) for v in VOWELS} for s, p in profiles.items()}
    audio = (previous.ValidationAudio if split == "validation" else previous.TestAudio)(
        inputs["sources"], 16
    )
    cache = EmbeddingCache(
        audio,
        encoder.identity,
        lambda pcm, sid: encoder.embed(torch.from_numpy(pcm.copy()), sid),
        3,
        already_unit=True,
    )
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    baseline = {
        (r["query_id"], r["claimed_speaker_id"]): r
        for r in iter_rows(phase6 / f"scores/{split}.jsonl")
        if r["enrollment_count"] == 10
    }
    tested, parity, complete_parity, slots = set(), 0, 0, 0
    with (
        writer(output / "scores.jsonl.gz") as write,
        writer(output / "query-validation.jsonl.gz") as write_query,
    ):
        for index, item in enumerate(inputs["queries"], 1):
            query = previous.VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**r) for r in item["segments"]),
            )
            prepared = prepare_available(
                query, next(iter(profiles.values())), encoder, audio_root=ROOT
            )
            if prepared.counts != item["counts"] or any(
                r["status"] != "used" for r in prepared.records
            ):
                raise ValueError("query eligibility changed from frozen manifest")
            present = [v for v in VOWELS if prepared.counts[v] > 0]
            grouped = {v: [] for v in VOWELS}
            if len(present) >= 3:
                for segment, pcm, reason in prepared.slices:
                    if reason is not None:
                        continue
                    first, last = previous.effective_window(
                        segment.start_frame,
                        segment.end_frame,
                        inputs["sources"][segment.source_file]["frame_count"],
                    )
                    if not np.array_equal(
                        pcm.numpy(),
                        audio.slice(
                            segment.source_file, segment.start_frame, segment.end_frame
                        ),
                    ):
                        raise ValueError("validated query PCM differs")
                    grouped[segment.vowel].append(
                        cache.embed(
                            segment.source_file, first, last, segment.segment_id
                        )
                    )
            prepared.require_unchanged_sources()
            write_query(
                {
                    "query_id": query.query_id,
                    "counts": prepared.counts,
                    "present_vowels": present,
                    "records": prepared.records,
                }
            )
            for trial in trials[query.query_id]:
                method, claimed = trial["method_id"], trial["claimed_speaker_id"]
                values = fuse_available(references[claimed], grouped, METHODS[method])
                original = baseline[query.query_id, claimed]["scores"]
                if method == "strict5":
                    if values != original:
                        raise ValueError("strict5 baseline score mismatch")
                    parity += 1
                if len(present) == 5:
                    if values != original or values != vowel_scores(
                        references[claimed], grouped
                    ):
                        raise ValueError("complete five-vowel score changed")
                    complete_parity += 1
                sample = (
                    item["speaker_id"],
                    item["role"],
                    tuple(present),
                    method,
                    trial["is_genuine"],
                )
                if sample not in tested:
                    fresh = verify_available(
                        profiles[claimed],
                        query,
                        encoder,
                        audio_root=ROOT,
                        minimum_vowels=METHODS[method],
                    )
                    if values != fresh:
                        raise ValueError("cache-free partial verification mismatch")
                    tested.add(sample)
                write(
                    {
                        **trial,
                        "status": "scored" if values is not None else "no_score",
                        "scores": values,
                        "reason": None if values is not None else "insufficient_vowels",
                        "present_vowels": present,
                        "missing_vowels": [v for v in VOWELS if v not in present],
                        "profile_sha256": checksums[claimed],
                    }
                )
                slots += 1
            if index % 100 == 0:
                print(
                    f"{split}: {index}/1200 queries; {len(cache.vectors)} embeddings",
                    flush=True,
                )
    previous.require_unchanged(encoder, snapshot)
    if any(
        not torch.equal(a, b)
        for a, b in zip(
            feature_tensors, [encoder.pipeline.mean, encoder.pipeline.std], strict=True
        )
    ):
        raise ValueError("feature statistics changed")
    if any(p.checksum != checksums[s] for s, p in profiles.items()):
        raise ValueError("registration profile changed")
    audio.require_unchanged()
    cache.save(output)
    write_json(
        output / "report.json",
        {
            "status": "completed",
            "score_slots": slots,
            "profiles_reused_unchanged": len(profiles),
            "strict5_baseline_score_parity": parity,
            "complete_query_score_parity": complete_parity,
            "cache_free_verification_checks": len(tested),
            "unique_embeddings": len(cache.vectors),
            "embedding_repeats": 3,
            "all_repeats_bitwise_equal": True,
            "weights_statistics_profiles_sources_unchanged": True,
            "elapsed_seconds": perf_counter() - started,
            "outputs_sha256": {
                str(p.relative_to(output)): sha256_file(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        },
    )
    frozen_inputs(run, split)


def score_groups(run, split):
    inputs = json.loads((run / f"{split}-inputs.json").read_text())
    report = json.loads((run / split / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("completed inference required")
    for name, checksum in report["outputs_sha256"].items():
        checked(run / split / name, checksum)
    expected = {r["score_id"]: r for r in inputs["trials"]}
    queries = {q["query_id"]: q for q in inputs["queries"]}
    groups, seen = defaultdict(list), set()
    for row in iter_rows(run / split / "scores.jsonl.gz"):
        trial = expected.get(row["score_id"])
        if (
            trial is None
            or row["score_id"] in seen
            or any(row[k] != v for k, v in trial.items())
        ):
            raise ValueError("unexpected trial identity or label")
        seen.add(row["score_id"])
        present = [v for v in VOWELS if queries[row["query_id"]]["counts"][v] > 0]
        scored = len(present) >= METHODS[row["method_id"]]
        if (
            row["present_vowels"] != present
            or row["missing_vowels"] != [v for v in VOWELS if v not in present]
            or row["status"] != ("scored" if scored else "no_score")
        ):
            raise ValueError("missing-vowel eligibility changed")
        values = row["scores"]
        if (values is not None) != scored:
            raise ValueError("invalid score state")
        if scored and (
            set(values) != {"fused", *VOWELS}
            or any(values[v] is not None for v in VOWELS if v not in present)
            or not all(
                np.isfinite(values[v]) and -1 <= values[v] <= 1
                for v in ["fused", *present]
            )
            or values["fused"]
            != float(np.mean([values[v] for v in present], dtype=np.float64))
        ):
            raise ValueError("invalid finite available-vowel fusion")
        groups[row["method_id"], row["role"]].append(
            {**row, "score": values["fused"] if scored else None}
        )
    if len(seen) != len(expected):
        raise ValueError("missing score slot")
    return inputs, groups


def calibrate(run):
    frozen_inputs(run, "validation")
    inputs, groups = score_groups(run, "validation")
    thresholds, cells, curves = {}, {}, {}
    for method in METHODS:
        thresholds[method] = calibrate_cell(
            groups[method, "verification"], inputs["speakers"]
        )
        if thresholds[method]["status"] != "calibrated":
            raise ValueError("insufficient validation support")
        for role in ROLES:
            cells[f"{method}/{role}"], curves[f"{method}/{role}"] = evaluate_cell(
                groups[method, role], thresholds[method], inputs["speakers"]
            )
    old = json.loads(
        (
            ROOT / inputs["config"]["prior_run"] / "validation-thresholds.json"
        ).read_text()
    )
    if (
        thresholds["strict5"]["operating_points"]
        != old["conditions"]["n10"]["operating_points"]
    ):
        raise ValueError("strict5 validation threshold parity failed")
    if (
        thresholds["available3"]["operating_points"]
        != thresholds["available4"]["operating_points"]
    ):
        raise ValueError("unexpected 3-vowel validation calibration data")
    write_json(
        run / "validation-thresholds.json",
        {"split": "validation", "role": "verification", "conditions": thresholds},
    )
    write_json(run / "validation-metrics.json", {"conditions": cells, "curves": curves})
    indices, counts = speaker_draws(15, 10000, 20260929)
    if not np.array_equal(
        counts,
        np.load(
            ROOT / inputs["config"]["prior_run"] / "bootstrap-counts.npy",
            allow_pickle=False,
        ),
    ):
        raise ValueError("shared bootstrap draws changed")
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    np.save(run / "bootstrap-indices.npy", indices, allow_pickle=False)
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_new_test_inference",
            "new_test_inference_performed": False,
            "test_previously_observed": True,
            "files": {
                name: sha256_file(run / name)
                for name in (
                    "validation-thresholds.json",
                    "validation-metrics.json",
                    "validation/report.json",
                    "bootstrap-counts.npy",
                    "bootstrap-indices.npy",
                )
            },
        },
    )


def paired(cells, replicas):
    result, arrays = {}, {}
    for role in ROLES:
        for method, reference in (
            ("available4", "strict5"),
            ("available3", "strict5"),
            ("available3", "available4"),
        ):
            a, b = f"{method}/{role}", f"{reference}/{role}"
            for point, _ in POINTS:
                for metric in METRICS:
                    key = f"{role}/{method}_minus_{reference}/{point}/{metric}"
                    delta = 100 * (
                        replicas[a][f"{point}/{metric}"]
                        - replicas[b][f"{point}/{metric}"]
                    )
                    arrays[key] = delta
                    result[key] = {
                        "difference_percentage_points": 100
                        * (
                            cells[a]["operating_points"][point][metric]
                            - cells[b]["operating_points"][point][metric]
                        ),
                        "ci95_percentage_points": interval(delta),
                    }
    return result, arrays


def transitions(groups, thresholds):
    results = {}
    for role in ROLES:
        reference = {
            (r["query_id"], r["claimed_speaker_id"]): r for r in groups["strict5", role]
        }
        for method in ("available4", "available3"):
            for point, _ in POINTS:
                old_tau = numeric_threshold(
                    thresholds["strict5"]["operating_points"][point]["threshold"]
                )
                new_tau = numeric_threshold(
                    thresholds[method]["operating_points"][point]["threshold"]
                )
                totals = Counter(
                    {
                        name: 0
                        for name in (
                            "missing_genuine",
                            "rescued_missing_genuine",
                            "still_rejected_missing_genuine",
                            "rescued_scored_genuine",
                            "newly_rejected_scored_genuine",
                            "added_false_accepts_missing",
                            "added_false_accepts_complete",
                            "removed_false_accepts",
                            "baseline_all_false_rejects",
                            "candidate_all_false_rejects",
                        )
                    }
                )
                for row in groups[method, role]:
                    old = reference[row["query_id"], row["claimed_speaker_id"]]
                    accepted_old = old["score"] is not None and old["score"] >= old_tau
                    accepted_new = row["score"] is not None and row["score"] >= new_tau
                    missing = old["score"] is None
                    if row["is_genuine"]:
                        totals["baseline_all_false_rejects"] += not accepted_old
                        totals["candidate_all_false_rejects"] += not accepted_new
                        if missing:
                            totals["missing_genuine"] += 1
                            totals["rescued_missing_genuine"] += accepted_new
                            totals["still_rejected_missing_genuine"] += not accepted_new
                        elif not accepted_old and accepted_new:
                            totals["rescued_scored_genuine"] += 1
                        elif accepted_old and not accepted_new:
                            totals["newly_rejected_scored_genuine"] += 1
                    elif not accepted_old and accepted_new:
                        totals[
                            "added_false_accepts_missing"
                            if missing
                            else "added_false_accepts_complete"
                        ] += 1
                    elif accepted_old and not accepted_new:
                        totals["removed_false_accepts"] += 1
                if (
                    totals["baseline_all_false_rejects"]
                    - totals["candidate_all_false_rejects"]
                    != totals["rescued_missing_genuine"]
                    + totals["rescued_scored_genuine"]
                    - totals["newly_rejected_scored_genuine"]
                ):
                    raise ValueError("genuine decision decomposition inconsistent")
                results[f"{method}/{role}/{point}"] = dict(totals)
    return results


def missing_diagnostics(groups, thresholds):
    results = {}
    for (method, role), rows in groups.items():
        subsets = defaultdict(list)
        for row in rows:
            subsets[f"vowels{len(row['present_vowels'])}"].append(row)
            mask = "".join(row["missing_vowels"]) or "none"
            subsets[f"missing_{mask}"].append(row)
        for key, subset in subsets.items():
            results[f"{method}/{role}/{key}"] = {
                "queries": len({r["query_id"] for r in subset}),
                "speakers": len({r["speaker_id"] for r in subset}),
                "scored_queries": len(
                    {r["query_id"] for r in subset if r["status"] == "scored"}
                ),
                "purpose": "small_support_diagnostic_not_used_for_calibration_or_selection",
                "operating_points": {
                    point: rates(
                        subset,
                        numeric_threshold(
                            thresholds[method]["operating_points"][point]["threshold"]
                        ),
                    )
                    for point, _ in POINTS
                },
            }
    return results


def audit_rates(groups, cells, thresholds):
    checks = 0
    for (method, role), rows in groups.items():
        for point, _ in POINTS:
            p = cells[f"{method}/{role}"]["operating_points"][point]
            threshold = thresholds[method]["operating_points"][point]["threshold"]
            tau = numeric_threshold(threshold)
            genuine = [r for r in rows if r["is_genuine"]]
            impostor = [r for r in rows if not r["is_genuine"]]
            sg = [r for r in genuine if r["score"] is not None]
            si = [r for r in impostor if r["score"] is not None]
            fa = sum(r["score"] >= tau for r in si)
            fr = sum(r["score"] < tau for r in sg)
            expected = {
                "threshold": threshold,
                "false_accepts": fa,
                "false_rejects": fr,
                "genuine": len(sg),
                "impostor": len(si),
                "all_genuine": len(genuine),
                "all_impostor": len(impostor),
                "no_score_genuine": len(genuine) - len(sg),
                "no_score_impostor": len(impostor) - len(si),
                "far": fa / len(si),
                "frr": fr / len(sg),
                "all_input_far": fa / len(impostor),
                "all_input_frr": (fr + len(genuine) - len(sg)) / len(genuine),
            }
            if any(p[k] != v for k, v in expected.items()):
                raise ValueError("raw-score rate audit failed")
            checks += 1
    return checks


def evaluate(run):
    inputs = frozen_inputs(run, "test")
    _, groups = score_groups(run, "test")
    thresholds = json.loads((run / "validation-thresholds.json").read_text())[
        "conditions"
    ]
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, curves, replicas = {}, {}, {}
    for (method, role), rows in sorted(groups.items()):
        key = f"{method}/{role}"
        cells[key], curves[key] = evaluate_cell(
            rows, thresholds[method], inputs["speakers"]
        )
        cells[key]["ci95"], replicas[key] = bootstrap_cell(
            rows, cells[key], thresholds[method], inputs["speakers"], counts
        )
        print(f"test metrics: {key}", flush=True)
    old = json.loads(
        (ROOT / inputs["config"]["prior_run"] / "test-metrics.json").read_text()
    )["conditions"]
    for role in ROLES:
        for field in (
            "queries",
            "scored_queries",
            "query_coverage",
            "pooled_eer",
            "operating_points",
            "ci95",
        ):
            if cells[f"strict5/{role}"][field] != old[f"n10/{role}"][field]:
                raise ValueError(f"strict5 baseline metric/CI parity failed: {field}")
    differences, arrays = paired(cells, replicas)
    with (run / "bootstrap-metrics.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                f"{key}/{name}": a
                for key, items in replicas.items()
                for name, a in items.items()
            },
        )
    with (run / "bootstrap-paired-differences.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    validation = json.loads((run / "validation-metrics.json").read_text())
    _, vg = score_groups(run, "validation")
    audits = audit_rates(groups, cells, thresholds) + audit_rates(
        vg, validation["conditions"], thresholds
    )
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "curves": curves,
            "paired_differences": differences,
            "decision_transitions": transitions(groups, thresholds),
            "missing_diagnostics": missing_diagnostics(groups, thresholds),
            "validation_decision_transitions": transitions(vg, thresholds),
            "validation_missing_diagnostics": missing_diagnostics(vg, thresholds),
            "threshold_recalibrated": False,
            "independent_rate_audits": audits,
            "baseline_metric_and_ci_parity_roles": 2,
        },
    )
    # The previous study's generic persisted-array audit also applies to method names.
    audits = previous.audit_intervals(run)
    frozen_inputs(run, "test")
    return audits
