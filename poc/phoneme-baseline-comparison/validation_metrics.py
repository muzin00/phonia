"""Condition-specific validation thresholds and explicit missing-score denominators."""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from smoke import ROOT, sha256_file, write_json

sys.path.insert(0, str(ROOT / "poc/phoneme-speaker-encoder"))
from phase3_train.metrics import (
    eer_operating_threshold,
    error_rates,
    far_target_threshold,
    json_threshold,
    roc_eer,
)

from pilot import METHODS, digest, load_inputs
from validation import iter_rows, writer

POINTS = (("far_1pct", 0.01), ("far_0_1pct", 0.001), ("eer_operating", None))


def condition_key(method, count, cap, support):
    return f"{support}/{method}/n{count}/{cap}"


def arrays(rows):
    scored = [r for r in rows if r["status"] == "scored"]
    return np.array([r["score"] for r in scored], dtype=np.float64), np.array(
        [r["is_genuine"] for r in scored], dtype=bool
    )


def coverage(rows, speakers):
    queries = {r["query_id"]: r for r in rows}
    counts = {
        speaker: {
            "all": sum(r["speaker_id"] == speaker for r in queries.values()),
            "scored": sum(
                r["speaker_id"] == speaker and r["status"] == "scored"
                for r in queries.values()
            ),
        }
        for speaker in speakers
    }
    return {
        "queries": len(queries),
        "scored_queries": sum(r["status"] == "scored" for r in queries.values()),
        "query_coverage": sum(r["status"] == "scored" for r in queries.values())
        / len(queries)
        if queries
        else None,
        "by_speaker": counts,
        "zero_scored_speakers": [s for s, n in counts.items() if not n["scored"]],
    }


def rates(rows, threshold, *, conditional_evaluable=True):
    values, genuine = arrays(rows)
    result = error_rates(values, genuine, threshold)
    missing_g = sum(r["is_genuine"] and r["status"] == "no_score" for r in rows)
    missing_i = sum(not r["is_genuine"] and r["status"] == "no_score" for r in rows)
    all_g, all_i = result["genuine"] + missing_g, result["impostor"] + missing_i
    return {
        **result,
        "far": result["far"] if conditional_evaluable else None,
        "frr": result["frr"] if conditional_evaluable else None,
        "threshold": json_threshold(threshold),
        "no_score_genuine": missing_g,
        "no_score_impostor": missing_i,
        "all_genuine": all_g,
        "all_impostor": all_i,
        "all_input_far": result["false_accepts"] / all_i if all_i else None,
        "all_input_frr": (result["false_rejects"] + missing_g) / all_g
        if all_g
        else None,
    }


def calibrate_cell(rows, speakers):
    if not rows or any(
        r["split"] != "validation" or r["role"] != "verification" for r in rows
    ):
        raise ValueError("threshold calibration accepts validation/verification only")
    summary = coverage(rows, speakers)
    values, genuine = arrays(rows)
    if summary["zero_scored_speakers"] or not genuine.any() or genuine.all():
        return {
            "status": "not_evaluable",
            "reason": "insufficient_scored_speaker_support",
            **summary,
            "operating_points": None,
        }
    points = {}
    for name, target in POINTS:
        threshold, _ = (
            eer_operating_threshold(values, genuine)
            if target is None
            else far_target_threshold(values, genuine, target)
        )
        points[name] = {"target_far": target, **rates(rows, threshold)}
    return {
        "status": "calibrated",
        **summary,
        "score_rows_sha256": digest(rows),
        "far_resolution": 1 / int((~genuine).sum()),
        "operating_points": points,
    }


def numeric_threshold(value):
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return -float("inf")
    return float(value)


def evaluate_cell(rows, thresholds, speakers):
    summary = coverage(rows, speakers)
    values, genuine = arrays(rows)
    evaluable = (
        not summary["zero_scored_speakers"] and genuine.any() and not genuine.all()
    )
    curve = (
        roc_eer(values, genuine)
        if evaluable
        else {"eer": None, "thresholds": [], "far": [], "frr": []}
    )
    points = thresholds["operating_points"]
    rates_at_points = (
        None
        if points is None
        else {
            name: rates(
                rows, numeric_threshold(p["threshold"]), conditional_evaluable=evaluable
            )
            for name, p in points.items()
        }
    )
    return {
        **summary,
        "conditional_status": "evaluable" if evaluable else "not_evaluable",
        "conditional_reason": None
        if evaluable
        else "zero_scored_queries_for_speaker_role_or_empty_class",
        "pooled_eer": curve["eer"],
        "operating_points": rates_at_points,
        "far_resolution": 1 / int((~genuine).sum()) if evaluable else None,
    }, curve


def calibrate_run(run):
    inputs = load_inputs(run)
    groups = defaultdict(list)
    by_query = {q["query_id"]: q for q in inputs["queries"]}
    speakers = inputs["speakers"]
    if inputs["config"]["calibration_role"] != "verification" or inputs["config"][
        "target_fars"
    ] != [0.01, 0.001]:
        raise ValueError("calibration configuration differs from protocol")
    for worker in ("vowels", "ecapa"):
        for r in iter_rows(run / worker / "scores.jsonl.gz"):
            if r["split"] != "validation":
                raise ValueError("validation calibration cannot read other splits")
            q = by_query[r["query_id"]]
            groups[
                r["method_id"], r["enrollment_count"], r["condition_id"], r["role"]
            ].append(
                {
                    "query_id": q["query_id"],
                    "speaker_id": q["speaker_id"],
                    "claimed_speaker_id": r["claimed_speaker_id"],
                    "split": "validation",
                    "role": q["role"],
                    "status": r["status"],
                    "is_genuine": r["is_genuine"],
                    "score": r["scores"]["fused"] if r["status"] == "scored" else None,
                }
            )
    common = defaultdict(set)
    for w in inputs["windows"]:
        if w["complete_five_vowels"]:
            common[w["condition_id"], w["role"]].add(w["query_id"])
    long_queries = {
        q["query_id"]
        for q in inputs["queries"]
        if inputs["sources"][q["source_file"]]["frame_count"] >= 120000
    }
    thresholds, conditions = {}, {}
    curves_path = run / "validation-curves.jsonl.gz"
    with writer(curves_path) as write_curve:
        for support in inputs["config"]["supports"]:
            for method in METHODS:
                for count in inputs["protocol"]["enrollment"]["counts_per_vowel"]:
                    for cap in (
                        c["id"]
                        for c in inputs["protocol"]["query_windows"]["conditions"]
                    ):
                        key = condition_key(method, count, cap, support)
                        calibration = groups[method, count, cap, "verification"]
                        if support == "common":
                            calibration = [
                                r
                                for r in calibration
                                if r["query_id"] in common[cap, "verification"]
                            ]
                        threshold = calibrate_cell(calibration, speakers)
                        thresholds[key] = threshold
                        for role in inputs["protocol"]["queries_and_trials"]["roles"]:
                            rows = groups[method, count, cap, role]
                            native_query_count = len({r["query_id"] for r in rows})
                            if support == "common":
                                rows = [
                                    r
                                    for r in rows
                                    if r["query_id"] in common[cap, role]
                                ]
                            item, curve = evaluate_cell(rows, threshold, speakers)
                            item["support_selection_coverage"] = (
                                item["queries"] / native_query_count
                            )
                            condition = f"{key}/{role}"
                            conditions[condition] = item
                            write_curve(
                                {
                                    "condition": condition,
                                    "purpose": "validation_diagnostic_not_test",
                                    **curve,
                                    "tpr": (1 - np.asarray(curve["frr"])).tolist(),
                                    "det_axes": "far_frr",
                                }
                            )
                            if support == "native":
                                long_rows = [
                                    r for r in rows if r["query_id"] in long_queries
                                ]
                                long_item, _ = evaluate_cell(
                                    long_rows, threshold, speakers
                                )
                                item["source_at_least_5s_diagnostic"] = long_item
    # Exact/full/native must reproduce the existing Phase 6 calibration.
    phase6 = json.loads(
        (
            ROOT
            / inputs["protocol"]["phase6_reference"]["run"]
            / "thresholds/validation.json"
        ).read_text()
    )
    parity_checks = 0
    for count in (1, 5, 10):
        ours = thresholds[condition_key("vowel_exact", count, "full", "native")][
            "operating_points"
        ]
        for name, _ in POINTS:
            old = phase6["conditions"][f"n{count}/fused"][name]
            for field in (
                "threshold",
                "false_accepts",
                "false_rejects",
                "all_input_far",
                "all_input_frr",
            ):
                if ours[name][field] != old[field]:
                    raise ValueError(
                        f"Phase 6 full threshold parity failed: n{count}/{name}/{field}"
                    )
            parity_checks += 1
    threshold_document = {
        "schema_version": 1,
        "split": "validation",
        "role": "verification",
        "protocol_sha256": inputs["protocol_sha256"],
        "config_sha256": inputs["config_sha256"],
        "acceptance": "score_gte_threshold",
        "conditions": thresholds,
        "source_scores_sha256": {
            worker: sha256_file(run / worker / "scores.jsonl.gz")
            for worker in ("vowels", "ecapa")
        },
    }
    write_json(run / "validation-thresholds.json", threshold_document)
    document = {
        "schema_version": 1,
        "purpose": "validation_calibration_and_cross_text_diagnostics_before_test",
        "thresholds_sha256": sha256_file(run / "validation-thresholds.json"),
        "conditions": conditions,
        "phase6_full_threshold_parity_checks": parity_checks,
        "test_audio_or_scores_read": False,
        "bootstrap": "reserved_for_test_after_execution_freeze",
        "curves_sha256": sha256_file(curves_path),
    }
    write_json(run / "validation-metrics.json", document)
    load_inputs(run)
    return {
        "calibration_cells": len(thresholds),
        "operating_thresholds": sum(
            len(t["operating_points"] or {}) for t in thresholds.values()
        ),
        "evaluation_cells": len(conditions),
        "not_evaluable_cells": [
            k
            for k, c in conditions.items()
            if c["conditional_status"] == "not_evaluable"
        ],
        "phase6_full_threshold_parity_checks": parity_checks,
    }
