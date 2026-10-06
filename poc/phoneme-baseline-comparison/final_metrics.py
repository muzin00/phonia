"""Fixed-threshold test metrics, duration diagnostics and paired speaker CIs."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import pairwise

import numpy as np

from frozen_test import load_inputs
from smoke import ROOT, sha256_file, write_json
from validation import iter_rows, writer
from validation_metrics import evaluate_cell, numeric_threshold

sys.path.insert(0, str(ROOT / "poc/phoneme-speaker-encoder/scripts"))
from weighted_bootstrap import weighted_eers


def groups_for(run, split):
    inputs = json.loads((run / "inputs.json").read_text())
    queries = {q["query_id"]: q for q in inputs["queries"]}
    windows = {(w["query_id"], w["condition_id"]): w for w in inputs["windows"]}
    groups = defaultdict(list)
    for worker in ("vowels", "ecapa"):
        for r in iter_rows(run / worker / "scores.jsonl.gz"):
            if r["split"] != split:
                raise ValueError("metric split mismatch")
            q = queries[r["query_id"]]
            w = windows[r["query_id"], r["condition_id"]]
            method = r["method_id"]
            frames = (
                w["recording_frames"][1] - w["recording_frames"][0]
                if method == "ecapa_whole"
                else w["used_audio"][method]["unique_frames"]
            )
            groups[method, r["enrollment_count"], r["condition_id"], r["role"]].append(
                {
                    "query_id": q["query_id"],
                    "speaker_id": q["speaker_id"],
                    "claimed_speaker_id": r["claimed_speaker_id"],
                    "split": split,
                    "role": q["role"],
                    "status": r["status"],
                    "is_genuine": r["is_genuine"],
                    "score": r["scores"]["fused"] if r["status"] == "scored" else None,
                    "common": w["complete_five_vowels"],
                    "source_seconds": inputs["sources"][q["source_file"]]["frame_count"]
                    / 24000,
                    "actual_input_seconds": frames / 24000
                    if r["status"] == "scored"
                    else 0.0,
                }
            )
    return inputs, groups


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return {
        "lower": float(np.percentile(valid, 2.5, method="linear"))
        if len(valid)
        else None,
        "upper": float(np.percentile(valid, 97.5, method="linear"))
        if len(valid)
        else None,
        "valid_replicates": len(valid),
        "undefined_replicates": len(values) - len(valid),
    }


def divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 0,
    )


def pair_totals(mask, pairs, weights, speakers):
    matrix = np.bincount(pairs[mask], minlength=speakers * speakers)
    return weights @ matrix


def bootstrap_cell(rows, cell, thresholds, speakers, counts):
    """Use n_i for genuine and n_i*n_j for impostor, including no_score."""
    n = len(speakers)
    lookup = {s: i for i, s in enumerate(speakers)}
    qi = np.array([lookup[r["speaker_id"]] for r in rows], dtype=np.int64)
    ci = np.array([lookup[r["claimed_speaker_id"]] for r in rows], dtype=np.int64)
    pairs = qi * n + ci
    genuine = qi == ci
    scored = np.array([r["status"] == "scored" for r in rows], dtype=bool)
    scores = np.array(
        [r["score"] if r["status"] == "scored" else -np.inf for r in rows],
        dtype=np.float64,
    )
    weights = (counts[:, :, None] * counts[:, None, :]).reshape(len(counts), -1).copy()
    weights[:, np.arange(n) * (n + 1)] = counts
    ag = pair_totals(genuine, pairs, weights, n)
    ai = pair_totals(~genuine, pairs, weights, n)
    sg = pair_totals(genuine & scored, pairs, weights, n)
    si = pair_totals(~genuine & scored, pairs, weights, n)
    replicas = {"query_coverage": divide(sg, ag)}
    result = {
        "query_coverage": interval(replicas["query_coverage"]),
        "operating_points": {},
        "pooled_eer": None,
        "conditional_status": cell["conditional_status"],
    }
    for name, point in (thresholds["operating_points"] or {}).items():
        accepted = scored & (scores >= numeric_threshold(point["threshold"]))
        fa = pair_totals(~genuine & accepted, pairs, weights, n)
        fr = pair_totals(genuine & scored & ~accepted, pairs, weights, n)
        measures = {
            "all_input_far": divide(fa, ai),
            "all_input_frr": divide(fr + ag - sg, ag),
        }
        if cell["conditional_status"] == "evaluable":
            measures.update(far=divide(fa, si), frr=divide(fr, sg))
        result["operating_points"][name] = {k: interval(v) for k, v in measures.items()}
        replicas.update({f"{name}/{k}": v for k, v in measures.items()})
    if cell["conditional_status"] == "evaluable":
        replicas["pooled_eer"] = weighted_eers(
            scores[scored], qi[scored], ci[scored], counts
        )
        result["pooled_eer"] = interval(replicas["pooled_eer"])
    return result, replicas


def duration_diagnostics(rows, threshold, speakers, protocol):
    result = {}
    for field, edges in (
        ("source_seconds", protocol["metrics"]["query_duration_bins_seconds"]),
        (
            "actual_input_seconds",
            protocol["metrics"]["actual_input_duration_bins_seconds"],
        ),
    ):
        bins = {}
        for first, last in pairwise(edges):
            subset = [
                r
                for r in rows
                if r[field] >= first and (last is None or r[field] < last)
            ]
            item, _ = evaluate_cell(subset, threshold, speakers)
            item["bin_seconds"] = [first, last]
            bins[f"{first}:{last}"] = item
        result[field] = bins
    return result


def paired_differences(cells, replicas):
    differences, draws = {}, {}
    for key in cells:
        support, method, count, cap, role = key.split("/")
        references = []
        if method != "vowel_exact":
            references.append(
                ("method_minus_exact", f"{support}/vowel_exact/{count}/{cap}/{role}")
            )
        if cap != "full":
            references.append(
                ("cap_minus_full", f"{support}/{method}/{count}/full/{role}")
            )
        for kind, reference in references:
            for point in ("far_1pct", "far_0_1pct", "eer_operating"):
                name = f"{kind}/{key}/{point}"
                measure = f"{point}/all_input_frr"
                a, b = replicas[key].get(measure), replicas[reference].get(measure)
                if a is None or b is None:
                    differences[name] = {
                        "status": "not_evaluable",
                        "reference": reference,
                    }
                    continue
                delta = 100 * (a - b)
                draws[name] = delta
                differences[name] = {
                    "status": "evaluable",
                    "reference": reference,
                    "point_difference_percentage_points": 100
                    * (
                        cells[key]["operating_points"][point]["all_input_frr"]
                        - cells[reference]["operating_points"][point]["all_input_frr"]
                    ),
                    "ci95_percentage_points": interval(delta),
                    "population_note": "common support changes across caps"
                    if support == "common" and kind == "cap_minus_full"
                    else "same query cohort",
                }
    return differences, draws


def evaluate_test(run):
    frozen_inputs = load_inputs(run)
    validation = ROOT / frozen_inputs["config"]["validation_run"]
    threshold_path = validation / "validation-thresholds.json"
    threshold_doc = json.loads(threshold_path.read_text())
    if (
        threshold_doc["split"] != "validation"
        or threshold_doc["role"] != "verification"
        or threshold_doc["protocol_sha256"] != frozen_inputs["protocol_sha256"]
    ):
        raise ValueError("test cannot recalibrate thresholds")
    inputs, groups = groups_for(run, "test")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, replicas = {}, {}
    with writer(run / "test-curves.jsonl.gz") as write_curve:
        for support in ("native", "common"):
            for (method, count, cap, role), native in sorted(groups.items()):
                rows = (
                    native
                    if support == "native"
                    else [r for r in native if r["common"]]
                )
                threshold = threshold_doc["conditions"][
                    f"{support}/{method}/n{count}/{cap}"
                ]
                key = f"{support}/{method}/n{count}/{cap}/{role}"
                cell, curve = evaluate_cell(rows, threshold, inputs["speakers"])
                cell["support_selection_coverage"] = cell["queries"] / len(
                    {r["query_id"] for r in native}
                )
                cell["threshold_source"] = f"{support}/{method}/n{count}/{cap}"
                cell["ci95"], replicas[key] = bootstrap_cell(
                    rows, cell, threshold, inputs["speakers"], counts
                )
                cell["duration_strata"] = duration_diagnostics(
                    rows, threshold, inputs["speakers"], inputs["protocol"]
                )
                if support == "native":
                    cell["source_at_least_5s_diagnostic"], _ = evaluate_cell(
                        [r for r in rows if r["source_seconds"] >= 5],
                        threshold,
                        inputs["speakers"],
                    )
                cells[key] = cell
                write_curve(
                    {
                        "condition": key,
                        **curve,
                        "tpr": (1 - np.asarray(curve["frr"])).tolist(),
                        "det_axes": "far_frr",
                    }
                )
                print(f"test metrics: {len(cells)}/180 {key}", flush=True)
    differences, paired = paired_differences(cells, replicas)
    flattened = {
        f"{key}/{metric}": array
        for key, measures in replicas.items()
        for metric, array in measures.items()
    }
    with (run / "bootstrap-metrics.npz").open("xb") as stream:
        np.savez_compressed(stream, **flattened)
    with (run / "bootstrap-paired-differences.npz").open("xb") as stream:
        np.savez_compressed(stream, **paired)
    write_json(
        run / "test-metrics.json",
        {
            "schema_version": 1,
            "split": "test",
            "thresholds_sha256": sha256_file(threshold_path),
            "threshold_recalibrated": False,
            "conditions": cells,
            "paired_differences": differences,
            "bootstrap": {
                **inputs["protocol"]["bootstrap"],
                "speakers": inputs["speakers"],
                "counts_sha256": sha256_file(run / "bootstrap-counts.npy"),
                "conditional_ci_for_zero_scored_speaker": "not_evaluable",
                "undefined_replicates": "retained_as_nan_in_npz_and_counted_in_interval_records_no_redraw",
            },
            "outputs_sha256": {
                name: sha256_file(run / name)
                for name in (
                    "test-curves.jsonl.gz",
                    "bootstrap-metrics.npz",
                    "bootstrap-paired-differences.npz",
                )
            },
        },
    )
    # Complete validation duration diagnostics without touching its frozen run.
    vi, vg = groups_for(validation, "validation")
    diagnostics = {}
    for support in ("native", "common"):
        for (method, count, cap, role), native in sorted(vg.items()):
            rows = native if support == "native" else [r for r in native if r["common"]]
            t = threshold_doc["conditions"][f"{support}/{method}/n{count}/{cap}"]
            diagnostics[f"{support}/{method}/n{count}/{cap}/{role}"] = (
                duration_diagnostics(rows, t, vi["speakers"], vi["protocol"])
            )
    write_json(run / "validation-duration-strata.json", diagnostics)
    load_inputs(run)
    return {
        "evaluation_cells": len(cells),
        "paired_difference_cells": len(differences),
        "not_evaluable_cells": [
            k for k, v in cells.items() if v["conditional_status"] != "evaluable"
        ],
    }
