"""Independent streaming fixed-threshold and report checks after test."""

import json
from collections import defaultdict

import numpy as np

from final_metrics import interval
from frozen_test import load_inputs
from smoke import ROOT, sha256_file
from validation import iter_rows
from validation_metrics import numeric_threshold


def audit_final(run):
    inputs = load_inputs(run)
    validation = ROOT / inputs["config"]["validation_run"]
    thresholds = json.loads((validation / "validation-thresholds.json").read_text())
    metrics = json.loads((run / "test-metrics.json").read_text())
    if metrics["threshold_recalibrated"] or metrics["thresholds_sha256"] != sha256_file(
        validation / "validation-thresholds.json"
    ):
        raise ValueError("threshold changed after validation")
    windows = {(w["query_id"], w["condition_id"]): w for w in inputs["windows"]}
    totals = defaultdict(lambda: defaultdict(int))
    queries = defaultdict(set)
    scored_queries = defaultdict(set)
    for worker in ("vowels", "ecapa"):
        for row in iter_rows(run / worker / "scores.jsonl.gz"):
            for support in ("native", "common"):
                if (
                    support == "common"
                    and not windows[row["query_id"], row["condition_id"]][
                        "complete_five_vowels"
                    ]
                ):
                    continue
                base = f"{support}/{row['method_id']}/n{row['enrollment_count']}/{row['condition_id']}"
                key = f"{base}/{row['role']}"
                queries[key].add(row["query_id"])
                if row["status"] == "scored":
                    scored_queries[key].add(row["query_id"])
                for name, threshold in thresholds["conditions"][base][
                    "operating_points"
                ].items():
                    target = totals[key, name]
                    g = row["is_genuine"]
                    scored = row["status"] == "scored"
                    accept = scored and row["scores"]["fused"] >= numeric_threshold(
                        threshold["threshold"]
                    )
                    target["all_genuine" if g else "all_impostor"] += 1
                    target["genuine" if g else "impostor"] += int(scored)
                    target["no_score_genuine" if g else "no_score_impostor"] += int(
                        not scored
                    )
                    target["false_rejects"] += int(g and scored and not accept)
                    target["false_accepts"] += int(not g and accept)
    checked = 0
    for key, cell in metrics["conditions"].items():
        if (cell["queries"], cell["scored_queries"]) != (
            len(queries[key]),
            len(scored_queries[key]),
        ):
            raise ValueError("query denominator mismatch")
        for name, point in cell["operating_points"].items():
            actual = totals[key, name]
            for field, value in actual.items():
                if point[field] != value:
                    raise ValueError(f"count mismatch: {key}/{name}/{field}")
            far = actual["false_accepts"] / actual["all_impostor"]
            frr = (actual["false_rejects"] + actual["no_score_genuine"]) / actual[
                "all_genuine"
            ]
            if point["all_input_far"] != far or point["all_input_frr"] != frr:
                raise ValueError("no_score denominator mismatch")
            threshold = thresholds["conditions"][cell["threshold_source"]][
                "operating_points"
            ][name]["threshold"]
            if point["threshold"] != threshold:
                raise ValueError("cross-role/split threshold changed")
            checked += 1
    with np.load(run / "bootstrap-metrics.npz", allow_pickle=False) as arrays:
        for key, cell in metrics["conditions"].items():
            ci = cell["ci95"]
            if interval(arrays[f"{key}/query_coverage"]) != ci["query_coverage"]:
                raise ValueError("coverage CI mismatch")
            for name, measures in ci["operating_points"].items():
                for metric, bounds in measures.items():
                    if interval(arrays[f"{key}/{name}/{metric}"]) != bounds:
                        raise ValueError("metric CI mismatch")
            if (
                ci["pooled_eer"] is not None
                and interval(arrays[f"{key}/pooled_eer"]) != ci["pooled_eer"]
            ):
                raise ValueError("EER CI mismatch")
            if cell["conditional_status"] == "not_evaluable" and (
                ci["pooled_eer"] is not None
                or any(
                    "frr" in c or "far" in c for c in ci["operating_points"].values()
                )
            ):
                raise ValueError("unsupported conditional CI claim")
    with np.load(
        run / "bootstrap-paired-differences.npz", allow_pickle=False
    ) as arrays:
        for key, difference in metrics["paired_differences"].items():
            if (
                difference["status"] == "evaluable"
                and interval(arrays[key]) != difference["ci95_percentage_points"]
            ):
                raise ValueError("paired difference CI mismatch")
    for name, checksum in metrics["outputs_sha256"].items():
        if sha256_file(run / name) != checksum:
            raise ValueError("metrics output hash mismatch")
    manifest = json.loads((run / "report/manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256_file(run / "report" / name) != checksum:
            raise ValueError("report output changed")
    if (
        len(metrics["conditions"]),
        manifest["condition_cells"],
        manifest["operating_point_rows"],
        manifest["figures_png"],
        manifest["figures_pdf"],
    ) != (180, 360, 1080, 60, 60):
        raise ValueError("incomplete condition/report coverage")
    return {
        "status": "passed",
        "fixed_threshold_points_checked": checked,
        "all_raw_trial_counts_and_query_denominators_checked": True,
        "all_ci_percentiles_and_not_evaluable_rules_checked": True,
        "all_numeric_and_report_output_hashes_checked": True,
        "execution_freeze_unchanged": True,
    }
