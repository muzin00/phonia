"""Describe changes at each model's frozen validation threshold."""

import src_evaluation as study
from diagnostics import paired_decisions


def run_diagnostics(run):
    _, groups = study.score_groups(run, "test")
    thresholds = study.read_json(run / "validation-thresholds.json")["conditions"]
    metrics = study.read_json(run / "test-metrics.json")["conditions"]
    transitions = {}
    for role in study.ROLES:
        for point, _ in study.POINTS:
            before, after = "cv70_u30000", "src70_u30000"
            result = paired_decisions(
                groups[before, role],
                groups[after, role],
                thresholds[before]["operating_points"][point]["threshold"],
                thresholds[after]["operating_points"][point]["threshold"],
            )
            for condition, side in ((before, "before"), (after, "after")):
                p = metrics[f"{condition}/{role}"]["operating_points"][point]
                for label in ("genuine", "impostor"):
                    counts = result["counts"][label]
                    accepted = counts["accepted_both"] + counts[f"accepted_{side}_only"]
                    expected = (
                        p["all_genuine"] - p["false_rejects"] - p["no_score_genuine"]
                        if label == "genuine"
                        else p["false_accepts"]
                    )
                    if accepted != expected:
                        raise ValueError("paired decisions differ from measured rates")
            transitions[f"{role}/{point}"] = result
    study.write_json(
        run / "diagnostics.json",
        {
            "comparison": "src70_u30000_minus_cv70_u30000",
            "thresholds": "each_model_frozen_validation_thresholds",
            "test_used_for_threshold_selection": False,
            "decision_transitions": transitions,
            "interpretation": "descriptive post_test diagnostics; not a causal attribution of remaining errors",
        },
    )
