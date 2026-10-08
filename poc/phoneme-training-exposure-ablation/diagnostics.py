"""Describe paired decision changes at frozen validation thresholds."""

from collections import Counter

import exposure_evaluation as study


def paired_decisions(before, after, threshold_before, threshold_after):
    indexed = [{r["trial_id"]: r for r in rows} for rows in (before, after)]
    if any(
        len(index) != len(rows) for index, rows in zip(indexed, (before, after))
    ) or set(indexed[0]) != set(indexed[1]):
        raise ValueError("paired decisions require identical unique trials")
    tau_before = study.old.previous.numeric_threshold(threshold_before)
    tau_after = study.old.previous.numeric_threshold(threshold_after)
    counts = {name: Counter() for name in ("genuine", "impostor")}
    pairs = {}
    for trial_id, a in indexed[0].items():
        b = indexed[1][trial_id]
        for key in ("is_genuine", "speaker_id", "claimed_speaker_id", "status"):
            if a[key] != b[key]:
                raise ValueError("paired trial label/coverage changed")
        accepted_before = a["status"] == "scored" and a["score"] >= tau_before
        accepted_after = b["status"] == "scored" and b["score"] >= tau_after
        transition = {
            (True, True): "accepted_both",
            (True, False): "accepted_before_only",
            (False, True): "accepted_after_only",
            (False, False): "rejected_both",
        }[accepted_before, accepted_after]
        label = "genuine" if a["is_genuine"] else "impostor"
        counts[label][transition] += 1
        counts[label]["all_trials"] += 1
        counts[label]["no_score"] += a["status"] == "no_score"
        if not a["is_genuine"]:
            key = (a["speaker_id"], a["claimed_speaker_id"])
            pair = pairs.setdefault(key, Counter())
            pair["false_accepts_before"] += accepted_before
            pair["false_accepts_after"] += accepted_after
    fields = (
        "all_trials",
        "no_score",
        "accepted_both",
        "accepted_before_only",
        "accepted_after_only",
        "rejected_both",
    )
    return {
        "counts": {name: {k: int(c[k]) for k in fields} for name, c in counts.items()},
        "impostor_speaker_pairs": [
            {
                "speaker_id": speaker,
                "claimed_speaker_id": claimed,
                "false_accepts_before": c["false_accepts_before"],
                "false_accepts_after": c["false_accepts_after"],
            }
            for (speaker, claimed), c in sorted(pairs.items())
        ],
    }


def run_diagnostics(run):
    _, groups = study.score_groups(run, "test")
    thresholds = study.read_json(run / "validation-thresholds.json")["conditions"]
    metrics = study.read_json(run / "test-metrics.json")["conditions"]
    transitions = {}
    for role in study.ROLES:
        for point, _ in study.POINTS:
            before, after = "cv70_u15000", "cv70_u30000"
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
            "comparison": "cv70_u30000_minus_cv70_u15000",
            "thresholds": "each_model_frozen_validation_thresholds",
            "test_used_for_threshold_selection": False,
            "decision_transitions": transitions,
            "interpretation": "descriptive post_test diagnostics; not a causal attribution of remaining errors",
        },
    )
