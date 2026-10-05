"""Validation-only calibration, explicit denominators and shared speaker bootstrap."""

import math
from collections import Counter

import numpy as np
from phase3_data.manifest import VOWELS, json_sha256
from phase3_train.metrics import (
    eer_operating_threshold,
    error_rates,
    far_target_threshold,
    json_threshold,
    roc_eer,
)
from weighted_bootstrap import speaker_draws, weighted_eers

KINDS = ("fused", *VOWELS)
OPERATING_POINTS = (("far_1pct", 0.01), ("far_0_1pct", 0.001), ("eer_operating", None))


def condition_key(count: int, kind: str) -> str:
    return f"n{count}/{kind}"


def numeric_threshold(value):
    if value == "+inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid threshold")
    return float(value)


def validate_scores(rows: list[dict], expected: list[dict]) -> None:
    if len(rows) != len(expected):
        raise ValueError("score/trial count mismatch")
    indexed = {row["trial_id"]: row for row in expected}
    if len(indexed) != len(expected):
        raise ValueError("duplicate expected trial")
    seen = set()
    states = {}
    for row in rows:
        identity = row["trial_id"]
        trial = indexed.get(identity)
        if (
            trial is None
            or identity in seen
            or any(row.get(key) != value for key, value in trial.items())
        ):
            raise ValueError("score identity/label mismatch")
        if type(row["is_genuine"]) is not bool or row["is_genuine"] != (
            row["speaker_id"] == row["claimed_speaker_id"]
        ):
            raise ValueError("invalid ground truth")
        seen.add(identity)
        query = row["query_id"]
        if query in states and states[query] != row["status"]:
            raise ValueError("inconsistent query score state")
        states[query] = row["status"]
        if row["status"] == "scored":
            if set(row["scores"]) != set(KINDS) or any(
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not -1 <= score <= 1
                for score in row["scores"].values()
            ):
                raise ValueError("invalid finite score")
            if not math.isclose(
                row["scores"]["fused"],
                np.mean([row["scores"][v] for v in VOWELS]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("fusion mismatch")
        elif (
            row["status"] != "no_score"
            or row.get("scores") is not None
            or not row.get("reason")
        ):
            raise ValueError("invalid no_score result")


def arrays(rows: list[dict], kind: str):
    scored = [row for row in rows if row["status"] == "scored"]
    return (
        np.array([row["scores"][kind] for row in scored], dtype=np.float64),
        np.array([row["is_genuine"] for row in scored], dtype=bool),
        scored,
    )


def operating_rates(rows: list[dict], kind: str, threshold: float) -> dict:
    scores, genuine, _ = arrays(rows, kind)
    result = error_rates(scores, genuine, threshold)
    missing_positive = sum(
        row["status"] == "no_score" and row["is_genuine"] for row in rows
    )
    missing_negative = sum(
        row["status"] == "no_score" and not row["is_genuine"] for row in rows
    )
    all_positive = result["genuine"] + missing_positive
    all_negative = result["impostor"] + missing_negative
    return {
        **result,
        "threshold": json_threshold(threshold),
        "no_score_genuine": missing_positive,
        "no_score_impostor": missing_negative,
        "all_genuine": all_positive,
        "all_impostor": all_negative,
        "all_input_far": result["false_accepts"] / all_negative
        if all_negative
        else None,
        "all_input_frr": (result["false_rejects"] + missing_positive) / all_positive
        if all_positive
        else None,
    }


def require_evaluable(rows: list[dict], speakers: list[str]) -> None:
    if any(
        not any(
            row["speaker_id"] == speaker and row["status"] == "scored" for row in rows
        )
        for speaker in speakers
    ):
        raise ValueError(
            "zero scored queries for speaker/role; condition not evaluable"
        )


def calibrate(rows: list[dict], counts: list[int], speakers: list[str]) -> dict:
    if not rows or any(
        row["split"] != "validation" or row["role"] != "verification" for row in rows
    ):
        raise ValueError("calibration requires validation/verification only")
    conditions = {}
    for count in counts:
        selected = [row for row in rows if row["enrollment_count"] == count]
        require_evaluable(selected, speakers)
        for kind in KINDS:
            scores, genuine, _ = arrays(selected, kind)
            operating = {}
            for name, target in OPERATING_POINTS:
                threshold, _ = (
                    eer_operating_threshold(scores, genuine)
                    if target is None
                    else far_target_threshold(scores, genuine, target)
                )
                operating[name] = {
                    "target_far": target,
                    **operating_rates(selected, kind, threshold),
                }
            conditions[condition_key(count, kind)] = operating
    return {
        "schema_version": 1,
        "split": "validation",
        "role": "verification",
        "score_rows_sha256": json_sha256(rows),
        "conditions": conditions,
    }


def bootstrap_draws(speakers: list[str], *, replicates=10000, seed=20260929) -> dict:
    indices, counts = speaker_draws(len(speakers), replicates=replicates, seed=seed)
    return {
        "schema_version": 1,
        "speakers": speakers,
        "seed": seed,
        "replicates": replicates,
        "indices": indices.tolist(),
        "multiplicities": counts.tolist(),
    }


def interval(values) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("nonfinite bootstrap estimate")
    return np.percentile(values, [2.5, 97.5], method="linear").tolist()


def bootstrap_intervals(rows: list[dict], kind: str, points: dict, draws: dict) -> dict:
    speakers = draws["speakers"]
    indices = {speaker: i for i, speaker in enumerate(speakers)}
    n = len(speakers)
    counts = np.asarray(draws["multiplicities"], dtype=np.int64)
    if (
        counts.ndim != 2
        or counts.shape[1] != n
        or (counts < 0).any()
        or not np.all(counts.sum(axis=1) == n)
        or np.any((counts > 0).sum(axis=1) < 2)
    ):
        raise ValueError("invalid shared bootstrap draws")
    scores, _genuine, scored = arrays(rows, kind)
    q = np.array([indices[row["speaker_id"]] for row in scored])
    c = np.array([indices[row["claimed_speaker_id"]] for row in scored])
    weights = (counts[:, :, None] * counts[:, None, :]).reshape(len(counts), -1)
    diagonal = np.arange(n) * (n + 1)
    weights[:, diagonal] = counts

    def total(selected):
        pair = [
            indices[row["speaker_id"]] * n + indices[row["claimed_speaker_id"]]
            for row in selected
        ]
        frequencies = np.bincount(pair, minlength=n * n)
        return weights @ frequencies

    g = total([row for row in scored if row["is_genuine"]])
    i = total([row for row in scored if not row["is_genuine"]])
    all_g = total([row for row in rows if row["is_genuine"]])
    all_i = total([row for row in rows if not row["is_genuine"]])
    if any((denominator <= 0).any() for denominator in (g, i, all_g, all_i)):
        raise ValueError("empty bootstrap denominator")
    result = {
        "pooled_eer": interval(weighted_eers(scores, q, c, counts)),
        "operating_points": {},
    }
    for name, point in points.items():
        threshold = numeric_threshold(point["threshold"])
        accepted = scores >= threshold
        fa = total(
            [
                row
                for row, accept in zip(scored, accepted, strict=True)
                if accept and not row["is_genuine"]
            ]
        )
        fr = total(
            [
                row
                for row, accept in zip(scored, accepted, strict=True)
                if not accept and row["is_genuine"]
            ]
        )
        result["operating_points"][name] = {
            "far": interval(fa / i),
            "frr": interval(fr / g),
            "all_input_far": interval(fa / all_i),
            "all_input_frr": interval((fr + all_g - g) / all_g),
        }
    return result


def evaluate(
    rows: list[dict],
    thresholds: dict,
    counts: list[int],
    roles: list[str],
    speakers: list[str],
    *,
    draws: dict | None = None,
) -> tuple[dict, list[dict]]:
    if (
        thresholds.get("split") != "validation"
        or thresholds.get("role") != "verification"
    ):
        raise ValueError("evaluation requires validation/verification thresholds")
    conditions, curves = {}, []
    for role in roles:
        for count in counts:
            selected = [
                row
                for row in rows
                if row["enrollment_count"] == count and row["role"] == role
            ]
            require_evaluable(selected, speakers)
            query_states = {row["query_id"]: row["status"] for row in selected}
            for kind in KINDS:
                scores, genuine, _ = arrays(selected, kind)
                curve = roc_eer(scores, genuine)
                points = {
                    name: operating_rates(
                        selected, kind, numeric_threshold(point["threshold"])
                    )
                    for name, point in thresholds["conditions"][
                        condition_key(count, kind)
                    ].items()
                }
                item = {
                    "pooled_eer": curve["eer"],
                    "all_queries": len(query_states),
                    "scored_queries": sum(s == "scored" for s in query_states.values()),
                    "score_coverage": sum(s == "scored" for s in query_states.values())
                    / len(query_states),
                    "far_resolution": 1 / int((~genuine).sum()),
                    "operating_points": points,
                }
                item["by_speaker"] = {
                    speaker: {
                        "query": {
                            name: operating_rates(
                                [r for r in selected if r["speaker_id"] == speaker],
                                kind,
                                numeric_threshold(p["threshold"]),
                            )
                            for name, p in points.items()
                        },
                        "claim": {
                            name: operating_rates(
                                [
                                    r
                                    for r in selected
                                    if r["claimed_speaker_id"] == speaker
                                ],
                                kind,
                                numeric_threshold(p["threshold"]),
                            )
                            for name, p in points.items()
                        },
                    }
                    for speaker in speakers
                }
                if draws is not None:
                    if draws["speakers"] != speakers:
                        raise ValueError("bootstrap speaker mismatch")
                    item["bootstrap_95pct"] = bootstrap_intervals(
                        selected, kind, points, draws
                    )
                key = f"{role}/{condition_key(count, kind)}"
                conditions[key] = item
                curves.append(
                    {
                        "condition": key,
                        **curve,
                        "tpr": (1 - np.asarray(curve["frr"])).tolist(),
                    }
                )
    macro = {
        f"{role}/n{count}": float(
            np.mean(
                [
                    conditions[f"{role}/{condition_key(count, v)}"]["pooled_eer"]
                    for v in VOWELS
                ]
            )
        )
        for role in roles
        for count in counts
    }
    first_count = counts[0]
    unique = {
        row["query_id"]: row
        for row in rows
        if row["enrollment_count"] == first_count and row["is_genuine"]
    }
    reasons = Counter(
        f"{r['role']}/{r['reason']}"
        for r in unique.values()
        if r["status"] == "no_score"
    )
    return {
        "schema_version": 1,
        "thresholds_sha256": json_sha256(thresholds),
        "conditions": conditions,
        "macro_vowel_eer_diagnostic": macro,
        "no_score_queries_by_reason": dict(reasons),
    }, curves
