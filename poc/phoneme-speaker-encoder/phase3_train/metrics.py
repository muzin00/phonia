"""Tie-aware verification curves, EER and validation-only thresholds."""

from __future__ import annotations

import math

import numpy as np


def _arrays(scores, genuine):
    scores = np.asarray(scores, dtype=np.float64)
    genuine = np.asarray(genuine, dtype=np.bool_)
    if (
        scores.ndim != 1
        or genuine.shape != scores.shape
        or not np.isfinite(scores).all()
    ):
        raise ValueError("scores and genuine labels must be finite 1-D arrays")
    return scores, genuine


def error_rates(scores, genuine, threshold: float) -> dict:
    scores, genuine = _arrays(scores, genuine)
    accepted = scores >= threshold
    positive = int(genuine.sum())
    negative = int((~genuine).sum())
    false_accepts = int((accepted & ~genuine).sum())
    false_rejects = int((~accepted & genuine).sum())
    return {
        "genuine": positive,
        "impostor": negative,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "far": false_accepts / negative if negative else None,
        "frr": false_rejects / positive if positive else None,
    }


def roc_eer(scores, genuine) -> dict:
    scores, genuine = _arrays(scores, genuine)
    positive = int(genuine.sum())
    negative = len(genuine) - positive
    if positive == 0 or negative == 0:
        return {"eer": None, "thresholds": [], "far": [], "frr": []}
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_genuine = scores[order], genuine[order]
    end_of_tie = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    accepted_positive = np.cumsum(sorted_genuine, dtype=np.int64)[end_of_tie]
    accepted_total = end_of_tie + 1
    far = np.r_[0.0, (accepted_total - accepted_positive) / negative]
    frr = np.r_[1.0, (positive - accepted_positive) / positive]
    thresholds = ["+inf", *sorted_scores[end_of_tie].tolist()]
    difference = far - frr
    exact = np.flatnonzero(difference == 0)
    if len(exact):
        eer = float(far[exact[0]])
    else:
        right = int(np.flatnonzero(difference > 0)[0])
        left = right - 1
        fraction = -difference[left] / (difference[right] - difference[left])
        eer = float(far[left] + fraction * (far[right] - far[left]))
    return {
        "eer": eer,
        "thresholds": thresholds,
        "far": far.tolist(),
        "frr": frr.tolist(),
    }


def far_target_threshold(scores, genuine, target_far: float) -> tuple[float, dict]:
    scores, genuine = _arrays(scores, genuine)
    if not 0 <= target_far <= 1:
        raise ValueError("target FAR must be in [0,1]")
    impostors = np.sort(scores[~genuine])
    if not len(impostors):
        raise ValueError("no impostor scores for FAR threshold")
    allowed = math.floor(target_far * len(impostors))
    threshold = (
        -math.inf
        if allowed == len(impostors)
        else float(np.nextafter(impostors[len(impostors) - allowed - 1], np.inf))
    )
    rates = error_rates(scores, genuine, threshold)
    if rates["far"] > target_far:
        raise AssertionError("selected FAR threshold exceeds target")
    return threshold, rates


def eer_operating_threshold(scores, genuine) -> tuple[float, dict]:
    scores, genuine = _arrays(scores, genuine)
    if not genuine.any() or genuine.all():
        raise ValueError("EER threshold requires genuine and impostor scores")
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_genuine = scores[order], genuine[order]
    end_of_tie = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    rejected_genuine = np.cumsum(ordered_genuine, dtype=np.int64)[end_of_tie]
    rejected_impostor = end_of_tie + 1 - rejected_genuine
    far = np.r_[
        1.0, (int((~genuine).sum()) - rejected_impostor) / int((~genuine).sum())
    ]
    frr = np.r_[0.0, rejected_genuine / int(genuine.sum())]
    candidates = np.r_[-math.inf, np.nextafter(ordered_scores[end_of_tie], np.inf)]
    best = int(np.lexsort((candidates, far, np.abs(far - frr)))[0])
    threshold = float(candidates[best])
    return threshold, error_rates(scores, genuine, threshold)


def json_threshold(threshold: float) -> float | str:
    if math.isinf(threshold):
        return "+inf" if threshold > 0 else "-inf"
    if not math.isfinite(threshold):
        raise ValueError("NaN threshold")
    return threshold
