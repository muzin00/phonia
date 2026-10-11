"""A random forward pass, accepted subsets only, and later retry passes."""

from __future__ import annotations

import math

from common import VOWELS, ordered, rank


def draw_order(candidates, seed):
    if set(candidates) & set(VOWELS) or len(set(candidates)) != len(candidates):
        raise ValueError("box must contain unique additional speech phones")
    return sorted(candidates, key=lambda p: (rank(seed, p), p))


def baseline_state(order, eer):
    if not math.isfinite(eer) or not 0 <= eer <= 1:
        raise ValueError("invalid baseline EER")
    return {
        "pass": 1,
        "position": 0,
        "accepted": list(VOWELS),
        "best_eer": eer,
        "best_trial": "trial-000-baseline",
        "pass_acceptances": 0,
        "order": order,
        "attempts": [],
        "completed": False,
    }


def next_candidate(state):
    while not state["completed"]:
        if state["position"] == len(state["order"]):
            if (
                state["pass_acceptances"] == 0
                or len(state["accepted"]) == len(state["order"]) + 5
            ):
                state["completed"] = True
                return None
            state["pass"] += 1
            state["position"] = 0
            state["pass_acceptances"] = 0
        phone = state["order"][state["position"]]
        state["position"] += 1
        if phone not in state["accepted"]:
            return phone
    return None


def record_attempt(state, phone, trial, eer, reason=None):
    before = state["best_eer"]
    accepted = eer is not None and math.isfinite(eer) and eer < before
    entry = {
        "step": len(state["attempts"]) + 1,
        "pass": state["pass"],
        "phoneme": phone,
        "trial": trial,
        "phonemes": ordered([*state["accepted"], phone]),
        "previous_best_eer": before,
        "validation_eer": eer,
        "eer_delta": eer - before if eer is not None else None,
        "accepted": accepted,
        "status": "accepted"
        if accepted
        else ("unavailable" if eer is None else "rejected"),
        "reason": reason,
    }
    if accepted:
        state["accepted"] = entry["phonemes"]
        state["best_eer"] = eer
        state["best_trial"] = trial
        state["pass_acceptances"] += 1
    entry["best_eer_after"] = state["best_eer"]
    state["attempts"].append(entry)
    return entry
