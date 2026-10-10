"""Deterministic source slices with equal PCM budgets and no padding/repetition."""

from __future__ import annotations

from collections import defaultdict

MINIMUM = 720
MAXIMUM = 6000
QUANTUM = 120
VOWELS = ("a", "i", "u", "e", "o")
PHONEMES = (*VOWELS, "m", "n")


def capacity(row):
    return min(MAXIMUM, row["end_frame"] - row["start_frame"])


def plan(candidates, phones, budget):
    """Water-fill complete phone groups; repair a last sub-30ms remainder.

    Each selected interval is 30..250ms, with at least one per phone. Increasing
    time within a phone extends its last interval before starting another.
    Candidate order is fixed externally and never depends on embedding scores.
    Return None if the requested duration cannot be packed into legal slices.
    """
    groups = defaultdict(list)
    for row in candidates:
        if row["vowel"] in phones:
            if capacity(row) < MINIMUM:
                raise ValueError("short candidate")
            groups[row["vowel"]].append(row)
    if len(set(phones)) != len(phones) or any(not groups[p] for p in phones):
        return None
    if budget < MINIMUM * len(phones) or budget > sum(
        capacity(r) for p in phones for r in groups[p]
    ):
        return None
    allocations = {p: [MINIMUM] for p in phones}
    totals = {p: MINIMUM for p in phones}
    remaining = budget - MINIMUM * len(phones)
    while remaining:
        options = []
        for order, phone in enumerate(phones):
            sizes = allocations[phone]
            room = capacity(groups[phone][len(sizes) - 1]) - sizes[-1]
            if room or (len(sizes) < len(groups[phone]) and remaining >= MINIMUM):
                options.append((totals[phone], order, phone, room))
        if not options:
            # All selected slices are full and the residual cannot start a new
            # slice. Shorten existing slices (never below 30ms) to make room.
            additional = next(
                (p for p in phones if len(allocations[p]) < len(groups[p])), None
            )
            needed = MINIMUM - remaining
            if (
                additional is None
                or sum(n - MINIMUM for a in allocations.values() for n in a) < needed
            ):
                return None
            for phone in sorted(phones, key=lambda p: (-totals[p], phones.index(p))):
                for index in reversed(range(len(allocations[phone]))):
                    reduction = min(needed, allocations[phone][index] - MINIMUM)
                    allocations[phone][index] -= reduction
                    totals[phone] -= reduction
                    needed -= reduction
                    if not needed:
                        break
                if not needed:
                    break
            allocations[additional].append(MINIMUM)
            totals[additional] += MINIMUM
            remaining = 0
            break
        _, _, phone, room = min(options)
        if room:
            amount = min(remaining, room, QUANTUM)
            allocations[phone][-1] += amount
        else:
            amount = MINIMUM
            allocations[phone].append(amount)
        remaining -= amount
        totals[phone] += amount
    result = []
    for phone in phones:
        for row, size in zip(groups[phone], allocations[phone]):
            first = (
                row["start_frame"] + (row["end_frame"] - row["start_frame"] - size) // 2
            )
            result.append(
                {
                    **row,
                    "original_segment_id": row["segment_id"],
                    "segment_id": f"{row['segment_id']}--center-{first}-{size}",
                    "start_frame": first,
                    "end_frame": first + size,
                }
            )
    if sum(r["end_frame"] - r["start_frame"] for r in result) != budget:
        raise ValueError("matched budget mismatch")
    return result


def matched_query(vowels, nasals, cap):
    present = {r["vowel"] for r in vowels}
    if not set(VOWELS) <= present:
        return {"vowels5": None, "vowels_mn_available": None}, False
    budget = min(cap, sum(capacity(r) for r in vowels))
    first = plan(vowels, VOWELS, budget)
    if first is None:
        return {"vowels5": None, "vowels_mn_available": None}, False
    extra = tuple(p for p in ("m", "n") if any(r["vowel"] == p for r in nasals))
    second = plan([*vowels, *nasals], (*VOWELS, *extra), budget)
    if second is None:
        second = first
    common = {r["vowel"] for r in second} == set(PHONEMES)
    return {"vowels5": first, "vowels_mn_available": second}, common
