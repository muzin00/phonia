"""Deterministic train-only frozen-encoder hard-negative schedules."""

import numpy as np


def corpus(speaker):
    return (
        "cv"
        if speaker.startswith("cv17_")
        else "jt"
        if speaker.startswith("jtube_")
        else "jvs"
    )


def score_table(arrays, data):
    enrollment, queries = arrays["enrollment"], arrays["queries"]
    emask, qmask = arrays["emask"], arrays["qmask"]
    owner = np.array([data["speakers"].index(q["speaker_id"]) for q in data["queries"]])
    allowed = qmask & emask[owner]
    assert allowed[:, :5].all() and emask[:, :5].all()
    result = np.zeros((len(queries), len(enrollment)), dtype=np.float64)
    counts = np.zeros(result.shape, dtype=np.int16)
    # Use float64 to make ranking and its independent direct-dot audit stable.
    for phone in range(queries.shape[1]):
        mask = allowed[:, phone, None] & emask[None, :, phone]
        result += (
            queries[:, phone].astype(np.float64)
            @ enrollment[:, phone].astype(np.float64).T
        ) * mask
        counts += mask
    result /= counts
    sources = np.array([corpus(v) for v in data["speakers"]])
    result[sources[owner, None] != sources[None, :]] = -np.inf
    result[np.arange(len(queries)), owner] = -np.inf
    return result, owner


def make_schedule(original, table, owner, config):
    rng = np.random.default_rng(config["mining_seed"])
    queries = original["queries"]
    negatives = original["negative_claims"].copy()
    start = config["budgets"][0]
    hard = np.zeros(queries.shape, dtype=bool)
    for update in range(start, len(queries)):
        hard[
            update,
            rng.choice(
                queries.shape[1], config["hard_queries_per_update"], replace=False
            ),
        ] = True
    positions = np.flatnonzero(hard.ravel())
    q = queries.ravel()[positions]
    old = negatives.ravel()[positions]
    candidates = np.empty((len(q), config["candidates"]), dtype=np.int16)
    candidates[:, 0] = old
    pools = {
        int(i): np.flatnonzero(np.isfinite(table[np.flatnonzero(owner == i)[0]]))
        for i in np.unique(owner)
    }
    for i, (qid, original_claim) in enumerate(zip(q, old, strict=True)):
        pool = pools[int(owner[qid])]
        pool = pool[pool != original_claim]
        candidates[i, 1:] = rng.choice(pool, config["candidates"] - 1, replace=False)
    scores = table[q[:, None], candidates]
    assert np.isfinite(scores).all()
    selected = candidates[np.arange(len(q)), scores.argmax(axis=1)]
    assert (table[q, selected] >= table[q, old] - 1e-12).all()
    negatives.ravel()[positions] = selected
    return {"queries": queries.copy(), "negative_claims": negatives}, {
        "positions": positions,
        "candidates": candidates,
    }
