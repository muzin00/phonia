"""Choose difficult CV impostors using only the fixed encoder and training data."""

from shared import rt


def cv_score_table(arrays, owners, corpora):
    cv = corpora[owners] == "common_voice"
    queries = rt.np.flatnonzero(cv)
    claims = rt.np.flatnonzero(corpora == "common_voice")
    result = rt.np.full((len(owners), len(corpora)), -rt.np.inf)
    qmask = arrays["qmask"].numpy()[queries]
    emask = arrays["emask"].numpy()
    common = qmask & emask[owners[queries]]
    if not common[:, :5].all() or not emask[claims, :5].all():
        raise ValueError("Mining five-vowel eligibility changed")
    total = rt.np.zeros((len(queries), len(claims)), rt.np.float64)
    counts = rt.np.zeros_like(total, dtype=rt.np.int16)
    for phone in range(common.shape[1]):
        mask = common[:, phone, None] & emask[None, claims, phone]
        q = arrays["queries"].numpy()[queries, phone].astype(rt.np.float64)
        e = arrays["enrollment"].numpy()[claims, phone].astype(rt.np.float64)
        total += (q @ e.T) * mask
        counts += mask
    result[rt.np.ix_(queries, claims)] = total / counts
    result[rt.np.arange(len(owners)), owners] = -rt.np.inf
    return result


def make_schedule(original, table, owners, corpora, config, seed):
    queries = original["queries"][: config["final_fusion_updates"]].copy()
    negatives = original["negative_claims"][: len(queries)].copy()
    rng = rt.np.random.default_rng(rt.seed_for(config["mining_seed"], seed))
    hard = rt.np.zeros(queries.shape, bool)
    for update in range(config["final_fusion_initial_updates"], len(queries)):
        hard[
            update,
            rng.choice(
                queries.shape[1], config["hard_queries_per_update"], replace=False
            ),
        ] = True
    hard &= corpora[owners[queries]] == "common_voice"
    positions = rt.np.flatnonzero(hard.ravel())
    q = queries.ravel()[positions]
    old = negatives.ravel()[positions]
    candidates = rt.np.empty((len(q), config["candidates"]), rt.np.int16)
    candidates[:, 0] = old
    cv_claims = rt.np.flatnonzero(corpora == "common_voice")
    for i, (qid, prior) in enumerate(zip(q, old, strict=True)):
        available = cv_claims[(cv_claims != owners[qid]) & (cv_claims != prior)]
        candidates[i, 1:] = rng.choice(
            available, config["candidates"] - 1, replace=False
        )
    scores = table[q[:, None], candidates]
    if not rt.np.isfinite(scores).all():
        raise ValueError("Invalid training-only mining candidates")
    selected = candidates[rt.np.arange(len(q)), scores.argmax(1)]
    negatives.ravel()[positions] = selected
    if (owners[queries] == negatives).any():
        raise ValueError("Genuine speaker used as negative")
    if not (corpora[owners[queries]] == corpora[negatives]).all():
        raise ValueError("Cross-corpus negative")
    if not (table[q, selected] >= table[q, old] - 1e-12).all():
        raise ValueError("Hard mining reduced frozen cosine similarity")
    return {"queries": queries, "negative_claims": negatives}, {
        "positions": positions,
        "candidates": candidates,
        "original_scores": table[q, old],
        "selected_scores": table[q, selected],
    }
