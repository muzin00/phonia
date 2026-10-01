"""Exact, tie-aware speaker-weighted EER with shared bootstrap draws."""

import numpy as np


def speaker_draws(speaker_count: int, replicates: int = 10000, seed: int = 20260929):
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = []
    while len(draws) < replicates:
        draw = rng.choice(speaker_count, size=speaker_count, replace=True)
        if len(np.unique(draw)) >= 2:
            draws.append(draw)
    indices = np.asarray(draws, dtype=np.int16)
    counts = np.stack([np.bincount(draw, minlength=speaker_count) for draw in indices])
    return indices, counts


def weighted_eers(scores, query_speaker, claimed_speaker, counts):
    """Reweight original pairs, then interpolate the weighted ROC at score ties.

    Prefix counts per ordered speaker pair allow exact binary searches for all
    replicates without materializing repeated trials or approximating scores.
    """
    scores = np.asarray(scores, dtype=np.float64)
    query_speaker = np.asarray(query_speaker, dtype=np.int64)
    claimed_speaker = np.asarray(claimed_speaker, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not len(scores):
        raise ValueError("finite nonempty scores required")
    if query_speaker.shape != scores.shape or claimed_speaker.shape != scores.shape:
        raise ValueError("speaker arrays must match scores")
    if counts.ndim != 2 or (counts < 0).any():
        raise ValueError("nonnegative replicate speaker counts required")
    speakers = counts.shape[1]
    if (
        (query_speaker < 0).any()
        or (query_speaker >= speakers).any()
        or (claimed_speaker < 0).any()
        or (claimed_speaker >= speakers).any()
    ):
        raise ValueError("speaker index outside counts")
    order = np.argsort(-scores, kind="stable")
    pairs = query_speaker[order] * speakers + claimed_speaker[order]
    cumulative = np.zeros((len(scores) + 1, speakers * speakers), dtype=np.int32)
    cumulative[np.arange(1, len(scores) + 1), pairs] = 1
    np.cumsum(cumulative, axis=0, dtype=np.int32, out=cumulative)
    tie_ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0) + 1, len(scores)]
    prefix = cumulative[np.r_[0, tie_ends]]
    del cumulative
    diagonal = np.arange(speakers) * (speakers + 1)
    weights = (counts[:, :, None] * counts[:, None, :]).reshape(len(counts), -1)
    weights[:, diagonal] = counts  # genuine weight n_s, not n_s squared
    positive_total = counts @ prefix[-1, diagonal]
    negative_total = weights @ prefix[-1] - positive_total
    if (positive_total <= 0).any() or (negative_total <= 0).any():
        raise ValueError("each replicate requires genuine and impostor weight")

    def accepted(indices):
        selected = prefix[indices]
        positive = np.einsum("ij,ij->i", selected[:, diagonal], counts, dtype=np.int64)
        negative = np.einsum("ij,ij->i", selected, weights, dtype=np.int64) - positive
        return positive, negative

    lo = np.zeros(len(counts), dtype=np.int64)
    hi = np.full(len(counts), len(prefix) - 1, dtype=np.int64)
    while np.any(hi - lo > 1):
        active = hi - lo > 1
        mid = (lo + hi) // 2
        positive, negative = accepted(mid)
        crossed = (
            negative * positive_total >= (positive_total - positive) * negative_total
        )
        hi = np.where(active & crossed, mid, hi)
        lo = np.where(active & ~crossed, mid, lo)
    lp, ln = accepted(lo)
    hp, hn = accepted(hi)
    lfar, hfar = ln / negative_total, hn / negative_total
    ldifference = lfar - (positive_total - lp) / positive_total
    hdifference = hfar - (positive_total - hp) / positive_total
    fraction = -ldifference / (hdifference - ldifference)
    return lfar + fraction * (hfar - lfar)
