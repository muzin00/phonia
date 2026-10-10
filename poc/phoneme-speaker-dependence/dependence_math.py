"""Trace variance components with nuisance adjustment, and exact 1D clustering."""

import numpy as np

CONTEXTS = (
    "a",
    "i",
    "u",
    "e",
    "o",
    "unvoiced_vowel",
    "nasal",
    "voiced_obstruent",
    "unvoiced_obstruent",
    "approximant",
    "other",
)


def context_class(phone):
    if phone in ["a", "i", "u", "e", "o"]:
        return phone
    if phone in ("A", "I", "U", "E", "O"):
        return "unvoiced_vowel"
    if phone in ("m", "n", "N", "my", "ny"):
        return "nasal"
    if phone in ("b", "d", "g", "z", "j", "v", "by", "dy", "gy"):
        return "voiced_obstruent"
    if phone in (
        "p",
        "t",
        "k",
        "s",
        "sh",
        "ch",
        "ts",
        "h",
        "f",
        "ky",
        "py",
        "ty",
        "hy",
        "cl",
    ):
        return "unvoiced_obstruent"
    if phone in ("r", "ry", "w", "y"):
        return "approximant"
    return "other"


def design(rows):
    return np.asarray(
        [
            [1.0, np.log((r["end_frame"] - r["start_frame"]) / 2400)]
            + [float(context_class(r["previous_phone"]) == c) for c in CONTEXTS[:-1]]
            + [float(context_class(r["following_phone"]) == c) for c in CONTEXTS[:-1]]
            for r in rows
        ],
        dtype=np.float64,
    )


def sufficient_statistics(x, y, labels, speakers):
    """Speaker blocks allow resampling clusters without resampling audio frames."""
    labels = np.asarray(labels)
    blocks = []
    for s in speakers:
        a, b = x[labels == s], y[labels == s]
        if len(a) < 2:
            raise ValueError("each speaker requires independent repeated tokens")
        blocks.append(
            {
                "n": len(a),
                "xx": a.T @ a,
                "xy": a.T @ b,
                "yy": float(np.sum(b * b)),
                "zx": a.sum(axis=0),
                "zy": b.sum(axis=0),
            }
        )
    return blocks, y.shape[1]


def fit_components(blocks, dimensions, draws=None):
    """Y = X beta + Z u + error; common B/W ratio over embedding coordinates.

    Profile restricted likelihood in lambda=B/W. Each bootstrap copy gets its
    own speaker intercept. Coordinate correlation is not used to form a CI:
    uncertainty is obtained by resampling whole speakers.
    """
    selected = blocks if draws is None else [blocks[i] for i in draws]
    xx = sum(b["xx"] for b in selected)
    xy = sum(b["xy"] for b in selected)
    yy = sum(b["yy"] for b in selected)
    zx, zy = (
        np.stack([b["zx"] for b in selected]),
        np.stack([b["zy"] for b in selected]),
    )
    n = np.asarray([b["n"] for b in selected], dtype=float)
    eigen, rotation = np.linalg.eigh(xx)
    keep = eigen > max(float(eigen[-1]) * 1e-10, 1e-12)
    inverse = (rotation[:, keep] / eigen[keep]) @ rotation[:, keep].T
    rank = int(keep.sum())
    residual_df = int(n.sum()) - rank
    if residual_df <= len(selected):
        raise ValueError("too few residual degrees of freedom")
    residual_energy = float(yy - np.sum(xy * (inverse @ xy)))
    gram = np.diag(n) - zx @ inverse @ zx.T
    gram = (gram + gram.T) / 2
    values, vectors = np.linalg.eigh(gram)
    keep = values > max(float(values[-1]) * 1e-9, 1e-10)
    values = values[keep]
    coordinates = vectors[:, keep].T @ (zy - zx @ inverse @ xy)
    energy = np.sum(coordinates * coordinates, axis=1) / values
    remainder = max(0.0, residual_energy - float(energy.sum()))
    df = dimensions * residual_df

    def objective(lam):
        factors = 1 + lam * values
        variance = max(float((remainder + np.sum(energy / factors)) / df), 1e-15)
        return df * np.log(variance) + dimensions * np.log(factors).sum(), variance

    grid = np.linspace(-18, 18, 37)
    index = min(range(len(grid)), key=lambda i: objective(np.exp(grid[i]))[0])
    lo, hi = grid[max(index - 1, 0)], grid[min(index + 1, len(grid) - 1)]
    ratio = (np.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = objective(np.exp(a))[0], objective(np.exp(b))[0]
    for _ in range(50):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = objective(np.exp(a))[0]
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = objective(np.exp(b))[0]
    lam = float(np.exp((lo + hi) / 2))
    if objective(0)[0] <= objective(lam)[0]:
        lam = 0.0
    _, w = objective(lam)
    return {
        "R": lam / (1 + lam),
        "B_trace": dimensions * lam * w,
        "W_trace": dimensions * w,
        "lambda": lam,
        "fixed_effect_rank": rank,
        "tokens": int(n.sum()),
        "speakers": len(selected),
        "residual_df": residual_df,
    }


def cluster(scores, minimum=2):
    order = sorted(scores, key=lambda p: (scores[p], p))
    if len(order) < 2 * minimum:
        raise ValueError("too few phones for two groups")
    values = np.array([scores[p] for p in order])

    def loss(k):
        return float(
            np.sum((values[:k] - values[:k].mean()) ** 2)
            + np.sum((values[k:] - values[k:].mean()) ** 2)
        )

    k = min(range(minimum, len(order) - minimum + 1), key=lambda k: (loss(k), k))
    single = float(np.sum((values - values.mean()) ** 2))
    return {
        "low": order[:k],
        "high": order[k:],
        "boundary": float((values[k - 1] + values[k]) / 2),
        "two_group_sse": loss(k),
        "single_group_sse": single,
        "relative_sse_reduction": 1 - loss(k) / single if single > 1e-15 else 0.0,
        "candidate_boundaries": len(order) - 2 * minimum + 1,
        "note": "A forced two-group solution does not establish natural bimodality.",
    }


def rank_correlation(x, y):
    def ranks(a):
        a = np.asarray(a)
        return np.array(
            [np.sum(a < v) + (np.sum(a == v) - 1) / 2 for v in a], dtype=float
        )

    a, b = ranks(x), ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])
