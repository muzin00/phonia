"""Describe score separation and speaker-balanced embedding dispersion."""

import numpy as np
from support import metrics


def distribution(values, histogram=True):
    values = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(values, bins=np.linspace(-1, 1, 81))
    result = {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": dict(
            zip(
                ("p01", "p25", "median", "p75", "p99"),
                map(float, np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])),
            )
        ),
    }
    if histogram:
        result.update(histogram_counts=counts.tolist(), histogram_edges=edges.tolist())
    return result


def score_separation(rows, histogram=True):
    genuine = distribution([r["score"] for r in rows if r["is_genuine"]], histogram)
    impostor = distribution(
        [r["score"] for r in rows if not r["is_genuine"]], histogram
    )
    gap = genuine["mean"] - impostor["mean"]
    scale = np.sqrt((genuine["std"] ** 2 + impostor["std"] ** 2) / 2)
    return {
        "genuine": genuine,
        "impostor": impostor,
        "mean_score_gap": gap,
        "dprime": float(gap / scale) if scale else None,
    }


def embedding_scatter(vectors, labels, speakers):
    vectors = np.asarray(vectors, dtype=np.float64)
    groups = [vectors[np.array(labels) == speaker] for speaker in speakers]
    if any(len(group) < 2 for group in groups):
        raise ValueError("requires repeated queries for every speaker")
    centers = np.stack([group.mean(axis=0) for group in groups])
    overall = centers.mean(axis=0)
    between = float(np.mean(np.sum((centers - overall) ** 2, axis=1)))
    within = float(
        np.mean(
            [
                np.mean(np.sum((group - center) ** 2, axis=1))
                for group, center in zip(groups, centers)
            ]
        )
    )
    return {
        "speaker_balanced_between_scatter": between,
        "speaker_balanced_within_scatter": within,
        "between_within_ratio": between / within if within else None,
        "definition": "equal speaker weighting; squared L2 scatter of unit query embeddings",
    }


def describe(rows, inputs, phone, role, query_vectors, threshold):
    selected = [q for q in inputs["queries"] if q["role"] == role]
    vectors = [query_vectors[f"{phone}/{q['query_id']}"] for q in selected]
    labels = [q["speaker_id"] for q in selected]
    by_speaker = {}
    for speaker in inputs["speakers"]:
        claimed = [r for r in rows if r["claimed_speaker_id"] == speaker]
        scores = np.array([r["score"] for r in claimed])
        genuine = np.array([r["is_genuine"] for r in claimed], dtype=bool)
        by_speaker[speaker] = {
            "eer": metrics.roc_eer(scores, genuine)["eer"],
            "separation": score_separation(claimed, histogram=False),
            "fixed_far_1pct_rates": metrics.error_rates(
                scores,
                genuine,
                float(threshold["operating_points"]["far_1pct"]["threshold"]),
            ),
        }
    return {
        "score_separation": score_separation(rows),
        "embedding_scatter": embedding_scatter(vectors, labels, inputs["speakers"]),
        "claimed_speaker_diagnostics": by_speaker,
    }


def primary_results(config, cells, replicas, interval):
    result = {}
    role = config["primary_role"]
    for phone, reference in config["primary_contrasts"]:
        a, b = f"{phone}/{role}", f"{reference}/{role}"
        delta = 100 * (replicas[a]["pooled_eer"] - replicas[b]["pooled_eer"])
        valid = delta[np.isfinite(delta)]
        alpha = (1 - config["primary_ci_confidence"]) / 2
        low, high = np.quantile(valid, [alpha, 1 - alpha])
        result[f"{phone}_minus_{reference}"] = {
            "eer_difference_percentage_points": 100
            * (cells[a]["pooled_eer"] - cells[b]["pooled_eer"]),
            "ci95_exploratory_percentage_points": interval(delta),
            "ci975_bonferroni_percentage_points": {
                "lower": float(low),
                "upper": float(high),
            },
            "valid_replicates": len(valid),
            "supports_lower_eer_than_s": bool(high < 0),
            "supports_higher_eer_than_s": bool(low > 0),
        }
    return {
        "role": role,
        "family_size": config["primary_family_size"],
        "family_confidence": 0.95,
        "per_contrast_confidence": 0.975,
        "contrasts": result,
        "both_m_and_n_better_than_s": all(
            r["supports_lower_eer_than_s"] for r in result.values()
        ),
        "scope": "fixed_model_50ms_tokens_previously_observed_test_one_seed; exploratory",
    }
