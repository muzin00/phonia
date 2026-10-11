"""Integrated verification scores; test is accessible only after selection freezes."""

from __future__ import annotations

from collections import Counter

from common import (
    VOWELS,
    checked,
    model_embeddings,
    np,
    read_json,
    sha256_file,
    write_json,
    write_rows,
)
from phase3_train.metrics import (
    eer_operating_threshold,
    error_rates,
    far_target_threshold,
    json_threshold,
    roc_eer,
)


def scores_for(data, vectors, phones):
    use = set(phones) & set(data["universally_registered"])
    profiles = {}
    for speaker in data["speakers"]:
        for phone in use:
            mean = vectors[data["profiles"][speaker][phone]].mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm < 1e-12:
                raise ValueError("zero enrollment mean")
            profiles[speaker, phone] = mean / norm
    rows = []
    for query in data["queries"]:
        available = sorted(p for p in use if query["groups"][p])
        means = (
            {p: vectors[query["groups"][p]].mean(axis=0) for p in available}
            if query["scorable"]
            else {}
        )
        if query["scorable"] != all(query["groups"][p] for p in VOWELS):
            raise ValueError("missing-vowel support changed")
        for speaker in data["speakers"]:
            components = (
                {p: float(means[p] @ profiles[speaker, p]) for p in available}
                if means
                else {}
            )
            rows.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": speaker,
                    "role": query["role"],
                    "split": data["split"],
                    "is_genuine": speaker == query["speaker_id"],
                    "status": "scored" if means else "no_score",
                    "score": float(np.mean(list(components.values())))
                    if means
                    else None,
                    "phone_scores": components,
                    "used_phones": available if means else [],
                    "reason": None if means else "missing_fixed_vowel",
                }
            )
    return rows


def arrays(rows):
    scored = [r for r in rows if r["status"] == "scored"]
    return (
        np.array([r["score"] for r in scored]),
        np.array([r["is_genuine"] for r in scored]),
        scored,
    )


def thresholds(rows):
    normal = [r for r in rows if r["role"] == "verification"]
    if any(r["split"] != "validation" for r in normal):
        raise ValueError("thresholds require validation normal only")
    scores, genuine, _ = arrays(normal)
    result = {}
    for name, target in (
        ("far_1pct", 0.01),
        ("far_0_1pct", 0.001),
        ("eer_operating", None),
    ):
        threshold, _ = (
            eer_operating_threshold(scores, genuine)
            if target is None
            else far_target_threshold(scores, genuine, target)
        )
        result[name] = json_threshold(threshold)
    return result


def metrics(rows, points):
    result = {}
    for role in ("verification", "cross_text_verification"):
        chosen = [r for r in rows if r["role"] == role]
        scores, genuine, scored = arrays(chosen)
        if not genuine.any() or genuine.all():
            raise ValueError("missing genuine/impostor scores")
        all_genuine = sum(r["is_genuine"] for r in chosen)
        all_impostor = len(chosen) - all_genuine
        no_score_genuine = all_genuine - int(genuine.sum())
        operating = {}
        for name, threshold in points.items():
            numeric = (
                np.inf
                if threshold == "+inf"
                else (-np.inf if threshold == "-inf" else threshold)
            )
            rates = error_rates(scores, genuine, numeric)
            operating[name] = {
                **rates,
                "threshold": threshold,
                "all_input_far": rates["false_accepts"] / all_impostor,
                "all_input_frr": (rates["false_rejects"] + no_score_genuine)
                / all_genuine,
            }
        result[role] = {
            "eer": roc_eer(scores, genuine)["eer"],
            "all_queries": all_genuine,
            "scored_queries": int(genuine.sum()),
            "coverage": int(genuine.sum()) / all_genuine,
            "genuine_trials": int(genuine.sum()),
            "impostor_trials": int((~genuine).sum()),
            "operating_points": operating,
            "used_phone_counts": dict(
                Counter(p for r in scored if r["is_genuine"] for p in r["used_phones"])
            ),
        }
    return result


def evaluate_trial(run, trial, model, phones, split):
    if split not in ("validation", "test"):
        raise ValueError("unknown split")
    if split == "test":
        frozen = read_json(run / "selection-freeze.json")
        if trial.name not in (frozen["baseline_trial"], frozen["final_trial"]):
            raise ValueError("test cannot be used to choose candidate subsets")
        selected = frozen["models"][trial.name]
        if selected["phonemes"] != phones:
            raise ValueError("final test phones changed")
        checked(trial / "encoder.pt", selected["encoder_sha256"])
        checked(trial / "validation-thresholds.json", selected["thresholds_sha256"])
    output = trial / f"{split}-metrics.json"
    if output.exists():
        return read_json(output)
    data = read_json(run / f"{split}-inputs.json")
    raw_features = np.memmap(
        run / f"{split}-features.f32",
        mode="r",
        dtype="<f4",
        shape=(data["feature_count"], 128),
    )
    vectors = model_embeddings(model, raw_features)
    rows = scores_for(data, vectors, phones)
    np.save(trial / f"{split}-embeddings.npy", vectors, allow_pickle=False)
    write_rows(trial / f"{split}-scores.jsonl", rows)
    if split == "validation":
        points = thresholds(rows)
        write_json(
            trial / "validation-thresholds.json",
            {"split": "validation", "role": "verification", "thresholds": points},
        )
    else:
        points = read_json(trial / "validation-thresholds.json")["thresholds"]
    report = {
        "split": split,
        "phonemes": phones,
        "encoder_sha256": sha256_file(trial / "encoder.pt"),
        "input_sha256": sha256_file(run / f"{split}-inputs.json"),
        "scores_sha256": sha256_file(trial / f"{split}-scores.jsonl"),
        "embeddings_sha256": sha256_file(trial / f"{split}-embeddings.npy"),
        "metrics": metrics(rows, points),
    }
    write_json(output, report)
    return report
