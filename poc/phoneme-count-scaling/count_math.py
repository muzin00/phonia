"""Predetermined balanced nested phoneme sets and integrated score arithmetic."""

import hashlib

import numpy as np

VOWELS = ("a", "i", "u", "e", "o")


def schedule(phones, seed):
    consonants = [p for p in phones if p not in VOWELS]
    if list(phones[:5]) != list(VOWELS) or len(consonants) != 9:
        raise ValueError("requires the fixed fourteen-phone inventory")
    shuffled = sorted(
        consonants, key=lambda p: hashlib.sha256(f"{seed}/{p}".encode()).hexdigest()
    )
    orders = [shuffled[i:] + shuffled[:i] for i in range(9)]
    conditions = {"p5": list(VOWELS), "p14": list(phones)}
    by_count = {"5": ["p5"], "14": ["p14"], "7": [], "10": []}
    for count in (7, 10):
        for i, order in enumerate(orders):
            key = f"p{count}_o{i}"
            selected = {*VOWELS, *order[: count - 5]}
            conditions[key] = [p for p in phones if p in selected]
            by_count[str(count)].append(key)
    return {"orders": orders, "conditions": conditions, "by_count": by_count}


def fused_scores(data, vectors):
    profiles = {}
    for speaker, profile in data["profiles"].items():
        for condition, selected in profile["conditions"].items():
            for p in {r["vowel"] for r in selected}:
                mean = np.mean(
                    [vectors[r["segment_id"]] for r in selected if r["vowel"] == p],
                    axis=0,
                )
                profiles[speaker, condition, p] = mean / np.linalg.norm(mean)
    result = []
    for q in data["queries"]:
        for condition, selected in q["conditions"].items():
            phone_means = {
                p: np.mean(
                    [vectors[r["segment_id"]] for r in selected if r["vowel"] == p],
                    axis=0,
                )
                for p in sorted({r["vowel"] for r in selected})
            }
            for speaker in data["speakers"]:
                components = {
                    p: float(v @ profiles[speaker, condition, p])
                    for p, v in phone_means.items()
                }
                result.append(
                    {
                        "query_id": q["query_id"],
                        "speaker_id": q["speaker_id"],
                        "claimed_speaker_id": speaker,
                        "split": data["split"],
                        "role": q["role"],
                        "condition": condition,
                        "is_genuine": speaker == q["speaker_id"],
                        "status": "scored",
                        "score": float(np.mean(list(components.values()))),
                        "phone_scores": components,
                        "used_frames": q["budget_frames"],
                    }
                )
    return result


def count_means(cells, by_count, role, field="pooled_eer"):
    def value(condition):
        cell = cells[f"{condition}/{role}"]
        if field == "pooled_eer":
            return cell[field]
        point, metric = field.split("/")
        return cell["operating_points"][point][metric]

    result = {}
    for count in (5, 7, 10, 14):
        values = [value(k) for k in by_count[str(count)]]
        if any(v is None for v in values):
            result[str(count)] = None
        else:
            result[str(count)] = {
                "mean": float(np.mean(values)),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
                "combinations": len(values),
            }
    return result
