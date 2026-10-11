"""Observed-data diagnostics; preserve the frozen recording pilot and all attempts."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-live-recording"))
import runtime as r

np = r.np
VOWELS = tuple(r.VOWELS)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_profile(rows, vectors, config):
    profiles, chosen = {}, {}
    for phone in sorted({row["phoneme"] for row in rows}):
        selected = r.t.h.primitives.diverse(
            [row for row in rows if row["phoneme"] == phone],
            config["limit_per_phone"],
            config["selection_seed"],
        )
        ids = [row["cache_index"] for row in selected]
        mean = vectors[ids].mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-12:
            raise ValueError("Degenerate profile")
        profiles[phone] = mean / norm
        chosen[phone] = selected
    eligible = all(
        len(chosen.get(p, [])) >= config["minimum_per_vowel"] for p in VOWELS
    )
    return profiles, chosen, eligible


def score(profile, eligible, query, vectors, phones):
    available = sorted(p for p in phones if p in profile and query["groups"].get(p))
    reason = None
    if not eligible:
        reason = "insufficient_enrollment"
    elif not set(VOWELS) <= set(available):
        reason = "insufficient_query_vowels"
    elif query["status"] != "complete":
        reason = "episode_not_complete"
    if reason:
        return {
            "query_id": query["id"],
            "status": "no_score",
            "score": None,
            "reason": reason,
        }
    parts = {
        p: float(vectors[query["groups"][p]].mean(axis=0) @ profile[p])
        for p in available
    }
    return {
        "query_id": query["id"],
        "status": "scored",
        "score": float(np.mean(list(parts.values()))),
        "phone_scores": parts,
        "used_phones": available,
    }


def differences(before, after):
    values = []
    for old, new in zip(before, after, strict=True):
        if old["query_id"] != new["query_id"]:
            raise ValueError("Queries were reordered")
        if old["score"] is not None and new["score"] is not None:
            values.append(
                {"query_id": old["query_id"], "difference": new["score"] - old["score"]}
            )
    return {
        "paired_queries": len(values),
        "values": values,
        "mean_difference": float(np.mean([v["difference"] for v in values]))
        if values
        else None,
        "max_absolute_difference": max(
            (abs(v["difference"]) for v in values), default=None
        ),
    }


def main():
    config = read(BASE / "config.json")
    source, output = ROOT / config["source_run"], ROOT / config["output_run"]
    source_freeze = read(source / "protocol-freeze.json")
    for name, checksum in source_freeze["files"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    for key in ("selection_seed",):
        assert config[key] == source_freeze["config"][key]
    assert (
        config["minimum_per_vowel"]
        == source_freeze["config"]["enrollment_minimum_per_vowel"]
    )
    assert (
        config["limit_per_phone"]
        == source_freeze["config"]["enrollment_limit_per_phone"]
    )
    episodes = sorted(
        [(p, read(p)) for p in (source / "episodes").glob("*/episode.json")],
        key=lambda pair: pair[1]["created_at"],
    )
    enrollments = [
        (p, e)
        for p, e in episodes
        if e["role"] == "enrollment" and e["status"] == "complete"
    ]
    if len(enrollments) != 1:
        raise ValueError("This diagnostic expects exactly one complete enrollment")
    enrollment_path, enrollment = enrollments[0]
    queries = [(p, e) for p, e in episodes if e["role"] == "query"]
    assert all(e["participant"] == enrollment["participant"] for _, e in queries)
    assert len(enrollment["clips"]) == max(config["enrollment_prefixes"]) == 20
    pins = {
        str(source / "protocol-freeze.json"): digest(source / "protocol-freeze.json")
    }
    for path in [
        *BASE.glob("*.py"),
        *BASE.glob("*.json"),
        BASE / "README.md",
        *BASE.glob("tests/*.py"),
        source / "query-five-review.json",
        enrollment_path.parent / "profile.json",
    ]:
        pins[str(path.relative_to(ROOT))] = digest(path)
    for path, e in episodes:
        pins[str(path.relative_to(ROOT))] = digest(path)
        if e["role"] not in ("enrollment", "query"):
            continue
        for clip in e["clips"]:
            rp = path.parent / clip["id"] / "result.json"
            result = read(rp)
            pins[str(rp.relative_to(ROOT))] = digest(rp)
            for name, checksum in result["files"].items():
                assert digest(ROOT / name) == checksum
                pins[name] = checksum
    frozen = {
        "config": config,
        "files": pins,
        "observed_data": True,
        "source_episode_ids": [e["id"] for _, e in episodes],
    }
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "design-freeze.json"
    if freeze_path.exists():
        if read(freeze_path) != frozen:
            raise ValueError("Analysis or inputs changed: use a new output run")
    else:
        write(freeze_path, frozen)

    vectors, features, rows_by_clip = [], [], {}
    seen_hashes = set()
    for path, e in [(enrollment_path, enrollment), *queries]:
        for clip in e["clips"]:
            assert clip["status"] == "processed" and clip["origin"] == "microphone"
            assert clip["pcm_sha256"] not in seen_hashes
            seen_hashes.add(clip["pcm_sha256"])
            directory = path.parent / clip["id"]
            rows = list(r.s.rows(directory / "segments.jsonl"))
            emb = np.load(directory / "embeddings.npy", allow_pickle=False)
            feat = np.load(directory / "features.npy", allow_pickle=False)
            assert emb.shape == feat.shape == (len(rows), 128)
            assert np.isfinite(emb).all() and np.isfinite(feat).all()
            assert dict(Counter(row["phoneme"] for row in rows)) == clip["counts"]
            selected = []
            for row, vector, feature in zip(rows, emb, feat, strict=True):
                selected.append({**row, "cache_index": len(vectors)})
                vectors.append(vector)
                features.append(feature)
            rows_by_clip[clip["id"]] = selected
    matrix = np.asarray(vectors, dtype=np.float32)
    engine = r.Runtime(source_freeze["config"])
    replay = r.s.original.model_embeddings(
        engine.model, np.asarray(features, dtype=np.float32)
    )
    embedding_error = float(np.abs(matrix - replay).max())
    assert embedding_error < 1e-6
    query_inputs = []
    for _, e in queries:
        groups = {}
        for clip in e["clips"]:
            for row in rows_by_clip[clip["id"]]:
                groups.setdefault(row["phoneme"], []).append(row["cache_index"])
        query_inputs.append({**e, "groups": groups})
    prefix_profiles, selected_profiles, eligible, prefix_results = {}, {}, {}, {}
    for n in config["enrollment_prefixes"]:
        clips = enrollment["clips"][:n]
        rows = [row for c in clips for row in rows_by_clip[c["id"]]]
        profile, selected, ok = make_profile(rows, matrix, config)
        prefix_profiles[n], selected_profiles[n], eligible[n] = profile, selected, ok
        prefix_results[n] = {
            "recording_seconds": sum(c["seconds"] for c in clips),
            "vowel_counts": {
                v: sum(row["phoneme"] == v for row in rows) for v in VOWELS
            },
            "selected_vowel_counts": {v: len(selected.get(v, [])) for v in VOWELS},
            "registration_eligible": ok,
            "profile_phones": sorted(profile),
        }
    common = sorted(
        set.intersection(*(set(p) for n, p in prefix_profiles.items() if eligible[n]))
    )
    assert set(VOWELS) <= set(common)
    saved = read(enrollment_path.parent / "profile.json")
    for p, values in saved["phones"].items():
        assert values["segments"] == [
            row["segment_id"] for row in selected_profiles[20][p]
        ]
        assert np.array_equal(
            prefix_profiles[20][p], np.asarray(values["vector"], dtype=np.float32)
        )
    score_error = 0.0
    for n in config["enrollment_prefixes"]:
        profile = prefix_profiles[n]
        for mode, phones in (("native", profile), ("common", common)):
            scores = [
                score(profile, eligible[n], q, matrix, phones) for q in query_inputs
            ]
            prefix_results[n][mode] = scores
            # Independent float64 segment-dot averaging, without query renormalization.
            for q, value in zip(query_inputs, scores, strict=True):
                if value["score"] is None:
                    continue
                parts = []
                for p in value["used_phones"]:
                    ids = [row["cache_index"] for row in selected_profiles[n][p]]
                    mean = matrix[ids].astype(np.float64).mean(axis=0)
                    mean /= np.linalg.norm(mean)
                    parts.append(
                        float((matrix[q["groups"][p]].astype(np.float64) @ mean).mean())
                    )
                error = abs(value["score"] - float(np.mean(parts)))
                score_error = max(score_error, error)
                assert error < 1e-6
        prefix_results[n]["vowel_profile_cosines_to_20"] = (
            {
                v: float(
                    np.dot(
                        profile[v].astype(np.float64),
                        prefix_profiles[20][v].astype(np.float64),
                    )
                    / (
                        np.linalg.norm(profile[v].astype(np.float64))
                        * np.linalg.norm(prefix_profiles[20][v].astype(np.float64))
                    )
                )
                for v in VOWELS
                if v in profile
            }
            if eligible[n]
            else None
        )
    baseline = {
        e["episode_id"]: e["diagnostic_score"]["score"]
        for e in read(source / "query-five-review.json")["episodes"]
    }
    for value in prefix_results[20]["native"]:
        old = baseline[value["query_id"]]
        assert (old is None) == (value["score"] is None)
        if old is not None:
            assert abs(old - value["score"]) < 1e-7
    for n in config["enrollment_prefixes"]:
        prefix_results[n]["native_difference_from_20"] = differences(
            prefix_results[20]["native"], prefix_results[n]["native"]
        )
        prefix_results[n]["common_difference_from_20"] = differences(
            prefix_results[20]["common"], prefix_results[n]["common"]
        )
    phone_diagnostics = {}
    for p in prefix_profiles[20]:
        observations = []
        for value in prefix_results[20]["native"]:
            if value["score"] is None or p not in value["phone_scores"]:
                continue
            parts = value["phone_scores"]
            without = float(np.mean([v for name, v in parts.items() if name != p]))
            observations.append(
                {
                    "query_id": value["query_id"],
                    "score": parts[p],
                    "score_change_if_omitted": without - value["score"],
                }
            )
        phone_diagnostics[p] = {
            "enrollment_segments": len(selected_profiles[20][p]),
            "query_observations": observations,
            "mean_phone_score": float(np.mean([v["score"] for v in observations]))
            if observations
            else None,
            "max_absolute_omission_effect": max(
                (abs(v["score_change_if_omitted"]) for v in observations), default=None
            ),
        }
    result = {
        "scope": config["scope"],
        "common_registered_phones": common,
        "prefixes": prefix_results,
        "phone_diagnostics": phone_diagnostics,
        "source_query_status_counts": dict(Counter(e["status"] for _, e in queries)),
        "query_episodes_with_audio": sum(bool(e["clips"]) for _, e in queries),
        "query_episodes_total": len(queries),
        "checks": {
            "source_frozen_files": len(source_freeze["files"]),
            "analysis_pinned_files": len(pins),
            "embedding_max_error": embedding_error,
            "float64_score_max_error": score_error,
            "original_profile_and_scores_replayed": True,
        },
        "threshold": None,
        "far": None,
        "frr": None,
    }
    write(output / "results.json", result)
    for n, values in prefix_results.items():
        print(
            json.dumps(
                {
                    "clips": n,
                    **{
                        k: values[k]
                        for k in (
                            "registration_eligible",
                            "recording_seconds",
                            "vowel_counts",
                            "common_difference_from_20",
                            "vowel_profile_cosines_to_20",
                        )
                    },
                },
                ensure_ascii=False,
            )
        )
    print(json.dumps(result["checks"]))


if __name__ == "__main__":
    main()
