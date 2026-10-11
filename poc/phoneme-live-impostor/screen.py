"""Fixed one-target, cross-corpus screening; no live authentication decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed, name):
    return hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()


def threshold(values, target):
    if not values or not 0 <= target < 1:
        raise ValueError("Nonempty scores and target in [0, 1) required")
    values = sorted(values, reverse=True)
    allowed = math.floor(len(values) * target + 1e-12)
    cut = float(np.nextafter(values[allowed], np.inf))
    assert sum(v >= cut for v in values) <= allowed
    return cut


def combine(sequence, arm, phones):
    groups = {p: [] for p in phones}
    selected = []
    for q in sequence:
        selected.append(q)
        for p in phones:
            groups[p].extend(q["groups"].get(p, []))
        if arm == "single" or all(groups.get(p) for p in VOWELS):
            break
    return groups, selected


def score(groups, vectors, profile):
    phones = sorted(p for p in profile if groups.get(p))
    if not set(VOWELS) <= set(phones):
        return None, {}, 0.0
    parts, independent = {}, []
    for p in phones:
        x = vectors[groups[p]]
        parts[p] = float(x.mean(axis=0) @ profile[p])
        independent.append(
            float((x.astype(np.float64) @ profile[p].astype(np.float64)).mean())
        )
    result = float(np.mean(list(parts.values())))
    error = abs(result - float(np.mean(independent)))
    assert error < 1e-6
    return result, parts, error


def rates(rows, cut):
    scored = [row for row in rows if row["score"] is not None]
    accepted = [row for row in scored if row["score"] >= cut]
    return {
        "inputs": len(rows),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "accepted": len(accepted),
        "accepted_fraction_all": len(accepted) / len(rows) if rows else None,
        "accepted_fraction_scored": len(accepted) / len(scored) if scored else None,
        "speaker_labels": len({(v.get("corpus"), v["speaker_id"]) for v in rows}),
        "speaker_labels_with_accept": len(
            {(v.get("corpus"), v["speaker_id"]) for v in accepted}
        ),
    }


def freeze(config, run):
    if run.exists():
        raise ValueError("Use a fresh output directory")
    live = ROOT / config["live_run"]
    pilot = read(live / "protocol-freeze.json")
    files = dict(pilot["files"])
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name

    def pin(path, expected=None):
        checksum = sha(path)
        if expected is not None:
            assert checksum == expected, str(path)
        files[str(path.relative_to(ROOT))] = checksum

    old = read(ROOT / config["corpus_freeze"])
    pin(ROOT / config["corpus_freeze"])
    partitions = {}
    for corpus, directory in config["corpora"].items():
        source = ROOT / directory
        prior = read(source / "design-freeze.json")
        assert prior["config"]["encoder_sha256"] == pilot["config"]["encoder_sha256"]
        for key in ("minimum_frames", "maximum_frames", "minimum_rms_dbfs"):
            assert prior["extract_config"][key] == pilot["config"][key]
        pin(source / "design-freeze.json")
        for split in ("validation", "test"):
            for name in (
                "inputs.json",
                "features.npy",
                "embeddings.npy",
                "segments.jsonl",
                "utterances.jsonl",
            ):
                path = source / split / name
                pin(path, old["files"][str(path.relative_to(ROOT))])
            data = read(source / split / "inputs.json")
            assert len(data["speakers"]) == 30 and len(data["queries"]) == 900
            partitions[f"{corpus}/{split}"] = data["speakers"]
    labels = [v for group in partitions.values() for v in group]
    assert len(labels) == len(set(labels)) == 120
    episodes = sorted(
        [(p, read(p)) for p in (live / "episodes").glob("*/episode.json")],
        key=lambda x: x[1]["created_at"],
    )
    enroll = [
        (p, e)
        for p, e in episodes
        if e["role"] == "enrollment" and e["status"] == "complete"
    ]
    assert len(enroll) == 1
    enrollment = enroll[0][1]["id"]
    assert len(enroll[0][1]["clips"]) == 20
    for path, e in episodes:
        pin(path)
        if e["role"] not in ("enrollment", "query"):
            continue
        for clip in e["clips"]:
            rp = path.parent / clip["id"] / "result.json"
            pin(rp)
            for name, digest in read(rp)["files"].items():
                pin(ROOT / name, digest)
    pin(enroll[0][0].parent / "profile.json")
    pin(live / "enrollment-review.json")
    pin(live / "query-five-review.json")
    pin(live / "protocol-freeze.json")
    for path in [
        *BASE.glob("*.py"),
        *BASE.glob("*.json"),
        BASE / "README.md",
        *BASE.glob("tests/*.py"),
    ]:
        pin(path)
    design = {
        "config": config,
        "files": files,
        "live_config": pilot["config"],
        "enrollment_episode_id": enrollment,
        "live_query_ids": [e["id"] for _, e in episodes if e["role"] == "query"],
        "corpus_partitions": partitions,
        "status": "frozen_before_scoring_against_live_profile; sources and genuine scores previously observed",
    }
    write(run / "design-freeze.json", design)
    print(f"Frozen {len(files)} files and 120 corpus speaker labels", flush=True)


def verify(config, run):
    design = read(run / "design-freeze.json")
    assert design["config"] == config
    for name, digest in design["files"].items():
        assert sha(ROOT / name) == digest, name
    return design


def load_profile(config, design, engine):
    live = ROOT / config["live_run"]
    directory = live / "episodes" / design["enrollment_episode_id"]
    episode = read(directory / "episode.json")
    saved = read(directory / "profile.json")
    prior = read(live / "enrollment-review.json")
    assert sha(directory / "profile.json") == prior["profile_sha256"]
    assert (
        saved["threshold"] is None
        and saved["encoder_sha256"] == design["live_config"]["encoder_sha256"]
    )
    vectors = []
    features = []
    indices = {}
    for clip in episode["clips"]:
        d = directory / clip["id"]
        rows = list(r.s.rows(d / "segments.jsonl"))
        v = np.load(d / "embeddings.npy")
        f = np.load(d / "features.npy")
        for row, vec, feat in zip(rows, v, f, strict=True):
            indices[row["segment_id"]] = len(vectors)
            vectors.append(vec)
            features.append(feat)
    matrix = np.asarray(vectors, dtype=np.float32)
    replay = r.s.original.model_embeddings(
        engine.model, np.asarray(features, dtype=np.float32)
    )
    assert np.allclose(matrix, replay, atol=1e-6, rtol=0)
    profile = {}
    for p, data in saved["phones"].items():
        ids = [indices[i] for i in data["segments"]]
        mean = matrix[ids].mean(axis=0)
        mean /= np.linalg.norm(mean)
        assert np.array_equal(mean, np.asarray(data["vector"], dtype=np.float32))
        profile[p] = mean
    assert set(VOWELS) <= set(profile)
    return profile


def corpus_scores(config, design, corpus, split, profile, engine):
    source = ROOT / config["corpora"][corpus] / split
    data = read(source / "inputs.json")
    vectors = np.load(source / "embeddings.npy", allow_pickle=False)
    features = np.load(source / "features.npy", allow_pickle=False)
    segments = list(r.s.rows(source / "segments.jsonl"))
    metadata = {v["source_file"]: v for v in r.s.rows(source / "utterances.jsonl")}
    assert vectors.shape == features.shape == (len(segments), 128)
    assert np.isfinite(vectors).all() and np.isfinite(features).all()
    replay = r.s.original.model_embeddings(engine.model, features)
    embedding_error = float(np.abs(vectors - replay).max())
    assert embedding_error < 1e-6
    grouped = defaultdict(list)
    for q in data["queries"]:
        assert q["role"] == "verification"
        for p, ids in q["groups"].items():
            assert all(
                segments[i]["phoneme"] == p
                and segments[i]["source_file"] == q["source_file"]
                and segments[i]["speaker_id"] == q["speaker_id"]
                for i in ids
            )
        grouped[q["speaker_id"]].append(q)
    result = {arm: [] for arm in config["arms"]}
    max_error = 0.0
    for speaker, queries in grouped.items():
        assert len(queries) == 30
        queries = sorted(
            queries, key=lambda q: rank(config["sequence_seed"], q["query_id"])
        )
        for i, first in enumerate(queries):
            sequence = [
                queries[(i + j) % len(queries)] for j in range(config["maximum_clips"])
            ]
            for arm in config["arms"]:
                groups, selected = combine(sequence, arm, profile)
                value, parts, error = score(groups, vectors, profile)
                max_error = max(max_error, error)
                seconds = [metadata[q["source_file"]]["duration_sec"] for q in selected]
                result[arm].append(
                    {
                        "corpus": corpus,
                        "speaker_id": speaker,
                        "query_id": first["query_id"],
                        "source_query_ids": [q["query_id"] for q in selected],
                        "clips": len(selected),
                        "recording_seconds": sum(seconds),
                        "within_live_duration_limits": max(seconds) <= 8
                        and sum(seconds) <= 20,
                        "score": value,
                        "phone_scores": parts,
                        "vowel_counts": {p: len(groups[p]) for p in VOWELS},
                    }
                )
    return result, {
        "embedding_max_absolute_error": embedding_error,
        "float64_score_max_absolute_error": max_error,
        "cache_segments": len(segments),
    }


def genuine_scores(config, design, profile, engine):
    live = ROOT / config["live_run"]
    result = {arm: [] for arm in config["arms"]}
    max_error = 0.0
    old = {
        v["episode_id"]: v["diagnostic_score"]["score"]
        for v in read(live / "query-five-review.json")["episodes"]
    }
    for identifier in design["live_query_ids"]:
        directory = live / "episodes" / identifier
        e = read(directory / "episode.json")
        vectors = []
        features = []
        sequence = []
        for clip in e["clips"]:
            d = directory / clip["id"]
            rows = list(r.s.rows(d / "segments.jsonl"))
            v = np.load(d / "embeddings.npy")
            f = np.load(d / "features.npy")
            groups = {p: [] for p in profile}
            for row, vec, feat in zip(rows, v, f, strict=True):
                if row["phoneme"] in groups:
                    groups[row["phoneme"]].append(len(vectors))
                vectors.append(vec)
                features.append(feat)
            sequence.append({"groups": groups, "seconds": clip["seconds"]})
        matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, 128)
        if len(matrix):
            assert np.allclose(
                matrix,
                r.s.original.model_embeddings(
                    engine.model, np.asarray(features, dtype=np.float32)
                ),
                atol=1e-6,
                rtol=0,
            )
        for arm in config["arms"]:
            groups, selected = combine(sequence, arm, profile)
            value, parts, error = score(groups, matrix, profile)
            max_error = max(max_error, error)
            if e["status"] != "complete":
                value = None
            if arm == "adaptive1":
                assert (value is None) == (old[identifier] is None)
                if value is not None:
                    assert abs(value - old[identifier]) < 1e-7
            result[arm].append(
                {
                    "corpus": "live",
                    "speaker_id": e["participant"],
                    "query_id": identifier,
                    "episode_status": e["status"],
                    "clips": len(selected),
                    "recording_seconds": sum(v["seconds"] for v in selected),
                    "score": value,
                    "phone_scores": parts,
                }
            )
    return result, max_error


def describe(rows):
    values = [v["score"] for v in rows if v["score"] is not None]
    return {
        "inputs": len(rows),
        "scored": len(values),
        "unscored": len(rows) - len(values),
        "quantiles": dict(
            zip(
                ("min", "p50", "p90", "p95", "p99", "max"),
                [float(v) for v in np.quantile(values, [0, 0.5, 0.9, 0.95, 0.99, 1])],
                strict=True,
            )
        )
        if values
        else None,
        "within_live_duration_limits": sum(
            v.get("within_live_duration_limits", True) for v in rows
        ),
    }


def evaluate(config, run, split):
    design = verify(config, run)
    if split == "test":
        frozen = read(run / "threshold-freeze.json")
        assert frozen["design_sha256"] == sha(run / "design-freeze.json")
        for name, digest in frozen["files"].items():
            assert sha(ROOT / name) == digest
    engine = r.Runtime(design["live_config"])
    profile = load_profile(config, design, engine)
    pooled = {arm: [] for arm in config["arms"]}
    checks = {}
    for corpus in config["corpora"]:
        result, audit = corpus_scores(config, design, corpus, split, profile, engine)
        write(run / split / f"{corpus}-scores.json", result)
        checks[corpus] = audit
        for arm, rows in result.items():
            pooled[arm].extend(rows)
        print(f"Scored {split}/{corpus}", flush=True)
    write(run / split / "audit.json", checks)
    if split == "validation":
        cuts = {
            arm: {
                str(target): threshold(
                    [v["score"] for v in rows if v["score"] is not None], target
                )
                for target in config["reference_far_targets"]
            }
            for arm, rows in pooled.items()
        }
        summary = {
            arm: {
                "distribution": describe(rows),
                "reference_points": {
                    target: {"threshold": cut, **rates(rows, cut)}
                    for target, cut in cuts[arm].items()
                },
            }
            for arm, rows in pooled.items()
        }
        write(run / "validation-summary.json", summary)
        pins = {
            str(p.relative_to(ROOT)): sha(p)
            for p in (run / "validation").glob("*.json")
        }
        pins[str((run / "validation-summary.json").relative_to(ROOT))] = sha(
            run / "validation-summary.json"
        )
        write(
            run / "threshold-freeze.json",
            {
                "design_sha256": sha(run / "design-freeze.json"),
                "thresholds": cuts,
                "files": pins,
                "purpose": "reference_only_not_installed_in_live_server",
            },
        )
        print(json.dumps(cuts), flush=True)
        return
    genuine, error = genuine_scores(config, design, profile, engine)
    write(run / "genuine-diagnostics.json", genuine)
    summary = {}
    for arm, rows in pooled.items():
        values = [v["score"] for v in genuine[arm] if v["score"] is not None]
        minimum = min(values) if values else None
        points = {}
        for target, cut in frozen["thresholds"][arm].items():
            points[target] = {
                "threshold": cut,
                "corpus_pooled": rates(rows, cut),
                "by_corpus": {
                    c: rates([v for v in rows if v["corpus"] == c], cut)
                    for c in config["corpora"]
                },
                "known_genuine_all_started": rates(genuine[arm], cut),
                "known_genuine_with_audio": rates(
                    [v for v in genuine[arm] if v["clips"]], cut
                ),
            }
        summary[arm] = {
            "corpus_distribution": describe(rows),
            "by_corpus": {
                c: describe([v for v in rows if v["corpus"] == c])
                for c in config["corpora"]
            },
            "known_genuine_distribution": describe(genuine[arm]),
            "reference_points": points,
            "overlap_at_known_genuine_minimum": {
                "boundary": minimum,
                "diagnostic_only": True,
                **rates(rows, minimum),
            }
            if minimum is not None
            else None,
        }
    write(
        run / "results.json",
        {
            "config": config,
            "results": summary,
            "checks": checks,
            "genuine_float64_max_error": error,
            "live_authentication_threshold_changed": False,
            "formal_authentication_accuracy_established": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "validation", "test"))
    args = parser.parse_args()
    config = read(BASE / "config.json")
    run = ROOT / config["run_directory"]
    if args.phase == "freeze":
        freeze(config, run)
    else:
        evaluate(config, run, args.phase)


if __name__ == "__main__":
    main()
