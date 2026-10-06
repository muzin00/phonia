"""Shared, auditable input selection and float64 aggregation for the v2 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from smoke import BASE, ROOT, read_pcm, sha256_file, write_json
from utterance_windows import describe_query_window

VOWELS = tuple("aiueo")
METHODS = ("vowel_exact", "vowel_context20", "ecapa_whole")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(map(json.loads, stream))


def write_rows(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def select_queries(
    queries, sources, speakers, *, first_speakers=3
) -> tuple[list[dict], dict]:
    """Freeze metadata-only edge cases before any embeddings or scores are read."""
    if any(
        q["split"] != "validation" or q["speaker_id"] not in speakers for q in queries
    ):
        raise ValueError("pilot accepts validation queries only")
    if type(first_speakers) is not int or not 1 <= first_speakers <= len(speakers):
        raise ValueError("invalid pilot speaker count")
    selected, reasons = {}, {}

    def add(query, reason):
        selected[query["query_id"]] = query
        reasons.setdefault(query["query_id"], []).append(reason)

    for role in ("verification", "cross_text_verification"):
        pool = sorted(
            (q for q in queries if q["role"] == role), key=lambda q: q["utterance_id"]
        )
        for speaker in sorted(speakers)[:first_speakers]:
            candidates = [q for q in pool if q["speaker_id"] == speaker]
            if not candidates:
                raise ValueError("missing pilot speaker/role")
            add(candidates[0], f"first_speaker_first_{role}")
        duration = sorted(
            pool,
            key=lambda q: (sources[q["source_file"]]["frame_count"], q["utterance_id"]),
        )
        add(duration[0], f"shortest_{role}")
        add(duration[-1], f"longest_{role}")
        eligible = [
            q
            for q in pool
            if describe_query_window(q, sources[q["source_file"]], 1)[
                "complete_five_vowels"
            ]
        ]
        if not eligible:
            raise ValueError("no complete one-second validation example")
        add(eligible[0], f"first_complete_1s_{role}")
    return sorted(selected.values(), key=lambda q: q["query_id"]), reasons


def prepare() -> dict:
    from scripts.inspect_comparison_validation import inspect

    protocol_path = BASE / "config/comparison-protocol.json"
    config_path = BASE / "config/validation-pilot.json"
    protocol, config = (
        json.loads(protocol_path.read_text()),
        json.loads(config_path.read_text()),
    )
    if sha256_file(protocol_path) != config["protocol_sha256"]:
        raise ValueError("pilot protocol checksum mismatch")
    if config["calibrate_thresholds"] or config["read_test_audio_or_scores"]:
        raise ValueError("pilot cannot calibrate or access test audio/scores")
    if config["selection"] != {
        "split": "validation",
        "first_speakers": 3,
        "first_utterance_per_speaker_and_role": True,
        "additional_per_role": ["shortest", "longest", "first_complete_at_1s"],
        "ordering": "utterance_id_lexicographic_duration_ties_use_utterance_id",
        "claimed_speakers": "all_speakers_of_selected_queries",
    }:
        raise ValueError("unsupported pilot selection configuration")
    inspect(protocol)
    refs = protocol["metadata_readiness_inputs"]
    sources = {
        r["source_file"]: r
        for r in read_rows(ROOT / refs["utterance_manifest"]["path"])
        if r["split"] == "validation"
    }
    split = json.loads((ROOT / refs["speaker_split"]["path"]).read_text())
    all_speakers = [s for group in split["speaker_splits"].values() for s in group]
    if len(all_speakers) != len(set(all_speakers)):
        raise ValueError("speaker split overlap")
    queries, reasons = select_queries(
        read_rows(ROOT / refs["queries"]["path"]),
        sources,
        split["speaker_splits"]["validation"],
        first_speakers=config["selection"]["first_speakers"],
    )
    speakers = sorted({q["speaker_id"] for q in queries})
    enrollment = sorted(
        (
            r
            for r in read_rows(ROOT / refs["enrollment"]["path"])
            if r["user_id"] in speakers
        ),
        key=lambda r: (r["user_id"], r["enrollment_count"]),
    )
    query_ids = {q["query_id"] for q in queries}
    run = ROOT / protocol["phase6_reference"]["run"]
    for name, key in (
        ("evaluation-plan.json", "plan_sha256"),
        ("execution.json", "execution_sha256"),
    ):
        if sha256_file(run / name) != protocol["phase6_reference"][key]:
            raise ValueError(f"Phase 6 freeze checksum mismatch: {name}")
    plan = json.loads((run / "evaluation-plan.json").read_text())
    for name, expected_hash in plan["implementation_files"].items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"Phase 6 implementation changed: {name}")
    trials_path = run / "trials/validation.jsonl"
    scores_path = run / "scores/validation.jsonl"
    for path, expected in (
        (trials_path, config["phase6_trials_sha256"]),
        (scores_path, config["phase6_validation_scores_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Phase 6 reference checksum mismatch: {path}")
    trials = [
        r
        for r in read_rows(trials_path)
        if r["query_id"] in query_ids and r["claimed_speaker_id"] in speakers
    ]
    expected = {
        (q["query_id"], s, c) for q in queries for s in speakers for c in (1, 5, 10)
    }
    if (
        len(trials) != len(expected)
        or {
            (t["query_id"], t["claimed_speaker_id"], t["enrollment_count"])
            for t in trials
        }
        != expected
        or any(
            t["split"] != "validation"
            or t["is_genuine"] != (t["speaker_id"] == t["claimed_speaker_id"])
            for t in trials
        )
    ):
        raise ValueError("incomplete or mislabeled pilot trials")
    filenames = {q["source_file"] for q in queries} | {
        s["source_file"] for r in enrollment for s in r["segments"]
    }
    enrollment_hashes = {
        sources[s["source_file"]]["source_sha256"]
        for r in enrollment
        for s in r["segments"]
    }
    if any(q["source_sha256"] in enrollment_hashes for q in queries):
        raise ValueError("enrollment/query audio overlap")
    windows = [
        {
            **describe_query_window(q, sources[q["source_file"]], c["maximum_seconds"]),
            "condition_id": c["id"],
        }
        for c in protocol["query_windows"]["conditions"]
        for q in queries
    ]
    return {
        "schema_version": 1,
        "purpose": config["purpose"],
        "config": config,
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": sha256_file(config_path),
        "speakers": speakers,
        "selection_reasons": reasons,
        "queries": queries,
        "enrollment": enrollment,
        "sources": {name: sources[name] for name in sorted(filenames)},
        "windows": windows,
        "trials": trials,
    }


def unit(vector) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(result)
    if (
        result.ndim != 1
        or not np.isfinite(result).all()
        or not np.isfinite(norm)
        or norm < 1e-12
    ):
        raise ValueError("invalid embedding or mean")
    return result / norm


def mean_profile(vectors) -> np.ndarray:
    if not vectors:
        raise ValueError("empty enrollment")
    for vector in vectors:
        require_unit(vector)
    return unit(np.mean(vectors, axis=0, dtype=np.float64))


def require_unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    if (
        value.ndim != 1
        or not np.isfinite(value).all()
        or not np.isclose(np.linalg.norm(value), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("expected finite unit embedding")
    return value


def vowel_scores(profile: dict, grouped: dict) -> dict | None:
    if any(not grouped[v] for v in VOWELS):
        return None
    scores = {
        v: float(
            np.mean(
                [np.clip(np.dot(profile[v], e), -1, 1) for e in grouped[v]],
                dtype=np.float64,
            )
        )
        for v in VOWELS
    }
    return {
        "fused": float(np.mean([scores[v] for v in VOWELS], dtype=np.float64)),
        **scores,
    }


def score_row(inputs, method, window, trial, profile, scores) -> dict:
    return {
        **trial,
        "method_id": method,
        "condition_id": window["condition_id"],
        "score_id": digest(
            [
                inputs["protocol"]["protocol_version"],
                method,
                window["condition_id"],
                trial["trial_id"],
            ]
        ),
        "window_id": window["window_id"],
        "profile_sha256": digest(profile),
        "status": "scored" if scores is not None else "no_score",
        "reason": None if scores is not None else "missing_vowels",
        "scores": scores,
        "missing_vowels": []
        if scores is not None
        else [v for v in VOWELS if not window["counts"][v]],
    }


class AudioCache:
    def __init__(self, sources):
        self.sources, self.cache = sources, {}

    def get(self, filename):
        if filename not in self.cache:
            self.cache[filename] = np.frombuffer(
                read_pcm(self.sources[filename]), dtype="<i2"
            ).copy()
        return self.cache[filename]

    def slice(self, filename, first, last):
        if not 0 <= first < last <= self.sources[filename]["frame_count"]:
            raise ValueError("invalid embedding input frames")
        return self.get(filename)[first:last].astype(np.float32) / 32768.0

    def require_unchanged(self):
        for filename in self.cache:
            if sha256_file(ROOT / filename) != self.sources[filename]["source_sha256"]:
                raise ValueError("source audio changed during inference")


class EmbeddingCache:
    """Actual source bounds identify vectors; cap names never substitute for bounds."""

    def __init__(self, audio, identity, forward, repeats, *, already_unit=False):
        self.audio, self.identity, self.forward, self.repeats = (
            audio,
            identity,
            forward,
            repeats,
        )
        self.already_unit = already_unit
        self.vectors, self.records = {}, {}

    def embed(self, filename, first, last, segment_id):
        row = {
            "model_identity": self.identity,
            "source_file": filename,
            "source_sha256": self.audio.sources[filename]["source_sha256"],
            "input_frames": [first, last],
        }
        key = digest(row)
        if key not in self.vectors:
            pcm = self.audio.slice(filename, first, last)
            vectors = [
                np.asarray(self.forward(pcm, segment_id), dtype=np.float64)
                for _ in range(self.repeats)
            ]
            if not all(np.array_equal(vectors[0], v) for v in vectors[1:]):
                raise ValueError("repeated embedding differs")
            vector = require_unit(vectors[0]) if self.already_unit else unit(vectors[0])
            self.vectors[key] = vector
            self.records[key] = {
                **row,
                "embedding_id": key,
                "pcm_float32_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                "dimension": len(vector),
                "bitwise_equal_repeats": True,
            }
        return self.vectors[key].copy()

    def save(self, output):
        keys = sorted(self.vectors)
        np.save(
            output / "embeddings.npy",
            np.stack([self.vectors[k] for k in keys]),
            allow_pickle=False,
        )
        write_rows(output / "embedding-inputs.jsonl", [self.records[k] for k in keys])


def load_inputs(run: Path):
    inputs = json.loads((run / "inputs.json").read_text())
    frozen = json.loads((run / "preflight.json").read_text())
    if sha256_file(run / "inputs.json") != frozen["inputs_sha256"]:
        raise ValueError("pilot inputs changed")
    for name, expected in frozen["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"pilot implementation changed: {name}")
    return inputs


def save_worker(output, profiles, scores, cache, report):
    cache.audio.require_unchanged()
    cache.save(output)
    write_rows(output / "profiles.jsonl", profiles)
    write_rows(output / "scores.jsonl", scores)
    report["outputs_sha256"] = {
        name: sha256_file(output / name)
        for name in (
            "embeddings.npy",
            "embedding-inputs.jsonl",
            "profiles.jsonl",
            "scores.jsonl",
        )
    }
    report["unique_embeddings"] = len(cache.vectors)
    report["all_repeats_bitwise_equal"] = True
    report["source_audio_unchanged"] = True
    write_json(output / "report.json", report)
