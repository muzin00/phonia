"""Bounded-memory validation inputs, checked score records, and artifact I/O."""

from __future__ import annotations

import gzip
import json
from collections import OrderedDict
from contextlib import contextmanager

from pilot import METHODS, AudioCache, digest, prepare, read_rows
from smoke import BASE, ROOT, read_pcm, sha256_file, write_json
from utterance_windows import describe_query_window


def prepare_validation():
    # Reuse every frozen metadata, split, Phase 6 implementation and reference check.
    checked = prepare()
    protocol = checked["protocol"]
    path = BASE / "config/validation.json"
    config = json.loads(path.read_text())
    if (
        config["protocol_sha256"] != checked["protocol_sha256"]
        or config["split"] != "validation"
        or config["read_test_audio_or_scores"]
    ):
        raise ValueError("validation configuration/split mismatch")
    refs = protocol["metadata_readiness_inputs"]
    all_sources = {
        r["source_file"]: r
        for r in read_rows(ROOT / refs["utterance_manifest"]["path"])
        if r["split"] == "validation"
    }
    queries = sorted(
        read_rows(ROOT / refs["queries"]["path"]), key=lambda r: r["query_id"]
    )
    enrollment = sorted(
        read_rows(ROOT / refs["enrollment"]["path"]),
        key=lambda r: (r["user_id"], r["enrollment_count"]),
    )
    speakers = sorted({q["speaker_id"] for q in queries})
    reference = ROOT / protocol["phase6_reference"]["run"]
    trials = read_rows(reference / "trials/validation.jsonl")
    if (len(speakers), len(queries), len(trials)) != (
        config["expected_speakers"],
        config["expected_queries"],
        config["expected_trials"],
    ):
        raise ValueError("validation population mismatch")
    for name, key in (
        ("trials/validation.jsonl", "phase6_trials_sha256"),
        ("scores/validation.jsonl", "phase6_validation_scores_sha256"),
        ("thresholds/validation.json", "phase6_validation_thresholds_sha256"),
    ):
        if sha256_file(reference / name) != config[key]:
            raise ValueError(f"Phase 6 validation checksum mismatch: {name}")
    validate_trials(queries, trials, speakers)
    filenames = {q["source_file"] for q in queries} | {
        s["source_file"] for r in enrollment for s in r["segments"]
    }
    sources = {name: all_sources[name] for name in sorted(filenames)}
    enroll_hashes = {
        sources[s["source_file"]]["source_sha256"]
        for r in enrollment
        for s in r["segments"]
    }
    if any(q["source_sha256"] in enroll_hashes for q in queries):
        raise ValueError("enrollment/query audio overlap")
    windows = [
        {
            **describe_query_window(q, sources[q["source_file"]], c["maximum_seconds"]),
            "condition_id": c["id"],
        }
        for q in queries
        for c in protocol["query_windows"]["conditions"]
    ]
    return {
        "schema_version": 1,
        "purpose": config["purpose"],
        "protocol": protocol,
        "protocol_sha256": checked["protocol_sha256"],
        "config": config,
        "config_sha256": sha256_file(path),
        "speakers": speakers,
        "queries": queries,
        "enrollment": enrollment,
        "trials": trials,
        "sources": sources,
        "windows": windows,
    }


def validate_trials(queries, trials, speakers):
    by_query = {q["query_id"]: q for q in queries}
    expected = {
        (q["query_id"], s, c) for q in queries for s in speakers for c in (1, 5, 10)
    }
    seen, ids = set(), set()
    for trial in trials:
        q = by_query[trial["query_id"]]
        key = trial["query_id"], trial["claimed_speaker_id"], trial["enrollment_count"]
        if (
            key in seen
            or trial["trial_id"] in ids
            or key not in expected
            or any(trial[name] != q[name] for name in ("split", "speaker_id", "role"))
            or trial["split"] != "validation"
            or type(trial["is_genuine"]) is not bool
            or trial["is_genuine"]
            != (trial["speaker_id"] == trial["claimed_speaker_id"])
        ):
            raise ValueError("invalid validation trial identity/label")
        seen.add(key)
        ids.add(trial["trial_id"])
    if seen != expected:
        raise ValueError("missing validation trial")


def iter_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


@contextmanager
def writer(path):
    # Incomplete run directories are retained with failure.json; no file is replaced.
    with path.open("xb") as raw:
        compressed = (
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=3)
            if path.suffix == ".gz"
            else raw
        )

        def write(row):
            compressed.write(
                (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )

        try:
            yield write
        finally:
            if compressed is not raw:
                compressed.close()


class BoundedAudio(AudioCache):
    def __init__(self, sources, maximum):
        super().__init__(sources)
        self.cache, self.checked, self.maximum = OrderedDict(), set(), maximum

    def get(self, filename):
        import numpy as np

        if filename not in self.cache:
            self.cache[filename] = np.frombuffer(
                read_pcm(self.sources[filename]), dtype="<i2"
            ).copy()
            self.checked.add(filename)
            if len(self.cache) > self.maximum:
                self.cache.popitem(last=False)
        self.cache.move_to_end(filename)
        return self.cache[filename]

    def require_unchanged(self):
        for filename in self.checked:
            if sha256_file(ROOT / filename) != self.sources[filename]["source_sha256"]:
                raise ValueError("source changed during validation")


def scored_row(inputs, method, window, trial, profile_hash, scores):
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
        "profile_sha256": profile_hash,
        "status": "scored" if scores is not None else "no_score",
        "reason": None if scores is not None else "missing_vowels",
        "scores": scores,
        "missing_vowels": []
        if scores is not None
        else [v for v, n in window["counts"].items() if not n],
    }


def preflight(run, inputs):
    write_json(run / "inputs.json", inputs)
    files = {
        *BASE.glob("*.py"),
        *BASE.glob("scripts/*.py"),
        *BASE.glob("tests/*.py"),
        *BASE.glob("config/*.json"),
        BASE / "pyproject.toml",
        BASE / "uv.lock",
        ROOT / "poc/phoneme-verification-evaluation/uv.lock",
        *(ROOT / "poc/phoneme-user-registration/registration").glob("*.py"),
        *(ROOT / "poc/phoneme-verification/verification").glob("*.py"),
        *(ROOT / "poc/phoneme-speaker-encoder/phase3_data").glob("*.py"),
        ROOT / "poc/phoneme-speaker-encoder/phase3_train/models.py",
        ROOT / "poc/phoneme-speaker-encoder/phase3_train/metrics.py",
        ROOT / "poc/phoneme-verification-evaluation/evaluation/inference.py",
    }
    write_json(
        run / "preflight.json",
        {
            "inputs_sha256": sha256_file(run / "inputs.json"),
            "implementation_sha256": {
                str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)
            },
        },
    )


def audit_scores(run, inputs):
    """Check all labels, bounds, profiles, coverage and IDs while streaming rows."""
    trials = {t["trial_id"]: t for t in inputs["trials"]}
    windows = {(w["query_id"], w["condition_id"]): w for w in inputs["windows"]}
    profiles, workers, seen = {}, {}, set()
    counts = {m: 0 for m in METHODS}
    for worker in ("vowels", "ecapa"):
        root = run / worker
        report = json.loads((root / "report.json").read_text())
        if report["status"] != "completed":
            raise ValueError("validation worker not completed")
        for name, checksum in report["outputs_sha256"].items():
            if sha256_file(root / name) != checksum:
                raise ValueError("worker artifact checksum mismatch")
        workers[worker] = report
        for profile in iter_rows(root / "profiles.jsonl"):
            key = (
                profile["method_id"],
                profile["speaker_id"],
                profile["enrollment_count"],
            )
            if key in profiles:
                raise ValueError("duplicate profile")
            profiles[key] = digest(profile)
        for row in iter_rows(root / "scores.jsonl.gz"):
            trial = trials[row["trial_id"]]
            w = windows[row["query_id"], row["condition_id"]]
            method = row["method_id"]
            key = method, row["condition_id"], row["trial_id"]
            if (
                key in seen
                or method not in METHODS
                or any(row[k] != v for k, v in trial.items())
            ):
                raise ValueError("duplicate/mislabeled validation score")
            seen.add(key)
            if (
                row["score_id"]
                != digest([inputs["protocol"]["protocol_version"], *key])
                or row["window_id"] != w["window_id"]
            ):
                raise ValueError("condition score/window identity mismatch")
            if (
                row["profile_sha256"]
                != profiles[method, row["claimed_speaker_id"], row["enrollment_count"]]
            ):
                raise ValueError("profile changed across query caps")
            eligible = method == "ecapa_whole" or w["complete_five_vowels"]
            if (
                row["status"] != ("scored" if eligible else "no_score")
                or (row["scores"] is not None) != eligible
            ):
                raise ValueError("score coverage differs from metadata")
            if eligible:
                import math

                expected_kinds = (
                    {"fused"} if method == "ecapa_whole" else {"fused", *"aiueo"}
                )
                if set(row["scores"]) != expected_kinds or any(
                    not math.isfinite(v) or not -1 <= v <= 1
                    for v in row["scores"].values()
                ):
                    raise ValueError("invalid score")
            counts[method] += 1
    expected_each = len(inputs["trials"]) * len(
        inputs["protocol"]["query_windows"]["conditions"]
    )
    if any(n != expected_each for n in counts.values()) or len(profiles) != len(
        METHODS
    ) * len(inputs["enrollment"]):
        raise ValueError("validation scores/profiles missing")
    return {
        "score_slots_by_method": counts,
        "profiles": len(profiles),
        "workers": workers,
        "all_score_id_label_window_profile_coverage_checks_passed": True,
    }
