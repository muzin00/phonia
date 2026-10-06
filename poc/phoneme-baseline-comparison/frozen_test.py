"""Freeze checked validation results and test metadata before test inference."""

from __future__ import annotations

import json
import shutil
import sys
import wave
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pilot import AudioCache, prepare, read_rows
from pilot import load_inputs as load_preflight
from smoke import BASE, ROOT, check_model_snapshot, sha256_file, write_json
from utterance_windows import describe_query_window
from validation import preflight


def checked_file(path, checksum):
    if sha256_file(path) != checksum:
        raise ValueError(f"frozen artifact changed: {path}")


def prepare_test():
    checked = prepare()  # Also verifies every Phase 6 implementation reference.
    config_path = BASE / "config/test.json"
    config = json.loads(config_path.read_text())
    protocol = checked["protocol"]
    if (
        config["split"] != "test"
        or config["protocol_sha256"] != checked["protocol_sha256"]
    ):
        raise ValueError("test protocol mismatch")
    validation = ROOT / config["validation_run"]
    load_preflight(validation)
    for name, checksum in config["validation_artifacts_sha256"].items():
        checked_file(validation / name, checksum)
    report = json.loads((validation / "validation-report.json").read_text())
    if report["status"] != "completed" or report["test_audio_or_scores_read"]:
        raise ValueError("requires successful validation before test")
    phase6 = ROOT / protocol["phase6_reference"]["run"]
    execution = json.loads((phase6 / "execution.json").read_text())
    inputs = {}
    for name in ("enrollment", "queries"):
        path = f"inputs/test/{name}.jsonl"
        checked_file(phase6 / path, execution["files"][path])
        inputs[name] = read_rows(phase6 / path)
    checked_file(phase6 / "trials/test.jsonl", execution["files"]["trials/test.jsonl"])
    inputs["trials"] = read_rows(phase6 / "trials/test.jsonl")
    inputs["queries"].sort(key=lambda q: q["query_id"])
    inputs["enrollment"].sort(key=lambda e: (e["user_id"], e["enrollment_count"]))
    refs = protocol["metadata_readiness_inputs"]
    split = json.loads((ROOT / refs["speaker_split"]["path"]).read_text())
    speakers = sorted(split["speaker_splits"]["test"])
    if (
        len(speakers),
        len(inputs["queries"]),
        len(inputs["trials"]),
        len(inputs["enrollment"]),
    ) != (15, 1200, 54000, 45):
        raise ValueError("test population mismatch")
    all_sources = {
        r["source_file"]: r
        for r in read_rows(ROOT / refs["utterance_manifest"]["path"])
    }
    filenames = {q["source_file"] for q in inputs["queries"]} | {
        s["source_file"] for e in inputs["enrollment"] for s in e["segments"]
    }
    sources = {name: all_sources[name] for name in sorted(filenames)}
    for source in sources.values():
        if source["split"] != "test" or source["speaker_id"] not in speakers:
            raise ValueError("test source assignment mismatch")
    enrolled = set()
    enrollment_keys = set()
    for e in inputs["enrollment"]:
        key = e["user_id"], e["enrollment_count"]
        if (
            key in enrollment_keys
            or e["split"] != "test"
            or key[0] not in speakers
            or key[1] not in (1, 5, 10)
        ):
            raise ValueError("invalid test enrollment")
        enrollment_keys.add(key)
        for v in "aiueo":
            if sum(s["vowel"] == v for s in e["segments"]) != key[1]:
                raise ValueError("enrollment quantity mismatch")
        for segment in e["segments"]:
            source = sources[segment["source_file"]]
            if (
                source["speaker_id"] != key[0]
                or source["evaluation_role"] != "enrollment"
                or source["source_sha256"] != segment["source_sha256"]
            ):
                raise ValueError("enrollment provenance mismatch")
            enrolled.add(segment["source_sha256"])
    by_query = {}
    for q in inputs["queries"]:
        source = sources[q["source_file"]]
        if (
            q["query_id"] in by_query
            or q["split"] != "test"
            or q["speaker_id"] not in speakers
            or q["source_sha256"] in enrolled
            or any(
                q[a] != source[b]
                for a, b in (
                    ("speaker_id", "speaker_id"),
                    ("role", "evaluation_role"),
                    ("source_sha256", "source_sha256"),
                    ("utterance_id", "utterance_id"),
                )
            )
        ):
            raise ValueError("test query provenance/overlap mismatch")
        by_query[q["query_id"]] = q
    for speaker in speakers:
        for role, number in (("verification", 50), ("cross_text_verification", 30)):
            if (
                sum(
                    q["speaker_id"] == speaker and q["role"] == role
                    for q in inputs["queries"]
                )
                != number
            ):
                raise ValueError("test speaker/role count mismatch")
    expected = {(q, s, n) for q in by_query for s in speakers for n in (1, 5, 10)}
    seen, ids = set(), set()
    for t in inputs["trials"]:
        q = by_query[t["query_id"]]
        key = t["query_id"], t["claimed_speaker_id"], t["enrollment_count"]
        if (
            key in seen
            or t["trial_id"] in ids
            or key not in expected
            or any(t[k] != q[k] for k in ("split", "speaker_id", "role"))
            or type(t["is_genuine"]) is not bool
            or t["is_genuine"] != (t["speaker_id"] == t["claimed_speaker_id"])
        ):
            raise ValueError("test trial identity/label mismatch")
        seen.add(key)
        ids.add(t["trial_id"])
    if seen != expected:
        raise ValueError("missing test trial")
    windows = [
        {
            **describe_query_window(q, sources[q["source_file"]], c["maximum_seconds"]),
            "condition_id": c["id"],
        }
        for q in inputs["queries"]
        for c in protocol["query_windows"]["conditions"]
    ]
    return {
        "schema_version": 1,
        "purpose": "test_fixed_validation_thresholds",
        "protocol": protocol,
        "protocol_sha256": checked["protocol_sha256"],
        "config": config,
        "config_sha256": sha256_file(config_path),
        "speakers": speakers,
        "sources": sources,
        "windows": windows,
        **inputs,
    }


def freeze(run, inputs, model_dir, verification):
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "verification.json", verification)
    if verification["status"] != "passed":
        raise ValueError("freeze requires passed implementation checks")
    sys.path.insert(0, str(ROOT / "poc/phoneme-speaker-encoder/scripts"))
    from weighted_bootstrap import speaker_draws

    indices, counts = speaker_draws(
        len(inputs["speakers"]),
        inputs["protocol"]["bootstrap"]["replicates"],
        inputs["protocol"]["bootstrap"]["seed"],
    )
    for name, data in (
        ("bootstrap-indices.npy", indices),
        ("bootstrap-counts.npy", counts),
    ):
        with (run / name).open("xb") as stream:
            np.save(stream, data, allow_pickle=False)
    preflight(run, inputs)
    validation = ROOT / inputs["config"]["validation_run"]
    phase6 = ROOT / inputs["protocol"]["phase6_reference"]["run"]
    pinned = {}

    def pin(path):
        pinned[str(path.relative_to(ROOT))] = sha256_file(path)

    # All numerical validation artifacts, plus its successful independent audit.
    for path in sorted(validation.rglob("*")):
        if path.is_file():
            pin(path)
    for path in (
        BASE / "comparison-design.md",
        BASE / "model-selection.md",
        BASE / "config/comparison-protocol.json",
        BASE / "config/test.json",
    ):
        pin(path)
    for p, checksum in json.loads((run / "preflight.json").read_text())[
        "implementation_sha256"
    ].items():
        pinned[p] = checksum
        dest = run / "source-snapshot" / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, dest)
    # Test score contents are intentionally not read until freeze exists. Their
    # expected checksums come from the already-pinned Phase 6 execution manifest.
    execution = json.loads((phase6 / "execution.json").read_text())
    baseline = {
        name: checksum
        for name, checksum in execution["files"].items()
        if name == "scores/test.jsonl" or name.startswith("profiles/test/")
    }
    for name in (
        "inputs/test/enrollment.jsonl",
        "inputs/test/queries.jsonl",
        "trials/test.jsonl",
        "execution.json",
        "evaluation-plan.json",
    ):
        pin(phase6 / name)
    bundle = ROOT / inputs["protocol"]["models"]["vowel"]["bundle"]
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            pin(path)
    check_model_snapshot(
        model_dir, json.loads((BASE / "config/embedding-smoke.json").read_text())
    )
    for path in model_dir.iterdir():
        if path.is_file():
            pin(path)
    for name, source in inputs["sources"].items():
        path = (ROOT / name).resolve()
        if (
            Path(name).is_absolute()
            or not path.is_relative_to(ROOT)
            or sha256_file(path) != source["source_sha256"]
        ):
            raise ValueError("test source path/checksum mismatch before freeze")
        pinned[name] = source["source_sha256"]
        with wave.open(str(path), "rb") as wav:
            if (
                wav.getframerate(),
                wav.getnchannels(),
                wav.getsampwidth(),
                wav.getcomptype(),
                wav.getnframes(),
            ) != (24000, 1, 2, "NONE", source["frame_count"]):
                raise ValueError("test WAV format mismatch before freeze")
    write_json(
        run / "resource-plan.json",
        {
            "scope": "test_and_final_report",
            "score_slots": 810000,
            "inference": inputs["protocol"]["execution"]["inference"],
            "repeats": 3,
            "audio_cache_maximum_sources": 16,
            "estimated_cpu_minutes": [10, 30],
            "estimated_peak_process_memory_gib": 4,
            "estimated_artifact_disk_gib": 2,
            "basis": "completed_validation_worker_times_and_artifact_sizes",
            "bootstrap": inputs["protocol"]["bootstrap"],
        },
    )
    pin(ROOT / "poc/phoneme-speaker-encoder/scripts/weighted_bootstrap.py")
    for name in (
        "inputs.json",
        "preflight.json",
        "resource-plan.json",
        "verification.json",
        "bootstrap-indices.npy",
        "bootstrap-counts.npy",
    ):
        pin(run / name)
    for path in (run / "source-snapshot").rglob("*"):
        if path.is_file():
            pin(path)
    write_json(
        run / "execution-freeze.json",
        {
            "schema_version": 1,
            "status": "execution_frozen_before_test",
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol_sha256": inputs["protocol_sha256"],
            "validation_thresholds_sha256": sha256_file(
                validation / "validation-thresholds.json"
            ),
            "test_model_inference_or_score_contents_read": False,
            "test_pcm_for_inference_read": False,
            "source_hash_and_wav_header_checked": True,
            "files": pinned,
            "phase6_baseline_expected_sha256": baseline,
            "threshold_recalibration_allowed": False,
        },
    )
    return load_inputs(run)


def load_inputs(run):
    inputs = load_preflight(run)
    frozen = json.loads((run / "execution-freeze.json").read_text())
    if (
        frozen["status"] != "execution_frozen_before_test"
        or inputs["config"]["split"] != "test"
        or frozen["threshold_recalibration_allowed"]
    ):
        raise ValueError("test requires execution freeze")
    # Validate every pinned file, including source snapshots and WAVs, on entry
    # and exit. Post-freeze artifacts are checked by their worker manifest.
    for path, checksum in frozen["files"].items():
        checked_file(ROOT / path, checksum)
    phase6 = ROOT / inputs["protocol"]["phase6_reference"]["run"]
    for name, checksum in frozen["phase6_baseline_expected_sha256"].items():
        checked_file(phase6 / name, checksum)
    return inputs


class BoundedAudio(AudioCache):
    def __init__(self, sources, maximum):
        super().__init__(sources)
        if any(s["split"] != "test" for s in sources.values()):
            raise ValueError("frozen test worker cannot open other splits")
        self.cache, self.checked, self.maximum = OrderedDict(), set(), maximum

    def get(self, filename):
        source = self.sources[filename]
        if filename not in self.cache:
            path = (ROOT / filename).resolve()
            if Path(filename).is_absolute() or not path.is_relative_to(ROOT):
                raise ValueError("source outside audio root")
            checked_file(path, source["source_sha256"])
            with wave.open(str(path), "rb") as wav:
                if (
                    wav.getframerate(),
                    wav.getnchannels(),
                    wav.getsampwidth(),
                    wav.getcomptype(),
                    wav.getnframes(),
                ) != (24000, 1, 2, "NONE", source["frame_count"]):
                    raise ValueError("source PCM metadata mismatch")
                self.cache[filename] = np.frombuffer(
                    wav.readframes(wav.getnframes()), dtype="<i2"
                ).copy()
            self.checked.add(filename)
            if len(self.cache) > self.maximum:
                self.cache.popitem(last=False)
        self.cache.move_to_end(filename)
        return self.cache[filename]

    def require_unchanged(self):
        for name in self.checked:
            checked_file(ROOT / name, self.sources[name]["source_sha256"])
