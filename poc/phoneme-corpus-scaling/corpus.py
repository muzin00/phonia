"""Checkpointed corpus preparation with bounded alignment and explicit failures."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

from acquire import BASE, ROOT, read, sha

CONFIG = BASE / "corpus-config.json"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def fixed(path, value):
    if path.exists():
        assert read(path) == value, f"Frozen input changed: {path}"
    else:
        atomic(path, value)


def inputs(config):
    rows = read(ROOT / config["source_run"] / "download.json")
    assert len(rows) == 9000 and len({r["sha256"] for r in rows}) == 9000
    assert Counter(r["split"] for r in rows) == {
        "train": 7000,
        "validation": 1000,
        "test": 1000,
    }
    # Process train first; ordering never depends on ASR/quality/verification scores.
    return sorted(
        rows,
        key=lambda r: (
            {"train": 0, "validation": 1, "test": 2}[r["split"]],
            r["speaker_id"],
            r["position"],
        ),
    )


def verify(run):
    frozen = read(run / "preparation-freeze.json")
    for name, digest in frozen["files"].items():
        assert sha(ROOT / name) == digest, name
    return frozen


def prepare(config, run):
    source = ROOT / config["source_run"]
    review = ROOT / config["review_run"]
    files = [
        CONFIG,
        BASE / "corpus.py",
        BASE / "corpus_asr.py",
        BASE / "acquire.py",
        source / "selection-freeze.json",
        source / "download.json",
        ROOT / config["runtime_freeze"],
        ROOT / config["asr_model_metadata"],
        review / "alignment-resource-policy.json",
    ]
    # Pin local pipeline code without traversing virtual environments.
    for directory in ROOT.joinpath("poc").iterdir():
        if directory.is_dir() and directory != BASE:
            for folder in [
                directory,
                directory / "scripts",
                directory / "phase3_data",
                directory / "phase3_train",
            ]:
                if folder.is_dir():
                    files.extend(folder.glob("*.py"))
    fixed(run / "inputs.json", inputs(config))
    files.append(run / "inputs.json")
    fixed(
        run / "preparation-freeze.json",
        {
            "config": config,
            "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
            "review_answers_synthesized": False,
            "training_started": False,
        },
    )
    print(
        "Frozen 9000 source clips: train 7000 / validation 1000 / test 1000", flush=True
    )


def progress(run, stage, processed, total, counts, started):
    state = {
        "stage": stage,
        "processed": processed,
        "total": total,
        "counts": dict(counts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "updated_at_epoch": time.time(),
        "pid": os.getpid(),
    }
    atomic(run / f"{stage}-progress.json", state)
    if processed % 25 == 0 or processed == total:
        print(json.dumps(state), flush=True)


def bounded_julius(original, config):
    def call(command, *args, **kwargs):
        if isinstance(command, list) and Path(command[0]).name == "julius":
            if "input" in kwargs:
                with wave.open(kwargs["input"].strip()) as wav:
                    seconds = wav.getnframes() / wav.getframerate()
                if seconds > config["maximum_utterance_seconds"]:
                    raise ValueError("alignment_duration_limit")
            kwargs["timeout"] = config["julius_timeout_seconds"]
        return original(command, *args, **kwargs)

    return call


def source_metadata(row):
    return {
        "utterance_id": "jtube_" + row["sha256"][:24],
        "speaker_id": row["speaker_id"].lower(),
        "origin": "jtubespeech_asv_corpus_scaling",
        "role": "training" if row["split"] == "train" else "held_out",
        "session_id": None,
        "original_audio_sha256": row["sha256"],
        "original_audio_path": row["path"],
    }


def official_segment(segment, row):
    assert row["split"] in ("train", "validation", "test")
    assert segment["speaker_id"] == row["speaker_id"].lower()
    return {
        **segment,
        "extraction_split": segment["split"],
        "split": row["split"],
        "role": "training" if row["split"] == "train" else "held_out",
        "corpus": "jtubespeech_asv",
        "source_position": row["position"],
        "original_mp3": row["local_path"],
        "original_mp3_sha256": row["sha256"],
    }


def extract(config, run, producer=None):
    verify(run)
    sys.path.insert(0, str(BASE.parent / "phoneme-live-recording"))
    import runtime as r

    engine = r.Runtime(read(ROOT / config["runtime_freeze"])["config"])
    fixed(
        run / "extraction-freeze.json",
        {
            "configuration": engine.extract_config,
            "runtime_files": engine.fingerprints(),
            "phones": list(engine.phones),
            "preparation_sha256": sha(run / "preparation-freeze.json"),
            "normalization_policy": "reuse existing train-only statistics, identical for all comparison arms",
        },
    )
    original = subprocess.run
    subprocess.run = bounded_julius(original, config)
    rows = read(run / "inputs.json")
    counts, started = Counter(), time.monotonic()
    try:
        for index, row in enumerate(rows):
            saved = run / "extraction" / row["sha256"] / "receipt.json"
            if saved.exists():
                receipt = read(saved)
                assert receipt["input_sha256"] == row["sha256"]
                for name, digest in receipt["files"].items():
                    assert sha(ROOT / name) == digest, name
            else:
                asr_path = run / "asr" / (row["sha256"] + ".json")
                while not asr_path.exists():
                    if producer is not None and producer.poll() is not None:
                        raise RuntimeError(
                            "ASR process exited before producing all results; see asr.log"
                        )
                    state_path = run / "asr-state.json"
                    if state_path.exists() and read(state_path)["status"] in (
                        "failed",
                        "complete",
                    ):
                        raise RuntimeError(
                            f"ASR ended without result for {row['path']}"
                        )
                    time.sleep(1)
                asr = read(asr_path)
                assert asr["input_sha256"] == row["sha256"]
                output = saved.parent
                output.mkdir(parents=True, exist_ok=True)
                files, detail = {}, {}
                if asr["status"] != "transcribed":
                    status = asr["status"]
                    detail["reason"] = asr.get("reason", status)
                else:
                    source = ROOT / row["local_path"]
                    assert sha(source) == row["sha256"]
                    wav = output / "input24k.wav"
                    subprocess.run(
                        [
                            "/usr/bin/afconvert",
                            "-f",
                            "WAVE",
                            "-d",
                            "LEI16@24000",
                            "-c",
                            "1",
                            str(source),
                            str(wav),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
                    with wave.open(str(wav)) as f:
                        assert (
                            f.getframerate(),
                            f.getsampwidth(),
                            f.getnchannels(),
                        ) == (24000, 2, 1)
                        pcm = r.np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
                    destination = output / "processed"
                    if (destination / "result.json").exists():
                        result = read(destination / "result.json")
                        for name, digest in result["files"].items():
                            assert sha(ROOT / name) == digest, name
                    else:
                        result = engine.process(
                            pcm,
                            destination,
                            {
                                **source_metadata(row),
                                "transcript": asr["asr"]["text"].strip(),
                            },
                        )
                    expected = read(destination / "expected.json")
                    raw_path = (
                        destination
                        / "alignment/raw"
                        / expected["speaker_id"]
                        / (expected["utterance_id"] + ".json")
                    )
                    raw = (
                        [
                            official_segment(v, row)
                            for v, _ in r.t.h.primitives.extract(
                                raw_path, expected, engine.extract_config, {}, Counter()
                            )
                        ]
                        if raw_path.exists()
                        else []
                    )
                    post = [
                        official_segment(json.loads(v), row)
                        for v in (destination / "segments.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    for segment in raw:
                        segment["exclusion_reasons"] = segment["quality_flags"] + (
                            []
                            if segment["phoneme"] in engine.phones
                            else ["outside_model_inventory"]
                        )
                    assert {v["segment_id"] for v in post} == {
                        v["segment_id"] for v in raw if not v["exclusion_reasons"]
                    }
                    fixed(output / "pre-qc.json", raw)
                    fixed(output / "post-qc.json", post)
                    status = result["status"]
                    detail = {
                        "pre_qc": len(raw),
                        "post_qc": len(post),
                        "phone_counts": dict(Counter(v["phoneme"] for v in post)),
                        "alignment_failures": [
                            v.get("reason", v["status"])
                            for v in result["alignment"]
                            if v["status"] != "success"
                        ],
                    }
                    files = {
                        str(p.relative_to(ROOT)): sha(p)
                        for p in [
                            output / "pre-qc.json",
                            output / "post-qc.json",
                            destination / "result.json",
                            destination / "features.npy",
                        ]
                    }
                receipt = {
                    "input_sha256": row["sha256"],
                    "asr_sha256": sha(asr_path),
                    "split": row["split"],
                    "speaker_id": row["speaker_id"],
                    "status": status,
                    "files": files,
                    **detail,
                }
                atomic(saved, receipt)
            counts[receipt["status"]] += 1
            progress(run, "extraction", index + 1, len(rows), counts, started)
    finally:
        subprocess.run = original


def finalize(config, run):
    import numpy as np

    verify(run)
    rows = read(run / "inputs.json")
    report = {
        "splits": {},
        "training_started": False,
        "formal_review_answers": 0,
        "source_count": len(rows),
        "preparation_sha256": sha(run / "preparation-freeze.json"),
    }
    for split in ("train", "validation", "test"):
        count, pre_count = 0, 0
        phones, states, exclusions = Counter(), Counter(), Counter()
        support, failures = defaultdict(Counter), []
        manifest = run / f"{split}-segments.jsonl"
        cache = run / f"{split}-features.f32"
        before = run / f"{split}-pre-qc.jsonl"
        with (
            manifest.with_suffix(".tmp").open("w") as stream,
            cache.with_suffix(".tmp").open("wb") as features,
            before.with_suffix(".tmp").open("w") as raw_stream,
        ):
            for row in rows:
                if row["split"] != split:
                    continue
                output = run / "extraction" / row["sha256"]
                receipt = read(output / "receipt.json")
                states[receipt["status"]] += 1
                for name, digest in receipt["files"].items():
                    assert sha(ROOT / name) == digest, name
                if receipt["status"] != "processed":
                    failures.append(
                        {
                            "source": row["path"],
                            "status": receipt["status"],
                            "reason": receipt.get(
                                "reason", receipt.get("alignment_failures")
                            ),
                        }
                    )
                if not (output / "post-qc.json").exists():
                    continue
                raw = read(output / "pre-qc.json")
                pre_count += len(raw)
                for segment in raw:
                    raw_stream.write(json.dumps(segment, ensure_ascii=False) + "\n")
                    exclusions.update(segment["exclusion_reasons"])
                post = read(output / "post-qc.json")
                matrix = np.load(output / "processed/features.npy", allow_pickle=False)
                assert matrix.shape == (len(post), 128) and np.isfinite(matrix).all()
                for local, segment in enumerate(post):
                    assert (
                        segment["cache_index"] == local
                        and segment["split"] == split
                        and not segment["quality_flags"]
                    )
                    stream.write(
                        json.dumps(
                            {**segment, "cache_index": count}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    count += 1
                    phones[segment["phoneme"]] += 1
                    support[segment["speaker_id"]][segment["phoneme"]] += 1
                features.write(matrix.astype("<f4").tobytes())
        for path in (manifest, cache, before):
            path.with_suffix(".tmp").replace(path)
        report["splits"][split] = {
            "utterance_status": dict(states),
            "pre_qc_segments": pre_count,
            "segments": count,
            "speakers_with_segments": len(support),
            "phone_counts": dict(sorted(phones.items())),
            "exclusion_reason_counts": dict(exclusions),
            "speaker_phone_counts": {s: dict(v) for s, v in support.items()},
            "failures": failures,
            "files": {
                str(p.relative_to(ROOT)): sha(p) for p in (manifest, cache, before)
            },
        }
    fixed(run / "preparation-report.json", report)
    print(
        json.dumps(
            {
                "stage": "prepared",
                "split_segments": {
                    k: v["segments"] for k, v in report["splits"].items()
                },
            }
        ),
        flush=True,
    )


def pipeline(config, run):
    verify(run)
    state_path = run / "pipeline-state.json"
    atomic(
        state_path,
        {
            "status": "running",
            "pid": os.getpid(),
            "started_at_epoch": time.time(),
            "training_started": False,
        },
    )
    atomic(run / "asr-state.json", {"status": "starting"})
    producer = None
    try:
        with (run / "asr.log").open("a") as log:
            producer = subprocess.Popen(
                [
                    str(ROOT / "artifacts/phoneme-corpus-scaling/asr-venv/bin/python"),
                    str(BASE / "corpus_asr.py"),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            extract(config, run, producer)
            if producer.wait() != 0:
                raise RuntimeError("ASR subprocess failed; see asr.log")
        finalize(config, run)
        atomic(
            state_path,
            {
                "status": "complete",
                "finished_at_epoch": time.time(),
                "training_started": False,
            },
        )
    except BaseException as error:
        if producer is not None and producer.poll() is None:
            producer.terminate()
            producer.wait()
        atomic(
            state_path,
            {
                "status": "failed",
                "error": str(error),
                "finished_at_epoch": time.time(),
                "training_started": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "pipeline", "finalize", "status"])
    args = parser.parse_args()
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    if args.phase == "status":
        for name in (
            "pipeline-state.json",
            "asr-progress.json",
            "extraction-progress.json",
        ):
            if (run / name).exists():
                print(json.dumps({name: read(run / name)}, ensure_ascii=False))
    elif args.phase == "pipeline":
        with (run / "pipeline.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(config, run)
    else:
        {"prepare": prepare, "finalize": finalize}[args.phase](config, run)


if __name__ == "__main__":
    main()
