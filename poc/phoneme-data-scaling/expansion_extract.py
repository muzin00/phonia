"""Resumable extraction of reviewed additions; never trains a model."""

import argparse
import json
import subprocess
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

from expansion_common import (
    BASE,
    CONFIG,
    ROOT,
    atomic,
    fixed,
    progress,
    read,
    sha,
    verify,
)


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
        "utterance_id": row["corpus"] + "_" + row["sha256"][:24],
        "speaker_id": row["speaker_id"],
        "origin": row["corpus"] + "_data_expansion",
        "role": "training",
        "session_id": None,
        "original_audio_sha256": row["sha256"],
        "original_audio_path": row["path"],
    }


def official_segment(segment, row):
    assert row["split"] == "train"
    assert segment["speaker_id"] == row["speaker_id"]
    return {
        **segment,
        "extraction_split": segment["split"],
        "split": "train",
        "role": "training",
        "corpus": row["corpus"],
        "addition_group": row["addition_group"],
        "input_id": row["input_id"],
        "source_position": row["position"],
        "original_mp3": row["local_path"],
        "original_mp3_sha256": row["sha256"],
    }


def extract(config, run, worker=0):
    verify(run)
    sys.path = [p for p in sys.path if Path(p).resolve() != BASE]
    sys.path.insert(0, str(BASE.parent / "phoneme-live-recording"))
    import pyopenjtalk
    import runtime as r

    engine = r.Runtime(read(ROOT / config["runtime_freeze"])["config"])
    fixed(
        run / f"extraction-freeze-{worker}.json",
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
    rows = read(run / "inputs.json")[worker :: config["extraction_workers"]]
    counts, started = Counter(), time.monotonic()
    try:
        for index, row in enumerate(rows):
            saved = run / "extraction" / row["input_id"] / "receipt.json"
            if saved.exists():
                receipt = read(saved)
                assert receipt["input_sha256"] == row["sha256"]
                for name, digest in receipt["files"].items():
                    assert sha(ROOT / name) == digest, name
            else:
                asr_path = run / "asr" / (row["input_id"] + ".json")
                while not asr_path.exists():
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
                    reason = None
                    if len(pcm) / 24000 > config["maximum_utterance_seconds"]:
                        reason = "duration_limit"
                    elif not pyopenjtalk.g2p(asr["asr"]["text"].strip()).split():
                        reason = "empty_g2p"
                    if reason:
                        receipt = {
                            "input_sha256": row["sha256"],
                            "asr_sha256": sha(asr_path),
                            "split": row["split"],
                            "speaker_id": row["speaker_id"],
                            "status": reason,
                            "files": {str(wav.relative_to(ROOT)): sha(wav)},
                        }
                        atomic(saved, receipt)
                        counts[reason] += 1
                        progress(
                            run,
                            f"extraction-{worker}",
                            index + 1,
                            len(rows),
                            counts,
                            started,
                        )
                        continue
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
            progress(run, f"extraction-{worker}", index + 1, len(rows), counts, started)
    finally:
        subprocess.run = original


def finalize(config, run):
    import numpy as np

    verify(run)
    rows = read(run / "inputs.json")
    report = {
        "groups": {},
        "training_started": False,
        "human_review": config["human_review"],
        "source_count": len(rows),
        "preparation_sha256": sha(run / "preparation-freeze.json"),
    }
    for split in config["groups"]:
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
                if row["addition_group"] != split:
                    continue
                output = run / "extraction" / row["input_id"]
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
                        and segment["split"] == "train"
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
        report["groups"][split] = {
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
                "groups": {
                    g: {
                        k: v
                        for k, v in d.items()
                        if k
                        in ("segments", "speakers_with_segments", "utterance_status")
                    }
                    for g, d in report["groups"].items()
                },
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["extract", "finalize"])
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    if args.phase == "extract":
        assert 0 <= args.worker < config["extraction_workers"]
        extract(config, run, args.worker)
    else:
        finalize(config, run)


if __name__ == "__main__":
    main()
