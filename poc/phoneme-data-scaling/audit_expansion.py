"""Audit complete additions and decoded-PCM overlap before any training."""

import hashlib
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from expansion_common import (
    BASE,
    CONFIG,
    ROOT,
    check_files,
    fixed,
    pin,
    read,
    sha,
    verify,
)


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def pcm_hash(path, expected):
    assert sha(path) == expected, path
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (
            24000,
            2,
            1,
        ), path
        content = wav.readframes(wav.getnframes())
    return hashlib.sha256(content).hexdigest()


def main():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    verify(run)
    assert read(run / "pipeline-state.json")["status"] == "complete"
    source_rows = read(run / "inputs.json")
    assert len(source_rows) == 11200
    for worker in range(config["extraction_workers"]):
        frozen = read(run / f"extraction-freeze-{worker}.json")
        check_files(frozen["runtime_files"])
    report = read(run / "preparation-report.json")
    files = {}
    additions = {}
    totals = Counter()
    for group in config["groups"]:
        check_files(report["groups"][group]["files"])
        files.update(report["groups"][group]["files"])
        manifest = list(rows(run / f"{group}-segments.jsonl"))
        cache = np.memmap(
            run / f"{group}-features.f32",
            mode="r",
            dtype="<f4",
            shape=(len(manifest), 128),
        )
        offset = 0
        for source in source_rows:
            if source["addition_group"] != group:
                continue
            assert sha(ROOT / source["local_path"]) == source["sha256"]
            out = run / "extraction" / source["input_id"]
            receipt = read(out / "receipt.json")
            assert receipt["input_sha256"] == source["sha256"]
            assert receipt["asr_sha256"] == sha(
                run / "asr" / (source["input_id"] + ".json")
            )
            assert (
                receipt["speaker_id"] == source["speaker_id"]
                and receipt["split"] == "train"
            )
            check_files(receipt["files"])
            totals[receipt["status"]] += 1
            if not (out / "post-qc.json").exists():
                continue
            result = read(out / "processed/result.json")
            check_files(result["files"])
            before = read(out / "pre-qc.json")
            after = read(out / "post-qc.json")
            assert {v["segment_id"] for v in after} == {
                v["segment_id"] for v in before if not v["exclusion_reasons"]
            }
            matrix = np.load(out / "processed/features.npy", allow_pickle=False)
            assert matrix.shape == (len(after), 128)
            assert np.array_equal(cache[offset : offset + len(after)], matrix)
            for local, item in enumerate(after):
                stored = manifest[offset + local]
                assert stored == {**item, "cache_index": offset + local}
                assert item["split"] == "train" and not item["quality_flags"]
                assert (
                    item["speaker_id"] == source["speaker_id"]
                    and item["input_id"] == source["input_id"]
                )
                assert (
                    720 <= item["end_frame"] - item["start_frame"] <= 6000
                    and item["rms_dbfs"] >= -50
                )
                additions[item["source_file"]] = (
                    item["source_sha256"],
                    source["input_id"],
                )
            offset += len(after)
        assert offset == len(manifest)
    baseline = ROOT / read(BASE / "config.json")["baseline_run"]
    prior = read(baseline / "design-freeze.json")["config"]
    manifests = [baseline / "expanded280/train-segments.jsonl"]
    evaluation_labels = set()
    for split in ("validation", "test"):
        manifests += [
            ROOT / prior["source_run"] / f"{split}-segments.jsonl",
            ROOT / prior["cv_evaluation_run"] / split / "segments.jsonl",
            ROOT / prior["corpus_run"] / f"{split}-segments.jsonl",
        ]
        for corpus in ("jvs", "common_voice", "jtube"):
            evaluation_labels.update(
                read(baseline / "evaluation-inputs" / f"{corpus}-{split}.json")[
                    "speakers"
                ]
            )
    assert not evaluation_labels & {r["speaker_id"] for r in source_rows}
    known_sources = {
        v["source_file"]: v["source_sha256"] for path in manifests for v in rows(path)
    }
    known_hashes = defaultdict(list)
    for path, checksum in known_sources.items():
        known_hashes[pcm_hash(ROOT / path, checksum)].append(path)
    new_hashes = defaultdict(list)
    for path, (checksum, input_id) in additions.items():
        new_hashes[pcm_hash(ROOT / path, checksum)].append(
            {"source_file": path, "input_id": input_id}
        )
    excluded = {}
    overlaps = []
    for digest, values in new_hashes.items():
        if digest in known_hashes or len(values) > 1:
            reason = (
                "matches_original_train_or_evaluation_pcm"
                if digest in known_hashes
                else "duplicate_added_pcm"
            )
            for value in values:
                excluded[value["input_id"]] = reason
            overlaps.append(
                {
                    "pcm_sha256": digest,
                    "added": values,
                    "known": known_hashes.get(digest, []),
                    "reason": reason,
                }
            )
    fixed(
        run / "overlap-audit.json",
        {
            "excluded_input_ids": excluded,
            "overlaps": overlaps,
            "original_source_waveforms": len(known_sources),
            "added_usable_source_waveforms": len(additions),
            "identity_overlap_across_corpora": "unknown",
            "policy": config["duplicate_policy"],
        },
    )
    files.update(
        pin(
            [
                Path(__file__),
                run / "overlap-audit.json",
                run / "preparation-report.json",
                run / "inputs.json",
                *manifests,
            ]
        )
    )
    fixed(
        run / "complete-audit.json",
        {
            "receipts_verified": sum(totals.values()),
            "states": dict(totals),
            "evaluation_label_overlap": 0,
            "excluded_added_sources": len(excluded),
            "files": files,
            "cache_equality": "all rows verified against per-source features",
            "pcm_comparison": "decoded 24kHz mono PCM16 bytes, independent of WAV headers",
            "training_started": False,
        },
    )
    print(
        json.dumps(
            {
                "verified": sum(totals.values()),
                "states": dict(totals),
                "excluded_added_sources": len(excluded),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
