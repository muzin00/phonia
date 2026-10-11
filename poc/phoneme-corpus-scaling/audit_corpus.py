"""Audit corpus split provenance and completed extraction receipts."""

import argparse
import json
from collections import Counter

import numpy as np
from acquire import ROOT, read, sha
from corpus import CONFIG, atomic, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    verify(run)
    rows = read(run / "inputs.json")
    source_freeze = read(ROOT / config["source_run"] / "selection-freeze.json")
    speakers = {}
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        labels = Counter(r["source_speaker_id"] for r in subset)
        assert set(labels) == set(source_freeze["partitions"][split])
        assert set(labels.values()) == {50}
        speakers[split] = set(labels)
    assert not speakers["train"] & (speakers["validation"] | speakers["test"])
    assert not speakers["validation"] & speakers["test"]
    extraction = read(run / "extraction-freeze.json")
    for name, digest in extraction["runtime_files"].items():
        assert sha(ROOT / name) == digest, name
    processed, states, features = 0, Counter(), 0
    for row in rows:
        output = run / "extraction" / row["sha256"]
        saved = output / "receipt.json"
        if not saved.exists():
            assert args.partial, f"Missing receipt: {row['path']}"
            continue
        receipt = read(saved)
        assert receipt["input_sha256"] == sha(ROOT / row["local_path"])
        assert (
            receipt["split"] == row["split"]
            and receipt["speaker_id"] == row["speaker_id"]
        )
        asr_path = run / "asr" / (row["sha256"] + ".json")
        assert sha(asr_path) == receipt["asr_sha256"]
        for name, digest in receipt["files"].items():
            assert sha(ROOT / name) == digest, name
        if (output / "post-qc.json").exists():
            post = read(output / "post-qc.json")
            raw = read(output / "pre-qc.json")
            assert {v["segment_id"] for v in post} == {
                v["segment_id"] for v in raw if not v["exclusion_reasons"]
            }
            matrix = np.load(output / "processed/features.npy", allow_pickle=False)
            assert matrix.shape == (len(post), 128) and np.isfinite(matrix).all()
            result = read(output / "processed/result.json")
            assert (
                result["utterance"]["transcript"]
                == read(asr_path)["asr"]["text"].strip()
            )
            for index, segment in enumerate(post):
                assert segment["cache_index"] == index
                assert segment["original_mp3_sha256"] == row["sha256"]
                assert (
                    segment["split"] == row["split"]
                    and segment["extraction_split"] == "pilot"
                )
                assert (segment["role"] == "training") == (row["split"] == "train")
                assert segment["speaker_id"] == row["speaker_id"].lower()
                assert not segment["quality_flags"] and segment["rms_dbfs"] >= -50
                assert 720 <= segment["end_frame"] - segment["start_frame"] <= 6000
            features += len(post)
        processed += 1
        states[receipt["status"]] += 1
    if not args.partial:
        report = read(run / "preparation-report.json")
        for split, item in report["splits"].items():
            for name, digest in item["files"].items():
                assert sha(ROOT / name) == digest, name
            path = run / f"{split}-features.f32"
            assert path.stat().st_size == item["segments"] * 128 * 4
            with (run / f"{split}-segments.jsonl").open() as stream:
                count = 0
                for index, line in enumerate(stream):
                    segment = json.loads(line)
                    assert segment["cache_index"] == index and segment["split"] == split
                    count += 1
                assert count == item["segments"]
    result = {
        "audit_scope": "completed receipts only" if args.partial else "complete corpus",
        "receipts_verified": processed,
        "total_expected": len(rows),
        "features_verified": features,
        "statuses": dict(states),
        "split_provenance": "passed",
        "training_started": False,
    }
    atomic(
        run / ("partial-audit.json" if args.partial else "complete-audit.json"), result
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
