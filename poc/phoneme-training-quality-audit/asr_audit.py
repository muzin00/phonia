"""Inspect saved ASR diagnostics for accepted JTube sources; never rerun ASR."""

from collections import Counter
from pathlib import Path

import numpy as np
from audit import OUTPUT, ROOT, read, sha, summary, write


def main():
    rows = []
    counts = Counter()
    fingerprints = {}
    source_path = OUTPUT / "source-measures.jsonl"
    fingerprints[str(source_path.relative_to(ROOT))] = sha(source_path)
    for line in source_path.open():
        import json

        source = json.loads(line)
        if source["corpus"] != "jtube":
            continue
        parts = Path(source["source_file"]).parts
        position = parts.index("extraction")
        path = ROOT / Path(*parts[:position]) / "asr" / (parts[position + 1] + ".json")
        counts["accepted_sources"] += 1
        counts["accepted_segments"] += source["segments"]
        if not path.exists():
            counts["missing_asr_file"] += 1
            continue
        fingerprints[str(path.relative_to(ROOT))] = sha(path)
        data = read(path)
        segments = data.get("asr", {}).get("segments", [])
        if not segments:
            counts["missing_asr_segments"] += 1
            continue
        flags = {
            "any_avg_logprob_below_minus1": any(
                s["avg_logprob"] < -1 for s in segments
            ),
            "any_compression_ratio_above_2_4": any(
                s["compression_ratio"] > 2.4 for s in segments
            ),
            "any_no_speech_above_0_6": any(s["no_speech_prob"] > 0.6 for s in segments),
            "joint_no_speech_and_low_logprob": any(
                s["no_speech_prob"] > 0.6 and s["avg_logprob"] < -1 for s in segments
            ),
        }
        for name, flag in flags.items():
            counts[name + "_sources"] += flag
            counts[name + "_accepted_segments"] += flag * source["segments"]
        rows.append(
            {
                "source_file": source["source_file"],
                "speaker_id": source["speaker_id"],
                "accepted_segments": source["segments"],
                "asr_file": str(path.relative_to(ROOT)),
                "text": data["asr"]["text"],
                "flags": flags,
                "minimum_avg_logprob": min(s["avg_logprob"] for s in segments),
                "maximum_compression_ratio": max(
                    s["compression_ratio"] for s in segments
                ),
                "maximum_no_speech_prob": max(s["no_speech_prob"] for s in segments),
            }
        )
    results = {
        "counts": dict(counts),
        "distributions": {
            k: summary([r[k] for r in rows])
            for k in (
                "minimum_avg_logprob",
                "maximum_compression_ratio",
                "maximum_no_speech_prob",
            )
        },
        "examples": sorted(rows, key=lambda r: r["minimum_avg_logprob"])[:10],
        "limitations": "These are decoder diagnostics, not verified transcription error labels. Segment counts refer to all accepted phone intervals from flagged recordings, not proven erroneous intervals. Whisper performs an internal silence skip; this is distinct from a dataset acceptance gate.",
        "source_files": fingerprints,
    }
    assert all(np.isfinite(r["minimum_avg_logprob"]) for r in rows)
    write(OUTPUT / "asr-diagnostics.json", rows)
    write(OUTPUT / "asr-results.json", results)
    print({"counts": dict(counts), "distributions": results["distributions"]})


if __name__ == "__main__":
    main()
