"""Inventory human microphone episodes without inventing authentication results."""

import argparse
from collections import Counter
from pathlib import Path

from store import read, write


def summarize(run):
    episodes = [read(p) for p in sorted((run / "episodes").glob("*/episode.json"))]
    # File imports and corpus replays are useful smoke tests, never fresh microphone evidence.
    human = [
        e
        for e in episodes
        if e["purpose"] == "human_pilot"
        and e["clips"]
        and all(c["origin"] == "microphone" for c in e["clips"])
    ]
    enrollment = [
        e for e in human if e["role"] == "enrollment" and e["status"] == "complete"
    ]
    queries = [e for e in human if e["role"] == "query"]
    pairs = []
    for q in queries:
        candidates = [e for e in enrollment if e["participant"] == q["participant"]]
        pairs.append(
            {
                "query_id": q["id"],
                "participant": q["participant"],
                "status": q["status"],
                "enrollment_count": len(candidates),
                "different_date_and_device": len(candidates) == 1
                and candidates[0]["recorded_on"] != q["recorded_on"]
                and candidates[0]["device"] != q["device"],
            }
        )
    return {
        "episodes": len(episodes),
        "microphone_episodes": len(human),
        "participants": len({e["participant"] for e in human}),
        "practice_episodes": sum(e["role"] == "practice" for e in human),
        "complete_enrollments": len(enrollment),
        "query_episodes": len(queries),
        "query_status_counts": dict(Counter(e["status"] for e in queries)),
        "microphone_clip_status_counts": dict(
            Counter(c["status"] for e in human for c in e["clips"])
        ),
        "recording_seconds": sum(e["seconds"] for e in human),
        "processing_seconds": sum(
            c.get("processing_seconds", 0) for e in human for c in e["clips"]
        ),
        "query_metadata_checks": pairs,
        "state": "awaiting_new_recordings" if not human else "operational_pilot_only",
        "threshold": None,
        "far": None,
        "frr": None,
        "limitations": [
            "Declared dates and device labels are not independently verified.",
            "Practice, imported files, and corpus replays do not establish authentication accuracy.",
            "New independent calibration and test speakers are required for FAR/FRR.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if not (args.run / "protocol-freeze.json").exists():
        parser.error("Missing protocol-freeze.json")
    result = summarize(args.run)
    write(args.run / "pilot-inventory.json", result)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
