#!/usr/bin/env python3
"""Compare Wav2Vec2 vowel intervals with Julius atomic vowel intervals."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JULIUS = PROJECT_DIR / "data" / "alignments" / "julius" / "normalized.jsonl"
DEFAULT_WAV2VEC2 = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "normalized.jsonl"
)
DEFAULT_JULIUS_REVIEWS = (
    PROJECT_DIR / "data" / "reviews" / "julius" / "review-records.jsonl"
)
DEFAULT_JULIUS_RUN = PROJECT_DIR / "data" / "alignments" / "julius" / "run.json"
DEFAULT_WAV2VEC2_RUN = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "run.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "comparison.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare atomic Wav2Vec2 and Julius vowel intervals."
    )
    parser.add_argument("--julius", type=Path, default=DEFAULT_JULIUS)
    parser.add_argument("--wav2vec2", type=Path, default=DEFAULT_WAV2VEC2)
    parser.add_argument(
        "--julius-reviews", type=Path, default=DEFAULT_JULIUS_REVIEWS
    )
    parser.add_argument("--julius-run", type=Path, default=DEFAULT_JULIUS_RUN)
    parser.add_argument("--wav2vec2-run", type=Path, default=DEFAULT_WAV2VEC2_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def target_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["utterance_id"]),
        int(record["vowel_index"]),
        str(record["normalized_phoneme"]),
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def latest_review_statuses(path: Path) -> dict[tuple[str, int, str], str]:
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        item_id = str(record["itemId"])
        if item_id not in latest or int(record["revision"]) >= int(
            latest[item_id]["revision"]
        ):
            latest[item_id] = record
    statuses: dict[tuple[str, int, str], str] = {}
    for item_id, record in latest.items():
        parts = item_id.rsplit("-julius-vowel-", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Unexpected Julius review item ID: {item_id}")
        # The phone label is recovered later from the normalized Julius record.
        statuses[(parts[0], int(parts[1]), "")] = str(record["reviewStatus"])
    return statuses


def main() -> None:
    args = parse_args()
    julius_records = {target_key(record): record for record in read_jsonl(args.julius)}
    wav2vec2_records = {
        target_key(record): record for record in read_jsonl(args.wav2vec2)
    }
    if julius_records.keys() != wav2vec2_records.keys():
        missing_from_wav2vec2 = sorted(julius_records.keys() - wav2vec2_records.keys())
        missing_from_julius = sorted(wav2vec2_records.keys() - julius_records.keys())
        raise ValueError(
            "Comparison targets differ: "
            f"missing_from_wav2vec2={missing_from_wav2vec2}, "
            f"missing_from_julius={missing_from_julius}"
        )

    review_statuses = latest_review_statuses(args.julius_reviews)
    start_deltas: list[float] = []
    end_deltas: list[float] = []
    intersections_over_union: list[float] = []
    julius_durations: list[float] = []
    wav2vec2_durations: list[float] = []
    reviewed_accepted_no_overlap = 0
    boundary_over_100ms = 0
    for key in sorted(julius_records):
        julius = julius_records[key]
        wav2vec2 = wav2vec2_records[key]
        start_delta = abs(float(julius["start_sec"]) - float(wav2vec2["start_sec"]))
        end_delta = abs(float(julius["end_sec"]) - float(wav2vec2["end_sec"]))
        intersection = max(
            0.0,
            min(float(julius["end_sec"]), float(wav2vec2["end_sec"]))
            - max(float(julius["start_sec"]), float(wav2vec2["start_sec"])),
        )
        union = max(float(julius["end_sec"]), float(wav2vec2["end_sec"])) - min(
            float(julius["start_sec"]), float(wav2vec2["start_sec"])
        )
        overlap = intersection / union
        start_deltas.append(start_delta)
        end_deltas.append(end_delta)
        intersections_over_union.append(overlap)
        julius_durations.append(float(julius["duration_sec"]))
        wav2vec2_durations.append(float(wav2vec2["duration_sec"]))
        if start_delta > 0.1 or end_delta > 0.1:
            boundary_over_100ms += 1
        status = review_statuses.get((key[0], key[1], ""))
        if status == "accepted" and overlap == 0:
            reviewed_accepted_no_overlap += 1

    wav2vec2_run = json.loads(args.wav2vec2_run.read_text(encoding="utf-8"))
    julius_run = json.loads(args.julius_run.read_text(encoding="utf-8"))
    edit_count = sum(
        int(record["greedy_token_edit_distance"])
        for record in wav2vec2_run["utterances"]
    )
    target_count = sum(
        int(record["target_token_count"])
        for record in wav2vec2_run["utterances"]
    )

    thresholds = {}
    for threshold_ms in (10, 20, 30, 50, 100):
        threshold_sec = threshold_ms / 1000
        thresholds[str(threshold_ms)] = sum(
            start <= threshold_sec and end <= threshold_sec
            for start, end in zip(start_deltas, end_deltas, strict=True)
        ) / len(start_deltas)

    result = {
        "comparison": "wav2vec2-ctc-onnx-vs-julius",
        "pair_count": len(start_deltas),
        "greedy_token_error": {
            "edit_count": edit_count,
            "target_token_count": target_count,
            "rate": edit_count / target_count,
        },
        "absolute_start_delta_ms": distribution(
            [value * 1000 for value in start_deltas]
        ),
        "absolute_end_delta_ms": distribution(
            [value * 1000 for value in end_deltas]
        ),
        "both_boundaries_within_threshold_rate": thresholds,
        "intersection_over_union": distribution(intersections_over_union),
        "no_overlap_count": sum(value == 0 for value in intersections_over_union),
        "iou_below_0_5_count": sum(
            value < 0.5 for value in intersections_over_union
        ),
        "boundary_over_100ms_count": boundary_over_100ms,
        "reviewed_julius_accepted_no_overlap_count": reviewed_accepted_no_overlap,
        "duration_ms": {
            "julius": distribution([value * 1000 for value in julius_durations]),
            "wav2vec2": distribution([value * 1000 for value in wav2vec2_durations]),
        },
        "runtime_sec": {
            "julius": float(julius_run["duration_sec"]),
            "wav2vec2": float(wav2vec2_run["duration_sec"]),
            "wav2vec2_to_julius_ratio": (
                float(wav2vec2_run["duration_sec"])
                / float(julius_run["duration_sec"])
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
