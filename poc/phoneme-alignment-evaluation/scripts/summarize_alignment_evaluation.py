#!/usr/bin/env python3
"""Aggregate machine and assisted-review results by Japanese vowel."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "evaluation-summary.json"
VOWELS = ("a", "i", "u", "e", "o")
METHOD_PATHS = {
    "mfa": {
        "alignment": PROJECT_DIR / "data" / "alignments" / "mfa" / "normalized.jsonl",
        "segments": PROJECT_DIR / "data" / "segments" / "mfa" / "manifest.jsonl",
        "reviews": PROJECT_DIR / "data" / "reviews" / "review-records.jsonl",
    },
    "julius": {
        "alignment": PROJECT_DIR
        / "data"
        / "alignments"
        / "julius"
        / "normalized.jsonl",
        "segments": PROJECT_DIR / "data" / "segments" / "julius" / "manifest.jsonl",
        "reviews": PROJECT_DIR
        / "data"
        / "reviews"
        / "julius"
        / "review-records.jsonl",
    },
    "wav2vec2": {
        "alignment": PROJECT_DIR
        / "data"
        / "alignments"
        / "wav2vec2"
        / "normalized.jsonl",
        "segments": PROJECT_DIR
        / "data"
        / "segments"
        / "wav2vec2"
        / "manifest.jsonl",
        "reviews": PROJECT_DIR
        / "data"
        / "reviews"
        / "wav2vec2"
        / "review-records.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize alignment and review results by vowel."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def rounded(value: float) -> float:
    return round(value, 6)


def duration_distribution(records: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(record["duration_sec"]) * 1000 for record in records]
    return {
        "mean": rounded(statistics.mean(values)),
        "median": rounded(statistics.median(values)),
        "min": rounded(min(values)),
        "max": rounded(max(values)),
    }


def latest_reviews(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = str(record["itemId"])
        if item_id not in latest or int(record["revision"]) >= int(
            latest[item_id]["revision"]
        ):
            latest[item_id] = record
    return latest


def review_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record["reviewStatus"]) for record in records)
    question_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        candidate = next(
            answer
            for answer in record["candidateAnswers"]
            if answer["candidateId"] == "A"
        )
        for question, value in candidate["answers"].items():
            question_counts[str(question)][str(value)] += 1
    completed = len(records)
    return {
        "reviewed_count": completed,
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_rate": (
            rounded(status_counts["accepted"] / completed) if completed else None
        ),
        "question_counts": {
            question: dict(sorted(counts.items()))
            for question, counts in sorted(question_counts.items())
        },
    }


def summarize_method(
    alignments: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    alignment_by_id = {
        str(record["vowel_interval_id"]): record for record in alignments
    }
    segment_by_id = {
        str(record["vowel_interval_id"]): record for record in segments
    }
    latest = latest_reviews(reviews)
    unknown_reviews = sorted(set(latest) - set(alignment_by_id))
    if unknown_reviews:
        raise ValueError(f"Reviews do not match alignments: {unknown_reviews}")

    vowel_results: dict[str, Any] = {}
    for vowel in VOWELS:
        vowel_alignments = [
            record
            for record in alignments
            if record["normalized_phoneme"] == vowel
        ]
        interval_ids = {
            str(record["vowel_interval_id"]) for record in vowel_alignments
        }
        vowel_segments = [segment_by_id[item_id] for item_id in interval_ids]
        vowel_reviews = [latest[item_id] for item_id in interval_ids if item_id in latest]
        flagged = [
            record for record in vowel_segments if record.get("quality_flags")
        ]
        vowel_results[vowel] = {
            "interval_count": len(vowel_alignments),
            "expected_unit_count": sum(
                int(record["expected_units"]) for record in vowel_alignments
            ),
            "duration_ms": duration_distribution(vowel_alignments),
            "quality_flagged_count": len(flagged),
            "review": review_summary(vowel_reviews),
        }
    return {
        "interval_count": len(alignments),
        "expected_unit_count": sum(
            int(record["expected_units"]) for record in alignments
        ),
        "reviewed_count": len(latest),
        "vowels": vowel_results,
    }


def group_by_utterance(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["utterance_id"])].append(record)
    for utterance_records in grouped.values():
        utterance_records.sort(key=lambda record: int(record["vowel_index"]))
    return grouped


def pair_mfa_with_julius(
    mfa_records: list[dict[str, Any]], julius_records: list[dict[str, Any]]
) -> list[tuple[str, float, float, float, float]]:
    mfa_by_utterance = group_by_utterance(mfa_records)
    julius_by_utterance = group_by_utterance(julius_records)
    if mfa_by_utterance.keys() != julius_by_utterance.keys():
        raise ValueError("MFA and Julius utterances differ")
    pairs: list[tuple[str, float, float, float, float]] = []
    for utterance_id in sorted(mfa_by_utterance):
        julius = julius_by_utterance[utterance_id]
        julius_index = 0
        for mfa in mfa_by_utterance[utterance_id]:
            unit_count = int(mfa["expected_units"])
            matched = julius[julius_index : julius_index + unit_count]
            if len(matched) != unit_count or any(
                record["normalized_phoneme"] != mfa["normalized_phoneme"]
                for record in matched
            ):
                raise ValueError(f"Could not match MFA target: {mfa}")
            pairs.append(
                (
                    str(mfa["normalized_phoneme"]),
                    float(mfa["start_sec"]),
                    float(mfa["end_sec"]),
                    float(matched[0]["start_sec"]),
                    float(matched[-1]["end_sec"]),
                )
            )
            julius_index += unit_count
        if julius_index != len(julius):
            raise ValueError(f"Unmatched Julius targets for {utterance_id}")
    return pairs


def pair_atomic_records(
    left_records: list[dict[str, Any]], right_records: list[dict[str, Any]]
) -> list[tuple[str, float, float, float, float]]:
    def key(record: dict[str, Any]) -> tuple[str, int, str]:
        return (
            str(record["utterance_id"]),
            int(record["vowel_index"]),
            str(record["normalized_phoneme"]),
        )

    left = {key(record): record for record in left_records}
    right = {key(record): record for record in right_records}
    if left.keys() != right.keys():
        raise ValueError("Atomic comparison targets differ")
    return [
        (
            target[2],
            float(left[target]["start_sec"]),
            float(left[target]["end_sec"]),
            float(right[target]["start_sec"]),
            float(right[target]["end_sec"]),
        )
        for target in sorted(left)
    ]


def comparison_summary(
    pairs: list[tuple[str, float, float, float, float]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for vowel in VOWELS:
        vowel_pairs = [pair for pair in pairs if pair[0] == vowel]
        start_deltas: list[float] = []
        end_deltas: list[float] = []
        overlaps: list[float] = []
        for _, left_start, left_end, right_start, right_end in vowel_pairs:
            start_deltas.append(abs(left_start - right_start) * 1000)
            end_deltas.append(abs(left_end - right_end) * 1000)
            intersection = max(
                0.0, min(left_end, right_end) - max(left_start, right_start)
            )
            union = max(left_end, right_end) - min(left_start, right_start)
            overlaps.append(intersection / union)
        results[vowel] = {
            "pair_count": len(vowel_pairs),
            "start_delta_ms_median": rounded(statistics.median(start_deltas)),
            "end_delta_ms_median": rounded(statistics.median(end_deltas)),
            "iou_mean": rounded(statistics.mean(overlaps)),
            "iou_median": rounded(statistics.median(overlaps)),
            "no_overlap_count": sum(value == 0 for value in overlaps),
            "boundary_over_100ms_count": sum(
                start > 100 or end > 100
                for start, end in zip(start_deltas, end_deltas, strict=True)
            ),
        }
    return {"pair_count": len(pairs), "vowels": results}


def main() -> None:
    args = parse_args()
    inputs = {
        method: {
            kind: read_jsonl(path) for kind, path in paths.items()
        }
        for method, paths in METHOD_PATHS.items()
    }
    methods = {
        method: summarize_method(
            values["alignment"], values["segments"], values["reviews"]
        )
        for method, values in inputs.items()
    }
    result = {
        "schema_version": 1,
        "vowels": list(VOWELS),
        "methods": methods,
        "comparisons": {
            "mfa_vs_julius": comparison_summary(
                pair_mfa_with_julius(
                    inputs["mfa"]["alignment"], inputs["julius"]["alignment"]
                )
            ),
            "wav2vec2_vs_julius": comparison_summary(
                pair_atomic_records(
                    inputs["wav2vec2"]["alignment"],
                    inputs["julius"]["alignment"],
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote per-vowel evaluation summary to {args.output}")


if __name__ == "__main__":
    main()
