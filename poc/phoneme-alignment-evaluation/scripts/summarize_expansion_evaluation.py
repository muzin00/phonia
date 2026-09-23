#!/usr/bin/env python3
"""Summarize the reproducible 40-utterance machine evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_alignment_evaluation import (
    VOWELS,
    comparison_summary,
    pair_atomic_records,
    pair_mfa_with_julius,
    read_jsonl,
    rounded,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "expansion"
OUTPUT = DATA_DIR / "evaluation-summary.json"
METHODS = ("mfa", "julius", "wav2vec2")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def method_summary(method: str) -> dict[str, Any]:
    alignments = read_jsonl(DATA_DIR / "alignments" / method / "normalized.jsonl")
    run = read_json(DATA_DIR / "alignments" / method / "run.json")
    segment_validation = read_json(DATA_DIR / "segments" / method / "validation.json")
    vowel_counts = Counter(record["normalized_phoneme"] for record in alignments)
    result = {
        "utterance_count": int(run["utterance_count"]),
        "vowel_interval_count": len(alignments),
        "vowel_counts": {vowel: vowel_counts[vowel] for vowel in VOWELS},
        "duration_sec": float(run["duration_sec"]),
        "mean_duration_ms_per_utterance": rounded(
            float(run["duration_sec"]) * 1000 / int(run["utterance_count"])
        ),
        "quality_flag_counts": segment_validation["flag_counts"],
        "quality_flagged_count": (
            int(segment_validation["segment_count"])
            - int(segment_validation["unflagged_segment_count"])
        ),
    }
    if method == "mfa":
        validation = read_json(DATA_DIR / "alignments" / method / "validation.json")
        matches = sum(item["sequence_match"] for item in validation["utterances"])
        result["expected_sequence_match_count"] = matches
        result["expected_sequence_match_rate"] = rounded(
            matches / int(validation["utterance_count"])
        )
    if method == "wav2vec2":
        edit_distance = sum(
            int(item["greedy_token_edit_distance"]) for item in run["utterances"]
        )
        target_count = sum(int(item["target_token_count"]) for item in run["utterances"])
        result["greedy_token_error_rate"] = rounded(edit_distance / target_count)
    return result


def main() -> None:
    manifest = read_jsonl(DATA_DIR / "manifest.jsonl")
    expected = read_jsonl(DATA_DIR / "expected-vowels.jsonl")
    speaker_counts = Counter(record["speaker_id"] for record in manifest)
    expected_counts = Counter(record["normalized_phoneme"] for record in expected)

    alignments = {
        method: read_jsonl(DATA_DIR / "alignments" / method / "normalized.jsonl")
        for method in METHODS
    }
    mfa_validation = read_json(DATA_DIR / "alignments" / "mfa" / "validation.json")
    matched_utterances = {
        item["utterance_id"]
        for item in mfa_validation["utterances"]
        if item["sequence_match"]
    }
    matched_mfa = [
        item for item in alignments["mfa"] if item["utterance_id"] in matched_utterances
    ]
    matched_julius = [
        item
        for item in alignments["julius"]
        if item["utterance_id"] in matched_utterances
    ]

    result = {
        "schema_version": 1,
        "dataset": {
            "utterance_count": len(manifest),
            "speaker_count": len(speaker_counts),
            "speaker_utterance_counts": dict(sorted(speaker_counts.items())),
            "audio_duration_sec": rounded(
                sum(float(record["duration_sec"]) for record in manifest)
            ),
            "expected_vowel_unit_count": sum(expected_counts.values()),
            "expected_vowel_counts": {
                vowel: expected_counts[vowel] for vowel in VOWELS
            },
        },
        "methods": {method: method_summary(method) for method in METHODS},
        "comparisons": {
            "mfa_vs_julius_sequence_matched_utterances": {
                "utterance_count": len(matched_utterances),
                **comparison_summary(
                    pair_mfa_with_julius(matched_mfa, matched_julius)
                ),
            },
            "wav2vec2_vs_julius": comparison_summary(
                pair_atomic_records(alignments["wav2vec2"], alignments["julius"])
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote expansion evaluation summary to {OUTPUT}")


if __name__ == "__main__":
    main()
