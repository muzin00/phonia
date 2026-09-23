#!/usr/bin/env python3
"""Convert raw Julius phone intervals into common atomic vowel records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "phonemes" / "expected-vowels.jsonl"
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "alignments" / "julius" / "raw"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "julius" / "normalized.jsonl"
)
DEFAULT_VALIDATION_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "julius" / "validation.json"
)
JULIUS_VERSION = "4.6"
MODEL_ID = "segmentation-kit-monophone"
MODEL_VERSION = "e0e8bbaf98e27d19dfc6fe8312be607ad03592ad"
NORMALIZATION_VERSION = "2"
JULIUS_VOWELS = {"a", "i", "u", "e", "o"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Julius phone intervals into Japanese vowel intervals."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def classify_julius_vowel(phone: str) -> tuple[str, int, bool] | None:
    if phone not in JULIUS_VOWELS:
        return None
    return phone, 1, False


def extract_vowel_intervals(
    record: dict[str, Any], phone_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    duration = float(record["duration_sec"])
    intervals: list[dict[str, Any]] = []
    previous_end = 0.0
    for phone_index, entry in enumerate(phone_entries):
        raw_phone = str(entry["phoneme"])
        start_sec = float(entry["start_sec"])
        end_sec = float(entry["end_sec"])
        if (
            start_sec < previous_end - 1e-9
            or end_sec <= start_sec
            or end_sec > duration + 0.025 + 1e-6
        ):
            raise ValueError(f"Invalid Julius interval at index {phone_index}: {entry}")
        previous_end = end_sec
        classified = classify_julius_vowel(raw_phone)
        if classified is None:
            continue
        normalized, expected_units, is_devoiced = classified
        vowel_index = len(intervals)
        intervals.append(
            {
                "utterance_id": record["utterance_id"],
                "speaker_id": record["speaker_id"],
                "session_id": record["session_id"],
                "source_file": record["source_file"],
                "transcript": record["transcript"],
                "aligner": "julius",
                "aligner_version": JULIUS_VERSION,
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "score": float(entry["score"]),
                "score_kind": "julius_n_score",
                "raw_phonemes": [raw_phone],
                "raw_intervals": [
                    {
                        "phoneme": raw_phone,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "begin_frame": int(entry["begin_frame"]),
                        "end_frame": int(entry["end_frame"]),
                        "score": float(entry["score"]),
                    }
                ],
                "normalized_phoneme": normalized,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "source_phoneme_start": phone_index,
                "source_phoneme_end": phone_index + 1,
                "expected_units": expected_units,
                "normalization_version": NORMALIZATION_VERSION,
                "vowel_index": vowel_index,
                "vowel_interval_id": (
                    f"{record['utterance_id']}-julius-vowel-{vowel_index:03d}"
                ),
                "duration_sec": round(end_sec - start_sec, 9),
                "is_long": False,
                "is_devoiced": is_devoiced,
                "contains_devoiced": is_devoiced,
            }
        )
    return intervals


def expected_by_utterance(path: Path) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = defaultdict(list)
    for record in read_jsonl(path):
        phoneme = str(record["normalized_phoneme"])
        units = int(record["expected_units"])
        expected[record["utterance_id"]].extend([phoneme] * units)
    return expected


def validate_expected(
    utterance_id: str,
    intervals: list[dict[str, Any]],
    expected: dict[str, list[str]],
) -> None:
    if utterance_id not in expected:
        raise ValueError(f"No expected vowels for {utterance_id}")
    actual = [interval["normalized_phoneme"] for interval in intervals]
    if actual != expected[utterance_id]:
        raise ValueError(
            f"{utterance_id}: Julius vowels differ from expected vowels\n"
            f"expected={expected[utterance_id]}\nactual={actual}"
        )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    expected = expected_by_utterance(args.expected)
    all_intervals: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    for record in manifest:
        raw_path = args.raw_directory / f"{Path(record['source_file']).stem}.json"
        with raw_path.open(encoding="utf-8") as source:
            alignment = json.load(source)
        phone_entries = alignment.get("intervals")
        if not isinstance(phone_entries, list):
            raise ValueError(f"{raw_path}: intervals are missing")
        intervals = extract_vowel_intervals(record, phone_entries)
        validate_expected(record["utterance_id"], intervals, expected)
        all_intervals.extend(intervals)
        validation_records.append(
            {
                "utterance_id": record["utterance_id"],
                "phone_interval_count": len(phone_entries),
                "expected_vowel_phone_count": len(expected[record["utterance_id"]]),
                "actual_vowel_interval_count": len(intervals),
                "expected_units": len(expected[record["utterance_id"]]),
                "actual_units": sum(item["expected_units"] for item in intervals),
                "sequence_match": True,
                "intervals_valid": True,
            }
        )
        print(
            f"{record['utterance_id']}: "
            f"{len(intervals)} atomic vowel intervals matched"
        )

    write_jsonl(args.output, all_intervals)
    validation = {
        "aligner": "julius",
        "normalization_version": NORMALIZATION_VERSION,
        "utterance_count": len(validation_records),
        "vowel_interval_count": len(all_intervals),
        "all_sequences_match": True,
        "all_intervals_valid": True,
        "utterances": validation_records,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(all_intervals)} vowel intervals to {args.output}")


if __name__ == "__main__":
    main()
