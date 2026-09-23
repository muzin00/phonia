#!/usr/bin/env python3
"""Convert raw MFA phone intervals into common atomic vowel records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "phonemes" / "expected-vowels.jsonl"
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "alignments" / "mfa" / "raw"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "alignments" / "mfa" / "normalized.jsonl"
DEFAULT_VALIDATION_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "mfa" / "validation.json"
)
NORMALIZATION_VERSION = "2"

MFA_VOWELS: dict[str, tuple[str, int, bool]] = {
    "a": ("a", 1, False),
    "aː": ("a", 2, False),
    "i": ("i", 1, False),
    "iː": ("i", 2, False),
    "i̥": ("i", 1, True),
    "ɨ": ("u", 1, False),
    "ɨː": ("u", 2, False),
    "ɨ̥": ("u", 1, True),
    "ɯ": ("u", 1, False),
    "ɯː": ("u", 2, False),
    "ɯ̥": ("u", 1, True),
    "e": ("e", 1, False),
    "eː": ("e", 2, False),
    "o": ("o", 1, False),
    "oː": ("o", 2, False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize MFA phone intervals into atomic Japanese vowel intervals."
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


def classify_mfa_vowel(phone: str) -> tuple[str, int, bool] | None:
    return MFA_VOWELS.get(phone)


def extract_vowel_intervals(
    record: dict[str, Any], phone_entries: list[list[Any]]
) -> list[dict[str, Any]]:
    duration = float(record["duration_sec"])
    intervals: list[dict[str, Any]] = []
    previous_end = 0.0

    for phone_index, entry in enumerate(phone_entries):
        if len(entry) != 3:
            raise ValueError(f"Invalid MFA phone entry at index {phone_index}: {entry}")
        start_sec, end_sec, raw_phone = entry
        start_sec = float(start_sec)
        end_sec = float(end_sec)
        if (
            start_sec < previous_end
            or end_sec <= start_sec
            or end_sec > duration + 1e-6
        ):
            raise ValueError(f"Invalid MFA interval at index {phone_index}: {entry}")
        previous_end = end_sec

        classified = classify_mfa_vowel(raw_phone)
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
                "aligner": "mfa",
                "aligner_version": "3.4.2",
                "model_id": "japanese_mfa",
                "model_version": "3.0.0",
                "score": None,
                "score_kind": None,
                "raw_phonemes": [raw_phone],
                "raw_intervals": [
                    {
                        "phoneme": raw_phone,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
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
                    f"{record['utterance_id']}-mfa-vowel-{vowel_index:03d}"
                ),
                "duration_sec": round(end_sec - start_sec, 9),
                "is_long": expected_units > 1,
                "is_devoiced": is_devoiced,
                "contains_devoiced": is_devoiced,
            }
        )

    return intervals


def expected_by_utterance(path: Path) -> dict[str, list[tuple[str, int]]]:
    expected: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for record in read_jsonl(path):
        expected[record["utterance_id"]].append(
            (record["normalized_phoneme"], record["expected_units"])
        )
    return expected


def validate_expected(
    utterance_id: str,
    intervals: list[dict[str, Any]],
    expected: dict[str, list[tuple[str, int]]],
) -> None:
    expected_units = expected.get(utterance_id)
    if expected_units is None:
        raise ValueError(f"No expected vowels for {utterance_id}")

    def expand(records: list[tuple[str, int]]) -> list[str]:
        return [phoneme for phoneme, count in records for _ in range(count)]

    actual_records = [
        (interval["normalized_phoneme"], interval["expected_units"])
        for interval in intervals
    ]
    actual_sequence = expand(actual_records)
    expected_sequence = expand(expected_units)
    if actual_sequence != expected_sequence:
        raise ValueError(
            f"{utterance_id}: MFA vowels differ from expected vowels\n"
            f"expected={expected_sequence}\nactual={actual_sequence}"
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
        source_stem = Path(record["source_file"]).stem
        raw_path = args.raw_directory / f"{source_stem}.json"
        with raw_path.open(encoding="utf-8") as source:
            alignment = json.load(source)
        try:
            phone_entries = alignment["tiers"]["phones"]["entries"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{raw_path}: phones tier is missing") from error
        intervals = extract_vowel_intervals(record, phone_entries)
        validate_expected(record["utterance_id"], intervals, expected)
        all_intervals.extend(intervals)
        validation_records.append(
            {
                "utterance_id": record["utterance_id"],
                "phone_interval_count": len(phone_entries),
                "expected_vowel_phone_count": len(expected[record["utterance_id"]]),
                "actual_vowel_interval_count": len(intervals),
                "expected_units": sum(
                    units for _, units in expected[record["utterance_id"]]
                ),
                "actual_units": sum(
                    interval["expected_units"] for interval in intervals
                ),
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
        "aligner": "mfa",
        "normalization_version": NORMALIZATION_VERSION,
        "utterance_count": len(validation_records),
        "vowel_interval_count": len(all_intervals),
        "all_sequences_match": True,
        "all_intervals_valid": True,
        "utterances": validation_records,
    }
    args.validation_output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(all_intervals)} vowel intervals to {args.output}")


if __name__ == "__main__":
    main()
