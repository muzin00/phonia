#!/usr/bin/env python3
"""Create model-independent vowel evaluation units from G2P output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "phonemes" / "expected.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "phonemes" / "expected-vowels.jsonl"
NORMALIZATION_VERSION = "1"
VOWELS = frozenset("aiueo")
DEVOICED_VOWELS = frozenset("AIUEO")
LONG_MARKS = (":", "ː")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expected vowel units from pyopenjtalk phonemes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("utterance_id"):
                raise ValueError(f"{path}:{line_number}: utterance_id is required")
            if not isinstance(record.get("raw_phonemes"), list):
                raise ValueError(f"{path}:{line_number}: raw_phonemes must be a list")
            records.append(record)
    return records


def classify_vowel(phone: str) -> tuple[str, int, bool] | None:
    """Return normalized vowel, expected unit count, and devoicing flag."""
    normalized = phone
    expected_units = 1
    if normalized.endswith(LONG_MARKS):
        normalized = normalized[:-1]
        expected_units = 2

    is_devoiced = normalized in DEVOICED_VOWELS
    normalized = normalized.lower()
    if normalized not in VOWELS:
        return None
    return normalized, expected_units, is_devoiced


def extract_vowel_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract vowels and merge only directly adjacent identical vowels."""
    utterance_id = record["utterance_id"]
    units: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        nonlocal current
        if current is None:
            return
        current["vowel_index"] = len(units)
        current["vowel_unit_id"] = f"{utterance_id}-vowel-{len(units):03d}"
        current["is_long"] = current["expected_units"] > 1
        current["is_devoiced"] = all(current.pop("_devoiced_flags"))
        current["contains_devoiced"] = current.pop("_contains_devoiced")
        units.append(current)
        current = None

    for index, raw_phone in enumerate(record["raw_phonemes"]):
        classified = classify_vowel(raw_phone)
        if classified is None:
            finish_current()
            continue

        normalized, expected_units, is_devoiced = classified
        can_merge = (
            current is not None
            and current["normalized_phoneme"] == normalized
            and current["source_phoneme_end"] == index
        )
        if not can_merge:
            finish_current()
            current = {
                "utterance_id": utterance_id,
                "speaker_id": record.get("speaker_id"),
                "text": record.get("text"),
                "reading_kana": record.get("reading_kana"),
                "normalized_phoneme": normalized,
                "raw_phonemes": [],
                "source_phoneme_start": index,
                "source_phoneme_end": index,
                "expected_units": 0,
                "phoneme_review_status": record.get("review_status", "pending"),
                "normalization_version": NORMALIZATION_VERSION,
                "_devoiced_flags": [],
                "_contains_devoiced": False,
            }

        current["raw_phonemes"].append(raw_phone)
        current["source_phoneme_end"] = index + 1
        current["expected_units"] += expected_units
        current["_devoiced_flags"].append(is_devoiced)
        current["_contains_devoiced"] |= is_devoiced

    finish_current()
    return units


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def main() -> None:
    args = parse_args()
    source_records = read_jsonl(args.input)
    vowel_units = [
        unit for record in source_records for unit in extract_vowel_units(record)
    ]
    write_jsonl(args.output, vowel_units)

    for record in source_records:
        count = sum(
            unit["utterance_id"] == record["utterance_id"] for unit in vowel_units
        )
        print(f"{record['utterance_id']}: {count} vowel units")
    print(f"Wrote {len(vowel_units)} vowel units to {args.output}")


if __name__ == "__main__":
    main()
