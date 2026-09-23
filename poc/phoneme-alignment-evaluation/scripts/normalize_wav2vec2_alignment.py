#!/usr/bin/env python3
"""Convert raw Wav2Vec2 CTC intervals into common atomic vowel records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "phonemes" / "expected-vowels.jsonl"
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "raw"
DEFAULT_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "normalized.jsonl"
)
DEFAULT_VALIDATION_OUTPUT = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "validation.json"
)
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_VERSION = "2c733782da5604684829819a5eb744c193fe9398"
NORMALIZATION_VERSION = "2"
VOWELS = {"a", "i", "u", "e", "o"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Wav2Vec2 CTC phone intervals into vowel intervals."
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


def extract_vowel_intervals(
    record: dict[str, Any], phone_entries: list[dict[str, Any]], aligner_version: str
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
            or end_sec > duration + 1e-6
        ):
            raise ValueError(
                f"Invalid Wav2Vec2 interval at index {phone_index}: {entry}"
            )
        previous_end = end_sec
        if raw_phone not in VOWELS:
            continue
        vowel_index = len(intervals)
        score = entry.get("mean_log_probability")
        intervals.append(
            {
                "utterance_id": record["utterance_id"],
                "speaker_id": record["speaker_id"],
                "session_id": record["session_id"],
                "source_file": record["source_file"],
                "transcript": record["transcript"],
                "aligner": "wav2vec2-ctc-onnx",
                "aligner_version": aligner_version,
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "score": float(score) if score is not None else None,
                "score_kind": "ctc_token_mean_log_probability",
                "raw_phonemes": [raw_phone],
                "raw_intervals": [entry],
                "normalized_phoneme": raw_phone,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "source_phoneme_start": phone_index,
                "source_phoneme_end": phone_index + 1,
                "expected_units": 1,
                "normalization_version": NORMALIZATION_VERSION,
                "vowel_index": vowel_index,
                "vowel_interval_id": (
                    f"{record['utterance_id']}-wav2vec2-vowel-{vowel_index:03d}"
                ),
                "duration_sec": round(end_sec - start_sec, 9),
                "is_long": False,
                "is_devoiced": False,
                "contains_devoiced": False,
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
    aligner_version: str | None = None
    for record in manifest:
        raw_path = args.raw_directory / f"{Path(record['source_file']).stem}.json"
        alignment = json.loads(raw_path.read_text(encoding="utf-8"))
        phone_entries = alignment.get("intervals")
        if not isinstance(phone_entries, list):
            raise ValueError(f"{raw_path}: intervals are missing")
        current_version = str(alignment["aligner_version"])
        if aligner_version is not None and current_version != aligner_version:
            raise ValueError("Wav2Vec2 results use multiple runtime versions")
        aligner_version = current_version
        intervals = extract_vowel_intervals(record, phone_entries, current_version)
        actual = [item["normalized_phoneme"] for item in intervals]
        expected_vowels = expected[record["utterance_id"]]
        if actual != expected_vowels:
            raise ValueError(
                f"{record['utterance_id']}: vowels differ from expected\n"
                f"expected={expected_vowels}\nactual={actual}"
            )
        all_intervals.extend(intervals)
        validation_records.append(
            {
                "utterance_id": record["utterance_id"],
                "phone_interval_count": len(phone_entries),
                "expected_vowel_phone_count": len(expected_vowels),
                "actual_vowel_interval_count": len(intervals),
                "expected_units": len(expected_vowels),
                "actual_units": len(intervals),
                "sequence_match": True,
                "intervals_valid": True,
            }
        )
        print(f"{record['utterance_id']}: {len(intervals)} vowels matched")

    write_jsonl(args.output, all_intervals)
    validation = {
        "aligner": "wav2vec2-ctc-onnx",
        "normalization_version": NORMALIZATION_VERSION,
        "utterance_count": len(validation_records),
        "vowel_interval_count": len(all_intervals),
        "all_sequences_match": True,
        "all_intervals_valid": True,
        "utterances": validation_records,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(all_intervals)} vowel intervals to {args.output}")


if __name__ == "__main__":
    main()
