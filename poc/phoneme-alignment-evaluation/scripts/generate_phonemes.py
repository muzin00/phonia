#!/usr/bin/env python3
"""Generate reproducible Open JTalk readings and phoneme sequences."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import pyopenjtalk

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "phonemes" / "expected.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate readings and raw phoneme sequences with pyopenjtalk."
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
            if not record.get("utterance_id") or not record.get("transcript"):
                raise ValueError(
                    f"{path}:{line_number}: utterance_id and transcript are required"
                )
            records.append(record)
    return records


def convert(record: dict[str, Any], version: str) -> dict[str, Any]:
    text = record["transcript"]
    raw_sequence = pyopenjtalk.g2p(text)
    return {
        "utterance_id": record["utterance_id"],
        "speaker_id": record.get("speaker_id"),
        "text": text,
        "reading_kana": pyopenjtalk.g2p(text, kana=True),
        "g2p_engine": "pyopenjtalk",
        "g2p_version": version,
        "g2p_dictionary": Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).name,
        "raw_phoneme_sequence": raw_sequence,
        "raw_phonemes": raw_sequence.split(),
        "review_status": "pending",
        "corrections": [],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def main() -> None:
    args = parse_args()
    version = importlib.metadata.version("pyopenjtalk")
    records = [convert(record, version) for record in read_jsonl(args.input)]
    write_jsonl(args.output, records)

    for record in records:
        print(f"{record['utterance_id']}: {record['raw_phoneme_sequence']}")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
