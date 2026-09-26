#!/usr/bin/env python3
"""Generate reproducible G2P records for the Phase 2 utterance manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyopenjtalk

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "vowel-dataset.json"
DEFAULT_INPUT = PROJECT_DIR / "data" / "utterance-manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "generated" / "expected-phonemes.jsonl"
DEFAULT_FAILURES = (
    PROJECT_DIR / "data" / "generated" / "expected-phonemes-failures.jsonl"
)
DEFAULT_RUN_METADATA = PROJECT_DIR / "data" / "generated" / "expected-phonemes-run.json"
SCHEMA_VERSION = 1
DESIGN_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expected phonemes for the Phase 2 utterance manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N utterances for a development smoke test.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = record.get("utterance_id")
            if not utterance_id or not record.get("transcript"):
                raise ValueError(
                    f"{path}:{line_number}: utterance_id and transcript are required"
                )
            if utterance_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate utterance_id {utterance_id}"
                )
            seen.add(str(utterance_id))
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def convert_record(
    record: dict[str, Any],
    version: str,
    dictionary: str,
    g2p: Callable[..., str] = pyopenjtalk.g2p,
) -> dict[str, Any]:
    text = str(record["transcript"])
    raw_sequence = g2p(text)
    raw_phonemes = raw_sequence.split()
    if not raw_phonemes:
        raise ValueError("G2P produced an empty phoneme sequence")
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "utterance_id": record["utterance_id"],
        "speaker_id": record["speaker_id"],
        "source_file": record["source_file"],
        "source_sha256": record["source_sha256"],
        "source_subset": record["source_subset"],
        "transcript": text,
        "split": record["split"],
        "evaluation_role": record["evaluation_role"],
        "learning_curve_cohorts": record["learning_curve_cohorts"],
        "session_id": record.get("session_id"),
        "duration_sec": record["duration_sec"],
        "g2p_engine": "pyopenjtalk",
        "g2p_version": version,
        "g2p_dictionary": dictionary,
        "reading_kana": g2p(text, kana=True),
        "raw_phonemes": raw_phonemes,
    }


def generate_records(
    records: list[dict[str, Any]],
    version: str,
    dictionary: str,
    g2p: Callable[..., str] = pyopenjtalk.g2p,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            generated.append(convert_record(record, version, dictionary, g2p))
        # Dataset construction must report an individual bad utterance and continue.
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "design_version": DESIGN_VERSION,
                    "utterance_id": record["utterance_id"],
                    "speaker_id": record.get("speaker_id"),
                    "source_file": record.get("source_file"),
                    "stage": "g2p",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
    return generated, failures


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset config schema_version does not match the generator")
    if config.get("design_version") != DESIGN_VERSION:
        raise ValueError("dataset config design_version does not match the generator")
    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    version = importlib.metadata.version("pyopenjtalk")
    dictionary = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)).name
    generated, failures = generate_records(records, version, dictionary)
    atomic_write_jsonl(args.output, generated)
    atomic_write_jsonl(args.failures, failures)
    run = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "stage": "g2p",
        "status": "pass" if not failures else "completed_with_failures",
        "input_manifest": str(args.input),
        "input_manifest_sha256": sha256(args.input),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "g2p_engine": "pyopenjtalk",
        "g2p_version": version,
        "g2p_dictionary": dictionary,
        "input_count": len(records),
        "success_count": len(generated),
        "failure_count": len(failures),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "failures": str(args.failures),
        "failures_sha256": sha256(args.failures),
    }
    atomic_write_json(args.run_metadata, run)
    print(
        f"Generated {len(generated)} expected phoneme records; failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
