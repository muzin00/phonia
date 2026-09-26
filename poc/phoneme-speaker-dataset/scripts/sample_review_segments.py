#!/usr/bin/env python3
"""Select a deterministic stratified sample of Phase 2 vowel segments."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "review-sampling.json"
DEFAULT_SEGMENTS = PROJECT_DIR / "data" / "generated" / "vowel-segments.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "generated" / "expected-phonemes.jsonl"
DEFAULT_MEDIA_ROOT = PROJECT_DIR / "data" / "source" / "jvs_ver1"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "reviews" / "stratified"
DEFAULT_SAMPLE = DEFAULT_OUTPUT_DIR / "sample.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "sampling-summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a reproducible stratified Phase 2 review sample."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def stable_priority(seed: int, item_id: str) -> int:
    value = f"{seed}:{item_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def mapped_utterances(path: Path) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for record in read_jsonl(path):
        phones = set(record["raw_phonemes"])
        if "ty" in phones:
            mapped[record["utterance_id"]] = "ty_to_ch"
        elif "v" in phones:
            mapped[record["utterance_id"]] = "v_to_b"
    return mapped


def duration_bin(duration: float, config: dict[str, Any]) -> str:
    bins = config["duration_bins_sec"]
    if duration <= float(bins["short_max"]):
        return "short"
    if duration <= float(bins["middle_max"]):
        return "middle"
    return "long"


def quality_group(record: dict[str, Any]) -> str:
    return "near_silent" if "near_silent" in record["quality_flags"] else "normal"


def sampling_stratum(record: dict[str, Any], config: dict[str, Any]) -> str:
    return "/".join(
        (
            str(record["normalized_phoneme"]),
            str(record["split"]),
            quality_group(record),
            duration_bin(float(record["duration_sec"]), config),
        )
    )


def media_url(source_file: str, media_root: Path) -> str:
    path = Path(source_file)
    resolved = path if path.is_absolute() else REPOSITORY_DIR / path
    try:
        relative = resolved.resolve().relative_to(media_root.resolve())
    except ValueError as error:
        raise ValueError(f"{source_file}: source is outside media root") from error
    return "/media/" + quote(relative.as_posix(), safe="/")


def review_record(
    record: dict[str, Any],
    config: dict[str, Any],
    media_root: Path,
    mapping: str | None,
) -> dict[str, Any]:
    copied = dict(record)
    copied["expected_units"] = 1
    copied["audio_url"] = media_url(str(record["source_file"]), media_root)
    copied["sampling"] = {
        "stratum": sampling_stratum(record, config),
        "quality_group": quality_group(record),
        "duration_bin": duration_bin(float(record["duration_sec"]), config),
        "phone_mapping": mapping,
    }
    return copied


def retain_smallest(
    heap: list[tuple[int, str, dict[str, Any]]],
    capacity: int,
    priority: int,
    item_id: str,
    record: dict[str, Any],
) -> None:
    entry = (-priority, item_id, record)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def select_records(
    records: Iterable[dict[str, Any]],
    config: dict[str, Any],
    media_root: Path,
    mapping_by_utterance: dict[str, str],
) -> list[dict[str, Any]]:
    seed = int(config["seed"])
    per_stratum = int(config["base_per_stratum"])
    target_count = int(config["target_count"])
    mapping_target = int(config["phone_mapping_target_count"])
    strata: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    mapped: list[tuple[int, int, str, dict[str, Any]]] = []
    fallback: list[tuple[int, str, dict[str, Any]]] = []

    for raw in records:
        item_id = str(raw["vowel_interval_id"])
        priority = stable_priority(seed, item_id)
        mapping = mapping_by_utterance.get(str(raw["utterance_id"]))
        record = review_record(raw, config, media_root, mapping)
        stratum = record["sampling"]["stratum"]
        retain_smallest(
            strata.setdefault(stratum, []), per_stratum, priority, item_id, record
        )
        retain_smallest(fallback, target_count * 2, priority, item_id, record)
        if mapping is not None:
            mapping_order = 0 if mapping == "ty_to_ch" else 1
            mapped.append((mapping_order, priority, item_id, record))

    selected: dict[str, dict[str, Any]] = {}
    for heap in strata.values():
        for _, item_id, record in heap:
            selected[item_id] = record

    added_mapping = 0
    for _, _, item_id, record in sorted(mapped):
        if item_id in selected:
            continue
        selected[item_id] = record
        added_mapping += 1
        if added_mapping >= mapping_target:
            break

    for _, item_id, record in sorted(fallback, reverse=True):
        if len(selected) >= target_count:
            break
        selected.setdefault(item_id, record)

    if len(selected) < target_count:
        raise ValueError(
            f"Could select only {len(selected)} records, expected {target_count}"
        )
    ordered = sorted(
        selected.values(),
        key=lambda record: (
            record["sampling"]["stratum"],
            stable_priority(seed, record["vowel_interval_id"]),
        ),
    )
    return ordered[:target_count]


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


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mappings = mapped_utterances(args.expected)
    selected = select_records(read_jsonl(args.segments), config, args.media_root, mappings)
    atomic_write_jsonl(args.output, selected)
    summary = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "seed": config["seed"],
        "target_count": config["target_count"],
        "selected_count": len(selected),
        "stratum_counts": dict(
            sorted(Counter(item["sampling"]["stratum"] for item in selected).items())
        ),
        "vowel_counts": dict(
            sorted(Counter(item["normalized_phoneme"] for item in selected).items())
        ),
        "split_counts": dict(sorted(Counter(item["split"] for item in selected).items())),
        "quality_group_counts": dict(
            sorted(
                Counter(item["sampling"]["quality_group"] for item in selected).items()
            )
        ),
        "duration_bin_counts": dict(
            sorted(
                Counter(item["sampling"]["duration_bin"] for item in selected).items()
            )
        ),
        "phone_mapping_counts": dict(
            sorted(
                Counter(
                    item["sampling"]["phone_mapping"] or "none" for item in selected
                ).items()
            )
        ),
        "speaker_count": len({item["speaker_id"] for item in selected}),
    }
    atomic_write_json(args.summary, summary)
    print(
        f"Selected {len(selected)} review segments from "
        f"{len(summary['stratum_counts'])} strata"
    )


if __name__ == "__main__":
    main()
