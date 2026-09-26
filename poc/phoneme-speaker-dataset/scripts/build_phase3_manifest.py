#!/usr/bin/env python3
"""Build and validate the Phase 3 vowel manifest from Phase 2 segments."""

from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .run_julius_dataset import (
        atomic_write_json,
        repository_relative,
        resolve_source_file,
        sha256,
    )
except ImportError:  # Direct script execution.
    from run_julius_dataset import (
        atomic_write_json,
        repository_relative,
        resolve_source_file,
        sha256,
    )

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "phase3-input.json"
DEFAULT_SPLIT_CONFIG = PROJECT_DIR / "config" / "dataset-split.json"
DEFAULT_INPUT = PROJECT_DIR / "data" / "generated" / "vowel-segments.jsonl"
DEFAULT_SOURCE_VALIDATION = PROJECT_DIR / "data" / "vowel-dataset-validation.json"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "generated" / "phase3-vowel-segments.jsonl"
DEFAULT_VALIDATION = PROJECT_DIR / "data" / "phase3-vowel-dataset-validation.json"


def split_by_speaker(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(speaker): str(split)
        for split, speakers in config["speaker_splits"].items()
        for speaker in speakers
    }


def learning_curve_cohorts(speaker: str, config: dict[str, Any]) -> list[int]:
    train_order = config["speaker_splits"]["train"]
    if speaker not in train_order:
        return []
    position = train_order.index(speaker)
    return [int(size) for size in config["learning_curve_sizes"] if position < size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the validated Phase 3 vowel segment manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--source-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--skip-source-audio-validation",
        action="store_true",
        help="Skip checking source WAV metadata and frame bounds (not recommended).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: record must be an object")
            yield line_number, record


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def validate_policy_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("phase3 input config schema_version must be 1")
    if not isinstance(config.get("policy_version"), str):
        raise TypeError("phase3 input config policy_version must be a string")
    flags = config.get("exclude_quality_flags")
    if (
        not isinstance(flags, list)
        or not flags
        or not all(isinstance(flag, str) and flag for flag in flags)
        or len(flags) != len(set(flags))
    ):
        raise ValueError("exclude_quality_flags must be a unique non-empty string list")
    expected = config.get("expected_counts")
    if not isinstance(expected, dict) or set(expected) != {
        "input",
        "excluded",
        "eligible",
    }:
        raise ValueError("expected_counts must contain input, excluded, and eligible")
    if not all(isinstance(value, int) and value >= 0 for value in expected.values()):
        raise ValueError("expected_counts values must be non-negative integers")
    if expected["input"] != expected["excluded"] + expected["eligible"]:
        raise ValueError("expected_counts input must equal excluded plus eligible")
    coverage = config.get("coverage")
    if not isinstance(coverage, dict):
        raise TypeError("coverage must be an object")
    vowels = coverage.get("required_vowels")
    if (
        not isinstance(vowels, list)
        or not vowels
        or not all(isinstance(vowel, str) and vowel for vowel in vowels)
        or len(vowels) != len(set(vowels))
    ):
        raise ValueError("coverage.required_vowels must be a unique string list")
    roles = coverage.get("required_evaluation_roles_by_split")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("coverage evaluation roles must be a non-empty object")
    for split, values in roles.items():
        if (
            not isinstance(split, str)
            or not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError("coverage evaluation roles must be unique string lists")


def validate_split_config(config: dict[str, Any]) -> None:
    speaker_splits = config.get("speaker_splits")
    if not isinstance(speaker_splits, dict) or "train" not in speaker_splits:
        raise ValueError("speaker_splits must contain train")
    all_speakers: list[str] = []
    for split, speakers in speaker_splits.items():
        if (
            not isinstance(split, str)
            or not isinstance(speakers, list)
            or not all(isinstance(speaker, str) for speaker in speakers)
            or len(speakers) != len(set(speakers))
        ):
            raise ValueError("speaker splits must contain unique string lists")
        all_speakers.extend(speakers)
    if len(all_speakers) != len(set(all_speakers)):
        raise ValueError("speakers must not occur in multiple splits")
    sizes = config.get("learning_curve_sizes")
    if (
        not isinstance(sizes, list)
        or not sizes
        or not all(isinstance(size, int) and size > 0 for size in sizes)
        or sizes != sorted(set(sizes))
        or sizes[-1] != len(speaker_splits["train"])
    ):
        raise ValueError("learning_curve_sizes must be nested through all train speakers")


@dataclass
class ManifestCounts:
    count: int = 0
    total_duration_sec: float = 0.0
    by_vowel: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)
    by_evaluation_role: Counter[str] = field(default_factory=Counter)
    by_speaker: Counter[str] = field(default_factory=Counter)
    by_split_vowel: Counter[str] = field(default_factory=Counter)
    by_learning_curve_cohort: Counter[str] = field(default_factory=Counter)
    quality_flags: Counter[str] = field(default_factory=Counter)
    speaker_vowels: Counter[tuple[str, str]] = field(default_factory=Counter)
    speaker_roles: Counter[tuple[str, str]] = field(default_factory=Counter)
    source_files: set[str] = field(default_factory=set)
    split_speakers: dict[str, set[str]] = field(default_factory=dict)
    cohort_speakers: dict[str, set[str]] = field(default_factory=dict)

    def add(self, record: dict[str, Any]) -> None:
        vowel = str(record["normalized_phoneme"])
        split = str(record["split"])
        speaker = str(record["speaker_id"])
        role = str(record["evaluation_role"])
        self.count += 1
        self.total_duration_sec += float(record["duration_sec"])
        self.by_vowel[vowel] += 1
        self.by_split[split] += 1
        self.by_evaluation_role[role] += 1
        self.by_speaker[speaker] += 1
        self.by_split_vowel[f"{split}/{vowel}"] += 1
        self.speaker_vowels[(speaker, vowel)] += 1
        self.speaker_roles[(speaker, role)] += 1
        self.source_files.add(str(record["source_file"]))
        self.split_speakers.setdefault(split, set()).add(speaker)
        self.quality_flags.update(record["quality_flags"])
        for cohort in record["learning_curve_cohorts"]:
            cohort_name = str(cohort)
            self.by_learning_curve_cohort[cohort_name] += 1
            self.cohort_speakers.setdefault(cohort_name, set()).add(speaker)

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.count,
            "total_duration_sec": round(self.total_duration_sec, 6),
            "source_file_count": len(self.source_files),
            "speaker_count": len(self.by_speaker),
            "by_vowel_count": dict(sorted(self.by_vowel.items())),
            "by_split_count": dict(sorted(self.by_split.items())),
            "by_split_speaker_count": {
                split: len(speakers)
                for split, speakers in sorted(self.split_speakers.items())
            },
            "by_evaluation_role_count": dict(
                sorted(self.by_evaluation_role.items())
            ),
            "by_speaker_count": dict(sorted(self.by_speaker.items())),
            "by_split_vowel_count": dict(sorted(self.by_split_vowel.items())),
            "by_learning_curve_cohort_count": dict(
                sorted(self.by_learning_curve_cohort.items(), key=lambda item: int(item[0]))
            ),
            "by_learning_curve_cohort_speaker_count": {
                cohort: len(speakers)
                for cohort, speakers in sorted(
                    self.cohort_speakers.items(), key=lambda item: int(item[0])
                )
            },
            "quality_flag_counts": dict(sorted(self.quality_flags.items())),
        }


def validate_source_audio(
    record: dict[str, Any],
    cache: dict[str, tuple[int, int, int, int]],
) -> None:
    source_file = str(record["source_file"])
    metadata = cache.get(source_file)
    if metadata is None:
        source_path = resolve_source_file(source_file)
        with wave.open(str(source_path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError(f"{source_file}: compressed WAV is not supported")
            metadata = (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getnframes(),
            )
        cache[source_file] = metadata
    sample_rate, channels, sample_width, source_frames = metadata
    expected = (
        int(record["sample_rate_hz"]),
        int(record["channels"]),
        int(record["sample_width_bytes"]),
    )
    if (sample_rate, channels, sample_width) != expected:
        raise ValueError(f"{source_file}: source WAV metadata differs from manifest")
    start_frame = int(record["start_frame"])
    end_frame = int(record["end_frame"])
    if not 0 <= start_frame < end_frame <= source_frames:
        raise ValueError(
            f"{record['vowel_interval_id']}: frame range is outside source WAV"
        )
    if int(record["frame_count"]) != end_frame - start_frame:
        raise ValueError(f"{record['vowel_interval_id']}: frame_count is inconsistent")


def validate_record(
    record: dict[str, Any],
    line_number: int,
    policy_config: dict[str, Any],
    speaker_splits: dict[str, str],
    split_config: dict[str, Any],
    seen_ids: set[str],
) -> None:
    required = {
        "schema_version",
        "vowel_interval_id",
        "speaker_id",
        "source_file",
        "split",
        "evaluation_role",
        "learning_curve_cohorts",
        "normalized_phoneme",
        "start_frame",
        "end_frame",
        "frame_count",
        "sample_rate_hz",
        "channels",
        "sample_width_bytes",
        "duration_sec",
        "quality_flags",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"line {line_number}: missing fields {missing}")
    if record["schema_version"] != policy_config["source_schema_version"]:
        raise ValueError(f"line {line_number}: source schema_version differs from config")
    item_id = str(record["vowel_interval_id"])
    if item_id in seen_ids:
        raise ValueError(f"line {line_number}: duplicate vowel_interval_id {item_id}")
    seen_ids.add(item_id)
    speaker = str(record["speaker_id"])
    expected_split = speaker_splits.get(speaker)
    if expected_split is None:
        raise ValueError(f"line {line_number}: unknown speaker {speaker}")
    if record["split"] != expected_split:
        raise ValueError(f"line {line_number}: split does not match speaker assignment")
    vowel = str(record["normalized_phoneme"])
    if vowel not in policy_config["coverage"]["required_vowels"]:
        raise ValueError(f"line {line_number}: unexpected normalized vowel {vowel}")
    role = str(record["evaluation_role"])
    allowed_roles = policy_config["coverage"][
        "required_evaluation_roles_by_split"
    ][expected_split]
    if role not in allowed_roles:
        raise ValueError(f"line {line_number}: evaluation role does not match split")
    expected_cohorts = learning_curve_cohorts(speaker, split_config)
    if record["learning_curve_cohorts"] != expected_cohorts:
        raise ValueError(f"line {line_number}: learning curve cohorts do not match")
    flags = record["quality_flags"]
    if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
        raise TypeError(f"line {line_number}: quality_flags must be a string list")
    if len(flags) != len(set(flags)):
        raise ValueError(f"line {line_number}: quality_flags contains duplicates")
    start_frame = int(record["start_frame"])
    end_frame = int(record["end_frame"])
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"line {line_number}: invalid frame range")
    if int(record["frame_count"]) != end_frame - start_frame:
        raise ValueError(f"line {line_number}: frame_count is inconsistent")
    if float(record["duration_sec"]) <= 0:
        raise ValueError(f"line {line_number}: duration_sec must be positive")


def validate_coverage(
    eligible: ManifestCounts,
    config: dict[str, Any],
    split_config: dict[str, Any],
) -> None:
    coverage = config["coverage"]
    required_vowels = coverage["required_vowels"]
    required_roles = coverage["required_evaluation_roles_by_split"]
    for split, speakers in split_config["speaker_splits"].items():
        for speaker in speakers:
            if eligible.by_speaker[speaker] == 0:
                raise ValueError(f"{speaker}: no eligible Phase 3 segments")
            for vowel in required_vowels:
                if eligible.speaker_vowels[(speaker, vowel)] == 0:
                    raise ValueError(f"{speaker}/{vowel}: no eligible Phase 3 segments")
            for role in required_roles[split]:
                if eligible.speaker_roles[(speaker, role)] == 0:
                    raise ValueError(f"{speaker}/{role}: no eligible Phase 3 segments")

    actual_cohort_speakers = {
        int(cohort): len(
            {
                speaker
                for speaker in split_config["speaker_splits"]["train"]
                if int(cohort) in learning_curve_cohorts(speaker, split_config)
                and eligible.by_speaker[speaker] > 0
            }
        )
        for cohort in split_config["learning_curve_sizes"]
    }
    expected_cohort_speakers = {
        int(cohort): int(cohort) for cohort in split_config["learning_curve_sizes"]
    }
    if actual_cohort_speakers != expected_cohort_speakers:
        raise ValueError("eligible learning curve cohorts are incomplete")


def filter_phase3_manifest(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    split_config: dict[str, Any],
    *,
    verify_source_audio: bool,
) -> dict[str, Any]:
    validate_policy_config(config)
    validate_split_config(split_config)
    configured_splits = set(split_config["speaker_splits"])
    coverage_splits = set(config["coverage"]["required_evaluation_roles_by_split"])
    if coverage_splits != configured_splits:
        raise ValueError("coverage evaluation role splits differ from speaker splits")
    speaker_splits = split_by_speaker(split_config)
    excluded_flags = set(config["exclude_quality_flags"])
    seen_ids: set[str] = set()
    audio_cache: dict[str, tuple[int, int, int, int]] = {}
    input_counts = ManifestCounts()
    eligible_counts = ManifestCounts()
    excluded_counts = ManifestCounts()
    exclusion_reasons: Counter[str] = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for line_number, record in read_jsonl(input_path):
                validate_record(
                    record,
                    line_number,
                    config,
                    speaker_splits,
                    split_config,
                    seen_ids,
                )
                if verify_source_audio:
                    validate_source_audio(record, audio_cache)
                input_counts.add(record)
                matched = sorted(excluded_flags.intersection(record["quality_flags"]))
                if matched:
                    excluded_counts.add(record)
                    exclusion_reasons.update(matched)
                    continue
                eligible_counts.add(record)
                json.dump(
                    record, destination, ensure_ascii=False, separators=(",", ":")
                )
                destination.write("\n")

        expected = config["expected_counts"]
        actual = {
            "input": input_counts.count,
            "excluded": excluded_counts.count,
            "eligible": eligible_counts.count,
        }
        if actual != expected:
            raise ValueError(f"manifest counts differ from config: {actual} != {expected}")
        if excluded_flags.intersection(eligible_counts.quality_flags):
            raise ValueError("excluded quality flags remain in Phase 3 manifest")
        validate_coverage(eligible_counts, config, split_config)
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "input": input_counts.as_dict(),
        "eligible": eligible_counts.as_dict(),
        "excluded": excluded_counts.as_dict(),
        "exclusion_reason_counts": dict(sorted(exclusion_reasons.items())),
        "validated_source_file_count": len(audio_cache),
    }


def validate_source_manifest(
    input_path: Path,
    source_validation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if source_validation.get("schema_version") != config["source_schema_version"]:
        raise ValueError("source validation schema_version differs from policy")
    if source_validation.get("vowel_manifest_sha256") != sha256(input_path):
        raise ValueError("source vowel manifest checksum differs from validation")
    if source_validation.get("segments", {}).get("count") != config[
        "expected_counts"
    ]["input"]:
        raise ValueError("source validation segment count differs from policy")


def main() -> None:
    args = parse_args()
    config = load_json_object(args.config)
    split_config = load_json_object(args.split_config)
    source_validation = load_json_object(args.source_validation)
    validate_policy_config(config)
    validate_source_manifest(args.input, source_validation, config)
    summary = filter_phase3_manifest(
        args.input,
        args.output,
        config,
        split_config,
        verify_source_audio=not args.skip_source_audio_validation,
    )
    validation = {
        "schema_version": 1,
        "policy_version": config["policy_version"],
        "status": "pass",
        "config": repository_relative(args.config),
        "config_sha256": sha256(args.config),
        "split_config": repository_relative(args.split_config),
        "split_config_sha256": sha256(args.split_config),
        "source_manifest": repository_relative(args.input),
        "source_manifest_sha256": sha256(args.input),
        "source_validation": repository_relative(args.source_validation),
        "source_validation_sha256": sha256(args.source_validation),
        "phase3_manifest": repository_relative(args.output),
        "phase3_manifest_sha256": sha256(args.output),
        "source_audio_validation": (
            "metadata_and_frame_bounds"
            if not args.skip_source_audio_validation
            else "skipped"
        ),
        "exclude_quality_flags": config["exclude_quality_flags"],
        **summary,
    }
    atomic_write_json(args.validation, validation)
    print(
        f"Built {summary['eligible']['segment_count']} Phase 3 segments; "
        f"excluded={summary['excluded']['segment_count']}"
    )


if __name__ == "__main__":
    main()
