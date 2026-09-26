#!/usr/bin/env python3
"""Build and validate source-slice vowel records from Julius alignments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .run_julius_dataset import (
        DESIGN_VERSION,
        JULIUS_VERSION,
        MODEL_ID,
        REPOSITORY_DIR,
        SCHEMA_VERSION,
        SEGMENTATION_KIT_COMMIT,
        alignment_input_sha256,
        atomic_write_json,
        repository_relative,
        resolve_source_file,
        sha256,
        to_julius_phonemes,
    )
except ImportError:  # Direct script execution.
    from run_julius_dataset import (
        DESIGN_VERSION,
        JULIUS_VERSION,
        MODEL_ID,
        REPOSITORY_DIR,
        SCHEMA_VERSION,
        SEGMENTATION_KIT_COMMIT,
        alignment_input_sha256,
        atomic_write_json,
        repository_relative,
        resolve_source_file,
        sha256,
        to_julius_phonemes,
    )

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "vowel-dataset.json"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "generated" / "expected-phonemes.jsonl"
DEFAULT_ALIGNMENTS = (
    PROJECT_DIR / "data" / "generated" / "alignments" / "manifest.jsonl"
)
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "generated" / "vowel-segments.jsonl"
DEFAULT_FAILURES = PROJECT_DIR / "data" / "generated" / "failures.jsonl"
DEFAULT_VALIDATION = PROJECT_DIR / "data" / "vowel-dataset-validation.json"
VOWELS = {"a", "i", "u", "e", "o"}


@dataclass(frozen=True)
class WaveSource:
    channels: int
    sample_width: int
    sample_rate_hz: int
    frame_count: int
    frames: bytes

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.sample_rate_hz

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * self.sample_width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 source-slice vowel manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--alignments", type=Path, default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--minimum-duration-sec", type=float, default=0.03)
    parser.add_argument("--near-silent-dbfs", type=float, default=-50.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N utterances for a development smoke test.",
    )
    parser.add_argument(
        "--skip-source-checksums",
        action="store_true",
        help="Skip rechecking source WAV SHA-256 (not recommended for final output).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = record.get("utterance_id")
            if not utterance_id:
                raise ValueError(f"{path}:{line_number}: utterance_id is required")
            if utterance_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate utterance_id {utterance_id}"
                )
            seen.add(str(utterance_id))
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def resolve_artifact(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def read_wave(path: Path) -> WaveSource:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV is not supported")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV")
        return WaveSource(
            channels=source.getnchannels(),
            sample_width=source.getsampwidth(),
            sample_rate_hz=source.getframerate(),
            frame_count=source.getnframes(),
            frames=source.readframes(source.getnframes()),
        )


def seconds_to_frame(seconds: float, sample_rate_hz: int, frame_count: int) -> int:
    return min(max(round(seconds * sample_rate_hz), 0), frame_count)


def slice_frames(source: WaveSource, start_frame: int, end_frame: int) -> bytes:
    if not 0 <= start_frame < end_frame <= source.frame_count:
        raise ValueError(
            f"Invalid frame range {start_frame}:{end_frame} for {source.frame_count}"
        )
    byte_start = start_frame * source.bytes_per_frame
    byte_end = end_frame * source.bytes_per_frame
    return source.frames[byte_start:byte_end]


def rms_dbfs(frames: bytes, sample_width: int) -> float | None:
    if sample_width != 2:
        raise ValueError("RMS calculation supports only 16-bit PCM WAV")
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return None
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms == 0:
        return None
    return round(20 * math.log10(rms / 32768), 3)


def validate_raw_alignment(
    expected: dict[str, Any], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    if raw.get("utterance_id") != expected["utterance_id"]:
        raise ValueError("raw alignment utterance_id does not match")
    if raw.get("source_sha256") != expected["source_sha256"]:
        raise ValueError("raw alignment source checksum does not match")
    if raw.get("alignment_input_sha256") != alignment_input_sha256(expected):
        raise ValueError("raw alignment input fingerprint does not match")
    expected_phones = to_julius_phonemes(
        [str(phone) for phone in expected["raw_phonemes"]]
    )
    intervals = raw.get("intervals")
    if not isinstance(intervals, list):
        raise TypeError("raw alignment intervals must be a list")
    actual = [str(interval.get("phoneme")) for interval in intervals]
    wanted = ["silB", *expected_phones, "silE"]
    if actual != wanted:
        raise ValueError("raw alignment phoneme sequence does not match G2P input")
    return intervals[1:-1]


def segment_record(
    expected: dict[str, Any],
    phone_interval: dict[str, Any],
    source: WaveSource,
    source_phone_index: int,
    vowel_index: int,
    minimum_duration_sec: float,
    near_silent_dbfs: float,
) -> dict[str, Any]:
    raw_phone = str(expected["raw_phonemes"][source_phone_index])
    normalized = str(phone_interval["phoneme"])
    if normalized not in VOWELS:
        raise ValueError(f"{normalized} is not a normalized vowel")
    requested_start = float(phone_interval["start_sec"])
    requested_end = float(phone_interval["end_sec"])
    if requested_start < 0 or requested_end <= requested_start:
        raise ValueError("invalid vowel time interval")
    if requested_end > source.duration_sec + (1 / source.sample_rate_hz):
        raise ValueError("vowel interval exceeds source audio")
    start_frame = seconds_to_frame(
        requested_start, source.sample_rate_hz, source.frame_count
    )
    end_frame = seconds_to_frame(
        requested_end, source.sample_rate_hz, source.frame_count
    )
    frames = slice_frames(source, start_frame, end_frame)
    level_dbfs = rms_dbfs(frames, source.sample_width)
    start_sec = round(start_frame / source.sample_rate_hz, 9)
    end_sec = round(end_frame / source.sample_rate_hz, 9)
    duration_sec = round(end_sec - start_sec, 9)
    quality_flags: list[str] = []
    if duration_sec < minimum_duration_sec:
        quality_flags.append("short")
    if level_dbfs is None:
        quality_flags.append("silent")
    elif level_dbfs < near_silent_dbfs:
        quality_flags.append("near_silent")
    raw_phonemes = expected["raw_phonemes"]
    interval_id = f"{expected['utterance_id']}-julius-vowel-{vowel_index:03d}"
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "vowel_interval_id": interval_id,
        "utterance_id": expected["utterance_id"],
        "speaker_id": expected["speaker_id"],
        "source_file": expected["source_file"],
        "source_sha256": expected["source_sha256"],
        "source_subset": expected["source_subset"],
        "split": expected["split"],
        "evaluation_role": expected["evaluation_role"],
        "learning_curve_cohorts": expected["learning_curve_cohorts"],
        "session_id": expected.get("session_id"),
        "transcript": expected["transcript"],
        "vowel_index": vowel_index,
        "source_phoneme_index": source_phone_index,
        "raw_phoneme": raw_phone,
        "normalized_phoneme": normalized,
        "previous_phoneme": (
            raw_phonemes[source_phone_index - 1] if source_phone_index else None
        ),
        "next_phoneme": (
            raw_phonemes[source_phone_index + 1]
            if source_phone_index + 1 < len(raw_phonemes)
            else None
        ),
        "is_devoiced": raw_phone in {"A", "I", "U", "E", "O"},
        "is_long": raw_phone.endswith(("-", ":")),
        "aligner": "julius",
        "aligner_version": JULIUS_VERSION,
        "model_id": MODEL_ID,
        "model_version": SEGMENTATION_KIT_COMMIT,
        "alignment_score": float(phone_interval["score"]),
        "score_kind": "julius_n_score",
        "g2p_engine": expected["g2p_engine"],
        "g2p_version": expected["g2p_version"],
        "g2p_dictionary": expected["g2p_dictionary"],
        "requested_start_sec": requested_start,
        "requested_end_sec": requested_end,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "frame_count": end_frame - start_frame,
        "sample_rate_hz": source.sample_rate_hz,
        "channels": source.channels,
        "sample_width_bytes": source.sample_width,
        "storage_mode": "source_slice",
        "rms_dbfs": level_dbfs,
        "quality_flags": quality_flags,
    }


def process_utterance(
    expected: dict[str, Any],
    alignment: dict[str, Any] | None,
    minimum_duration_sec: float,
    near_silent_dbfs: float,
    verify_source_checksum: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    failure_base = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "utterance_id": expected["utterance_id"],
        "speaker_id": expected["speaker_id"],
        "source_file": expected["source_file"],
        "split": expected["split"],
        "evaluation_role": expected["evaluation_role"],
    }
    if alignment is None:
        return [], {
            **failure_base,
            "stage": "vowel_manifest",
            "error_type": "MissingAlignment",
            "reason": "alignment manifest has no record for utterance",
        }
    if alignment.get("status") != "success":
        failure = {
            **failure_base,
            "stage": alignment.get("stage", "julius_alignment"),
            "error_type": alignment.get("error_type", "AlignmentFailure"),
            "reason": alignment.get("reason", "alignment failed"),
        }
        if alignment.get("log_file"):
            failure["log_file"] = alignment["log_file"]
        return [], failure
    try:
        raw_path = resolve_artifact(str(alignment["raw_alignment_file"]))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        phone_intervals = validate_raw_alignment(expected, raw)
        source_path = resolve_source_file(str(expected["source_file"]))
        if verify_source_checksum and sha256(source_path) != expected["source_sha256"]:
            raise ValueError("source WAV checksum differs from utterance manifest")
        source = read_wave(source_path)
        segments: list[dict[str, Any]] = []
        for source_index, interval in enumerate(phone_intervals):
            if str(interval["phoneme"]) not in VOWELS:
                continue
            segments.append(
                segment_record(
                    expected,
                    interval,
                    source,
                    source_index,
                    len(segments),
                    minimum_duration_sec,
                    near_silent_dbfs,
                )
            )
        if not segments:
            raise ValueError("alignment contains no vowel intervals")
        return segments, None
    except Exception as error:  # noqa: BLE001 - record bad utterance and continue.
        return [], {
            **failure_base,
            "stage": "vowel_manifest",
            "error_type": type(error).__name__,
            "reason": str(error),
        }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[lower], 6)
    value = ordered[lower] * (upper - index) + ordered[upper] * (index - lower)
    return round(value, 6)


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "total_sec": round(sum(values), 6),
        "min_sec": percentile(values, 0),
        "p25_sec": percentile(values, 0.25),
        "median_sec": percentile(values, 0.5),
        "p75_sec": percentile(values, 0.75),
        "max_sec": percentile(values, 1),
    }


class Summary:
    def __init__(self) -> None:
        self.durations: list[float] = []
        self.by_vowel: defaultdict[str, list[float]] = defaultdict(list)
        self.by_split: defaultdict[str, list[float]] = defaultdict(list)
        self.by_speaker: defaultdict[str, list[float]] = defaultdict(list)
        self.by_session: defaultdict[str, list[float]] = defaultdict(list)
        self.by_evaluation_role: defaultdict[str, list[float]] = defaultdict(list)
        self.by_split_vowel: Counter[str] = Counter()
        self.quality_flags: Counter[str] = Counter()

    def add(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            duration = float(record["duration_sec"])
            vowel = str(record["normalized_phoneme"])
            split = str(record["split"])
            self.durations.append(duration)
            self.by_vowel[vowel].append(duration)
            self.by_split[split].append(duration)
            self.by_speaker[str(record["speaker_id"])].append(duration)
            self.by_session[str(record.get("session_id") or "unknown")].append(duration)
            self.by_evaluation_role[str(record["evaluation_role"])].append(duration)
            self.by_split_vowel[f"{split}/{vowel}"] += 1
            self.quality_flags.update(record["quality_flags"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "segments": distribution(self.durations),
            "by_vowel": {
                key: distribution(value) for key, value in sorted(self.by_vowel.items())
            },
            "by_split": {
                key: distribution(value) for key, value in sorted(self.by_split.items())
            },
            "by_speaker": {
                key: distribution(value)
                for key, value in sorted(self.by_speaker.items())
            },
            "by_session": {
                key: distribution(value)
                for key, value in sorted(self.by_session.items())
            },
            "by_evaluation_role": {
                key: distribution(value)
                for key, value in sorted(self.by_evaluation_role.items())
            },
            "by_split_vowel_count": dict(sorted(self.by_split_vowel.items())),
            "quality_flag_counts": dict(sorted(self.quality_flags.items())),
        }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset config schema_version does not match the builder")
    if config.get("design_version") != DESIGN_VERSION:
        raise ValueError("dataset config design_version does not match the builder")
    if config.get("segment", {}).get("storage_mode") != "source_slice":
        raise ValueError("dataset config storage_mode must be source_slice")
    if args.minimum_duration_sec < 0:
        raise ValueError("--minimum-duration-sec must not be negative")
    expected_records = read_jsonl(args.expected)
    if args.limit is not None:
        expected_records = expected_records[: args.limit]
    alignment_records = {
        record["utterance_id"]: record for record in read_jsonl(args.alignments)
    }
    summary = Summary()
    failures: list[dict[str, Any]] = []
    successful_utterances = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    with temporary_output.open("w", encoding="utf-8") as destination:
        for expected in expected_records:
            segments, failure = process_utterance(
                expected,
                alignment_records.get(expected["utterance_id"]),
                args.minimum_duration_sec,
                args.near_silent_dbfs,
                verify_source_checksum=not args.skip_source_checksums,
            )
            if failure is not None:
                failures.append(failure)
                continue
            successful_utterances += 1
            summary.add(segments)
            for record in segments:
                json.dump(
                    record, destination, ensure_ascii=False, separators=(",", ":")
                )
                destination.write("\n")
    temporary_output.replace(args.output)

    args.failures.parent.mkdir(parents=True, exist_ok=True)
    temporary_failures = args.failures.with_name(f".{args.failures.name}.tmp")
    with temporary_failures.open("w", encoding="utf-8") as destination:
        for failure in failures:
            json.dump(failure, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
    temporary_failures.replace(args.failures)

    validation = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "status": "pass" if not failures else "completed_with_failures",
        "storage_mode": "source_slice",
        "config": repository_relative(args.config),
        "config_sha256": sha256(args.config),
        "expected_phonemes": repository_relative(args.expected),
        "expected_phonemes_sha256": sha256(args.expected),
        "alignment_manifest": repository_relative(args.alignments),
        "alignment_manifest_sha256": sha256(args.alignments),
        "vowel_manifest": repository_relative(args.output),
        "vowel_manifest_sha256": sha256(args.output),
        "input_utterance_count": len(expected_records),
        "successful_utterance_count": successful_utterances,
        "failed_utterance_count": len(failures),
        "minimum_duration_sec": args.minimum_duration_sec,
        "near_silent_dbfs": args.near_silent_dbfs,
        "source_checksums_verified": not args.skip_source_checksums,
        **summary.as_dict(),
        "failure_reason_counts": dict(
            sorted(Counter(failure["reason"] for failure in failures).items())
        ),
        "failed_utterances": failures,
        "failures": repository_relative(args.failures),
        "failures_sha256": sha256(args.failures),
    }
    atomic_write_json(args.validation, validation)
    print(
        f"Built {len(summary.durations)} vowel segments from "
        f"{successful_utterances}/{len(expected_records)} utterances; "
        f"failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
