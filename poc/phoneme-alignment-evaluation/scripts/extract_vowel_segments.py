#!/usr/bin/env python3
"""Extract exact and review-context WAV files from normalized vowel intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "alignments" / "mfa" / "normalized.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "segments" / "mfa"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.jsonl"
DEFAULT_VALIDATION = DEFAULT_OUTPUT_DIR / "validation.json"
DEFAULT_CONTEXT_PADDING_SEC = 0.1
DEFAULT_MINIMUM_DURATION_SEC = 0.03
DEFAULT_NEAR_SILENT_DBFS = -50.0


@dataclass(frozen=True)
class WaveSource:
    channels: int
    sample_width: int
    sample_rate_hz: int
    frame_count: int
    compression_type: str
    compression_name: str
    frames: bytes

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.sample_rate_hz

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * self.sample_width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized vowel intervals as exact and context WAV files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--context-padding-sec", type=float, default=DEFAULT_CONTEXT_PADDING_SEC
    )
    parser.add_argument(
        "--minimum-duration-sec", type=float, default=DEFAULT_MINIMUM_DURATION_SEC
    )
    parser.add_argument(
        "--near-silent-dbfs", type=float, default=DEFAULT_NEAR_SILENT_DBFS
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in (
                "utterance_id",
                "vowel_interval_id",
                "vowel_index",
                "normalized_phoneme",
                "source_file",
                "start_sec",
                "end_sec",
            ):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: {field} is required")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def validate_vowel_indices(records: list[dict[str, Any]]) -> None:
    indices_by_utterance: dict[str, list[int]] = defaultdict(list)
    for record in records:
        indices_by_utterance[record["utterance_id"]].append(record["vowel_index"])
    for utterance_id, indices in indices_by_utterance.items():
        if indices != list(range(len(indices))):
            raise ValueError(
                f"{utterance_id}: vowel indices are not contiguous: {indices}"
            )


def resolve_source_file(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def read_wave(path: Path) -> WaveSource:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV is not supported")
        frames = source.readframes(source.getnframes())
        return WaveSource(
            channels=source.getnchannels(),
            sample_width=source.getsampwidth(),
            sample_rate_hz=source.getframerate(),
            frame_count=source.getnframes(),
            compression_type=source.getcomptype(),
            compression_name=source.getcompname(),
            frames=frames,
        )


def seconds_to_frame(seconds: float, sample_rate_hz: int, frame_count: int) -> int:
    return min(max(round(seconds * sample_rate_hz), 0), frame_count)


def slice_frames(source: WaveSource, start_frame: int, end_frame: int) -> bytes:
    if not 0 <= start_frame < end_frame <= source.frame_count:
        raise ValueError(
            f"Invalid frame range {start_frame}:{end_frame} for {source.frame_count} frames"
        )
    byte_start = start_frame * source.bytes_per_frame
    byte_end = end_frame * source.bytes_per_frame
    return source.frames[byte_start:byte_end]


def write_wave(path: Path, source: WaveSource, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(source.channels)
        destination.setsampwidth(source.sample_width)
        destination.setframerate(source.sample_rate_hz)
        destination.setcomptype(source.compression_type, source.compression_name)
        destination.writeframes(frames)


def rms_dbfs(frames: bytes, sample_width: int) -> float | None:
    if sample_width != 2:
        raise ValueError("RMS calculation currently supports only 16-bit PCM WAV")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def extract_record(
    record: dict[str, Any],
    source: WaveSource,
    output_directory: Path,
    context_padding_sec: float,
    minimum_duration_sec: float,
    near_silent_dbfs: float,
) -> dict[str, Any]:
    requested_start = float(record["start_sec"])
    requested_end = float(record["end_sec"])
    if requested_start < 0 or requested_end <= requested_start:
        raise ValueError(f"{record['vowel_interval_id']}: invalid time interval")
    if requested_end > source.duration_sec + (1 / source.sample_rate_hz):
        raise ValueError(
            f"{record['vowel_interval_id']}: interval exceeds source audio"
        )

    exact_start_frame = seconds_to_frame(
        requested_start, source.sample_rate_hz, source.frame_count
    )
    exact_end_frame = seconds_to_frame(
        requested_end, source.sample_rate_hz, source.frame_count
    )
    context_start_frame = seconds_to_frame(
        requested_start - context_padding_sec,
        source.sample_rate_hz,
        source.frame_count,
    )
    context_end_frame = seconds_to_frame(
        requested_end + context_padding_sec,
        source.sample_rate_hz,
        source.frame_count,
    )
    exact_frames = slice_frames(source, exact_start_frame, exact_end_frame)
    context_frames = slice_frames(source, context_start_frame, context_end_frame)

    filename = f"{record['vowel_interval_id']}_{record['normalized_phoneme']}.wav"
    exact_path = output_directory / "exact" / filename
    context_path = output_directory / "context" / filename
    write_wave(exact_path, source, exact_frames)
    write_wave(context_path, source, context_frames)

    actual_start_sec = round(exact_start_frame / source.sample_rate_hz, 9)
    actual_end_sec = round(exact_end_frame / source.sample_rate_hz, 9)
    actual_duration_sec = round(actual_end_sec - actual_start_sec, 9)
    level_dbfs = rms_dbfs(exact_frames, source.sample_width)
    quality_flags: list[str] = []
    if actual_duration_sec < minimum_duration_sec:
        quality_flags.append("short_exact")
    if level_dbfs is None:
        quality_flags.append("silent_exact")
    elif level_dbfs < near_silent_dbfs:
        quality_flags.append("near_silent_exact")
    if context_start_frame == 0:
        quality_flags.append("context_clipped_start")
    if context_end_frame == source.frame_count:
        quality_flags.append("context_clipped_end")

    return {
        "utterance_id": record["utterance_id"],
        "vowel_interval_id": record["vowel_interval_id"],
        "vowel_index": record["vowel_index"],
        "normalized_phoneme": record["normalized_phoneme"],
        "source_file": record["source_file"],
        "aligner": record["aligner"],
        "model_id": record["model_id"],
        "sample_rate_hz": source.sample_rate_hz,
        "channels": source.channels,
        "sample_width_bytes": source.sample_width,
        "requested_start_sec": requested_start,
        "requested_end_sec": requested_end,
        "exact_start_sec": actual_start_sec,
        "exact_end_sec": actual_end_sec,
        "exact_duration_sec": actual_duration_sec,
        "exact_frame_count": exact_end_frame - exact_start_frame,
        "exact_file": project_relative(exact_path),
        "exact_sha256": sha256_file(exact_path),
        "context_padding_sec": context_padding_sec,
        "context_start_sec": round(context_start_frame / source.sample_rate_hz, 9),
        "context_end_sec": round(context_end_frame / source.sample_rate_hz, 9),
        "context_frame_count": context_end_frame - context_start_frame,
        "context_file": project_relative(context_path),
        "context_sha256": sha256_file(context_path),
        "rms_dbfs": level_dbfs,
        "quality_flags": quality_flags,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def main() -> None:
    args = parse_args()
    if args.context_padding_sec < 0:
        raise ValueError("context padding must be non-negative")
    records = read_jsonl(args.input)
    validate_vowel_indices(records)
    aligners = {str(record["aligner"]) for record in records}
    if len(aligners) != 1:
        raise ValueError(f"Expected one aligner, found: {sorted(aligners)}")
    aligner = aligners.pop()
    wave_cache: dict[Path, WaveSource] = {}
    extracted: list[dict[str, Any]] = []

    for record in records:
        source_path = resolve_source_file(record["source_file"])
        if source_path not in wave_cache:
            wave_cache[source_path] = read_wave(source_path)
        extracted.append(
            extract_record(
                record,
                wave_cache[source_path],
                args.output_directory,
                args.context_padding_sec,
                args.minimum_duration_sec,
                args.near_silent_dbfs,
            )
        )

    write_jsonl(args.manifest, extracted)
    flag_counts = Counter(
        flag for record in extracted for flag in record["quality_flags"]
    )
    validation = {
        "aligner": aligner,
        "segment_count": len(extracted),
        "source_audio_count": len(wave_cache),
        "context_padding_sec": args.context_padding_sec,
        "minimum_duration_sec": args.minimum_duration_sec,
        "near_silent_dbfs": args.near_silent_dbfs,
        "flag_counts": dict(sorted(flag_counts.items())),
        "unflagged_segment_count": sum(
            not record["quality_flags"] for record in extracted
        ),
    }
    args.validation.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Extracted {len(extracted)} exact and context vowel segments "
        f"from {len(wave_cache)} source files"
    )
    print(f"Quality flags: {dict(flag_counts) or 'none'}")


if __name__ == "__main__":
    main()
