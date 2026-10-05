"""Validate query slices and reject enrollment/query source reuse before inference."""

from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from phase3_data.input import read_pcm_slice
from phase3_data.manifest import VOWELS, sha256_file
from registration import EnrollmentSegment, FrozenEncoder, UserProfile

from .policy import POLICY


@dataclass(frozen=True)
class VerificationInput:
    query_id: str
    segments: tuple[EnrollmentSegment, ...]

    def __post_init__(self):
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a nonempty string")
        if not isinstance(self.segments, tuple) or any(
            not isinstance(segment, EnrollmentSegment) for segment in self.segments
        ):
            raise ValueError("segments must be a tuple of EnrollmentSegment values")


class IncompleteVerification(ValueError):
    def __init__(self, query_id: str, counts: dict, records: list[dict]):
        self.query_id, self.counts, self.records = query_id, counts, records
        missing = [vowel for vowel in VOWELS if counts[vowel] < 1]
        self.missing_vowels = missing
        super().__init__(
            f"incomplete verification: missing {missing}; no score created"
        )


def load_verification_input(path: Path) -> VerificationInput:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data["schema_version"] != 1
        or not isinstance(data.get("segments"), list)
    ):
        raise ValueError("unsupported verification input")
    try:
        segments = tuple(EnrollmentSegment(**row) for row in data["segments"])
    except TypeError as exc:
        raise ValueError(f"invalid verification segment: {exc}") from exc
    return VerificationInput(data.get("query_id"), segments)


@dataclass
class PreparedQuery:
    slices: list[tuple[EnrollmentSegment, torch.Tensor, str | None]]
    records: list[dict]
    counts: dict[str, int]
    checksums: dict[Path, str]

    def require_unchanged_sources(self) -> None:
        for source, checksum in self.checksums.items():
            if sha256_file(source) != checksum:
                raise ValueError(f"source changed during verification: {source}")


def prepare_query(
    query: VerificationInput,
    profile: UserProfile,
    encoder: FrozenEncoder,
    *,
    audio_root: Path,
) -> PreparedQuery:
    root = Path(audio_root).resolve()
    enrollment_ids = {row["segment_id"] for row in profile.data["segments"]}
    enrollment_sources = {
        (root / row["source_file"]).resolve() for row in profile.data["segments"]
    }
    enrollment_hashes = {row["source_sha256"] for row in profile.data["segments"]}
    seen_ids, seen_intervals = set(), set()
    checksums, slices, records = {}, [], []
    for segment in sorted(query.segments, key=lambda item: item.segment_id):
        source = (root / segment.source_file).resolve()
        if Path(segment.source_file).is_absolute() or not source.is_relative_to(root):
            raise ValueError(f"source outside audio root: {segment.source_file}")
        if segment.segment_id in seen_ids:
            raise ValueError(f"duplicate query segment ID: {segment.segment_id}")
        seen_ids.add(segment.segment_id)
        if segment.segment_id in enrollment_ids or source in enrollment_sources:
            raise ValueError(f"enrollment/query source overlap: {segment.segment_id}")
        if source not in checksums:
            checksums[source] = sha256_file(source)
        checksum = checksums[source]
        if segment.source_sha256 is not None and segment.source_sha256 != checksum:
            raise ValueError(f"source checksum mismatch: {segment.segment_id}")
        if checksum in enrollment_hashes:
            raise ValueError(
                f"enrollment/query WAV content overlap: {segment.segment_id}"
            )
        interval = (checksum, segment.start_frame, segment.end_frame)
        if interval in seen_intervals:
            raise ValueError(f"duplicate query audio interval: {segment.segment_id}")
        seen_intervals.add(interval)
        canonical = EnrollmentSegment(
            segment.segment_id,
            segment.vowel,
            source.relative_to(root).as_posix(),
            segment.start_frame,
            segment.end_frame,
            checksum,
        )
        try:
            pcm = read_pcm_slice(root, canonical)
        except (wave.Error, EOFError) as exc:
            raise ValueError(
                f"invalid source WAV: {segment.segment_id}: {exc}"
            ) from exc
        reason = None
        if pcm.numel() < encoder.pipeline.minimum:
            reason = "too_short"
        else:
            rms = math.sqrt(float(pcm.to(torch.float64).square().mean()))
            if rms == 0:
                reason = "silent"
            elif round(20 * math.log10(rms), 3) < POLICY["near_silent_dbfs"]:
                reason = "near_silent"
        slices.append((canonical, pcm, reason))
        records.append(
            {
                **asdict(canonical),
                "status": "used" if reason is None else "excluded",
                "reason": reason,
                "score": None,
            }
        )
    counts = {
        vowel: sum(row["vowel"] == vowel and row["status"] == "used" for row in records)
        for vowel in VOWELS
    }
    if any(count < POLICY["minimum_segments_per_vowel"] for count in counts.values()):
        raise IncompleteVerification(query.query_id, counts, records)
    return PreparedQuery(slices, records, counts, checksums)
