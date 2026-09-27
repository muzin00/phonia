"""Manifest validation and compact indexing for source-slice WAV segments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

VOWELS = ("a", "i", "u", "e", "o")
COHORTS = (10, 25, 50, 70)
ROLES = {
    "train": {"training"},
    "validation": {"enrollment", "verification", "cross_text_verification"},
    "test": {"enrollment", "verification", "cross_text_verification"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def seeded_hash(*values: object) -> int:
    data = json.dumps(
        values, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return int.from_bytes(hashlib.sha256(data.encode("utf-8")).digest(), "big")


def ids_checksum(ids: Iterator[str] | list[str]) -> str:
    return json_sha256(sorted(ids))


@dataclass(frozen=True, slots=True)
class Segment:
    segment_id: str
    speaker_id: str
    vowel: str
    split: str
    role: str
    cohorts: tuple[int, ...]
    source_file: str
    start_frame: int
    end_frame: int
    source_sha256: str | None = None

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def iter_segments(manifest: Path) -> Iterator[Segment]:
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank manifest line {line_number}")
            row = json.loads(line)
            try:
                segment = Segment(
                    segment_id=row["vowel_interval_id"],
                    speaker_id=row["speaker_id"],
                    vowel=row["normalized_phoneme"],
                    split=row["split"],
                    role=row["evaluation_role"],
                    cohorts=tuple(row["learning_curve_cohorts"]),
                    source_file=row["source_file"],
                    start_frame=row["start_frame"],
                    end_frame=row["end_frame"],
                    source_sha256=row.get("source_sha256"),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
            if (
                row.get("schema_version") != 1
                or row.get("design_version") != "1.0.0"
                or row.get("sample_rate_hz") != 24000
                or row.get("channels") != 1
                or row.get("sample_width_bytes") != 2
                or row.get("storage_mode") != "source_slice"
                or row.get("frame_count") != segment.length
                or "near_silent" in row.get("quality_flags", [])
                or segment.vowel not in VOWELS
                or segment.split not in ROLES
                or segment.role not in ROLES[segment.split]
                or not isinstance(segment.start_frame, int)
                or not isinstance(segment.end_frame, int)
                or segment.start_frame < 0
                or segment.length < 720
                or any(c not in COHORTS for c in segment.cohorts)
                or (segment.split == "train" and 70 not in segment.cohorts)
                or (segment.split != "train" and segment.cohorts)
            ):
                raise ValueError(
                    f"invalid Phase 3 segment on line {line_number}: {segment.segment_id}"
                )
            yield segment


def load_segments(
    manifest: Path,
    *,
    split: str | None = None,
    role: str | None = None,
    cohort: int | None = None,
) -> list[Segment]:
    if split is not None and split not in ROLES:
        raise ValueError(f"invalid split: {split}")
    if cohort is not None and cohort not in COHORTS:
        raise ValueError(f"invalid cohort: {cohort}")
    if cohort is not None and (split != "train" or role != "training"):
        raise ValueError("cohorts may only select train/training segments")
    seen: set[str] = set()
    result = []
    for segment in iter_segments(manifest):
        if split is not None and segment.split != split:
            continue
        if role is not None and segment.role != role:
            continue
        if cohort is not None and cohort not in segment.cohorts:
            continue
        if segment.segment_id in seen:
            raise ValueError(f"duplicate segment ID: {segment.segment_id}")
        seen.add(segment.segment_id)
        result.append(segment)
    return result
