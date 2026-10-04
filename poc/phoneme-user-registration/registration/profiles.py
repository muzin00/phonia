"""Deterministic enrollment, versioned profiles and a Phase 5 loading boundary."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from phase3_data.input import read_pcm_slice
from phase3_data.manifest import VOWELS, json_sha256, sha256_file

from .encoder import POLICY, FrozenEncoder


def _identifier(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class EnrollmentSegment:
    segment_id: str
    vowel: str
    source_file: str
    start_frame: int
    end_frame: int
    source_sha256: str | None = None

    def __post_init__(self):
        _identifier(self.segment_id, "segment_id")
        _identifier(self.source_file, "source_file")
        if self.vowel not in VOWELS:
            raise ValueError(f"unsupported vowel: {self.vowel}")
        if (
            type(self.start_frame) is not int
            or type(self.end_frame) is not int
            or self.start_frame < 0
            or self.end_frame <= self.start_frame
        ):
            raise ValueError(f"invalid frame range: {self.segment_id}")
        if self.source_sha256 is not None and (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError(f"invalid source checksum: {self.segment_id}")

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


class IncompleteEnrollment(ValueError):
    def __init__(self, counts: dict, minimum: int, segments: list[dict]):
        self.counts, self.minimum, self.segments = counts, minimum, segments
        shortages = ", ".join(
            f"/{vowel}/: {counts[vowel]}/{minimum}"
            for vowel in VOWELS
            if counts[vowel] < minimum
        )
        super().__init__(f"incomplete enrollment ({shortages}); no profile created")


def load_registration_input(path: Path) -> tuple[str, list[EnrollmentSegment]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("segments"), list)
    ):
        raise ValueError("unsupported registration input")
    user_id = _identifier(data.get("user_id"), "user_id")
    try:
        segments = [EnrollmentSegment(**row) for row in data["segments"]]
    except TypeError as exc:
        raise ValueError(f"invalid registration segment: {exc}") from exc
    return user_id, segments


def _checked_vector(value, dimension: int) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != dimension
        or any(type(x) not in (float, int) for x in value)
    ):
        raise ValueError("invalid profile vector shape or values")
    vector = np.asarray(value, dtype=np.float64)
    if not np.isfinite(vector).all() or not np.isclose(
        np.linalg.norm(vector), 1.0, atol=1e-12, rtol=0
    ):
        raise ValueError("profile vector must be finite and unit normalized")
    return vector


@dataclass(frozen=True)
class UserProfile:
    data: dict

    def __post_init__(self):
        data = self.data
        if (
            data.get("schema_version") != 1
            or data.get("registration_version") != "1.0.0"
            or data.get("embedding_dimension") != 128
            or data.get("aggregation") != POLICY["aggregation"]
            or data.get("policy_version") != POLICY["policy_version"]
        ):
            raise ValueError("unsupported profile schema or policy")
        _identifier(data.get("user_id"), "user_id")
        minimum = data.get("minimum_segments_per_vowel")
        if type(minimum) is not int or minimum < 1:
            raise ValueError("invalid profile minimum segment count")
        encoder = data.get("encoder")
        if not isinstance(encoder, dict):
            raise ValueError("missing encoder identity")  # noqa: TRY004 -- external schema
        for key in (
            "checkpoint_sha256",
            "encoder_config_sha256",
            "feature_statistics_sha256",
            "preprocessing_sha256",
            "implementation_sha256",
        ):
            value = encoder.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"invalid encoder identity: {key}")
        if (
            encoder.get("design_version") != "2.0.0"
            or encoder.get("encoder") != POLICY["selected_encoder"]
            or encoder.get("seed") != POLICY["selected_seed"]
        ):
            raise ValueError("unsupported profile encoder")
        vowels = data.get("vowels")
        records = data.get("segments")
        if not isinstance(vowels, dict) or set(vowels) != set(VOWELS):
            raise ValueError("profile must contain all five vowels")
        if not isinstance(records, list):
            raise ValueError("missing profile segment provenance")  # noqa: TRY004 -- external schema
        counts = Counter()
        seen = set()
        for row in records:
            try:
                segment = EnrollmentSegment(
                    **{
                        key: row[key]
                        for key in (
                            "segment_id",
                            "vowel",
                            "source_file",
                            "start_frame",
                            "end_frame",
                            "source_sha256",
                        )
                    }
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid profile segment provenance") from exc
            if segment.source_sha256 is None or segment.segment_id in seen:
                raise ValueError("missing source checksum or duplicate profile segment")
            seen.add(segment.segment_id)
            if row.get("status") == "used" and row.get("reason") is None:
                counts[segment.vowel] += 1
            elif row.get("status") != "excluded" or row.get("reason") not in (
                "too_short",
                "silent",
                "near_silent",
            ):
                raise ValueError("invalid profile segment status")
        for vowel in VOWELS:
            item = vowels[vowel]
            if (
                not isinstance(item, dict)
                or type(item.get("segment_count")) is not int
                or item["segment_count"] < minimum
                or item["segment_count"] != counts[vowel]
            ):
                raise ValueError(f"profile segment count mismatch: /{vowel}/")
            _checked_vector(item.get("vector"), 128)

    @property
    def user_id(self) -> str:
        return self.data["user_id"]

    @property
    def checksum(self) -> str:
        return json_sha256(self.data)

    def vector(self, vowel: str) -> np.ndarray:
        if vowel not in VOWELS:
            raise ValueError(f"unsupported vowel: {vowel}")
        return _checked_vector(self.data["vowels"][vowel]["vector"], 128).copy()

    def require_compatible(self, encoder: FrozenEncoder) -> None:
        if self.data["encoder"] != encoder.identity:
            raise ValueError("profile encoder/preprocessing is incompatible")


def register_user(
    user_id: str,
    segments: list[EnrollmentSegment],
    encoder: FrozenEncoder,
    *,
    audio_root: Path,
    minimum_segments: int = POLICY["minimum_segments_per_vowel"],
) -> UserProfile:
    _identifier(user_id, "user_id")
    if type(minimum_segments) is not int or minimum_segments < 1:
        raise ValueError("minimum_segments must be a positive integer")
    root = Path(audio_root).resolve()
    seen_ids, seen_ranges = set(), set()
    prepared = []
    checksums = {}
    # Validate all identities/ranges before computing any embeddings.
    for segment in sorted(segments, key=lambda item: item.segment_id):
        source = (root / segment.source_file).resolve()
        if Path(segment.source_file).is_absolute() or not source.is_relative_to(root):
            raise ValueError(f"source outside audio root: {segment.source_file}")
        interval = (source, segment.start_frame, segment.end_frame)
        if segment.segment_id in seen_ids or interval in seen_ranges:
            raise ValueError(
                f"duplicate segment or source interval: {segment.segment_id}"
            )
        seen_ids.add(segment.segment_id)
        seen_ranges.add(interval)
        if source not in checksums:
            checksums[source] = sha256_file(source)
        checksum = checksums[source]
        if segment.source_sha256 is not None and segment.source_sha256 != checksum:
            raise ValueError(f"source checksum mismatch: {segment.segment_id}")
        canonical = EnrollmentSegment(
            segment.segment_id,
            segment.vowel,
            source.relative_to(root).as_posix(),
            segment.start_frame,
            segment.end_frame,
            checksum,
        )
        pcm = read_pcm_slice(root, canonical)
        reason = None
        if pcm.numel() < encoder.pipeline.minimum:
            reason = "too_short"
        else:
            # Match Phase 2: raw interval RMS, before crop/DC removal, rounded to 3 decimals.
            rms = math.sqrt(float(pcm.to(torch.float64).square().mean()))
            if rms == 0:
                reason = "silent"
            elif round(20 * math.log10(rms), 3) < POLICY["near_silent_dbfs"]:
                reason = "near_silent"
        prepared.append((canonical, pcm, reason))
    records = [
        {
            **asdict(segment),
            "status": "used" if reason is None else "excluded",
            "reason": reason,
        }
        for segment, _, reason in prepared
    ]
    counts = {
        vowel: sum(row["vowel"] == vowel and row["status"] == "used" for row in records)
        for vowel in VOWELS
    }
    if any(count < minimum_segments for count in counts.values()):
        raise IncompleteEnrollment(counts, minimum_segments, records)
    vectors = {vowel: [] for vowel in VOWELS}
    for segment, pcm, reason in prepared:
        if reason is None:
            vectors[segment.vowel].append(encoder.embed(pcm, segment.segment_id))
    vowels = {}
    for vowel in VOWELS:
        average = np.mean(vectors[vowel], axis=0, dtype=np.float64)
        norm = np.linalg.norm(average)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"invalid mean embedding: /{vowel}/")
        vowels[vowel] = {
            "segment_count": counts[vowel],
            "vector": (average / norm).tolist(),
        }
    for source, checksum in checksums.items():
        if sha256_file(source) != checksum:
            raise ValueError(f"source changed during registration: {source}")
    return UserProfile(
        {
            "schema_version": 1,
            "registration_version": "1.0.0",
            "policy_version": POLICY["policy_version"],
            "user_id": user_id,
            "embedding_dimension": 128,
            "aggregation": POLICY["aggregation"],
            "minimum_segments_per_vowel": minimum_segments,
            "encoder": dict(encoder.identity),
            "inference": {"device": "cpu", "dtype": "float32", "batch_size": 1},
            "vowels": vowels,
            "segments": records,
        }
    )


def save_profile(profile: UserProfile, path: Path) -> None:
    UserProfile(profile.data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {**profile.data, "sha256": profile.checksum}
    payload = (
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publish with no overwrite, including concurrent registrations.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_profile(path: Path, *, encoder: FrozenEncoder | None = None) -> UserProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid profile document")  # noqa: TRY004 -- external schema
    checksum = data.pop("sha256", None)
    if checksum != json_sha256(data):
        raise ValueError("profile checksum mismatch")
    profile = UserProfile(data)
    if encoder is not None:
        profile.require_compatible(encoder)
    return profile
