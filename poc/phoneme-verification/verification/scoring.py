"""Same-vowel cosine scores, equal vowel fusion and auditable result files."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from phase3_data.manifest import VOWELS, json_sha256, sha256_file
from registration import EnrollmentSegment, FrozenEncoder, UserProfile

from .inputs import VerificationInput, prepare_query
from .policy import POLICY


def _score(value) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not -1 <= value <= 1
    ):
        raise ValueError("score must be finite and in [-1, 1]")
    return float(value)


def _checksum(value) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("invalid SHA-256 identity")


@dataclass(frozen=True)
class VerificationResult:
    data: dict

    def __post_init__(self):
        data = self.data
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("verification_version") != "1.0.0"
            or data.get("policy") != POLICY
            or data.get("policy_sha256") != json_sha256(POLICY)
        ):
            raise ValueError("unsupported verification result schema or policy")
        for name in ("query_id", "user_id"):
            if not isinstance(data.get(name), str) or not data[name].strip():
                raise ValueError(f"invalid result {name}")
        for name in ("profile_sha256", "implementation_sha256"):
            _checksum(data.get(name))
        encoder = data.get("encoder")
        if not isinstance(encoder, dict):
            raise ValueError("missing encoder identity")  # noqa: TRY004 -- external schema
        for name in (
            "checkpoint_sha256",
            "encoder_config_sha256",
            "feature_statistics_sha256",
            "preprocessing_sha256",
            "implementation_sha256",
        ):
            _checksum(encoder.get(name))
        if data.get("inference") != {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
        }:
            raise ValueError("unsupported inference conditions")
        if not isinstance(data.get("runtime"), dict) or any(
            not isinstance(data["runtime"].get(name), str)
            for name in ("python", "numpy", "torch", "platform", "machine")
        ):
            raise ValueError("missing runtime versions")
        rows, vowels = data.get("segments"), data.get("vowels")
        if (
            not isinstance(rows, list)
            or not isinstance(vowels, dict)
            or set(vowels) != set(VOWELS)
        ):
            raise ValueError("result must contain query provenance and all five vowels")
        grouped = defaultdict(list)
        seen_ids, seen_intervals = set(), set()
        for row in rows:
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
                raise ValueError("invalid query provenance") from exc
            _checksum(segment.source_sha256)
            interval = (segment.source_sha256, segment.start_frame, segment.end_frame)
            if segment.segment_id in seen_ids or interval in seen_intervals:
                raise ValueError("duplicate result query segment")
            seen_ids.add(segment.segment_id)
            seen_intervals.add(interval)
            if row.get("status") == "used" and row.get("reason") is None:
                grouped[segment.vowel].append(_score(row.get("score")))
            elif (
                row.get("status") != "excluded"
                or row.get("reason") not in ("too_short", "silent", "near_silent")
                or row.get("score") is not None
            ):
                raise ValueError("invalid result segment status or score")
        for vowel in VOWELS:
            item = vowels[vowel]
            if (
                not isinstance(item, dict)
                or type(item.get("query_segment_count")) is not int
                or item["query_segment_count"] != len(grouped[vowel])
                or item["query_segment_count"] < POLICY["minimum_segments_per_vowel"]
                or type(item.get("enrollment_segment_count")) is not int
                or item["enrollment_segment_count"] < 1
            ):
                raise ValueError(f"invalid result counts: /{vowel}/")
            expected = float(np.mean(grouped[vowel], dtype=np.float64))
            if not math.isclose(
                _score(item.get("score")), expected, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"result vowel score mismatch: /{vowel}/")
        expected = float(
            np.mean([vowels[vowel]["score"] for vowel in VOWELS], dtype=np.float64)
        )
        if not math.isclose(
            _score(data.get("verification_score")), expected, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("result fusion score mismatch")

    @property
    def score(self) -> float:
        return self.data["verification_score"]

    @property
    def checksum(self) -> str:
        return json_sha256(self.data)


def verify(
    profile: UserProfile,
    query: VerificationInput,
    encoder: FrozenEncoder,
    *,
    audio_root: Path,
) -> VerificationResult:
    # Revalidate mutable profile contents and require the exact inference identity.
    UserProfile(profile.data)
    profile.require_compatible(encoder)
    profile_checksum = profile.checksum
    prepared = prepare_query(query, profile, encoder, audio_root=audio_root)
    references = {vowel: profile.vector(vowel) for vowel in VOWELS}
    grouped = defaultdict(list)
    for (segment, pcm, reason), row in zip(
        prepared.slices, prepared.records, strict=True
    ):
        if reason is None:
            embedding = np.asarray(
                encoder.embed(pcm, segment.segment_id), dtype=np.float64
            )
            if (
                embedding.shape != (128,)
                or not np.isfinite(embedding).all()
                or not np.isclose(np.linalg.norm(embedding), 1.0, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"invalid unit embedding: {segment.segment_id}")
            # Unit vectors: cosine equals the dot product. Clamp round-off only.
            score = float(
                np.clip(np.dot(references[segment.vowel], embedding), -1.0, 1.0)
            )
            row["score"] = score
            grouped[segment.vowel].append(score)
    vowels = {
        vowel: {
            "enrollment_segment_count": profile.data["vowels"][vowel]["segment_count"],
            "query_segment_count": prepared.counts[vowel],
            "score": float(np.mean(grouped[vowel], dtype=np.float64)),
        }
        for vowel in VOWELS
    }
    prepared.require_unchanged_sources()
    if profile.checksum != profile_checksum:
        raise ValueError("profile changed during verification")
    return VerificationResult(
        {
            "schema_version": 1,
            "verification_version": "1.0.0",
            "query_id": query.query_id,
            "user_id": profile.user_id,
            "profile_sha256": profile_checksum,
            "encoder": copy.deepcopy(encoder.identity),
            "inference": {"device": "cpu", "dtype": "float32", "batch_size": 1},
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "platform": sys.platform,
                "machine": platform.machine(),
            },
            "policy": copy.deepcopy(POLICY),
            "policy_sha256": json_sha256(POLICY),
            "implementation_sha256": json_sha256(
                {
                    name: sha256_file(Path(__file__).parent / name)
                    for name in ("inputs.py", "scoring.py", "policy.py")
                }
            ),
            "vowels": vowels,
            "verification_score": float(
                np.mean([vowels[vowel]["score"] for vowel in VOWELS], dtype=np.float64)
            ),
            "segments": prepared.records,
        }
    )


def write_document(path: Path, document: dict) -> None:
    """Atomically publish a deterministic JSON document without replacing files."""
    path = Path(path)
    payload = (
        json.dumps(
            document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def save_result(result: VerificationResult, path: Path) -> None:
    VerificationResult(result.data)
    write_document(path, {**result.data, "sha256": result.checksum})


def load_result(
    path: Path, *, encoder: FrozenEncoder | None = None
) -> VerificationResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.pop("sha256", None) != json_sha256(data):
        raise ValueError("verification result checksum mismatch")
    result = VerificationResult(data)
    if encoder is not None and result.data["encoder"] != encoder.identity:
        raise ValueError("verification result encoder/preprocessing is incompatible")
    return result
