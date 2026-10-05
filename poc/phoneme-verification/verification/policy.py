"""Versioned, deliberately uncalibrated initial scoring policy."""

import json
from pathlib import Path

from registration.encoder import POLICY as REGISTRATION_POLICY

POLICY = json.loads(
    (
        Path(__file__).resolve().parents[1] / "config/verification-policy.json"
    ).read_text()
)
if any(
    POLICY.get(name) != value
    for name, value in {
        "schema_version": 1,
        "policy_version": "1.0.0",
        "minimum_segments_per_vowel": 1,
        "segment_score": "cosine_similarity",
        "within_vowel_aggregation": "arithmetic_mean_of_segment_cosines_float64",
        "fusion": "equal_mean_of_five_vowels",
        "source_separation": "different_wav_paths_and_sha256_from_enrollment",
    }.items()
):
    raise ValueError("unsupported verification policy")
if (
    POLICY["required_vowels"] != REGISTRATION_POLICY["required_vowels"]
    or POLICY["near_silent_dbfs"] != REGISTRATION_POLICY["near_silent_dbfs"]
):
    raise ValueError("verification and registration quality policies differ")
