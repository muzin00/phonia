"""Experimental missing-vowel fusion; frozen Phase 5 validation remains intact."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from phase3_data.input import read_pcm_slice
from phase3_data.manifest import VOWELS, sha256_file
from pilot import require_unit
from registration import EnrollmentSegment, UserProfile
from verification import IncompleteVerification
from verification.inputs import prepare_query


def fuse_available(profile, grouped, minimum_vowels):
    if type(minimum_vowels) is not int or minimum_vowels not in (3, 4, 5):
        raise ValueError("minimum_vowels must be 3, 4 or 5")
    if set(profile) != set(VOWELS) or set(grouped) != set(VOWELS):
        raise ValueError("explicit five-vowel references and groups required")
    present = [v for v in VOWELS if grouped[v]]
    if len(present) < minimum_vowels:
        return None
    scores = dict.fromkeys(VOWELS)
    for vowel in present:
        reference = require_unit(profile[vowel])
        if reference.shape != (128,):
            raise ValueError("128-dimensional reference required")
        values = []
        for vector in grouped[vowel]:
            vector = require_unit(vector)
            if vector.shape != reference.shape:
                raise ValueError("embedding dimension mismatch")
            values.append(np.clip(np.dot(reference, vector), -1, 1))
        scores[vowel] = float(np.mean(values, dtype=np.float64))
    return {
        "fused": float(np.mean([scores[v] for v in present], dtype=np.float64)),
        **scores,
    }


@dataclass
class AvailableQuery:
    slices: list
    records: list
    counts: dict
    checksums: dict

    def require_unchanged_sources(self):
        for path, expected in self.checksums.items():
            if sha256_file(path) != expected:
                raise ValueError("partial query source changed")


def prepare_available(query, profile, encoder, *, audio_root):
    UserProfile(profile.data)
    profile.require_compatible(encoder)
    root = Path(audio_root).resolve()
    try:
        original = prepare_query(query, profile, encoder, audio_root=root)
        prepared = AvailableQuery(
            original.slices, original.records, original.counts, original.checksums
        )
    except IncompleteVerification as missing:
        # The original validator completed identity, WAV, overlap and quality checks.
        # Catch only its missing-vowel exception; all other errors remain fatal.
        slices, checksums = [], {}
        for row in missing.records:
            segment = EnrollmentSegment(
                **{
                    k: row[k]
                    for k in (
                        "segment_id",
                        "vowel",
                        "source_file",
                        "start_frame",
                        "end_frame",
                        "source_sha256",
                    )
                }
            )
            path = root / segment.source_file
            checksums[path] = segment.source_sha256
            if row["status"] == "used":
                slices.append((segment, read_pcm_slice(root, segment), None))
        prepared = AvailableQuery(slices, missing.records, missing.counts, checksums)
    prepared.require_unchanged_sources()
    return prepared


def verify_available(profile, query, encoder, *, audio_root, minimum_vowels):
    checksum = profile.checksum
    prepared = prepare_available(query, profile, encoder, audio_root=audio_root)
    grouped = {v: [] for v in VOWELS}
    if sum(n > 0 for n in prepared.counts.values()) >= minimum_vowels:
        for segment, pcm, reason in prepared.slices:
            if reason is None:
                grouped[segment.vowel].append(encoder.embed(pcm, segment.segment_id))
    result = fuse_available(
        {v: profile.vector(v) for v in VOWELS}, grouped, minimum_vowels
    )
    prepared.require_unchanged_sources()
    if checksum != profile.checksum:
        raise ValueError("partial query profile changed")
    return result
