"""Fixed validation-only query selection for CLI examples and smoke verification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from phase3_data.manifest import VOWELS, load_segments, seeded_hash, sha256_file
from registration import EnrollmentSegment


def prepare_jvs_input(
    manifest: Path,
    speaker: str,
    *,
    role: str = "verification",
    count: int = 5,
    seed: int = 20261005,
) -> dict:
    if role not in ("verification", "cross_text_verification"):
        raise ValueError("only validation verification roles are available")
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError("count must be positive and seed must be an integer")
    segments = [
        segment
        for segment in load_segments(manifest, split="validation", role=role)
        if segment.speaker_id == speaker
    ]
    grouped = defaultdict(lambda: defaultdict(list))
    for segment in segments:
        grouped[segment.vowel][segment.source_file].append(segment)
    rows = []
    for vowel in VOWELS:
        sources = grouped[vowel]
        if sum(map(len, sources.values())) < count:
            raise ValueError(
                f"fewer than {count} validation {role} segments: {speaker}/{vowel}"
            )
        source_order = sorted(
            sources,
            key=lambda source: (
                seeded_hash("phase5_query_source", seed, speaker, role, vowel, source),
                source,
            ),
        )
        queues = {
            source: sorted(
                sources[source],
                key=lambda segment: (
                    seeded_hash("phase5_query_segment", seed, role, segment.segment_id),
                    segment.segment_id,
                ),
            )
            for source in source_order
        }
        chosen, round_number = [], 0
        while len(chosen) < count:
            for source in source_order:
                if round_number < len(queues[source]):
                    chosen.append(queues[source][round_number])
                    if len(chosen) == count:
                        break
            round_number += 1
        rows.extend(
            asdict(
                EnrollmentSegment(
                    segment.segment_id,
                    segment.vowel,
                    segment.source_file,
                    segment.start_frame,
                    segment.end_frame,
                    segment.source_sha256,
                )
            )
            for segment in chosen
        )
    return {
        "schema_version": 1,
        "query_id": f"{speaker}-{role}-n{count}-s{seed}",
        "segments": rows,
        "origin": {
            "split": "validation",
            "speaker_id": speaker,
            "role": role,
            "segments_per_vowel": count,
            "seed": seed,
            "manifest_sha256": sha256_file(manifest),
        },
    }
