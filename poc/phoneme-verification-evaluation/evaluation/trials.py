"""Score-independent enrollment, utterance queries and complete trial matrices."""

import json
import wave
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from phase3_data.artifacts import make_enrollment
from phase3_data.manifest import VOWELS, iter_segments, json_sha256, sha256_file
from registration import EnrollmentSegment

from . import ROOT


def checked_source(row: dict, root: Path) -> Path:
    source = (root / row["source_file"]).resolve()
    if Path(row["source_file"]).is_absolute() or not source.is_relative_to(
        root.resolve()
    ):
        raise ValueError("source outside audio root")
    if sha256_file(source) != row["source_sha256"]:
        raise ValueError(f"source checksum mismatch: {source}")
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
            wav.getnframes(),
        ) != (24000, 1, 2, "NONE", row["frame_count"]):
            raise ValueError(f"source PCM metadata mismatch: {source}")
    return source


def to_input(segment) -> dict:
    return asdict(
        EnrollmentSegment(
            segment.segment_id,
            segment.vowel,
            segment.source_file,
            segment.start_frame,
            segment.end_frame,
            segment.source_sha256,
        )
    )


def build_trials(
    queries: list[dict], speakers: list[str], counts: list[int], version: str
):
    if (
        len(speakers) != len(set(speakers))
        or len(speakers) < 2
        or not counts
        or len(set(counts)) != len(counts)
    ):
        raise ValueError("invalid trial speakers or counts")
    seen = set()
    for query in queries:
        if query["speaker_id"] not in speakers or query["query_id"] in seen:
            raise ValueError("query identity or speaker mismatch")
        seen.add(query["query_id"])
        for claimed in speakers:
            for count in counts:
                yield {
                    "schema_version": 1,
                    "trial_id": json_sha256(
                        [version, query["query_id"], claimed, count]
                    ),
                    "query_id": query["query_id"],
                    "split": query["split"],
                    "role": query["role"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": claimed,
                    "enrollment_count": count,
                    "is_genuine": query["speaker_id"] == claimed,
                }


def prepare_split(protocol: dict, split: str, *, audio_root: Path = ROOT) -> dict:
    if split not in ("validation", "test"):
        raise ValueError("only validation/test can be evaluated")
    settings = json.loads(
        (ROOT / protocol["inputs"]["speaker_split"]["path"]).read_text()
    )
    speaker_splits = settings["speaker_splits"]
    all_speakers = [s for names in speaker_splits.values() for s in names]
    if len(all_speakers) != len(set(all_speakers)):
        raise ValueError("speaker split leakage")
    speakers = sorted(speaker_splits[split])
    if len(speakers) != protocol["splits"]["expected_speakers_per_split"]:
        raise ValueError("incorrect speaker count")
    original = {}
    with (ROOT / protocol["inputs"]["utterance_manifest"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != split:
                continue
            if row["speaker_id"] not in speakers or row["evaluation_role"] not in (
                "enrollment",
                *protocol["splits"]["roles"],
            ):
                raise ValueError("original utterance assignment mismatch")
            if row["source_file"] in original:
                raise ValueError("duplicate original source")
            checked_source(row, audio_root)
            original[row["source_file"]] = row
    enrollment_sources = {
        row["source_sha256"]
        for row in original.values()
        if row["evaluation_role"] == "enrollment"
    }
    query_sources = {
        row["source_sha256"]
        for row in original.values()
        if row["evaluation_role"] != "enrollment"
    }
    if enrollment_sources & query_sources:
        raise ValueError("enrollment/query source content leakage")
    grouped = defaultdict(list)
    segments = {}
    for segment in iter_segments(ROOT / protocol["inputs"]["segment_manifest"]["path"]):
        if segment.split != split:
            continue
        row = original.get(segment.source_file)
        if (
            row is None
            or segment.speaker_id != row["speaker_id"]
            or segment.role != row["evaluation_role"]
            or segment.source_sha256 != row["source_sha256"]
            or segment.end_frame > row["frame_count"]
        ):
            raise ValueError("segment/original utterance mismatch")
        if segment.segment_id in segments:
            raise ValueError("duplicate segment ID")
        segments[segment.segment_id] = segment
        grouped[segment.source_file].append(segment)
    selected = make_enrollment(
        (s for s in segments.values() if s.role == "enrollment"),
        split=split,
        seed=protocol["enrollment"]["seed"],
        count=10,
    )
    if {row["speaker_id"] for row in selected} != set(speakers):
        raise ValueError("missing enrollment speaker")
    enrollment = []
    for speaker in speakers:
        for count in protocol["enrollment"]["counts_per_vowel"]:
            chosen = [
                row
                for row in selected
                if row["speaker_id"] == speaker and row["rank"] <= count
            ]
            if Counter(row["vowel"] for row in chosen) != Counter(
                {v: count for v in VOWELS}
            ):
                raise ValueError("incomplete enrollment ranks")
            enrollment.append(
                {
                    "schema_version": 1,
                    "split": split,
                    "user_id": speaker,
                    "enrollment_count": count,
                    "ranks": chosen,
                    "segments": [
                        to_input(segments[row["segment_id"]]) for row in chosen
                    ],
                }
            )
    queries = []
    for row in sorted(
        original.values(), key=lambda r: (r["evaluation_role"], r["utterance_id"])
    ):
        if row["evaluation_role"] == "enrollment":
            continue
        source_segments = sorted(
            grouped[row["source_file"]], key=lambda s: s.segment_id
        )
        counts = Counter(s.vowel for s in source_segments)
        queries.append(
            {
                "schema_version": 1,
                "query_id": json_sha256(
                    [split, row["evaluation_role"], row["utterance_id"]]
                ),
                "split": split,
                "role": row["evaluation_role"],
                "speaker_id": row["speaker_id"],
                "utterance_id": row["utterance_id"],
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "session_id": row["session_id"],
                "duration_sec": row["frame_count"] / 24000,
                "segment_duration_sec": sum(s.length for s in source_segments) / 24000,
                "counts": {v: counts[v] for v in VOWELS},
                "segments": [to_input(s) for s in source_segments],
            }
        )
    if any(
        not any(q["speaker_id"] == speaker and q["role"] == role for q in queries)
        for speaker in speakers
        for role in protocol["splits"]["roles"]
    ):
        raise ValueError("missing speaker/role utterances")
    return {
        "split": split,
        "speakers": speakers,
        "enrollment": enrollment,
        "queries": queries,
    }
