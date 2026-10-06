"""Limit the acquired query recording before choosing complete vowel anchors."""

from __future__ import annotations

import hashlib
import json

from comparison_inputs import effective_window, union_frames


def recording_window(frames: int, maximum_seconds: int | None) -> tuple[int, int]:
    """Use the common center and never lengthen a recording shorter than the cap."""
    if type(frames) is not int or frames < 1:
        raise ValueError("invalid source frame count")
    if maximum_seconds is not None and (
        type(maximum_seconds) is not int or maximum_seconds < 1
    ):
        raise ValueError("duration cap must be positive integer seconds")
    length = frames if maximum_seconds is None else min(frames, maximum_seconds * 24000)
    start = (frames - length) // 2
    return start, start + length


def describe_query_window(
    query: dict, source: dict, maximum_seconds: int | None, *, margin: int = 480
) -> dict:
    """Do not relabel partial vowels or read context outside the acquired window."""
    if (
        query["source_file"] != source["source_file"]
        or query["source_sha256"] != source["source_sha256"]
        or query["speaker_id"] != source["speaker_id"]
        or query["split"] != source["split"]
        or query["role"] != source["evaluation_role"]
    ):
        raise ValueError("query/source identity mismatch")
    first, last = recording_window(source["frame_count"], maximum_seconds)
    counts = {vowel: 0 for vowel in "aiueo"}
    anchors, excluded, seen = [], [], set()
    exact, context = [], []
    for segment in sorted(query["segments"], key=lambda row: row["segment_id"]):
        if segment["segment_id"] in seen:
            raise ValueError("duplicate query anchor")
        seen.add(segment["segment_id"])
        if (
            segment["source_file"] != source["source_file"]
            or segment["source_sha256"] != source["source_sha256"]
            or segment["vowel"] not in counts
        ):
            raise ValueError("anchor/source identity mismatch")
        start, end = segment["start_frame"], segment["end_frame"]
        if not 0 <= start < end <= source["frame_count"]:
            raise ValueError("invalid original anchor interval")
        if not first <= start < end <= last:
            excluded.append(segment["segment_id"])
            continue
        core = effective_window(start - first, end - first, last - first)
        extended = effective_window(
            start - first, end - first, last - first, margin=margin
        )
        core = first + core[0], first + core[1]
        extended = first + extended[0], first + extended[1]
        counts[segment["vowel"]] += 1
        exact.append(core)
        context.append(extended)
        anchors.append(
            {
                "segment_id": segment["segment_id"],
                "vowel": segment["vowel"],
                "original_frames": [start, end],
                "exact_frames": list(core),
                "context_frames": list(extended),
            }
        )
    window_identity = [source["source_sha256"], first, last]
    return {
        "query_id": query["query_id"],
        "speaker_id": query["speaker_id"],
        "role": query["role"],
        "split": query["split"],
        "source_file": query["source_file"],
        "source_sha256": query["source_sha256"],
        "maximum_seconds": maximum_seconds,
        "recording_frames": [first, last],
        "actual_recording_seconds": (last - first) / 24000,
        "source_shorter_than_cap": maximum_seconds is not None
        and source["frame_count"] < maximum_seconds * 24000,
        "window_id": hashlib.sha256(
            json.dumps(window_identity, separators=(",", ":")).encode()
        ).hexdigest(),
        "counts": counts,
        "complete_five_vowels": all(counts.values()),
        "anchors": anchors,
        "excluded_anchor_ids": excluded,
        "used_audio": {
            method: {
                "unique_frames": union_frames(intervals),
                "processed_frames": sum(end - start for start, end in intervals),
            }
            for method, intervals in (
                ("vowel_exact", exact),
                ("vowel_context20", context),
            )
        },
    }
