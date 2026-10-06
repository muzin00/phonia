"""Score-independent geometry for matched Phase 7 audio budgets."""

from __future__ import annotations

from collections import defaultdict


def effective_window(
    start: int,
    end: int,
    frames: int,
    *,
    margin: int = 0,
    maximum: int = 6000,
) -> tuple[int, int]:
    """Expand a source interval, clip to the WAV, then apply Phase 3 center crop."""
    if not 0 <= start < end <= frames or margin < 0 or maximum < 1:
        raise ValueError("invalid source interval or crop parameters")
    start, end = max(0, start - margin), min(frames, end + margin)
    offset = max(0, end - start - maximum) // 2
    return start + offset, min(end, start + offset + maximum)


def union_frames(intervals: list[tuple[int, int]]) -> int:
    """Count each source frame once, including when context windows overlap."""
    if any(not 0 <= start < end for start, end in intervals):
        raise ValueError("invalid interval for union")
    total = 0
    previous_end = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def matched_center_window(frames: int, budget: int) -> tuple[int, int] | None:
    if frames < 1 or not 0 <= budget <= frames:
        raise ValueError("audio budget exceeds source or is negative")
    if budget == 0:
        return None
    start = (frames - budget) // 2
    return start, start + budget


def describe_audio(
    segments: list[dict], sources: dict[str, dict], *, margin: int = 480
) -> dict:
    exact, context = defaultdict(list), defaultdict(list)
    changed = retained = exact_crops = context_crops = 0
    seen = set()
    for segment in segments:
        if segment["segment_id"] in seen:
            raise ValueError("duplicate anchor segment")
        seen.add(segment["segment_id"])
        source = sources[segment["source_file"]]
        if segment["source_sha256"] != source["source_sha256"]:
            raise ValueError("segment/source checksum identity mismatch")
        start, end = segment["start_frame"], segment["end_frame"]
        frames = source["frame_count"]
        core = effective_window(start, end, frames)
        extended = effective_window(start, end, frames, margin=margin)
        exact[segment["source_file"]].append(core)
        context[segment["source_file"]].append(extended)
        changed += core != extended
        retained += (
            extended[1]
            - extended[0]
            - max(0, min(extended[1], end) - max(extended[0], start))
        )
        exact_crops += end - start > 6000
        context_crops += min(frames, end + margin) - max(0, start - margin) > 6000
    by_source = {}
    for filename in sorted(exact):
        source = sources[filename]
        budgets = {
            "vowel_exact": union_frames(exact[filename]),
            "vowel_context20": union_frames(context[filename]),
        }
        by_source[filename] = {
            "source_frames": source["frame_count"],
            "source_sha256": source["source_sha256"],
            "budgets": budgets,
            "ecapa_duration_windows": {
                method: matched_center_window(source["frame_count"], budget)
                for method, budget in budgets.items()
            },
        }
    return {
        "anchor_count": len(segments),
        "source_count": len(by_source),
        "sources": by_source,
        "unique_frames": {
            method: sum(row["budgets"][method] for row in by_source.values())
            for method in ("vowel_exact", "vowel_context20")
        },
        "processed_frames": {
            "vowel_exact": sum(
                end - start for rows in exact.values() for start, end in rows
            ),
            "vowel_context20": sum(
                end - start for rows in context.values() for start, end in rows
            ),
        },
        "whole_source_frames": sum(row["source_frames"] for row in by_source.values()),
        "context_changed_anchor_count": changed,
        "retained_context_frames_including_repeated_frames": retained,
        "exact_cropped_anchor_count": exact_crops,
        "context_cropped_anchor_count": context_crops,
    }
