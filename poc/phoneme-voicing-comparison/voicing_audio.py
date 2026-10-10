"""Match real PCM time, keeping shortened stops at their candidate end."""

from __future__ import annotations

import math
import wave
from functools import lru_cache

import numpy as np
from matched_audio import MINIMUM, VOWELS, capacity
from matched_audio import plan as center_plan

STOPS = ("t", "d", "k", "g")
__all__ = ["MINIMUM", "VOWELS", "capacity", "pcm_rms_dbfs", "plan"]


def plan(candidates, phones, budget):
    rows = center_plan(candidates, phones, budget)
    if rows is None:
        return None
    originals = {row["segment_id"]: row for row in candidates}
    for row in rows:
        if row["vowel"] not in STOPS:
            continue
        original = originals[row["original_segment_id"]]
        size = row["end_frame"] - row["start_frame"]
        first = original["end_frame"] - size
        row.update(
            start_frame=first,
            end_frame=original["end_frame"],
            segment_id=f"{original['segment_id']}--end-{first}-{size}",
        )
    return rows


@lru_cache(maxsize=100000)
def _pcm_rms(root, source, first, last):
    with wave.open(str(root / source), "rb") as audio:
        audio.setpos(first)
        samples = np.frombuffer(audio.readframes(last - first), dtype="<i2")
    if len(samples) != last - first or not len(samples):
        raise ValueError("invalid matched PCM range")
    rms = float(np.sqrt(np.mean((samples.astype(np.float64) / 32768) ** 2)))
    return 20 * math.log10(rms) if rms else -math.inf


def pcm_rms_dbfs(root, row):
    return _pcm_rms(root, row["source_file"], row["start_frame"], row["end_frame"])
