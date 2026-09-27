"""Train-only feature statistics and fixed enrollment / verification selections."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

from .input import InputPipeline, read_pcm_slice
from .manifest import VOWELS, Segment, ids_checksum, json_sha256, seeded_hash


def compute_feature_statistics(
    root: Path,
    segments: Iterable[Segment],
    pipeline: InputPipeline,
    *,
    cohort: int,
    manifest_sha256: str,
    git_commit: str,
    calculation_code_sha256: str,
) -> dict:
    if pipeline.statistics is not None:
        raise ValueError("statistics must use unnormalized log-Mel")
    selected = sorted(segments, key=lambda s: s.segment_id)
    if not selected or any(
        s.split != "train" or s.role != "training" or cohort not in s.cohorts
        for s in selected
    ):
        raise ValueError(
            "feature statistics require only cohort train/training segments"
        )
    if len({s.segment_id for s in selected}) != len(selected):
        raise ValueError("duplicate segment IDs")
    speaker_ids = sorted({s.speaker_id for s in selected})
    if len(speaker_ids) != cohort:
        raise ValueError(f"expected {cohort} train speakers, got {len(speaker_ids)}")
    total = 0
    mean = torch.zeros(64, dtype=torch.float64)
    m2 = torch.zeros(64, dtype=torch.float64)
    for segment in selected:
        pcm = read_pcm_slice(root, segment)
        waveform, _ = pipeline.waveform(pcm, segment.segment_id, mode="center")
        frames = pipeline.log_mel(waveform, normalize=False).to(torch.float64)
        count = frames.shape[1]
        batch_mean = frames.mean(dim=1)
        batch_m2 = ((frames - batch_mean[:, None]) ** 2).sum(dim=1)
        delta = batch_mean - mean
        new_total = total + count
        mean += delta * (count / new_total)
        m2 += batch_m2 + delta.square() * (total * count / new_total)
        total = new_total
    if total <= 0:
        raise ValueError("empty feature statistics")
    variance = m2 / total
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(variance).all()
        or (variance < 0).any()
    ):
        raise ValueError("nonfinite feature statistics")
    result = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "cohort": cohort,
        "rms_enabled": pipeline.rms_enabled,
        "manifest_sha256": manifest_sha256,
        "speaker_ids_sha256": ids_checksum(speaker_ids),
        "segment_ids_sha256": ids_checksum(s.segment_id for s in selected),
        "preprocessing_sha256": pipeline.preprocessing_sha256,
        "segment_count": len(selected),
        "speaker_count": len(speaker_ids),
        "frame_count": total,
        "mean": mean.tolist(),
        "population_variance": variance.tolist(),
        "calculation_git_commit": git_commit,
        "calculation_code_sha256": calculation_code_sha256,
    }
    result["sha256"] = json_sha256(result)
    return result


def make_enrollment(
    segments: Iterable[Segment], *, split: str, seed: int = 20260930, count: int = 10
) -> list[dict]:
    if split not in ("validation", "test") or count < 1:
        raise ValueError("invalid enrollment split or count")
    by_group: dict[tuple[str, str], dict[str, list[Segment]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for segment in segments:
        if segment.split != split or segment.role != "enrollment":
            raise ValueError(
                "enrollment selection accepts only requested split/enrollment"
            )
        by_group[(segment.speaker_id, segment.vowel)][segment.source_file].append(
            segment
        )
    if not by_group:
        raise ValueError("no enrollment segments")
    speakers = {speaker for speaker, _ in by_group}
    if any(
        (speaker, vowel) not in by_group for speaker in speakers for vowel in VOWELS
    ):
        raise ValueError("enrollment missing speaker/vowel group")
    rows = []
    for speaker, vowel in sorted(by_group):
        sources = by_group[(speaker, vowel)]
        if sum(map(len, sources.values())) < count:
            raise ValueError(
                f"fewer than {count} enrollment segments for {speaker}/{vowel}"
            )
        source_order = sorted(
            sources,
            key=lambda source: (
                seeded_hash("enroll_source", seed, split, speaker, vowel, source),
                source.encode("utf-8"),
            ),
        )
        queues = {
            source: sorted(
                sources[source],
                key=lambda s: (
                    seeded_hash(
                        "enroll_segment",
                        seed,
                        split,
                        speaker,
                        vowel,
                        source,
                        s.segment_id,
                    ),
                    s.segment_id.encode("utf-8"),
                ),
            )
            for source in source_order
        }
        chosen = []
        round_number = 0
        while len(chosen) < count:
            for source in source_order:
                if round_number < len(queues[source]):
                    chosen.append(queues[source][round_number])
                    if len(chosen) == count:
                        break
            round_number += 1
        for rank, segment in enumerate(chosen, 1):
            rows.append(
                {
                    "schema_version": 1,
                    "split": split,
                    "speaker_id": speaker,
                    "vowel": vowel,
                    "rank": rank,
                    "segment_id": segment.segment_id,
                    "source_file": segment.source_file,
                    "seed": seed,
                }
            )
    return rows


def make_trials(
    queries: Iterable[Segment],
    enrollment: list[dict],
    *,
    split: str,
    counts: tuple[int, ...] = (1, 5, 10),
) -> Iterator[dict]:
    if split not in ("validation", "test") or not counts or any(c <= 0 for c in counts):
        raise ValueError("invalid trial split or enrollment counts")
    enrollment_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in enrollment:
        if row["split"] != split:
            raise ValueError("enrollment split mismatch")
        enrollment_by_group[(row["speaker_id"], row["vowel"])].append(row)
    speakers = sorted({speaker for speaker, _ in enrollment_by_group})
    if len(speakers) != 15 or any(
        (speaker, vowel) not in enrollment_by_group
        for speaker in speakers
        for vowel in VOWELS
    ):
        raise ValueError("trials require 15 enrolled speakers with all vowels")
    for group in enrollment_by_group.values():
        if sorted(row["rank"] for row in group) != list(range(1, max(counts) + 1)):
            raise ValueError("enrollment ranks are incomplete")
    for query in queries:
        if query.split != split or query.role not in (
            "verification",
            "cross_text_verification",
        ):
            raise ValueError("trial query has incorrect split/role")
        if query.speaker_id not in speakers:
            raise ValueError("query speaker lacks enrollment")
        for claimed in speakers:
            source_ids = {
                row["source_file"]
                for row in enrollment_by_group[(claimed, query.vowel)]
            }
            if query.source_file in source_ids:
                raise ValueError(f"enrollment/query source overlap: {query.segment_id}")
            for count in counts:
                identity = [split, query.role, query.segment_id, claimed, count]
                yield {
                    "schema_version": 1,
                    "trial_id": json_sha256(identity),
                    "split": split,
                    "role": query.role,
                    "speaker_id": query.speaker_id,
                    "claimed_speaker_id": claimed,
                    "vowel": query.vowel,
                    "segment_id": query.segment_id,
                    "source_file": query.source_file,
                    "duration_samples": query.length,
                    "enrollment_count": count,
                    "is_genuine": query.speaker_id == claimed,
                }


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            count += 1
    return count
