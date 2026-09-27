"""Frozen-checkpoint enrollment, cosine trials and validation metrics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import Segment, load_segments
from torch.utils.data import DataLoader

from .metrics import (
    eer_operating_threshold,
    error_rates,
    far_target_threshold,
    json_threshold,
    roc_eer,
)
from .models import SpeakerEncoder

VOWELS = ("a", "i", "u", "e", "o")
COUNTS = (1, 5, 10)
ROLES = ("verification", "cross_text_verification")


def _read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _embeddings(
    model: SpeakerEncoder,
    pipeline: InputPipeline,
    root: Path,
    segments: list[Segment],
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    if not segments:
        raise ValueError("no segments for evaluation")
    dataset = SegmentDataset(
        root, segments, pipeline, kind=model.input_kind, mode="center"
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_segments
    )
    model.eval()
    result = {}
    with torch.no_grad():
        for batch in loader:
            embedding = model(batch["input"].to(device), batch["mask"].to(device))
            vectors = embedding.detach().to(device="cpu", dtype=torch.float64).numpy()
            norms = np.linalg.norm(vectors, axis=1)
            if not np.isfinite(vectors).all() or np.any(norms < 1e-12):
                raise ValueError("invalid evaluation embedding")
            for segment_id, vector, norm in zip(batch["segment_ids"], vectors, norms):
                result[segment_id] = vector / norm
    return result


def _profiles(
    enrollment: list[dict], embeddings: dict[str, np.ndarray]
) -> dict[tuple[str, str, int], np.ndarray]:
    grouped = defaultdict(list)
    for row in enrollment:
        grouped[(row["speaker_id"], row["vowel"])].append(row)
    result = {}
    for (speaker, vowel), rows in grouped.items():
        rows.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in rows] != list(range(1, 11)):
            raise ValueError("enrollment ranks must be 1..10")
        for count in COUNTS:
            average = np.mean(
                [embeddings[row["segment_id"]] for row in rows[:count]], axis=0
            )
            norm = np.linalg.norm(average)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError("invalid enrollment profile")
            result[(speaker, vowel, count)] = average / norm
    if len(result) != 15 * 5 * 3:
        raise ValueError("expected 15 speakers x five vowels x three profile sizes")
    return result


def _group_report(
    scores: list[float], labels: list[bool], *, curve: bool = False
) -> dict:
    result = roc_eer(scores, labels)
    report = {
        "eer": result["eer"],
        "genuine_trials": int(sum(labels)),
        "impostor_trials": len(labels) - int(sum(labels)),
        "trial_count": len(labels),
    }
    if curve:
        report["roc_det_curve"] = {
            "thresholds": result["thresholds"],
            "far": result["far"],
            "frr": result["frr"],
        }
    return report


def _operating(scores: list[float], labels: list[bool]) -> dict:
    result = {}
    for name, target in (("far_1pct", 0.01), ("far_0_1pct", 0.001)):
        threshold, rates = far_target_threshold(scores, labels, target)
        result[name] = {"threshold": json_threshold(threshold), **rates}
    threshold, rates = eer_operating_threshold(scores, labels)
    result["eer_operating"] = {"threshold": json_threshold(threshold), **rates}
    return result


def _numeric_threshold(value: float | str) -> float:
    if value == "-inf":
        return -math.inf
    if value == "+inf":
        return math.inf
    return float(value)


def summarize_trials(
    groups: dict[tuple[str, int, str], tuple[list[float], list[bool]]],
    *,
    partial: bool = False,
) -> tuple[dict, dict]:
    metrics = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "partial": partial,
        "roles": {},
        "primary_metric": "validation_verification_10_enrollment_macro_eer",
    }
    thresholds = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "source_role": "verification",
        "per_count": {},
    }
    for role in ROLES:
        metrics["roles"][role] = {}
        for count in COUNTS:
            by_vowel = {}
            pooled_scores, pooled_labels = [], []
            for vowel in VOWELS:
                scores, labels = groups.get((role, count, vowel), ([], []))
                by_vowel[vowel] = _group_report(
                    scores, labels, curve=role == "verification" and count == 10
                )
                pooled_scores.extend(scores)
                pooled_labels.extend(labels)
            values = [by_vowel[vowel]["eer"] for vowel in VOWELS]
            report = {
                "vowels": by_vowel,
                "macro_eer": sum(values) / 5
                if all(v is not None for v in values)
                else None,
                "pooled": _group_report(
                    pooled_scores,
                    pooled_labels,
                    curve=role == "verification" and count == 10,
                ),
            }
            metrics["roles"][role][str(count)] = report
            if role == "verification":
                thresholds["per_count"][str(count)] = {}
                for vowel in VOWELS:
                    scores, labels = groups.get((role, count, vowel), ([], []))
                    if scores and any(labels) and not all(labels):
                        thresholds["per_count"][str(count)][vowel] = _operating(
                            scores, labels
                        )
                if pooled_scores and any(pooled_labels) and not all(pooled_labels):
                    thresholds["per_count"][str(count)]["pooled"] = _operating(
                        pooled_scores, pooled_labels
                    )
    for role in ROLES:
        for count in COUNTS:
            for vowel in VOWELS:
                score, label = groups.get((role, count, vowel), ([], []))
                reference = thresholds["per_count"].get(str(count), {}).get(vowel)
                if reference is not None:
                    metrics["roles"][role][str(count)]["vowels"][vowel][
                        "fixed_threshold_rates"
                    ] = {
                        name: error_rates(
                            score,
                            label,
                            _numeric_threshold(reference[name]["threshold"]),
                        )
                        for name in ("far_1pct", "far_0_1pct", "eer_operating")
                    }
    metrics["macro_eer"] = metrics["roles"]["verification"]["10"]["macro_eer"]
    return metrics, thresholds


def _stratum_values(segment: Segment) -> dict[str, list[str]]:
    milliseconds = segment.length / 24
    if milliseconds < 50:
        duration = "30-49ms"
    elif milliseconds < 100:
        duration = "50-99ms"
    else:
        duration = "100ms-plus"
    return {
        "duration": [duration],
        "cropped_over_250ms": ["yes" if segment.length > 6000 else "no"],
        "is_devoiced": [
            "unknown"
            if segment.is_devoiced is None
            else str(segment.is_devoiced).lower()
        ],
        "is_long": [
            "unknown" if segment.is_long is None else str(segment.is_long).lower()
        ],
        "quality_flags": list(segment.quality_flags) or ["none"],
    }


def _strata_report(strata: dict, thresholds: dict) -> dict:
    result = {}
    for (dimension, value, vowel), group in strata.items():
        report = _group_report(group["scores"], group["labels"])
        report.update(
            {
                "query_count": len(group["queries"]),
                "speaker_count": len(group["speakers"]),
            }
        )
        reference = thresholds["per_count"].get("10", {}).get(vowel)
        if reference is not None:
            report["fixed_threshold_rates"] = {
                name: error_rates(
                    group["scores"],
                    group["labels"],
                    _numeric_threshold(reference[name]["threshold"]),
                )
                for name in ("far_1pct", "far_0_1pct", "eer_operating")
            }
        result.setdefault(dimension, {}).setdefault(value, {})[vowel] = report
    for values in result.values():
        for vowels in values.values():
            eers = [vowels.get(vowel, {}).get("eer") for vowel in VOWELS]
            vowels["macro_eer"] = (
                sum(eers) / 5 if all(eer is not None for eer in eers) else None
            )
    return result


def evaluate_validation(
    model: SpeakerEncoder,
    pipeline: InputPipeline,
    *,
    root: Path,
    manifest: Path,
    selection_root: Path,
    output_root: Path,
    device: str = "cpu",
    batch_size: int = 32,
    max_queries_per_vowel: int | None = None,
) -> dict:
    root, manifest, selection_root, output_root = map(
        Path, (root, manifest, selection_root, output_root)
    )
    if model.input_kind == "log_mel" and pipeline.statistics is None:
        raise ValueError("log-Mel evaluation requires train feature statistics")
    metadata = json.loads((selection_root / "data.json").read_text(encoding="utf-8"))
    if metadata.get("split") != "validation" or metadata[
        "manifest_sha256"
    ] != sha256_file(manifest):
        raise ValueError("validation selection/manifest checksum mismatch")
    enrollment_path = selection_root / "selections/enrollment-segments.jsonl"
    trials_path = selection_root / "selections/trials.jsonl"
    if metadata["enrollment_sha256"] != sha256_file(enrollment_path) or metadata[
        "trials_sha256"
    ] != sha256_file(trials_path):
        raise ValueError("validation selection checksum mismatch")
    enrollment = list(_read_jsonl(enrollment_path))
    segments = load_segments(manifest, split="validation")
    by_id = {segment.segment_id: segment for segment in segments}
    enrollment_ids = {row["segment_id"] for row in enrollment}
    queries = [segment for segment in segments if segment.role in ROLES]
    if max_queries_per_vowel is not None:
        if max_queries_per_vowel < 1:
            raise ValueError("max_queries_per_vowel must be positive")
        selected = set()
        for role in ROLES:
            for vowel in VOWELS:
                eligible = sorted(
                    (s for s in queries if s.role == role and s.vowel == vowel),
                    key=lambda s: s.segment_id,
                )
                selected.update(s.segment_id for s in eligible[:max_queries_per_vowel])
        queries = [s for s in queries if s.segment_id in selected]
    query_ids = {segment.segment_id for segment in queries}
    if enrollment_ids & query_ids or not enrollment_ids <= by_id.keys():
        raise ValueError("invalid enrollment/query IDs")
    needed = [by_id[segment_id] for segment_id in sorted(enrollment_ids | query_ids)]
    vectors = _embeddings(
        model.to(device), pipeline, root, needed, device=device, batch_size=batch_size
    )
    profiles = _profiles(enrollment, vectors)
    groups = defaultdict(lambda: ([], []))
    strata = defaultdict(
        lambda: {"scores": [], "labels": [], "queries": set(), "speakers": set()}
    )
    output_root.mkdir(parents=True, exist_ok=True)
    score_path = output_root / "scores/validation.jsonl"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    trial_count = 0
    with score_path.open("w", encoding="utf-8") as output:
        for trial in _read_jsonl(trials_path):
            if trial["segment_id"] not in query_ids:
                continue
            profile = profiles[
                (trial["claimed_speaker_id"], trial["vowel"], trial["enrollment_count"])
            ]
            score = float(vectors[trial["segment_id"]] @ profile)
            if not math.isfinite(score):
                raise ValueError("nonfinite cosine score")
            scores, labels = groups[
                (trial["role"], trial["enrollment_count"], trial["vowel"])
            ]
            scores.append(score)
            labels.append(trial["is_genuine"])
            if trial["role"] == "verification" and trial["enrollment_count"] == 10:
                segment = by_id[trial["segment_id"]]
                for dimension, values in _stratum_values(segment).items():
                    for value in values:
                        group = strata[(dimension, value, trial["vowel"])]
                        group["scores"].append(score)
                        group["labels"].append(trial["is_genuine"])
                        group["queries"].add(segment.segment_id)
                        group["speakers"].add(segment.speaker_id)
            output.write(
                json.dumps(
                    {**trial, "cosine_score": score},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            trial_count += 1
    metrics, thresholds = summarize_trials(
        groups, partial=max_queries_per_vowel is not None
    )
    metrics["strata"] = _strata_report(strata, thresholds)
    metrics.update(
        {
            "split": "validation",
            "query_count": len(query_ids),
            "enrollment_segment_count": len(enrollment_ids),
            "trial_count": trial_count,
            "score_sha256": sha256_file(score_path),
            "manifest_sha256": metadata["manifest_sha256"],
            "selections_sha256": metadata["trials_sha256"],
        }
    )
    write_json(output_root / "metrics/validation.json", metrics)
    write_json(output_root / "selections/thresholds.json", thresholds)
    return metrics
