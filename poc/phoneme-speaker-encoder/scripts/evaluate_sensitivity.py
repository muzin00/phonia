"""Evaluate boundary and gain sensitivity of the selected 70-speaker checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from phase3_data.input import crop_start
from phase3_data.manifest import load_segments
from phase3_train.evaluation import (
    _embeddings,
    _numeric_threshold,
    _profiles,
    _read_jsonl,
)
from phase3_train.metrics import error_rates, roc_eer
from phase3_train.training import TrainSettings
from scripts.run_phase3 import _trainer

FULL_ROOT = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2"
)
OUTPUT = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/learning-curve-phase3-selected-v2/sensitivity"
)
SELECTION_ROOT = ROOT / "artifacts/phoneme-speaker-encoder/fixed-validation"
VOWELS = ("a", "i", "u", "e", "o")
SHIFTS = (-240, -120, 120, 240)
GAINS = (-6, 6)


class GainPipeline:
    def __init__(self, original, db: int):
        self.original = original
        self.factor = 10 ** (db / 20)

    def prepare(self, pcm, segment_id, **kwargs):
        start = crop_start(
            pcm.numel(),
            segment_id,
            kwargs["mode"],
            kwargs.get("run_seed", 0),
            kwargs.get("logical_update", 0),
            self.original.maximum,
        )
        cropped = pcm[start : start + self.original.maximum] * self.factor
        # Apply gain after crop, before DC removal, without clipping.
        return self.original.prepare(cropped, segment_id, **kwargs)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_lengths(segments) -> dict[str, int]:
    result = {}
    for segment in segments:
        if segment.source_file not in result:
            source = (ROOT / segment.source_file).resolve()
            if not source.is_relative_to(ROOT):
                raise ValueError("source WAV outside repository")
            with wave.open(str(source), "rb") as wav:
                result[segment.source_file] = wav.getnframes()
    return result


def _queries(manifest: Path):
    segments = load_segments(manifest, split="validation")
    by_id = {segment.segment_id: segment for segment in segments}
    queries = {s.segment_id: s for s in segments if s.role == "verification"}
    lengths = _source_lengths(queries.values())
    common = {
        segment_id
        for segment_id, segment in queries.items()
        if segment.start_frame >= 240
        and segment.end_frame + 240 <= lengths[segment.source_file]
    }
    return by_id, queries, common


def _baseline_scores(path: Path, query_ids: set[str]) -> list[dict]:
    rows = []
    for row in _read_jsonl(path):
        if row["role"] == "verification" and row["enrollment_count"] == 10:
            if row["segment_id"] not in query_ids:
                raise ValueError("baseline score has an unknown query")
            rows.append(row)
    if not rows:
        raise ValueError("no primary verification trials")
    return rows


def _summarize(rows: list[dict], thresholds: dict, output: Path) -> dict:
    groups = defaultdict(lambda: ([], []))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            score = row["cosine_score"]
            if not math.isfinite(score):
                raise ValueError("nonfinite diagnostic score")
            scores, genuine = groups[row["vowel"]]
            scores.append(score)
            genuine.append(row["is_genuine"])
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    vowels = {}
    for vowel in VOWELS:
        scores, genuine = groups[vowel]
        if not scores:
            raise ValueError(f"missing {vowel} diagnostic trials")
        fixed = thresholds["per_count"]["10"][vowel]
        vowels[vowel] = {
            "eer": roc_eer(scores, genuine)["eer"],
            "trial_count": len(scores),
            "fixed_threshold_rates": {
                name: error_rates(
                    scores,
                    genuine,
                    _numeric_threshold(fixed[name]["threshold"]),
                )
                for name in ("far_1pct", "far_0_1pct", "eer_operating")
            },
        }
    return {
        "macro_eer": sum(vowels[v]["eer"] for v in VOWELS) / 5,
        "vowels": vowels,
        "trial_count": len(rows),
        "score_sha256": sha256_file(output),
    }


def _evaluate(seed: int) -> None:
    bundle = FULL_ROOT / "selection-evaluation/selected-bundle" / str(seed)
    run = _read(bundle / "run.json")
    settings = TrainSettings(**run["settings"])
    if (
        settings.cohort != 70
        or settings.seed != seed
        or settings.encoder != "statistics_mlp"
    ):
        raise ValueError("unexpected selected checkpoint settings")
    manifest = Path(run["manifest"])
    model, pipeline, _ = _trainer(settings, manifest, Path(run["statistics"]))
    checkpoint = torch.load(bundle / "best.pt", map_location="cpu", weights_only=False)
    expected = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "run_seed": seed,
        "config_sha256": settings.sha256,
        "manifest_sha256": model.manifest_sha256,
        "feature_statistics_sha256": model.feature_statistics_sha256,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("selected checkpoint metadata mismatch")
    model.model.load_state_dict(checkpoint["model"])
    encoder = model.model
    thresholds = _read(bundle / "thresholds.json")
    selection_meta = _read(SELECTION_ROOT / "data.json")
    if selection_meta["split"] != "validation" or selection_meta[
        "manifest_sha256"
    ] != sha256_file(manifest):
        raise ValueError("validation selection/manifest mismatch")
    enrollment = list(
        _read_jsonl(SELECTION_ROOT / "selections/enrollment-segments.jsonl")
    )
    by_id, queries, common = _queries(manifest)
    enrollment_ids = {row["segment_id"] for row in enrollment}
    if enrollment_ids & queries.keys() or not enrollment_ids <= by_id.keys():
        raise ValueError("invalid enrollment IDs")
    enrollment_vectors = _embeddings(
        encoder,
        pipeline,
        ROOT,
        [by_id[segment_id] for segment_id in sorted(enrollment_ids)],
        device="cpu",
        batch_size=32,
    )
    profiles = _profiles(enrollment, enrollment_vectors)
    full_run = (
        FULL_ROOT
        / "runs"
        / (
            f"full-c70-s{seed}__log_mel__statistics_mlp__rms-off__"
            "aam_softmax_plus_supcon_within_vowel"
        )
    )
    summary = _read(full_run / "training/summary.json")
    original_scores = full_run / (
        f"validation/update-{summary['best']['update']:06d}/scores/validation.jsonl"
    )
    original_metrics = _read(
        full_run
        / (f"validation/update-{summary['best']['update']:06d}/metrics/validation.json")
    )
    if original_metrics["score_sha256"] != sha256_file(original_scores):
        raise ValueError("original validation scores changed")
    base_rows = _baseline_scores(original_scores, set(queries))
    sample_ids = sorted(queries)[:5]
    sample_vectors = _embeddings(
        encoder,
        pipeline,
        ROOT,
        [queries[segment_id] for segment_id in sample_ids],
        device="cpu",
        batch_size=5,
    )
    sample_rows = {
        row["segment_id"]: row for row in base_rows if row["segment_id"] in sample_ids
    }
    for segment_id, row in sample_rows.items():
        computed = float(
            sample_vectors[segment_id]
            @ profiles[(row["claimed_speaker_id"], row["vowel"], 10)]
        )
        if abs(computed - row["cosine_score"]) > 1e-12:
            raise ValueError(
                "selected bundle does not reproduce original validation scores"
            )
    out = OUTPUT / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    conditions = {}

    def save(name: str, rows: list[dict], expected_ids: set[str]) -> None:
        if {row["segment_id"] for row in rows} != expected_ids:
            raise ValueError(f"missing diagnostic queries: {name}")
        path = out / name / "scores.jsonl"
        metrics = _summarize(rows, thresholds, path)
        write_json(out / name / "metrics.json", metrics)
        conditions[name] = metrics

    save("baseline-full", base_rows, set(queries))
    common_rows = [row for row in base_rows if row["segment_id"] in common]
    save("baseline-common", common_rows, common)

    for shift in SHIFTS:
        name = f"boundary-{shift:+d}-samples"
        altered = [
            replace(
                queries[segment_id],
                start_frame=queries[segment_id].start_frame + shift,
                end_frame=queries[segment_id].end_frame + shift,
            )
            for segment_id in sorted(common)
        ]
        vectors = _embeddings(
            encoder, pipeline, ROOT, altered, device="cpu", batch_size=32
        )
        rows = [
            {
                **row,
                "transform_id": name,
                "source_trial_id": row["trial_id"],
                "source_cosine_score": row["cosine_score"],
                "cosine_score": float(
                    vectors[row["segment_id"]]
                    @ profiles[(row["claimed_speaker_id"], row["vowel"], 10)]
                ),
            }
            for row in common_rows
        ]
        save(name, rows, common)

    for gain in GAINS:
        name = f"gain-{gain:+d}-db"
        vectors = _embeddings(
            encoder,
            GainPipeline(pipeline, gain),
            ROOT,
            [queries[segment_id] for segment_id in sorted(queries)],
            device="cpu",
            batch_size=32,
        )
        rows = [
            {
                **row,
                "transform_id": name,
                "source_trial_id": row["trial_id"],
                "source_cosine_score": row["cosine_score"],
                "cosine_score": float(
                    vectors[row["segment_id"]]
                    @ profiles[(row["claimed_speaker_id"], row["vowel"], 10)]
                ),
            }
            for row in base_rows
        ]
        save(name, rows, set(queries))

    write_json(
        out / "summary.json",
        {
            "schema_version": 1,
            "seed": seed,
            "checkpoint_sha256": sha256_file(bundle / "best.pt"),
            "thresholds_sha256": sha256_file(bundle / "thresholds.json"),
            "manifest_sha256": sha256_file(manifest),
            "original_scores_sha256": sha256_file(original_scores),
            "common_query_count": len(common),
            "excluded_query_count": len(queries) - len(common),
            "excluded_by_speaker_vowel": {
                f"{speaker}:{vowel}": sum(
                    1
                    for segment_id, segment in queries.items()
                    if segment.speaker_id == speaker
                    and segment.vowel == vowel
                    and segment_id not in common
                )
                for speaker, vowel in sorted(
                    {
                        (segment.speaker_id, segment.vowel)
                        for segment in queries.values()
                    }
                )
            },
            "conditions": {
                name: {
                    "macro_eer": value["macro_eer"],
                    "macro_eer_difference": value["macro_eer"]
                    - conditions[
                        "baseline-common"
                        if name.startswith("boundary-")
                        else "baseline-full"
                    ]["macro_eer"]
                    if name not in ("baseline-full", "baseline-common")
                    else 0.0,
                    "trial_count": value["trial_count"],
                    "score_sha256": value["score_sha256"],
                }
                for name, value in conditions.items()
            },
            "test_used": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(20260926, 20260927, 20260928))
    args = parser.parse_args()
    seeds = (args.seed,) if args.seed else (20260926, 20260927, 20260928)
    for seed in seeds:
        _evaluate(seed)
        print(f"completed sensitivity validation: {seed}", flush=True)


if __name__ == "__main__":
    main()
