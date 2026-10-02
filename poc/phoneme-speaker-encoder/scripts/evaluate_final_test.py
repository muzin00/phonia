"""Run the one-time, frozen Phase 3 test evaluation for all selected seeds."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from phase3_data.manifest import load_segments
from phase3_train.evaluation import (
    COUNTS,
    ROLES,
    VOWELS,
    _embeddings,
    _group_report,
    _numeric_threshold,
    _profiles,
    _read_jsonl,
    _stratum_values,
)
from phase3_train.metrics import error_rates
from phase3_train.training import TrainSettings
from scripts.run_phase3 import _trainer

FULL_ROOT = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2"
)
BUNDLES = FULL_ROOT / "selection-evaluation/selected-bundle"
SELECTION = FULL_ROOT / "selection-evaluation/selection.json"
OUTPUT = ROOT / "artifacts/phoneme-speaker-encoder/final-test-phase3-selected-v2"
TEST_SELECTIONS = OUTPUT / "fixed-test"
SEEDS = (20260926, 20260927, 20260928)
CONFIG_ID = "log_mel__statistics_mlp__rms-off__aam_softmax_plus_supcon_within_vowel"
BUNDLE_FILES = ("best.pt", "thresholds.json", "run.json", "feature-statistics.json")
THRESHOLD_NAMES = ("far_1pct", "far_0_1pct", "eer_operating")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan() -> dict:
    selection = _read(SELECTION)
    if selection["selected_config_id"] != CONFIG_ID or selection["test_used"]:
        raise ValueError("selection is not the frozen validation choice")
    bundles = {}
    manifest_hashes = set()
    for seed in SEEDS:
        bundle = BUNDLES / str(seed)
        run = _read(bundle / "run.json")
        settings = TrainSettings(**run["settings"])
        if (
            settings.seed != seed
            or settings.cohort != 70
            or settings.encoder != "statistics_mlp"
            or settings.rms_enabled
            or not settings.supcon_enabled
        ):
            raise ValueError(f"unexpected selected settings: {seed}")
        manifest_hashes.add(sha256_file(Path(run["manifest"])))
        bundles[str(seed)] = {name: sha256_file(bundle / name) for name in BUNDLE_FILES}
    if len(manifest_hashes) != 1:
        raise ValueError("selected seeds use different manifests")
    return {
        "schema_version": 1,
        "design_version": "2.0.0",
        "selected_config_id": CONFIG_ID,
        "seeds": list(SEEDS),
        "single_distribution_seed": selection["single_distribution_seed"],
        "selection_sha256": sha256_file(SELECTION),
        "manifest_sha256": manifest_hashes.pop(),
        "bundles_sha256": bundles,
        "selection_builder_sha256": sha256_file(
            BASE / "scripts/build_data_artifacts.py"
        ),
        "evaluation_code_sha256": sha256_file(Path(__file__)),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "split": "test",
        "primary_role": "verification",
        "primary_enrollment_count": 10,
        "roles": list(ROLES),
        "enrollment_counts": list(COUNTS),
        "vowels": list(VOWELS),
        "fixed_thresholds": list(THRESHOLD_NAMES),
        "test_threshold_refit": False,
        "ensemble": False,
        "device": "cpu",
        "batch_size": 32,
    }


def _frozen_plan() -> dict:
    expected = _plan()
    path = OUTPUT / "plan.json"
    if path.exists():
        stored = _read(path)
        if stored != expected:
            raise ValueError(
                "selected bundle or evaluation protocol differs from frozen plan"
            )
        return stored
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return expected


def _rates(scores: list[float], labels: list[bool], thresholds: dict) -> dict:
    return {
        name: error_rates(
            scores, labels, _numeric_threshold(thresholds[name]["threshold"])
        )
        for name in THRESHOLD_NAMES
    }


def _summarize(groups: dict, strata: dict, thresholds: dict) -> dict:
    roles = {}
    for role in ROLES:
        roles[role] = {}
        for count in COUNTS:
            vowels = {}
            pooled_scores, pooled_labels = [], []
            for vowel in VOWELS:
                scores, labels = groups[(role, count, vowel)]
                report = _group_report(
                    scores, labels, curve=role == "verification" and count == 10
                )
                report["fixed_threshold_rates"] = _rates(
                    scores, labels, thresholds["per_count"][str(count)][vowel]
                )
                vowels[vowel] = report
                pooled_scores.extend(scores)
                pooled_labels.extend(labels)
            eers = [vowels[vowel]["eer"] for vowel in VOWELS]
            if any(eer is None for eer in eers):
                raise ValueError(f"undefined test EER: {role}, {count}")
            pooled = _group_report(
                pooled_scores,
                pooled_labels,
                curve=role == "verification" and count == 10,
            )
            pooled["fixed_threshold_rates"] = _rates(
                pooled_scores,
                pooled_labels,
                thresholds["per_count"][str(count)]["pooled"],
            )
            roles[role][str(count)] = {
                "vowels": vowels,
                "macro_eer": sum(eers) / len(VOWELS),
                "pooled": pooled,
            }
    strata_report = {}
    for (dimension, value, vowel), group in strata.items():
        report = _group_report(group["scores"], group["labels"])
        report.update(
            query_count=len(group["queries"]),
            speaker_count=len(group["speakers"]),
            fixed_threshold_rates=_rates(
                group["scores"],
                group["labels"],
                thresholds["per_count"]["10"][vowel],
            ),
        )
        strata_report.setdefault(dimension, {}).setdefault(value, {})[vowel] = report
    for values in strata_report.values():
        for vowels in values.values():
            eers = [vowels.get(vowel, {}).get("eer") for vowel in VOWELS]
            vowels["macro_eer"] = (
                sum(eers) / len(VOWELS)
                if all(eer is not None for eer in eers)
                else None
            )
    return {
        "roles": roles,
        "strata": strata_report,
        "macro_eer": roles["verification"]["10"]["macro_eer"],
    }


def _evaluate(seed: int, plan: dict) -> None:
    output = OUTPUT / str(seed)
    metric_path = output / "metrics/test.json"
    if metric_path.exists():
        raise FileExistsError(f"test already evaluated for seed {seed}")
    selection_data = _read(TEST_SELECTIONS / "data.json")
    enrollment_path = TEST_SELECTIONS / "selections/enrollment-segments.jsonl"
    trials_path = TEST_SELECTIONS / "selections/trials.jsonl"
    if (
        selection_data["split"] != "test"
        or selection_data["enrollment_seed"] != 20260930
        or selection_data["enrollment_counts"] != list(COUNTS)
        or selection_data["manifest_sha256"] != plan["manifest_sha256"]
        or selection_data["enrollment_sha256"] != sha256_file(enrollment_path)
        or selection_data["trials_sha256"] != sha256_file(trials_path)
    ):
        raise ValueError("test selection checksum or split differs")
    bundle = BUNDLES / str(seed)
    run = _read(bundle / "run.json")
    settings = TrainSettings(**run["settings"])
    trainer, pipeline, _ = _trainer(
        settings, Path(run["manifest"]), bundle / "feature-statistics.json"
    )
    checkpoint = torch.load(bundle / "best.pt", map_location="cpu", weights_only=False)
    expected = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "run_seed": seed,
        "config_sha256": settings.sha256,
        "manifest_sha256": trainer.manifest_sha256,
        "feature_statistics_sha256": trainer.feature_statistics_sha256,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("selected checkpoint metadata mismatch")
    trainer.model.load_state_dict(checkpoint["model"])
    thresholds = _read(bundle / "thresholds.json")
    if thresholds["source_role"] != "verification":
        raise ValueError("thresholds were not fit on validation verification")
    segments = load_segments(Path(run["manifest"]), split="test")
    by_id = {segment.segment_id: segment for segment in segments}
    enrollment = list(_read_jsonl(enrollment_path))
    enrollment_ids = {row["segment_id"] for row in enrollment}
    query_ids = {segment.segment_id for segment in segments if segment.role in ROLES}
    if enrollment_ids & query_ids or not enrollment_ids <= by_id.keys():
        raise ValueError("test enrollment/query IDs invalid")
    needed = [by_id[segment_id] for segment_id in sorted(enrollment_ids | query_ids)]
    started = time.monotonic()
    vectors = _embeddings(
        trainer.model, pipeline, ROOT, needed, device="cpu", batch_size=32
    )
    profiles = _profiles(enrollment, vectors)
    groups = defaultdict(lambda: ([], []))
    strata = defaultdict(
        lambda: {"scores": [], "labels": [], "queries": set(), "speakers": set()}
    )
    score_path = output / "scores/test.jsonl"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    trial_count = 0
    with score_path.open("w", encoding="utf-8") as stream:
        for trial in _read_jsonl(trials_path):
            segment_id = trial["segment_id"]
            if trial["split"] != "test" or segment_id not in query_ids:
                raise ValueError("invalid test trial")
            key = (
                trial["claimed_speaker_id"],
                trial["vowel"],
                trial["enrollment_count"],
            )
            score = float(vectors[segment_id] @ profiles[key])
            if not math.isfinite(score):
                raise ValueError("nonfinite test score")
            scores, labels = groups[
                (trial["role"], trial["enrollment_count"], trial["vowel"])
            ]
            scores.append(score)
            labels.append(trial["is_genuine"])
            if trial["role"] == "verification" and trial["enrollment_count"] == 10:
                segment = by_id[segment_id]
                for dimension, values in _stratum_values(segment).items():
                    for value in values:
                        group = strata[(dimension, value, trial["vowel"])]
                        group["scores"].append(score)
                        group["labels"].append(trial["is_genuine"])
                        group["queries"].add(segment_id)
                        group["speakers"].add(segment.speaker_id)
            stream.write(
                json.dumps(
                    {**trial, "cosine_score": score},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            trial_count += 1
    if trial_count != selection_data["trial_rows"]:
        raise ValueError("test trial count differs from frozen selections")
    metrics = _summarize(groups, strata, thresholds)
    metrics.update(
        {
            "schema_version": 1,
            "design_version": "2.0.0",
            "split": "test",
            "seed": seed,
            "selected_config_id": CONFIG_ID,
            "primary_metric": "test_verification_10_enrollment_macro_eer",
            "query_count": len(query_ids),
            "enrollment_segment_count": len(enrollment_ids),
            "embedding_count": len(needed),
            "trial_count": trial_count,
            "evaluation_seconds": time.monotonic() - started,
            "score_sha256": sha256_file(score_path),
            "test_selection_sha256": sha256_file(TEST_SELECTIONS / "data.json"),
            "validation_thresholds_sha256": plan["bundles_sha256"][str(seed)][
                "thresholds.json"
            ],
            "checkpoint_sha256": plan["bundles_sha256"][str(seed)]["best.pt"],
            "test_threshold_refit": False,
        }
    )
    write_json(metric_path, metrics)
    print(
        json.dumps(
            {
                "seed": seed,
                "macro_eer": metrics["macro_eer"],
                "evaluation_seconds": metrics["evaluation_seconds"],
                "trial_count": trial_count,
            }
        ),
        flush=True,
    )


def _report(plan: dict) -> None:
    rows = []
    for seed in SEEDS:
        metric_path = OUTPUT / str(seed) / "metrics/test.json"
        scores = OUTPUT / str(seed) / "scores/test.jsonl"
        metrics = _read(metric_path)
        if (
            metrics["split"] != "test"
            or metrics["seed"] != seed
            or metrics["test_threshold_refit"]
            or metrics["checkpoint_sha256"]
            != plan["bundles_sha256"][str(seed)]["best.pt"]
            or metrics["validation_thresholds_sha256"]
            != plan["bundles_sha256"][str(seed)]["thresholds.json"]
            or metrics["score_sha256"] != sha256_file(scores)
        ):
            raise ValueError(f"incomplete or inconsistent test result: {seed}")
        main = metrics["roles"]["verification"]["10"]
        row = {
            "seed": seed,
            "macro_eer": metrics["macro_eer"],
            "cross_text_macro_eer": metrics["roles"]["cross_text_verification"]["10"][
                "macro_eer"
            ],
            "enrollment_macro_eer": {
                str(count): metrics["roles"]["verification"][str(count)]["macro_eer"]
                for count in COUNTS
            },
            "vowel_eer": {vowel: main["vowels"][vowel]["eer"] for vowel in VOWELS},
            "vowel_fixed_threshold_rates": {
                vowel: main["vowels"][vowel]["fixed_threshold_rates"]
                for vowel in VOWELS
            },
            "evaluation_seconds": metrics["evaluation_seconds"],
            "embedding_count": metrics["embedding_count"],
            "trial_count": metrics["trial_count"],
            "metric_sha256": sha256_file(metric_path),
            "score_sha256": metrics["score_sha256"],
        }
        rows.append(row)
    result = {
        "schema_version": 1,
        "plan_sha256": sha256_file(OUTPUT / "plan.json"),
        "test_selection_sha256": sha256_file(TEST_SELECTIONS / "data.json"),
        "seeds": list(SEEDS),
        "selected_config_id": CONFIG_ID,
        "single_distribution_seed": plan["single_distribution_seed"],
        "macro_eer_mean": statistics.mean(row["macro_eer"] for row in rows),
        "macro_eer_std_population": statistics.pstdev(row["macro_eer"] for row in rows),
        "cross_text_macro_eer_mean": statistics.mean(
            row["cross_text_macro_eer"] for row in rows
        ),
        "fixed_threshold_macro_rates": {
            name: {
                rate: statistics.mean(
                    row["vowel_fixed_threshold_rates"][vowel][name][rate]
                    for row in rows
                    for vowel in VOWELS
                )
                for rate in ("far", "frr")
            }
            for name in THRESHOLD_NAMES
        },
        "rows": rows,
        "test_threshold_refit": False,
    }
    write_json(OUTPUT / "results.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    args = parser.parse_args()
    plan = _frozen_plan()
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
        return
    if args.command == "report":
        _report(plan)
        return
    for seed in SEEDS:
        _evaluate(seed, plan)


if __name__ == "__main__":
    main()
