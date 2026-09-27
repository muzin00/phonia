"""Generate version 2.0.0 train statistics or fixed evaluation selections."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_data import (
    InputPipeline,
    compute_feature_statistics,
    load_segments,
    make_enrollment,
    make_trials,
    sha256_file,
)
from phase3_data.artifacts import write_json, write_jsonl
from phase3_data.manifest import ids_checksum, json_sha256

BASE = ROOT / "poc/phoneme-speaker-encoder"
BASELINE = BASE / "config/baseline-log-mel.json"
MANIFEST = (
    ROOT / "poc/phoneme-speaker-dataset/data/generated/phase3-vowel-segments.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("statistics", "selections"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=int, choices=(10, 25, 50, 70))
    parser.add_argument("--rms", choices=("on", "off"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(args.manifest)
    if (
        args.manifest.resolve() == MANIFEST.resolve()
        and manifest_sha != baseline["data"]["manifest_sha256"]
    ):
        raise ValueError("canonical manifest SHA-256 differs from design config")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    code_files = sorted((BASE / "phase3_data").glob("*.py")) + [Path(__file__)]
    code_sha = json_sha256(
        {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files}
    )
    if args.command == "statistics":
        if args.cohort is None or args.rms is None:
            parser.error("statistics requires --cohort and --rms")
        segments = load_segments(
            args.manifest, split="train", role="training", cohort=args.cohort
        )
        pipeline = InputPipeline(baseline["input"], rms_enabled=args.rms == "on")
        result = compute_feature_statistics(
            ROOT,
            segments,
            pipeline,
            cohort=args.cohort,
            manifest_sha256=manifest_sha,
            git_commit=commit,
            calculation_code_sha256=code_sha,
        )
        write_json(args.output, result)
        data = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "cohort": args.cohort,
            "cohort_speaker_ids_sha256": result["speaker_ids_sha256"],
            "cohort_segment_ids_sha256": result["segment_ids_sha256"],
            "rms_enabled": pipeline.rms_enabled,
            "preprocessing_sha256": pipeline.preprocessing_sha256,
            "feature_statistics_path": str(args.output.resolve()),
            "feature_statistics_sha256": sha256_file(args.output),
            "calculation_git_commit": commit,
            "calculation_code_sha256": code_sha,
        }
        write_json(args.output.parent / "data.json", data)
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "sha256": sha256_file(args.output),
                    "segment_count": result["segment_count"],
                    "frame_count": result["frame_count"],
                }
            )
        )
    else:
        enrollment_segments = load_segments(
            args.manifest, split=args.split, role="enrollment"
        )
        enrollment = make_enrollment(enrollment_segments, split=args.split)
        output = args.output / "selections"
        enrollment_path = output / "enrollment-segments.jsonl"
        enrollment_count = write_jsonl(enrollment_path, enrollment)
        query_segments = load_segments(args.manifest, split=args.split)
        queries = (
            s
            for s in query_segments
            if s.role in ("verification", "cross_text_verification")
        )
        trials_path = output / "trials.jsonl"
        trial_count = write_jsonl(
            trials_path, make_trials(queries, enrollment, split=args.split)
        )
        data = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "split": args.split,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "enrollment_seed": 20260930,
            "enrollment_counts": [1, 5, 10],
            "enrollment_speaker_ids_sha256": ids_checksum(
                {s.speaker_id for s in enrollment_segments}
            ),
            "enrollment_segment_ids_sha256": ids_checksum(
                row["segment_id"] for row in enrollment
            ),
            "enrollment_path": str(enrollment_path.resolve()),
            "enrollment_sha256": sha256_file(enrollment_path),
            "enrollment_rows": enrollment_count,
            "trials_path": str(trials_path.resolve()),
            "trials_sha256": sha256_file(trials_path),
            "trial_rows": trial_count,
            "calculation_git_commit": commit,
            "calculation_code_sha256": code_sha,
        }
        write_json(args.output / "data.json", data)
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "enrollment_rows": enrollment_count,
                    "trial_rows": trial_count,
                    "data_sha256": sha256_file(args.output / "data.json"),
                }
            )
        )


if __name__ == "__main__":
    main()
