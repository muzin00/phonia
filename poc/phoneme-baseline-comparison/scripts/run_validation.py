"""Generate and calibrate all validation conditions before execution freeze."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import load_inputs
from smoke import BASE, ROOT, sha256_file, write_json
from validation import audit_scores, preflight, prepare_validation
from validation_metrics import calibrate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0",
    )
    args = parser.parse_args()
    inputs = prepare_validation()
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    preflight(run, inputs)
    write_json(
        run / "resource-plan.json",
        {
            "scope": "validation_only",
            "score_slots": 810000,
            "inference": inputs["protocol"]["execution"]["inference"],
            "repeats": 3,
            "audio_cache_maximum_sources": 16,
            "scores": "streamed_jsonl_gzip_level3",
            "estimated_cpu_minutes": [5, 20],
            "estimated_peak_process_memory_gib": 4,
            "estimated_artifact_disk_gib": 2,
            "basis": "pilot_worker_times_plus_full_metadata_counts",
            "execution_freeze_for_test": False,
        },
    )
    print(
        "validation: 15 speakers, 1200 queries, 810000 score slots; no test", flush=True
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    try:
        for project, method in (
            ("phoneme-verification-evaluation", "vowels"),
            ("phoneme-baseline-comparison", "ecapa"),
        ):
            subprocess.run(
                [
                    str(ROOT / "poc" / project / ".venv/bin/python"),
                    str(BASE / "scripts/run_validation_worker.py"),
                    "--method",
                    method,
                    "--run-dir",
                    str(run),
                    "--model-dir",
                    str(args.model_dir),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )
        audited = audit_scores(run, inputs)
        print(
            "validation: all 810000 score identities/labels/coverage checked; calibrating",
            flush=True,
        )
        calibrated = calibrate_run(run)
        load_inputs(run)
        report = {
            "schema_version": 1,
            "status": "completed",
            "stage": "validation_not_test_execution_freeze",
            "protocol_sha256": inputs["protocol_sha256"],
            "config_sha256": inputs["config_sha256"],
            "speakers": inputs["speakers"],
            "queries": len(inputs["queries"]),
            "phase6_trial_ids": len(inputs["trials"]),
            **audited,
            **calibrated,
            "test_audio_or_scores_read": False,
            "outputs_sha256": {
                name: sha256_file(run / name)
                for name in (
                    "inputs.json",
                    "preflight.json",
                    "resource-plan.json",
                    "validation-thresholds.json",
                    "validation-metrics.json",
                    "validation-curves.jsonl.gz",
                )
            },
        }
        write_json(run / "validation-report.json", report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "queries",
                        "profiles",
                        "calibration_cells",
                        "operating_thresholds",
                        "evaluation_cells",
                        "not_evaluable_cells",
                    )
                },
                indent=2,
            )
        )
    except Exception as exc:
        write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
