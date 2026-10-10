"""Evaluate all six fixed snapshots after every validation threshold is frozen."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import budget_evaluation as study
from budget_report import audit_report, publish, publish_training, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("training-report", "evaluate"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    study.old.previous.configure_runtime()
    if args.stage == "training-report":
        publish_training()
        return
    if not args.output_dir:
        parser.error("evaluate requires --output-dir")
    run = args.output_dir.resolve()
    checks = []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for command in (
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(study.BASE / "tests"),
            "-q",
        ],
        [sys.executable, "-m", "ruff", "check", "--ignore", "E402", str(study.BASE)],
        [sys.executable, "-m", "ruff", "format", "--check", str(study.BASE)],
    ):
        result = subprocess.run(
            command, cwd=study.ROOT, env=env, check=True, text=True, capture_output=True
        )
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
    study.prepare(run)
    study.write_json(run / "verification.json", {"checks": checks})
    print(
        "six fixed snapshots prepared; all validation inference runs before any test",
        flush=True,
    )
    try:
        study.infer_split(run, "validation")
        study.calibrate(run)
        print(
            "all validation thresholds frozen; starting fixed exploratory test",
            flush=True,
        )
        study.infer_split(run, "test")
        study.evaluate(run)
        interval_audits = study.audit_intervals(run)
        render(run)
        report_checks = audit_report(run)
        study.frozen_inputs(run, "test")
        study.write_json(
            run / "study-report.json",
            {
                "status": "completed",
                "conditions": list(study.CONDITIONS),
                "new_inference_conditions": 6,
                "new_training_trajectories": 2,
                "test_evaluated": True,
                "threshold_recalibrated_on_test": False,
                "checkpoint_selection_changed_after_training": False,
                "independent_interval_audits": interval_audits,
                "report_checks": report_checks,
                "outputs_sha256": {
                    str(p.relative_to(run)): study.sha256_file(p)
                    for p in sorted(run.rglob("*"))
                    if p.is_file()
                },
            },
        )
        publish(run)
        print(f"completed {run}; {interval_audits} interval audits", flush=True)
    except Exception as exc:
        study.write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
