"""Freeze, infer the fixed endpoint, compare and publish."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exposure_evaluation as study
from diagnostics import run_diagnostics
from exposure_report import audit_report, publish, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    study.old.previous.configure_runtime()
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
        "evaluation design frozen; one new endpoint and two pinned references",
        flush=True,
    )
    try:
        study.infer(run, "validation")
        study.calibrate(run)
        print("validation thresholds frozen; starting exploratory test", flush=True)
        study.infer(run, "test")
        study.evaluate(run)
        audits = study.audit_intervals(run)
        run_diagnostics(run)
        render(run)
        report_checks = audit_report(run)
        study.frozen_inputs(run, "test")
        study.write_json(
            run / "study-report.json",
            {
                "status": "completed",
                "conditions": list(study.CONDITIONS),
                "new_inference_conditions": 1,
                "test_evaluated": True,
                "threshold_recalibrated_on_test": False,
                "independent_interval_audits": audits,
                "report_checks": report_checks,
                "outputs_sha256": {
                    str(p.relative_to(run)): study.sha256_file(p)
                    for p in sorted(run.rglob("*"))
                    if p.is_file()
                },
            },
        )
        publish(run)
        print(f"completed: {run}; interval audits {audits}", flush=True)
    except Exception as exc:
        study.write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
