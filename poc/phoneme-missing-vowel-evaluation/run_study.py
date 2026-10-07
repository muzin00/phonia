"""Execute missing-vowel evaluation from validation through audited report."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import missing_study as study
from missing_report import audit_report, publish, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run = parser.parse_args().output_dir.resolve()
    study.previous.configure_runtime()
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
        [sys.executable, "-m", "ruff", "check", str(study.BASE), "--ignore", "E402"],
        [sys.executable, "-m", "ruff", "format", "--check", str(study.BASE)],
    ):
        result = subprocess.run(
            command, cwd=study.ROOT, env=env, capture_output=True, text=True, check=True
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
    print("design frozen; strict5/available4/available3, enrollment n10", flush=True)
    try:
        study.infer(run, "validation")
        study.calibrate(run)
        print("validation thresholds frozen; starting exploratory test", flush=True)
        study.infer(run, "test")
        interval_audits = study.evaluate(run)
        render(run)
        report_checks = audit_report(run)
        study.frozen_inputs(run, "test")
        study.write_json(
            run / "study-report.json",
            {
                "status": "completed",
                "claim_scope": "exploratory_on_previously_observed_JVS_test",
                "independent_interval_audits": interval_audits,
                "report_checks": report_checks,
                "outputs_sha256": {
                    str(p.relative_to(run)): study.sha256_file(p)
                    for p in run.rglob("*")
                    if p.is_file()
                },
            },
        )
        publish(run)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "interval_audits": interval_audits,
                    "run": str(run),
                }
            ),
            flush=True,
        )
    except Exception as exc:
        study.write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
