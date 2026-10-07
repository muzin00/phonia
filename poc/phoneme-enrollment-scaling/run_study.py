"""Run validation, fixed-threshold exploratory JVS evaluation, audit and report."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling import (
    BASE,
    ROOT,
    audit_intervals,
    calibrate,
    configure_runtime,
    evaluate,
    frozen_inputs,
    infer,
    prepare,
    sha256_file,
    write_json,
)
from scaling_report import audit_report, publish, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    configure_runtime()
    checks = []
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for command in (
        [sys.executable, "-m", "unittest", "discover", "-s", str(BASE / "tests"), "-q"],
        [sys.executable, "-m", "ruff", "check", str(BASE), "--ignore", "E402"],
        [sys.executable, "-m", "ruff", "format", "--check", str(BASE)],
    ):
        result = subprocess.run(
            command, cwd=ROOT, env=env, check=True, text=True, capture_output=True
        )
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
    prepare(run)
    write_json(run / "verification.json", {"checks": checks})
    print("design frozen; counts 10/20/30, unchanged full queries", flush=True)
    try:
        infer(run, "validation")
        calibrate(run)
        print("validation thresholds frozen; starting exploratory test", flush=True)
        infer(run, "test")
        evaluate(run)
        ci_audits = audit_intervals(run)
        render(run)
        report_checks = audit_report(run)
        frozen_inputs(run, "test")
        write_json(
            run / "study-report.json",
            {
                "status": "completed",
                "claim_scope": "exploratory_on_previously_observed_JVS_test",
                "counts_per_vowel": [10, 20, 30],
                "independent_interval_audits": ci_audits,
                "report_checks": report_checks,
                "outputs_sha256": {
                    str(p.relative_to(run)): sha256_file(p)
                    for p in sorted(run.rglob("*"))
                    if p.is_file()
                },
            },
        )
        publish(run)
        print(
            json.dumps(
                {"status": "completed", "interval_audits": ci_audits, "run": str(run)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
