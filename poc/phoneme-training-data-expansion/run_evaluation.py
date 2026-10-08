"""Run the fixed two-model utterance evaluation through publication."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_report import audit_report, publish, render
from evaluation_study import (
    BASE,
    ROOT,
    audit_intervals,
    calibrate,
    evaluate,
    frozen_inputs,
    infer,
    prepare,
    previous,
    sha256_file,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    previous.configure_runtime()
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
    print(
        "design frozen; two models, enrollment10, strict5, unchanged full queries",
        flush=True,
    )
    try:
        infer(run, "validation")
        calibrate(run)
        print("validation thresholds frozen; starting exploratory test", flush=True)
        infer(run, "test")
        evaluate(run)
        audits = audit_intervals(run)
        render(run)
        report_checks = audit_report(run)
        frozen_inputs(run, "test")
        write_json(
            run / "study-report.json",
            {
                "status": "completed",
                "conditions": ["jvs70", "jvs70_cv70"],
                "test_evaluated": True,
                "threshold_recalibrated_on_test": False,
                "claim_scope": "exploratory_on_previously_observed_JVS_test",
                "cross_corpus_person_overlap": "unknown",
                "independent_interval_audits": audits,
                "report_checks": report_checks,
                "outputs_sha256": {
                    str(p.relative_to(run)): sha256_file(p)
                    for p in sorted(run.rglob("*"))
                    if p.is_file()
                },
            },
        )
        publish(run)
        print(f"completed: {run}; interval audits {audits}", flush=True)
    except Exception as exc:
        write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
