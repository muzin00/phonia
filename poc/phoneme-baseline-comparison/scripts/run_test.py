"""Freeze, infer, evaluate and report all Phase 7 test conditions."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_audit import audit_final
from final_metrics import evaluate_test
from final_report import render_report
from frozen_test import freeze, load_inputs, prepare_test
from smoke import BASE, ROOT, sha256_file, write_json
from validation import audit_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0",
    )
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if run.exists() or not run.is_relative_to(ROOT):
        raise ValueError("requires unused run directory inside repository")
    checks = []
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "MPLCONFIGDIR": str(
            ROOT / "artifacts/phoneme-baseline-comparison/matplotlib-cache"
        ),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    for command in (
        [
            str(BASE / ".venv/bin/python"),
            "-m",
            "unittest",
            "discover",
            "-s",
            str(BASE / "tests"),
            "-q",
        ],
        [str(BASE / ".venv/bin/ruff"), "check", str(BASE)],
        [str(BASE / ".venv/bin/ruff"), "format", "--check", str(BASE)],
    ):
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, cwd=ROOT, env=env
        )
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
    inputs = prepare_test()
    freeze(run, inputs, args.model_dir, {"status": "passed", "checks": checks})
    print(
        "test: execution frozen; 15 speakers, 1200 queries, 810000 score slots",
        flush=True,
    )
    try:
        for project, method in (
            ("phoneme-verification-evaluation", "vowels"),
            ("phoneme-baseline-comparison", "ecapa"),
        ):
            subprocess.run(
                [
                    str(ROOT / "poc" / project / ".venv/bin/python"),
                    str(BASE / "scripts/run_test_worker.py"),
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
        evaluated = evaluate_test(run)
        rendered = render_report(run)
        checks = audit_final(run)
        write_json(run / "test-checks.json", checks)
        load_inputs(run)
        outputs = {
            str(p.relative_to(run)): sha256_file(p)
            for p in sorted(run.rglob("*"))
            if p.is_file() and "source-snapshot" not in p.parts
        }
        write_json(
            run / "test-report.json",
            {
                "schema_version": 1,
                "status": "completed",
                "stage": "phase7_test_and_report_complete",
                "execution_freeze_sha256": sha256_file(run / "execution-freeze.json"),
                "protocol_sha256": inputs["protocol_sha256"],
                "threshold_recalibration": False,
                "speakers": inputs["speakers"],
                "queries": len(inputs["queries"]),
                **audited,
                **evaluated,
                **rendered,
                "checks": checks,
                "outputs_sha256": outputs,
            },
        )
        print(
            json.dumps(
                {"status": "completed", **evaluated, **rendered, "checks": checks},
                indent=2,
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
