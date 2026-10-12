"""Continue from complete preprocessing through audit, training and evaluation."""

import fcntl
import os
import subprocess
import time

from expansion_common import BASE, CONFIG, ROOT, atomic, check_files, fixed, pin, read


def main():
    preparation = read(CONFIG)
    prepared = ROOT / preparation["run_directory"]
    config = read(BASE / "study-config.json")
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    state = run / "continuation-state.json"
    with (run / "continuation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed(
            run / "execution-intent.json",
            {
                "authorization": preparation["human_review"],
                "files": pin(
                    [
                        BASE / "continue_expansion.py",
                        BASE / "audit_expansion.py",
                        BASE / "scaling_study.py",
                        BASE / "check_feature_runtime.py",
                        BASE / "audit_study.py",
                        BASE / "study-config.json",
                    ]
                ),
            },
        )
        try:
            while True:
                status = read(prepared / "pipeline-state.json")
                if status["status"] == "complete":
                    break
                if status["status"] == "failed":
                    raise RuntimeError(status["error"])
                os.kill(status["pid"], 0)
                atomic(
                    state,
                    {
                        "status": "waiting_for_preparation",
                        "pid": os.getpid(),
                        "updated_at_epoch": time.time(),
                    },
                )
                time.sleep(10)
            check_files(read(run / "execution-intent.json")["files"])
            jobs = [
                (
                    "auditing_preparation",
                    "poc/phoneme-alignment-evaluation/.venv/bin/python",
                    "audit_expansion.py",
                    [],
                ),
                (
                    "training_and_evaluating",
                    "poc/phoneme-verification-evaluation/.venv/bin/python",
                    "scaling_study.py",
                    ["execute"],
                ),
                (
                    "auditing_results",
                    "poc/phoneme-verification-evaluation/.venv/bin/python",
                    "audit_study.py",
                    [],
                ),
            ]
            for stage, python, script, args in jobs:
                atomic(
                    state,
                    {
                        "status": stage,
                        "pid": os.getpid(),
                        "updated_at_epoch": time.time(),
                    },
                )
                with (run / (stage + ".log")).open("a") as log:
                    subprocess.run(
                        [str(ROOT / python), str(BASE / script), *args],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                        },
                    )
            atomic(
                state,
                {
                    "status": "complete",
                    "finished_at_epoch": time.time(),
                    "results": str((run / "results.md").relative_to(ROOT)),
                },
            )
        except BaseException as error:
            atomic(
                state,
                {
                    "status": "failed",
                    "error": str(error),
                    "updated_at_epoch": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
