"""Supervise one ASR producer and independent extraction shards with checkpoints."""

import fcntl
import os
import subprocess
import sys
import time

from expansion_common import (
    BASE,
    CONFIG,
    ROOT,
    atomic,
    check_files,
    fixed,
    pin,
    read,
    sha,
)


def freeze(config, run):
    acquired = read(run / "acquisition-complete.json")
    assert (
        acquired["source_count"] == 11200
        and sha(run / "inputs.json") == acquired["inputs_sha256"]
    )
    check_files(read(run / "acquisition-freeze.json")["files"])
    review = ROOT / config["review_run"]
    check_files(read(review / "runtime-inputs.json")["files"])
    paths = [
        CONFIG,
        BASE / "expansion_common.py",
        BASE / "expansion_pipeline.py",
        BASE / "expansion_asr.py",
        BASE / "expansion_extract.py",
        run / "acquisition-freeze.json",
        run / "inputs.json",
        run / "acquisition-complete.json",
        ROOT / config["runtime_freeze"],
        ROOT / config["asr_model_metadata"],
        review / "asr.json",
        review / "asr/freeze.json",
        review / "runtime-inputs.json",
        review / "review-dataset.json",
        review / "independent-audit.json",
    ]
    fixed(
        run / "preparation-freeze.json",
        {
            "config": config,
            "files": pin(paths),
            "user_authorized": config["human_review"],
            "synthetic_review_answers": False,
            "training_started": False,
        },
    )


def main():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    children = []
    logs = []
    with (run / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(config, run)
        if (run / "pipeline-state.json").exists() and read(run / "pipeline-state.json")[
            "status"
        ] == "complete":
            return
        atomic(
            run / "pipeline-state.json",
            {"status": "running", "pid": os.getpid(), "started_at_epoch": time.time()},
        )
        jobs = [
            (
                "asr",
                ROOT / "artifacts/phoneme-corpus-scaling/asr-venv/bin/python",
                BASE / "expansion_asr.py",
                [],
            )
        ]
        jobs += [
            (
                f"extraction-{i}",
                ROOT / "poc/phoneme-alignment-evaluation/.venv/bin/python",
                BASE / "expansion_extract.py",
                ["extract", "--worker", str(i)],
            )
            for i in range(config["extraction_workers"])
        ]
        try:
            # Clear only live producer state on resume; immutable clip receipts remain.
            atomic(run / "asr-state.json", {"status": "starting"})
            for name, python, script, args in jobs:
                stream = (run / (name + ".log")).open("a")
                logs.append(stream)
                child = subprocess.Popen(
                    [str(python), str(script), *args],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                children.append((name, child))
            while any(p.poll() is None for _, p in children):
                failed = [
                    (name, p.returncode)
                    for name, p in children
                    if p.poll() not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"Preparation workers failed: {failed}")
                time.sleep(5)
            assert all(p.returncode == 0 for _, p in children)
            atomic(
                run / "pipeline-state.json",
                {
                    "status": "finalizing",
                    "pid": os.getpid(),
                    "updated_at_epoch": time.time(),
                },
            )
            subprocess.run(
                [sys.executable, str(BASE / "expansion_extract.py"), "finalize"],
                cwd=ROOT,
                check=True,
            )
            atomic(
                run / "pipeline-state.json",
                {
                    "status": "complete",
                    "finished_at_epoch": time.time(),
                    "training_started": False,
                },
            )
        except BaseException as error:
            for _, p in children:
                if p.poll() is None:
                    p.terminate()
            for _, p in children:
                p.wait()
            atomic(
                run / "pipeline-state.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "updated_at_epoch": time.time(),
                },
            )
            raise
        finally:
            for stream in logs:
                stream.close()


if __name__ == "__main__":
    main()
