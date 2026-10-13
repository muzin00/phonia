"""Complete the frozen adjacent-pair experiment, with resumable worker jobs."""

import argparse
import fcntl
import os
import subprocess
import sys
import time

from pair_common import (
    ARMS,
    BASE,
    CONFIG,
    ROOT,
    check,
    np,
    pin,
    read,
    sha,
    state,
    torch,
    write,
)


def freeze(config, run):
    runtime = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    path = run / "design-freeze.json"
    if path.exists():
        frozen = read(path)
        assert frozen["config"] == config and frozen["runtime"] == runtime
        check(frozen["files"])
        return frozen
    from prepare import prepare

    prepare(config, run)
    prepared = read(run / "prepared-freeze.json")
    check(prepared["files"])
    paths = [
        CONFIG,
        BASE / "README.md",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        run / "prepared-freeze.json",
        run / "pair-selection.json",
    ]
    frozen = {
        "config": config,
        "runtime": runtime,
        "files": {**prepared["files"], **pin(paths)},
        "pair_selection_before_test": True,
        "fixed_final_updates": config["updates_per_pair_type"]
        * len(prepared["selected"]),
    }
    write(path, frozen, immutable=True)
    return frozen


def execute(config, run):
    frozen = freeze(config, run)
    if (run / "complete-audit.json").exists():
        check(read(run / "complete-audit.json")["files"])
        state(run, "complete", verified_existing=True)
        return
    children = []
    try:
        state(
            run,
            "training",
            seeds=config["seeds"],
            updates=frozen["fixed_final_updates"],
        )
        for seed in config["seeds"]:
            with (run / f"worker-{seed}.log").open("a") as log:
                children.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(BASE / "study.py"),
                            "worker",
                            "--seed",
                            str(seed),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                        },
                    )
                )
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("Training worker failed; see worker logs")
            time.sleep(5)
        assert all(p.returncode == 0 for p in children)
        paths = [
            run / f"seed-{seed}/{arm}/encoder.pt"
            for seed in config["seeds"]
            for arm in ARMS
        ]
        write(
            run / "models-freeze.json",
            {"files": pin(paths), "design_sha256": sha(run / "design-freeze.json")},
            immutable=True,
        )
        from evaluate import confidence, evaluate_split
        from prepare import prepare

        evaluate_split(config, run, "validation")
        state(run, "preparing_test")
        prepare(config, run, test=True)
        evaluate_split(config, run, "test")
        state(run, "bootstrap")
        confidence(config, run)
        from audit import audit
        from report import report

        state(run, "auditing")
        audit(config, run)
        report(config, run)
        check(frozen["files"])
        paths = [
            p
            for p in run.rglob("*")
            if p.is_file()
            and p.name not in ("state.json", "complete-audit.json")
            and p.suffix not in (".tmp", ".lock", ".log")
        ]
        write(
            run / "complete-audit.json",
            {
                "files": pin(paths),
                "training_runs_complete": 6,
                "source_files_unchanged": True,
            },
            immutable=True,
        )
        state(run, "complete")
    except BaseException:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("execute", "worker", "status"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "status":
        for path in [run / "state.json", *run.glob("seed-*/*/state.json")]:
            if path.exists():
                print(str(path.relative_to(run)), read(path))
        return
    lock = run / (
        f"worker-{args.seed}.lock" if args.phase == "worker" else "execution.lock"
    )
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase == "worker":
                assert args.seed in config["seeds"]
                from train import train_seed

                train_seed(config, run, args.seed)
            else:
                execute(config, run)
        except BaseException as e:
            state(
                run if args.phase != "worker" else run / f"seed-{args.seed}",
                "failed",
                error=repr(e),
            )
            raise


if __name__ == "__main__":
    main()
