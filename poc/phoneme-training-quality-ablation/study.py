"""Prepare, freeze, execute/resume and inspect the training quality ablation."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from quality_data import load_training, prepare
from quality_results import finalize
from shared_quality import (
    ARMS,
    BASE,
    ROOT,
    configuration,
    evaluation_prepared,
    fit,
    load_evaluations,
    prepared,
    rt,
    target,
)

# isort: split
from ablation_train import train_uniform


def freeze(config, run):
    if (
        config["updates"] <= config["warmup_updates"]
        or config["updates"] % config["block_updates"]
    ):
        raise ValueError("Invalid acoustic budget")
    if config["negative_sampling"] != "uniform_other_speaker_within_query_corpus":
        raise ValueError("Unsupported negative sampling protocol")
    if config["workers"] < 1 or not config["evaluation_seeds"]:
        raise ValueError("Invalid workers or seeds")
    seal = run / "design-freeze.json"
    if seal.exists():
        saved = rt.read(seal)
        if saved["config"] != config or saved["runtime"] != rt.runtime():
            raise ValueError("Frozen configuration/runtime changed; use a new run")
        rt.check(saved["files"])
        return
    prepare(config)
    paths = {
        BASE / "config.json",
        BASE / "README.md",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        prepared(config) / "freeze.json",
    }
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (
                path.is_relative_to(ROOT / "poc")
                and path.suffix == ".py"
                and ".venv" not in path.parts
            ):
                paths.add(path)
    files = {**rt.read(prepared(config) / "freeze.json")["files"], **rt.pin(paths)}
    rt.write(
        seal,
        {
            "config": config,
            "runtime": rt.runtime(),
            "files": files,
            "smoke_test_only": config.get("smoke_test_only", False),
        },
        immutable=True,
    )


def worker(config, run, arm, seed):
    rt.check(rt.read(run / "design-freeze.json")["files"])
    directory = target(run, arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        rt.check(rt.read(directory / "complete.json")["files"])
        return
    (directory / "failure.json").unlink(missing_ok=True)
    pool, bank = load_training(config, arm, seed)
    rt.write(
        directory / "state.json",
        {"phase": "acoustic_training", "arm": arm, "seed": seed},
    )
    student, coverage, done = train_uniform(config, directory / "acoustic", pool, seed)
    if not done:
        raise RuntimeError("Full acoustic budget was not completed")
    selection = load_evaluations(config, evaluation_prepared(config), "selection")
    with rt.np.load(
        prepared(config) / "schedules" / f"schedule-{seed}.npz", allow_pickle=False
    ) as archive:
        schedule = {k: archive[k].copy() for k in archive.files}
    rt.write(
        directory / "state.json", {"phase": "fusion_training", "arm": arm, "seed": seed}
    )
    fit(
        config,
        directory / "fusion",
        student,
        bank,
        selection,
        schedule_override=schedule,
    )
    corpora = rt.np.array([rt.corpus(s) for s in pool.speakers])[pool.segment_speaker]
    counts = {}
    for c in sorted(set(corpora)):
        mask = corpora == c
        selected = coverage.selected[mask] + coverage.singleton[mask]
        counts[c] = {
            "available_segments": int(mask.sum()),
            "unique_selected": int(rt.np.count_nonzero(selected)),
            "presentations": int(selected.sum()),
            "unique_candidates": int(rt.np.count_nonzero(coverage.candidate[mask])),
        }
    rt.write(directory / "coverage-by-corpus.json", counts)
    paths = [directory / "coverage-by-corpus.json"]
    for name, files in {
        "acoustic": ("encoder.pt", "complete.json", "coverage.json", "history.json"),
        "fusion": ("selected.pt", "final.pt", "complete.json", "history.json"),
    }.items():
        paths.extend(directory / name / f for f in files)
    rt.write(
        directory / "complete.json",
        {"status": "complete", "files": rt.pin(paths)},
        immutable=True,
    )
    rt.write(directory / "state.json", {"phase": "complete", "arm": arm, "seed": seed})


def execute(config, run):
    freeze(config, run)
    if (run / "complete.json").exists():
        rt.check(rt.read(run / "complete.json")["files"])
        return
    (run / "failure.json").unlink(missing_ok=True)
    pending = [(arm, seed) for seed in config["evaluation_seeds"] for arm in ARMS]
    running = {}
    previous = rt.read(run / "state.json") if (run / "state.json").exists() else {}
    state = {
        "phase": "running",
        "pid": os.getpid(),
        "started_at": previous.get("started_at", time.time()),
        "smoke_test_only": config.get("smoke_test_only", False),
        "encoder_updates": config["updates"],
        "fusion_updates": config["final_fusion_updates"],
        "seeds": config["evaluation_seeds"],
        "arms": list(ARMS),
    }
    rt.write(run / "state.json", state)
    try:
        while pending or running:
            while pending and len(running) < config["workers"]:
                arm, seed = pending.pop(0)
                label = f"{arm}-{seed}"
                with (run / f"{label}.log").open("a") as stream:
                    running[label] = subprocess.Popen(
                        [
                            sys.executable,
                            str(BASE / "study.py"),
                            "worker",
                            "--run",
                            str(run),
                            "--arm",
                            arm,
                            "--seed",
                            str(seed),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                        },
                    )
            for label, process in list(running.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{label} failed ({code}); see its log")
                    del running[label]
            state.update(
                heartbeat_at=time.time(),
                running={k: p.pid for k, p in running.items()},
                pending=[f"{a}-{s}" for a, s in pending],
            )
            rt.write(run / "state.json", state)
            if running:
                time.sleep(2)
        state.update(phase="finalizing", heartbeat_at=time.time())
        rt.write(run / "state.json", state)
        finalize(config, run)
        rt.check(rt.read(run / "design-freeze.json")["files"])
        products = [
            run / "results.json",
            run / "models-freeze.json",
            run / "selection-freeze.json",
            *(
                target(run, a, s) / "complete.json"
                for a in ARMS
                for s in config["evaluation_seeds"]
            ),
            *(run / "scores").glob("*.npz"),
        ]
        rt.write(
            run / "complete.json",
            {
                "status": "complete",
                "smoke_test_only": config.get("smoke_test_only", False),
                "files": rt.pin(products),
            },
            immutable=True,
        )
        state.update(phase="complete", finished_at=time.time())
        rt.write(run / "state.json", state)
    except BaseException as error:
        for process in running.values():
            if process.poll() is None:
                process.terminate()
        for process in running.values():
            process.wait()
        failure = {"error": repr(error), "time": time.time()}
        rt.write(run / "failure.json", failure)
        rt.write(run / "state.json", {**state, "phase": "failed", **failure})
        raise


def status(run):
    if not (run / "state.json").exists():
        print({"phase": "not_started", "run": str(run)})
        return
    print("state.json", rt.read(run / "state.json"))
    config = rt.read(run / "design-freeze.json")["config"]
    for seed in config["evaluation_seeds"]:
        for arm in ARMS:
            directory = target(run, arm, seed)
            state = (
                rt.read(directory / "state.json")
                if (directory / "state.json").exists()
                else {"phase": "pending"}
            )
            for name in ("acoustic", "fusion"):
                path = directory / name / "state.json"
                if path.exists():
                    state[name] = rt.read(path)
            print(f"{arm}/{seed}", state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "execute", "worker", "status"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rt.torch.set_num_threads(1)
    rt.torch.use_deterministic_algorithms(True)
    config = configuration(args.smoke)
    run = args.run.resolve() if args.run else ROOT / config["run_directory"]
    if args.phase == "status":
        status(run)
        return
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        config = rt.read(run / "design-freeze.json")["config"]
        if args.arm not in ARMS or args.seed not in config["evaluation_seeds"]:
            raise ValueError("Invalid worker identity")
        lock_path = run / f"{args.arm}-{args.seed}.lock"
    else:
        lock_path = run / "execution.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare":
            prepare(config)
        elif args.phase == "execute":
            execute(config, run)
        else:
            try:
                worker(config, run, args.arm, args.seed)
            except BaseException as error:
                rt.write(
                    target(run, args.arm, args.seed) / "failure.json",
                    {"error": repr(error), "time": time.time()},
                )
                raise


if __name__ == "__main__":
    main()
