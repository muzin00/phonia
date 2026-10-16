"""Freeze, run/resume and inspect the paired scoring experiment."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from scoring import FrozenModel, arm_configuration
from scoring_data import prepare, training_bank
from scoring_results import finalize
from shared import (
    ARMS,
    BASE,
    ROOT,
    cached_evaluations,
    configuration,
    fit,
    rt,
    target,
)


def freeze(config, run):
    seal = run / "design-freeze.json"
    if seal.exists():
        saved = rt.read(seal)
        if saved["config"] != config or saved["runtime"] != rt.runtime():
            raise ValueError("Frozen design/runtime changed; use a new run")
        rt.check(saved["files"])
        return
    if config["arms"] != list(ARMS) or config["workers"] < 1:
        raise ValueError("Invalid arms/workers")
    if not 0 < config["final_fusion_initial_updates"] < config["final_fusion_updates"]:
        raise ValueError("Invalid training budgets")
    prepare(config, run)
    paths = {
        BASE / "config.json",
        BASE / "README.md",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        run / "prepared/freeze.json",
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
    files = {**rt.read(run / "prepared/freeze.json")["files"], **rt.pin(paths)}
    rt.write(
        seal,
        {"config": config, "runtime": rt.runtime(), "files": files},
        immutable=True,
    )


def worker(config, run, arm, seed):
    rt.check(rt.read(run / "design-freeze.json")["files"])
    directory = target(run, arm, seed)
    directory.mkdir(parents=True, exist_ok=True)
    seal = directory / "worker-complete.json"
    if seal.exists():
        rt.check(rt.read(seal)["files"])
        return
    (directory / "failure.json").unlink(missing_ok=True)
    rt.write(
        directory / "worker-state.json", {"phase": "preparing", "pid": os.getpid()}
    )
    model = FrozenModel(config, arm, seed)
    before = rt.tree_hash(model.encoder.state_dict())
    bank = training_bank(config, seed)
    selection = cached_evaluations(config, model.encoder, "selection")
    path = run / "prepared" / f"seed-{seed}/schedule.npz"
    with rt.np.load(path, allow_pickle=False) as archive:
        schedule = {k: archive[k].copy() for k in archive.files}
    rt.write(
        directory / "worker-state.json",
        {"phase": "fusion_training", "pid": os.getpid()},
    )
    fit(
        arm_configuration(config, arm),
        directory,
        model,
        bank,
        selection,
        schedule_override=schedule,
    )
    after = rt.tree_hash(model.encoder.state_dict())
    if before != after or any(p.grad is not None for p in model.encoder.parameters()):
        raise AssertionError("Frozen encoder was updated")
    rt.write(
        seal,
        {
            "status": "complete",
            "encoder_state_sha256": after,
            "schedule_sha256": rt.sha(path),
            "files": rt.pin(
                directory / name
                for name in (
                    "selected.pt",
                    "final.pt",
                    "last.pt",
                    "history.json",
                    "complete.json",
                )
            ),
        },
        immutable=True,
    )
    rt.write(directory / "worker-state.json", {"phase": "complete", "pid": os.getpid()})


def execute(config, run):
    running = {}
    prior = rt.read(run / "state.json") if (run / "state.json").exists() else {}
    state = {
        "phase": "preparing",
        "pid": os.getpid(),
        "started_at": prior.get("started_at", time.time()),
        "arms": list(ARMS),
        "seeds": config["evaluation_seeds"],
        "updates_per_model": config["final_fusion_updates"],
        "smoke_test_only": config.get("smoke_test_only", False),
    }
    try:
        freeze(config, run)
        if (run / "complete.json").exists():
            rt.check(rt.read(run / "complete.json")["files"])
            return
        (run / "failure.json").unlink(missing_ok=True)
        pending = [(a, s) for s in config["evaluation_seeds"] for a in ARMS]
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
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "PYTHONUNBUFFERED": "1",
                        },
                    )
            for label, process in list(running.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{label} failed ({code}); see its log")
                    del running[label]
            state.update(
                phase="running",
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
                target(run, a, s) / "worker-complete.json"
                for s in config["evaluation_seeds"]
                for a in ARMS
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
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        failure = {"error": repr(error), "time": time.time()}
        rt.write(run / "failure.json", failure)
        rt.write(run / "state.json", {**state, "phase": "failed", **failure})
        raise


def status(run):
    for name in ("state.json", "failure.json", "complete.json"):
        if (run / name).exists():
            value = rt.read(run / name)
            print(name, {k: v for k, v in value.items() if k != "files"})
    if not (run / "design-freeze.json").exists():
        print("design not frozen", str(run))
        return
    config = rt.read(run / "design-freeze.json")["config"]
    for seed in config["evaluation_seeds"]:
        for arm in ARMS:
            directory = target(run, arm, seed)
            worker_state = directory / "worker-state.json"
            item = (
                rt.read(worker_state) if worker_state.exists() else {"phase": "pending"}
            )
            if (directory / "state.json").exists():
                item.update(rt.read(directory / "state.json"))
            if (directory / "failure.json").exists():
                item["failure"] = rt.read(directory / "failure.json")
            print(f"{arm}/{seed}", item)


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
        if args.seed not in config["evaluation_seeds"] or args.arm not in ARMS:
            raise ValueError("Invalid worker identity")
    lock_path = run / (
        f"{args.arm}-{args.seed}.lock" if args.phase == "worker" else "execution.lock"
    )
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare":
            freeze(config, run)
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
