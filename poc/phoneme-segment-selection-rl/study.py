"""Freeze, execute/resume the full experiment, and expose durable progress."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from final_fusion import fit
from pool import Pool
from prepare import load_training_bank, prepare
from runtime import (
    ARMS,
    BASE,
    ROOT,
    atomic_torch,
    check,
    pin,
    read,
    runtime,
    sha,
    torch,
    write,
)
from trainer import teacher_for, train_episode
from verification import load_evaluations


def prepared_path(config):
    return ROOT / config.get(
        "prepared_directory", config["run_directory"] + "/prepared"
    )


def validate_config(config):
    if not 0 <= config["warmup_updates"] < config["updates"]:
        raise ValueError("Invalid acoustic warmup/budget")
    if config["updates"] % config["block_updates"]:
        raise ValueError("Acoustic budget must contain complete blocks")
    if (
        not 0 < config["teacher_exploration"] < 1
        or not 0 <= config["long_reward_fraction"] <= 1
    ):
        raise ValueError("Invalid exploration or delayed reward fraction")
    if set(config["training_seeds"]) & set(config["evaluation_seeds"]):
        raise ValueError("Teacher and deployment student seeds overlap")
    if not 0 < config["hybrid_proxy_episodes"] < len(config["training_seeds"]):
        raise ValueError("Hybrid requires a proxy stage and an EER stage")
    if config["teacher_checkpoints"] != sorted(
        set(config["teacher_checkpoints"])
    ) or config["teacher_checkpoints"][-1] != len(config["training_seeds"]):
        raise ValueError("Teacher learning-curve checkpoints must include full budget")
    for key in (
        "candidate_pairs",
        "proxy_interval",
        "fusion_steps_per_block",
        "final_fusion_updates",
        "final_fusion_validation_every",
        "long_probe_every_blocks",
        "singleton_interval",
        "workers",
    ):
        if config[key] < 1:
            raise ValueError(f"Invalid positive parameter: {key}")


def freeze(config, run):
    validate_config(config)
    if (run / "design-freeze.json").exists():
        saved = read(run / "design-freeze.json")
        if saved["config"] != config or saved["runtime"] != runtime():
            raise ValueError("Frozen configuration/runtime changed; use a new run")
        check(saved["files"])
        return
    prepared = prepared_path(config)
    if prepared == run / "prepared":
        prepare(config, run)
    else:
        check(read(prepared / "freeze.json")["files"])
    files = dict(read(prepared / "freeze.json")["files"])
    paths = {
        BASE / "config.json",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        *(BASE.parent / "phoneme-speaker-encoder/config").glob("*.json"),
        prepared / "freeze.json",
        BASE.parent / "phoneme-speaker-encoder/scripts/weighted_bootstrap.py",
    }
    if (BASE / "README.md").exists():
        paths.add(BASE / "README.md")
    for imported in list(sys.modules.values()):
        name = getattr(imported, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (
                path.is_relative_to(ROOT / "poc")
                and path.suffix == ".py"
                and ".venv" not in path.parts
            ):
                paths.add(path)
    files.update(pin(paths))
    write(
        run / "design-freeze.json",
        {
            "config": config,
            "runtime": runtime(),
            "files": files,
            "smoke_test_only": config.get("smoke_test_only", False),
            "test_previously_observed": True,
        },
        immutable=True,
    )


def wait_for_proxy(config, run):
    episodes = config["hybrid_proxy_episodes"]
    path = run / f"proxy/teachers/after-{episodes}.pt"
    while not path.exists():
        failure = run / "proxy/failure.json"
        if failure.exists():
            raise RuntimeError(f"Proxy warm start failed: {read(failure)}")
        write(
            run / "hybrid/state.json",
            {"phase": "waiting_for_proxy", "episodes": episodes},
        )
        time.sleep(2)
    return path


def worker(config, run, arm):
    check(read(run / "design-freeze.json")["files"])
    directory = run / arm
    directory.mkdir(exist_ok=True)
    (directory / "failure.json").unlink(missing_ok=True)
    if (directory / "complete.json").exists():
        check(read(directory / "complete.json")["files"])
        return
    warm_start = wait_for_proxy(config, run) if arm == "hybrid" else None
    prepared = prepared_path(config)
    pool = Pool(ROOT / config["source_run"], prepared)
    bank = load_training_bank(config, prepared, pool)
    teacher, optimizer = teacher_for(config, len(pool.phones))
    reward = load_evaluations(config, prepared, "reward") if arm != "uniform" else None
    teacher_files = []
    if arm != "uniform":
        if warm_start:
            state = torch.load(warm_start, weights_only=True)
            teacher.load_state_dict(state["teacher"])
            optimizer.load_state_dict(state["optimizer"])
            write(
                directory / "warm-start.json",
                {
                    "file": str(warm_start.relative_to(ROOT)),
                    "sha256": sha(warm_start),
                    "shared_proxy_episodes": config["hybrid_proxy_episodes"],
                },
                immutable=True,
            )
        for episode in range(len(config["training_seeds"])):
            output = directory / f"teachers/after-{episode + 1}.pt"
            if warm_start and episode < config["hybrid_proxy_episodes"]:
                original = run / f"proxy/teachers/after-{episode + 1}.pt"
                atomic_torch(output, torch.load(original, weights_only=True))
            else:
                write(
                    directory / "state.json",
                    {
                        "phase": "teacher_training",
                        "episode": episode + 1,
                        "episodes": len(config["training_seeds"]),
                        "arm": arm,
                    },
                )
                episode_dir = directory / f"training/episode-{episode + 1}"
                train_episode(
                    config,
                    episode_dir,
                    pool,
                    bank,
                    reward,
                    arm,
                    episode,
                    teacher,
                    optimizer,
                )
                atomic_torch(
                    output, torch.load(episode_dir / "teacher.pt", weights_only=True)
                )
            teacher_files.append(output)
        write(
            directory / "teachers-freeze.json",
            {"files": pin(teacher_files)},
            immutable=True,
        )
    del reward
    selection = load_evaluations(config, prepared, "selection")
    budgets = [0] if arm == "uniform" else config["teacher_checkpoints"]
    products = teacher_files.copy()
    for episode_dir in (directory / "training").glob("episode-*"):
        products.extend(
            episode_dir / name
            for name in (
                "complete.json",
                "history.json",
                "coverage.json",
                "teacher.pt",
                "encoder.pt",
            )
        )
    for budget in budgets:
        if arm != "uniform":
            teacher.load_state_dict(
                torch.load(
                    directory / f"teachers/after-{budget}.pt", weights_only=True
                )["teacher"]
            )
        for seed in config["evaluation_seeds"]:
            target = directory / f"deployment/budget-{budget}/seed-{seed}"
            write(
                directory / "state.json",
                {
                    "phase": "fresh_student_evaluation",
                    "arm": arm,
                    "teacher_episodes": budget,
                    "student_seed": seed,
                    "encoder_updates": config["updates"],
                    "fusion_updates": config["final_fusion_updates"],
                },
            )
            student, _, _ = train_episode(
                config,
                target / "acoustic",
                pool,
                bank,
                None,
                arm,
                seed,
                None if arm == "uniform" else teacher,
                None,
                training_teacher=False,
            )
            fit(config, target / "fusion", student, bank, selection)
            products.extend(
                [
                    target / "acoustic/encoder.pt",
                    target / "acoustic/complete.json",
                    target / "acoustic/coverage.json",
                    target / "fusion/selected.pt",
                    target / "fusion/final.pt",
                    target / "fusion/history.json",
                    target / "fusion/complete.json",
                ]
            )
    write(
        directory / "complete.json",
        {"status": "complete", "files": pin(products)},
        immutable=True,
    )
    write(directory / "state.json", {"phase": "complete", "arm": arm})


def execute(config, run):
    freeze(config, run)
    if (run / "complete.json").exists():
        check(read(run / "complete.json")["files"])
        return
    pending = ["proxy", "eer", "hybrid", "uniform"]
    running = {}
    write(
        run / "state.json",
        {
            "phase": "running",
            "pid": os.getpid(),
            "arms": list(ARMS),
            "teacher_episodes": len(config["training_seeds"]),
            "smoke_test_only": config.get("smoke_test_only", False),
            "encoder_updates_per_trajectory": config["updates"],
            "started_at": time.time(),
        },
    )
    try:
        while pending or running:
            while pending and len(running) < config["workers"]:
                arm = pending.pop(0)
                (run / arm).mkdir(exist_ok=True)
                with (run / f"{arm}.log").open("a") as stream:
                    running[arm] = subprocess.Popen(
                        [
                            sys.executable,
                            str(BASE / "study.py"),
                            "worker",
                            "--run",
                            str(run),
                            "--arm",
                            arm,
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                        },
                    )
            for arm, process in list(running.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(
                            f"{arm} worker failed ({code}); see {run / (arm + '.log')}"
                        )
                    del running[arm]
            if running:
                time.sleep(2)
        from results import finalize

        finalize(config, run)
        check(read(run / "design-freeze.json")["files"])
        products = [
            run / "results.json",
            run / "selection-freeze.json",
            run / "models-freeze.json",
            *(run / arm / "complete.json" for arm in ARMS),
            *(run / "scores").glob("*.npz"),
        ]
        write(
            run / "complete.json",
            {
                "files": pin(products),
                "status": "complete",
                "smoke_test_only": config.get("smoke_test_only", False),
            },
            immutable=True,
        )
        write(
            run / "state.json",
            {
                "phase": "complete",
                "smoke_test_only": config.get("smoke_test_only", False),
                "finished_at": time.time(),
            },
        )
    except BaseException as error:
        for process in running.values():
            if process.poll() is None:
                process.terminate()
        for process in running.values():
            process.wait()
        write(
            run / "state.json",
            {"phase": "failed", "error": repr(error), "time": time.time()},
        )
        raise


def status(run):
    if not run.exists():
        print("not_started", run)
        return
    for path in [run / "state.json", *(run / arm / "state.json" for arm in ARMS)]:
        if path.exists():
            print(str(path.relative_to(run)), read(path))
    for arm in ARMS:
        path = run / arm / "state.json"
        if not path.exists():
            continue
        state = read(path)
        if state["phase"] == "teacher_training":
            details = run / arm / f"training/episode-{state['episode']}/state.json"
        elif state["phase"] == "fresh_student_evaluation":
            target = (
                run
                / arm
                / f"deployment/budget-{state['teacher_episodes']}/seed-{state['student_seed']}"
            )
            details = target / (
                "fusion/state.json"
                if (target / "fusion/state.json").exists()
                else "acoustic/state.json"
            )
        else:
            continue
        if details.exists():
            print(str(details.relative_to(run)), read(details))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "execute", "worker", "status"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = read(BASE / "config.json")
    if args.smoke:
        config.update(
            prepared_directory=config["run_directory"] + "/prepared",
            run_directory=config["run_directory"] + "-smoke",
            smoke_test_only=True,
            updates=400,
            warmup_updates=10,
            training_seeds=config["training_seeds"][:2],
            evaluation_seeds=config["evaluation_seeds"][:1],
            teacher_checkpoints=[1, 2],
            hybrid_proxy_episodes=1,
            final_fusion_updates=40,
            final_fusion_validation_every=20,
            long_probe_every_blocks=1,
            long_probe_updates=100,
        )
    run = args.run.resolve() if args.run else ROOT / config["run_directory"]
    if args.phase == "status":
        status(run)
        return
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        config = read(run / "design-freeze.json")["config"]
        lock_path = run / f"{args.arm}.lock"
    else:
        lock_path = run / "execution.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "prepare":
            prepare(config, run)
        elif args.phase == "execute":
            execute(config, run)
        else:
            try:
                worker(config, run, args.arm)
            except BaseException as error:
                write(
                    run / args.arm / "failure.json",
                    {"error": repr(error), "time": time.time()},
                )
                raise


if __name__ == "__main__":
    main()
