"""Freeze, train, evaluate and audit the fixed 34-phone retraining hypothesis."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time

from common import (
    BASE,
    CONFIG,
    ROOT,
    check,
    family,
    old_model_path,
    pin,
    prior,
    read,
    runtime,
    state,
    torch,
    write,
)
from trainer import run_training


def freeze(config, run):
    path = run / "design-freeze.json"
    if path.exists():
        frozen = read(path)
        assert frozen["config"] == config and frozen["runtime"] == runtime()
        check(frozen["files"])
        return frozen
    source = ROOT / config["source_run"]
    previous = ROOT / config["subset_run"]
    assert read(previous / "state.json")["status"] == "complete"
    previous_design = read(previous / "design-freeze.json")
    check(previous_design["files"])
    check(read(previous / "complete-audit.json")["files"])
    check(read(source / "complete-audit.json")["files"])
    phones = read(source / "design-freeze.json")["phones"]
    winner = read(previous / "winner-freeze.json")
    assert winner["removed"] == config["removed_phones"] == ["ch", "k"]
    assert len(winner["subset"]) == 34
    files = {
        **previous_design["files"],
        **read(previous / "complete-audit.json")["files"],
    }
    paths = [
        CONFIG,
        BASE / "README.md",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        previous / "winner-freeze.json",
        previous / "independent-audit.json",
        source / "complete-audit.json",
        prior.BASE / "core.py",
        prior.BACKEND_PATH,
        prior.METRICS_PATH,
        prior.BOOTSTRAP_PATH,
    ]
    for seed in config["encoder_seeds"]:
        root = family(config, seed)
        original = read(root / "fusion/design-freeze.json")["config"]
        for key in (
            "learning_rate",
            "weight_decay",
            "uniform_kl_coefficient",
            "validation_every",
        ):
            assert config[key] == original[key]
        assert config["prefix_updates"] == original["updates"]
        files.update(read(root / "fusion/prepared-freeze.json")["files"])
        files.update(read(root / "continued/schedule-freeze.json")["files"])
        paths += [
            root / "fusion" / f"transformer-{config['fusion_seed']}" / name
            for name in (
                "last.pt",
                "fusion.pt",
                "history.json",
                "training-summary.json",
            )
        ]
        paths += [old_model_path(config, seed, mode) for mode in ("selected", "final")]
        paths += [
            root
            / "continued"
            / f"seed-{config['fusion_seed']}/budget-{config['updates']}/summary.json"
        ]
        for c in config["corpora"]:
            for split in ("validation", "test"):
                paths += [
                    source
                    / "scores"
                    / f"{c}-src490-seed{seed}-u{config['updates']}-{mode}-{split}.json"
                    for mode in ("selected", "final")
                ]
    files.update(pin(paths))
    check(files)
    frozen = {
        "config": config,
        "files": files,
        "phones": phones,
        "subset": winner["subset"],
        "runtime": runtime(),
        "all_test_previously_observed": True,
        "new_training_runs": 3,
        "subset_selection_not_repeated": True,
    }
    write(path, frozen, immutable=True)
    return frozen


def execute(config, run):
    frozen = freeze(config, run)
    if (run / "complete-audit.json").exists():
        check(read(run / "complete-audit.json")["files"])
        state(run, "complete", verified_existing=True)
        return
    state(run, "reference_prefix")
    run_training(config, run, config["encoder_seeds"][0], reference=True)
    state(run, "training", seeds=config["encoder_seeds"])
    children = []
    try:
        for seed in config["encoder_seeds"]:
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
                raise RuntimeError("Training worker failed; see per-seed log")
            time.sleep(5)
        assert all(p.returncode == 0 for p in children)
        from evaluation import bootstrap_comparisons, evaluate

        evaluate(config, run, frozen)
        state(run, "bootstrap")
        bootstrap_comparisons(config, run)
        from audit import audit
        from report import report

        audit(config, run)
        report(config, run)
        check(frozen["files"])
        paths = [
            p
            for p in run.rglob("*")
            if p.is_file()
            and p.name not in ("state.json", "execution.lock", "complete-audit.json")
            and not p.name.endswith((".tmp", ".lock", ".log"))
        ]
        write(
            run / "complete-audit.json",
            {
                "files": pin(paths),
                "source_files_unchanged": True,
                "new_training_runs_complete": 3,
                "model_and_threshold_freeze_before_test": True,
            },
            immutable=True,
        )
        state(run, "complete")
    except BaseException:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            process.wait()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["execute", "worker", "status"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "status":
        for path in [run / "state.json", *run.glob("acoustic-*/state.json")]:
            if path.exists():
                print(str(path.relative_to(run)), read(path))
        return
    lock_path = run / (
        f"worker-{args.seed}.lock" if args.phase == "worker" else "execution.lock"
    )
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase == "worker":
                assert args.seed in config["encoder_seeds"]
                run_training(config, run, args.seed)
            else:
                execute(config, run)
        except BaseException as error:
            state(
                run if args.phase != "worker" else run / f"acoustic-{args.seed}",
                "failed",
                error=repr(error),
            )
            raise


if __name__ == "__main__":
    main()
