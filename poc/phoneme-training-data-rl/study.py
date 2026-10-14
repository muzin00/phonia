"""Freeze, run a training-data selection experiment, evaluate and audit without promotion."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time

from common import (
    ARMS,
    BASE,
    CONFIG,
    REUSE,
    ROOT,
    assert_nested_equal,
    atomic_torch,
    check,
    evaluation_inputs,
    family,
    initial_model,
    np,
    partitions,
    pin,
    prior,
    read,
    runtime,
    state,
    torch,
    training_data,
    write,
)
from selection import actions_for, make_draws
from trainer import run_training


def freeze(config, run):
    path = run / "design-freeze.json"
    if path.exists():
        frozen = read(path)
        assert frozen["config"] == config and frozen["runtime"] == runtime()
        check(frozen["files"])
        return frozen
    assert config["updates"] % config["validation_every"] == 0
    source = ROOT / config["source_run"]
    source_audit = read(source / "complete-audit.json")
    check(source_audit["files"])
    phones = read(source / "design-freeze.json")["phones"]
    assert len(phones) == 36
    files = dict(source_audit["files"])
    paths = [
        CONFIG,
        BASE / "README.md",
        *BASE.glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        REUSE,
        prior.BASE / "core.py",
        prior.BACKEND_PATH,
        prior.METRICS_PATH,
        prior.BOOTSTRAP_PATH,
        source / "complete-audit.json",
        source / "design-freeze.json",
    ]
    training_counts, actions, split_manifest = {}, {}, {}
    assert config["updates"] % config["block_updates"] == 0
    assert config["validation_every"] % config["block_updates"] == 0
    for seed in config["encoder_seeds"]:
        root = family(config, seed)
        for name in ("fusion/prepared-freeze.json", "continued/schedule-freeze.json"):
            files.update(read(root / name)["files"])
            paths.append(root / name)
        paths += [
            root / "trial/encoder.pt",
            root / "fusion/design-freeze.json",
            root / "fusion/train-inputs.json",
            root / "fusion/train.npz",
            root / "continued" / f"schedule-{config['fusion_seed']}.npz",
        ]
        arrays, owners, pools = training_data(config, seed)
        actions[str(seed)] = actions_for(pools)
        draws = make_draws(config, pools)
        train = read(root / "fusion/train-inputs.json")
        train_speakers = set(train["speakers"])
        assert len(train_speakers) == 485
        training_counts[str(seed)] = {
            "speakers": len(train_speakers),
            "queries": len(train["queries"]),
            "candidate_queries_per_update": pools["queries"].shape[1],
            "phone_candidate_counts": pools["common"].sum(axis=(0, 1)).tolist(),
        }
        schedule_path = run / f"draws-{seed}.npz"
        if schedule_path.exists():
            with np.load(schedule_path, allow_pickle=False) as old:
                for key, values in draws.items():
                    np.testing.assert_array_equal(old[key], values)
        else:
            np.savez_compressed(schedule_path, **draws)
        paths.append(schedule_path)
        initial_path = run / f"acoustic-{seed}/initial.pt"
        initial = {"state_dict": initial_model(config, seed).state_dict(), "update": 0}
        if initial_path.exists():
            assert_nested_equal(torch.load(initial_path, weights_only=True), initial)
        else:
            atomic_torch(initial_path, initial)
        paths.append(initial_path)
        for corpus in config["corpora"]:
            speaker_sets = []
            for split in ("validation", "test"):
                info_path = root / "fusion" / f"{corpus}-{split}-inputs.json"
                data = read(info_path)
                assert train_speakers.isdisjoint(data["speakers"])
                speaker_sets.append(set(data["speakers"]))
                paths += [
                    info_path,
                    root / "fusion" / f"{corpus}-{split}.npz",
                    root / "fusion" / f"{corpus}-{split}-baseline.json",
                ]
                if split == "validation":
                    partition = partitions(config, data["speakers"], corpus)
                    assert set(partition["reward"]).isdisjoint(partition["selection"])
                    if corpus in split_manifest:
                        assert split_manifest[corpus] == partition
                    split_manifest[corpus] = partition
            assert speaker_sets[0].isdisjoint(speaker_sets[1])
        del arrays, owners, pools
    write(run / "partitions.json", split_manifest, immutable=True)
    paths.append(run / "partitions.json")
    split_counts = {}
    for seed in config["encoder_seeds"]:
        for corpus in config["corpora"]:
            for split in ("reward", "selection"):
                arrays, _, rows = evaluation_inputs(config, run, seed, corpus, split)
                normal = [r for r in rows if r["role"] == "verification"]
                scorable = [r for r in normal if r["score"] is not None]
                assert any(r["is_genuine"] for r in scorable) and any(
                    not r["is_genuine"] for r in scorable
                )
                split_counts[f"{seed}/{corpus}/{split}"] = {
                    "speakers": len(split_manifest[corpus][split]),
                    "trials": len(normal),
                    "scorable": len(scorable),
                    "genuine": sum(r["is_genuine"] for r in scorable),
                }
                del arrays
    files.update(pin(paths))
    check(files)
    frozen = {
        "config": config,
        "files": files,
        "phones": phones,
        "runtime": runtime(),
        "training_counts": training_counts,
        "actions": actions,
        "split_counts": split_counts,
        "all_test_previously_observed": True,
        "new_training_runs": 9,
        "training_validation_test_speakers_disjoint": True,
    }
    write(path, frozen, immutable=True)
    return frozen


def execute(config, run):
    frozen = freeze(config, run)
    if (run / "complete-audit.json").exists():
        check(read(run / "complete-audit.json")["files"])
        state(run, "complete", verified_existing=True)
        return
    state(run, "training", seeds=config["encoder_seeds"], arms=ARMS)
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
            time.sleep(2)
        assert all(p.returncode == 0 for p in children)
        from evaluation import bootstrap_comparisons, evaluate

        evaluate(config, run)
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
                "new_training_runs_complete": 9,
                "model_and_threshold_freeze_before_test": True,
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
        for path in [run / "state.json", *run.glob("acoustic-*/*/state.json")]:
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
                check(read(run / "design-freeze.json")["files"])
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
