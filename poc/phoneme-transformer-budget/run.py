"""Extend frozen Transformer trajectories without changing the original experiment."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-corpus-transformer"))
import experiment as e

s, r = e.s, e.r
CONFIG = BASE / "config.json"


def source(config):
    return s.ROOT / config["reference_run"]


def original_config(config):
    return s.read(source(config) / "design-freeze.json")["config"]


def build_schedule(config, original, data, seed):
    with s.np.load(
        source(config) / f"schedule-{seed}.npz", allow_pickle=False
    ) as saved:
        blocks = [{k: saved[k].copy() for k in saved.files}]
    block_size = config["schedule_block_updates"]
    assert len(blocks[0]["queries"]) == block_size == config["budgets"][0]
    assert max(config["budgets"]) % block_size == 0
    for block in range(1, max(config["budgets"]) // block_size):
        blocks.append(
            e.make_schedule(
                {**original, "updates": block_size},
                data,
                seed + block * config["schedule_seed_stride"],
            )
        )
    return {k: s.np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def select_best(history):
    best = history[0]
    for item in history[1:]:
        if item["macro_eer"] < best["macro_eer"] - 1e-12:
            best = item
    return best


def freeze(config, run):
    ref = source(config)
    assert s.read(ref / "execution-state.json")["status"] == "complete"
    design = s.read(ref / "design-freeze.json")
    assert config["seeds"] == design["config"]["seeds"]
    assert config["budgets"] == [4000, 16000, 64000]
    assert config["validation_every"] == design["config"]["validation_every"]
    assert config["parallel_workers"] == len(config["seeds"])
    assert s.expanded.runtime() == design["runtime"]
    files = {
        **design["files"],
        **s.read(ref / "prepared-freeze.json")["files"],
        **s.read(ref / "complete-audit.json")["files"],
    }
    paths = [
        CONFIG,
        BASE / "run.py",
        *sorted((BASE / "tests").glob("*.py")),
        ref / "design-freeze.json",
        ref / "prepared-freeze.json",
        ref / "complete-audit.json",
    ]
    for seed in config["seeds"]:
        paths += [
            ref / f"transformer-{seed}" / name
            for name in (
                "fusion.pt",
                "last.pt",
                "history.json",
                "training-summary.json",
            )
        ]
    files.update(r.pin(paths))
    r.checked_files(files)
    frozen = {
        "config": config,
        "files": files,
        "runtime": s.expanded.runtime(),
        "source_design_sha256": s.sha(ref / "design-freeze.json"),
    }
    s.fixed(run / "design-freeze.json", frozen)
    if not (run / "schedule-freeze.json").exists():
        data = s.read(ref / "train-inputs.json")
        for seed in config["seeds"]:
            s.np.savez(
                run / f"schedule-{seed}.npz",
                **build_schedule(config, design["config"], data, seed),
            )
        s.fixed(
            run / "schedule-freeze.json",
            {"files": r.pin(run / f"schedule-{seed}.npz" for seed in config["seeds"])},
        )
    r.checked_files(s.read(run / "schedule-freeze.json")["files"])


def write_state(path, status, **details):
    s.atomic(
        path,
        {
            "status": status,
            "pid": os.getpid(),
            "updated_at_epoch": time.time(),
            **details,
        },
    )


def snapshot(directory, seed, budget, model, history, encoder_sha):
    destination = directory / f"budget-{budget}"
    destination.mkdir(exist_ok=True)
    best = select_best(history)
    for mode in ("selected", "final"):
        path = destination / f"{mode}.pt"
        if not path.exists():
            if mode == "selected":
                shutil.copyfile(directory / "best.pt", path)
            else:
                s.torch.save(
                    {
                        "kind": "transformer",
                        "seed": seed,
                        "update": budget,
                        "state_dict": model.state_dict(),
                    },
                    path,
                )
    s.fixed(
        destination / "summary.json",
        {
            "seed": seed,
            "completed_updates": budget,
            "selected_update": best["update"],
            "validation_macro_eer": best["macro_eer"],
            "final_validation_macro_eer": history[-1]["macro_eer"],
            "encoder_sha256": encoder_sha,
            "encoder_updated": False,
            "models": r.pin(
                destination / f"{mode}.pt" for mode in ("selected", "final")
            ),
        },
    )


def worker(config, run, seed):
    original = original_config(config)
    ref = source(config)
    directory = run / f"seed-{seed}"
    directory.mkdir(exist_ok=True)
    state_path = directory / "state.json"
    if (directory / "complete.json").exists():
        r.checked_files(s.read(directory / "complete.json")["files"])
        return
    with (directory / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            saved_path = directory / "last.pt"
            continued = saved_path.exists()
            saved = s.torch.load(
                saved_path if continued else ref / f"transformer-{seed}/last.pt",
                weights_only=False,
            )
            expected = run if continued else ref
            assert saved["design_sha256"] == s.sha(expected / "design-freeze.json")
            e.s.expanded.seed_everything(seed)
            model = e.create_model(original, "transformer")
            model.load_state_dict(saved["model"])
            optimizer = s.torch.optim.AdamW(
                model.parameters(),
                lr=original["learning_rate"],
                weight_decay=original["weight_decay"],
            )
            optimizer.load_state_dict(saved["optimizer"])
            s.torch.set_rng_state(saved["rng"])
            history = saved["history"]
            best = select_best(history)
            assert (
                saved["best"] == best["macro_eer"]
                and saved["selected"] == best["update"]
            )
            if not continued:
                assert saved["update"] == config["budgets"][0]
                shutil.copyfile(
                    ref / f"transformer-{seed}/fusion.pt", directory / "best.pt"
                )
                snapshot(
                    directory,
                    seed,
                    saved["update"],
                    model,
                    history,
                    original["encoder_sha256"],
                )
            else:
                # A crash may leave best.pt newer than the atomic last.pt.
                # Restore both states from the same saved boundary.
                s.torch.save(saved["best_bundle"], directory / "best.pt")
            arrays = e.load_arrays(ref / "train.npz")
            data = s.read(ref / "train-inputs.json")
            validation = {
                c: (
                    e.load_arrays(ref / f"{c}-validation.npz"),
                    s.read(ref / f"{c}-validation-baseline.json"),
                )
                for c in s.CORPORA
            }
            with s.np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as archive:
                schedule = {k: archive[k].copy() for k in archive.files}
            began = time.monotonic()
            running_loss = 0.0
            write_state(state_path, "training", seed=seed, update=saved["update"])
            for update in range(saved["update"] + 1, max(config["budgets"]) + 1):
                model.train()
                for group in optimizer.param_groups:
                    group["lr"] = config["continuation_learning_rate"]
                f, c, mask, labels = e.training_batch(
                    arrays, data, schedule, update - 1
                )
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(
                    f, c, mask, labels, original["uniform_kl_coefficient"]
                )
                assert s.torch.isfinite(loss)
                loss.backward()
                s.torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
                running_loss += float(loss.detach())
                if update % config["validation_every"]:
                    continue
                eers = {
                    c: e.eer(e.score(model, a, b)) for c, (a, b) in validation.items()
                }
                macro = sum(eers.values()) / len(eers)
                if macro < best["macro_eer"] - 1e-12:
                    best = {"update": update, "macro_eer": macro}
                    s.torch.save(
                        {
                            "kind": "transformer",
                            "seed": seed,
                            "update": update,
                            "state_dict": model.state_dict(),
                        },
                        directory / "best.pt",
                    )
                item = {
                    "update": update,
                    "validation_eer": eers,
                    "macro_eer": macro,
                    "selected_update": best["update"],
                    "mean_training_loss": running_loss / config["validation_every"],
                    "elapsed_seconds": time.monotonic() - began,
                }
                running_loss = 0.0
                history.append(item)
                s.atomic(directory / "history.json", history)
                if update in config["budgets"]:
                    snapshot(
                        directory,
                        seed,
                        update,
                        model,
                        history,
                        original["encoder_sha256"],
                    )
                temporary = directory / "last.tmp"
                s.torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": s.torch.get_rng_state(),
                        "best": best["macro_eer"],
                        "selected": best["update"],
                        "update": update,
                        "history": history,
                        "best_bundle": s.torch.load(
                            directory / "best.pt", weights_only=True
                        ),
                        "design_sha256": s.sha(run / "design-freeze.json"),
                    },
                    temporary,
                )
                temporary.replace(saved_path)
                write_state(state_path, "training", seed=seed, **item)
                print({"seed": seed, **item}, flush=True)
            files = [
                directory / "history.json",
                directory / "last.pt",
                *(p for p in directory.glob("budget-*/*") if p.is_file()),
            ]
            s.fixed(
                directory / "complete.json",
                {
                    "files": r.pin(files),
                    "updates": max(config["budgets"]),
                    "seed": seed,
                },
            )
            write_state(
                state_path, "complete", seed=seed, update=max(config["budgets"])
            )
        except BaseException as error:
            write_state(state_path, "failed", seed=seed, error=str(error))
            raise


def variants(config, run):
    return {
        f"u{budget}-{mode}-{seed}": run / f"seed-{seed}/budget-{budget}/{mode}.pt"
        for budget in config["budgets"]
        for mode in ("selected", "final")
        for seed in config["seeds"]
    }


def load_model(config, path):
    bundle = s.torch.load(path, weights_only=True)
    model = e.create_model(original_config(config), "transformer")
    model.load_state_dict(bundle["state_dict"])
    return model.eval()


def evaluate(config, run):
    ref = source(config)
    paths = variants(config, run)
    for seed in config["seeds"]:
        r.checked_files(s.read(run / f"seed-{seed}/complete.json")["files"])
    s.fixed(run / "all-models-freeze.json", {"files": r.pin(paths.values())})
    models = {name: load_model(config, path) for name, path in paths.items()}
    thresholds = {}
    validation_paths = []
    for corpus in s.CORPORA:
        write_state(run / "state.json", "calibrating", corpus=corpus)
        arrays = e.load_arrays(ref / f"{corpus}-validation.npz")
        baseline = s.read(ref / f"{corpus}-validation-baseline.json")
        thresholds[corpus] = {}
        for name in ("equal", *models):
            rows = (
                baseline if name == "equal" else e.score(models[name], arrays, baseline)
            )
            normal = [
                v
                for v in rows
                if v["role"] == "verification" and v["score"] is not None
            ]
            point, _ = s.far_target_threshold(
                [v["score"] for v in normal], [v["is_genuine"] for v in normal], 0.01
            )
            thresholds[corpus][name] = s.json_threshold(point)
            path = run / "scores" / f"{corpus}-{name}-validation.json"
            s.fixed(path, rows)
            validation_paths.append(path)
    s.fixed(
        run / "selection-freeze.json",
        {
            "models": r.pin(paths.values()),
            "thresholds": thresholds,
            "validation_files": r.pin(validation_paths),
            "all_18_checkpoints_fixed_before_new_test_scoring": True,
        },
    )
    results = {}
    for corpus in s.CORPORA:
        write_state(run / "state.json", "evaluating", corpus=corpus)
        arrays = e.load_arrays(ref / f"{corpus}-test.npz")
        baseline = s.read(ref / f"{corpus}-test-baseline.json")
        results[corpus] = {}
        for name in ("equal", *models):
            rows = (
                baseline if name == "equal" else e.score(models[name], arrays, baseline)
            )
            s.fixed(run / "scores" / f"{corpus}-{name}-test.json", rows)
            results[corpus][name] = {
                role: s.operating(
                    [v for v in rows if v["role"] == role], thresholds[corpus][name]
                )
                for role in sorted({v["role"] for v in rows})
            }
    aggregate = {
        c: {
            f"u{budget}-{mode}": {
                metric: {
                    "values": [
                        values[f"u{budget}-{mode}-{seed}"]["verification"][metric]
                        for seed in config["seeds"]
                    ],
                    "mean": float(
                        s.np.mean(
                            [
                                values[f"u{budget}-{mode}-{seed}"]["verification"][
                                    metric
                                ]
                                for seed in config["seeds"]
                            ]
                        )
                    ),
                }
                for metric in ("eer", "far", "frr", "coverage")
            }
            for budget in config["budgets"]
            for mode in ("selected", "final")
        }
        for c, values in results.items()
    }
    s.fixed(
        run / "results.json",
        {
            "conditions": results,
            "aggregate": aggregate,
            "config": config,
            "selection_sha256": s.sha(run / "selection-freeze.json"),
        },
    )
    lines = [
        "# Transformerの追加学習",
        "",
        "同じencoder・データ・3seedの継続学習。4,000更新以降の学習率は3e-5で固定。EERは採点可能分、FAR/FRRは全入力。",
        "",
        "| corpus | checkpoint | EER平均 | FAR平均 | FRR平均 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus, values in aggregate.items():
        for name, metrics in values.items():
            lines.append(
                f"| {corpus} | {name} | "
                + " | ".join(
                    f"{metrics[m]['mean'] * 100:.3f}%" for m in ("eer", "far", "frr")
                )
                + " |"
            )
    (run / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("execute", "freeze", "worker", "status"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = s.read(CONFIG)
    run = s.ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "status":
        print(
            {
                "run": s.read(run / "state.json"),
                "seeds": {
                    str(seed): s.read(run / f"seed-{seed}/state.json")
                    for seed in config["seeds"]
                    if (run / f"seed-{seed}/state.json").exists()
                },
            }
        )
        return
    if args.phase == "worker":
        assert args.seed in config["seeds"]
        worker(config, run, args.seed)
        return
    with (run / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_state(run / "state.json", "freezing")
            freeze(config, run)
            if args.phase == "freeze":
                return
            jobs = []
            for seed in config["seeds"]:
                log = (run / f"seed-{seed}.log").open("a")
                jobs.append(
                    (
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(BASE / "run.py"),
                                "worker",
                                "--seed",
                                str(seed),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        ),
                        log,
                    )
                )
            write_state(
                run / "state.json", "training", workers=[job.pid for job, _ in jobs]
            )
            codes = [job.wait() for job, _ in jobs]
            for _, log in jobs:
                log.close()
            assert codes == [0] * len(jobs), f"worker exits: {codes}"
            r.checked_files(s.read(run / "design-freeze.json")["files"])
            evaluate(config, run)
            write_state(run / "state.json", "complete")
        except BaseException as error:
            write_state(run / "state.json", "failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
