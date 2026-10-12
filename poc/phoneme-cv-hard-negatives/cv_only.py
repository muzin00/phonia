"""Restrict the audited hard-negative intervention to CV training queries."""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-data-budget"))
import data_budget as d
from schedule import restrict_to_cv

s, e, b = d.s, d.e, d.budget
CONFIG = BASE / "config.json"


def sources(config):
    old = d.read(d.ROOT / config["reference_run"] / "results.json")
    return {
        seed: old["sources"][f"combined420-seed{seed}-e{config['encoder_updates']}"]
        for seed in config["encoder_seeds"]
    }


def options(config, source):
    return {**config, "reference_run": source["fusion"]}


def freeze(config, run):
    run.mkdir(parents=True, exist_ok=True)
    prior = d.ROOT / config["all_hard_run"]
    previous = d.read(prior / "design-freeze.json")
    for key in (
        "reference_run",
        "encoder_seeds",
        "encoder_updates",
        "fusion_seed",
        "budgets",
        "validation_every",
        "continuation_learning_rate",
        "mining_seed",
        "candidates",
        "hard_queries_per_update",
    ):
        assert config[key] == previous["config"][key]
    files = dict(previous["files"])
    files.update(d.read(prior / "complete-audit.json")["files"])
    files.update(
        d.pin(
            [
                prior / "complete-audit.json",
                prior / "results.json",
                CONFIG,
                *BASE.glob("*.py"),
                *(BASE / "tests").glob("*.py"),
            ]
        )
    )
    d.check_files(files)
    d.fixed(
        run / "design-freeze.json",
        {"config": config, "files": files, "runtime": s.expanded.runtime()},
    )


def prepare(config, run, seed):
    source = sources(config)[seed]
    directory = run / f"acoustic-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "schedule-freeze.json").exists():
        d.check_files(d.read(directory / "schedule-freeze.json")["files"])
        return
    data = d.read(d.ROOT / source["fusion"] / "train-inputs.json")
    name = f"schedule-{config['fusion_seed']}.npz"
    with np.load(d.ROOT / source["budget"] / name, allow_pickle=False) as archive:
        original = {k: archive[k].copy() for k in archive.files}
    with np.load(
        d.ROOT / config["all_hard_run"] / f"acoustic-{seed}" / name, allow_pickle=False
    ) as archive:
        all_hard = {k: archive[k].copy() for k in archive.files}
    cv_queries = np.array(
        [q["speaker_id"].startswith("cv17_") for q in data["queries"]]
    )
    schedule = restrict_to_cv(original, all_hard, cv_queries, config["budgets"][0])
    np.savez(directory / name, **schedule)
    files = d.pin([directory / name])
    d.fixed(
        directory / "design-freeze.json",
        {
            "config": options(config, source),
            "root_design_sha256": d.sha(run / "design-freeze.json"),
            "files": files,
        },
    )
    d.fixed(
        directory / "schedule-freeze.json",
        {
            "files": files,
            "changed_cv_negatives": int(
                (schedule["negative_claims"] != original["negative_claims"]).sum()
            ),
            "restored_noncv_negatives": int(
                (schedule["negative_claims"] != all_hard["negative_claims"]).sum()
            ),
        },
    )


def worker(config, run, seed):
    s.torch.set_num_threads(1)
    d.check_files(d.read(run / "design-freeze.json")["files"])
    prepare(config, run, seed)
    b.worker(
        options(config, sources(config)[seed]),
        run / f"acoustic-{seed}",
        config["fusion_seed"],
    )


def model_paths(config, run):
    result = {}
    for seed, source in sources(config).items():
        for arm, directory in (
            ("random", d.ROOT / source["budget"]),
            ("mixed", d.ROOT / config["all_hard_run"] / f"acoustic-{seed}"),
            ("cv_only", run / f"acoustic-{seed}"),
        ):
            for updates in config["budgets"][1:]:
                for mode in ("selected", "final"):
                    label = f"{arm}-seed{seed}-u{updates}-{mode}"
                    result[label] = (
                        seed,
                        directory
                        / f"seed-{config['fusion_seed']}/budget-{updates}/{mode}.pt",
                    )
    return result


def evaluate(config, run):
    paths = model_paths(config, run)
    d.fixed(
        run / "all-models-freeze.json", {"files": d.pin([p for _, p in paths.values()])}
    )
    selection = {}
    score_files = []
    old = d.read(d.ROOT / config["all_hard_run"] / "results.json")
    results, maxdiff, audited_rows = {}, 0.0, 0
    spec = importlib.util.spec_from_file_location(
        "hard_negative_independent_audit",
        BASE.parent / "phoneme-corpus-transformer/audit.py",
    )
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    for split in ("validation", "test"):
        for seed, source in sources(config).items():
            fusion = d.ROOT / source["fusion"]
            original = d.read(fusion / "design-freeze.json")["config"]
            models = {}
            for label, (owner, path) in paths.items():
                if owner != seed:
                    continue
                model = e.create_model(original, "transformer")
                model.load_state_dict(
                    s.torch.load(path, weights_only=True)["state_dict"]
                )
                models[label] = model.eval()
            for corpus in s.CORPORA:
                d.state(
                    run / "state.json",
                    "evaluating",
                    split=split,
                    seed=seed,
                    corpus=corpus,
                )
                arrays = e.load_arrays(fusion / f"{corpus}-{split}.npz")
                baseline = d.read(fusion / f"{corpus}-{split}-baseline.json")
                data = d.read(fusion / f"{corpus}-{split}-inputs.json")
                for label, model in models.items():
                    rows = e.score(model, arrays, baseline)
                    path = run / "scores" / f"{corpus}-{label}-{split}.json"
                    d.fixed(path, rows)
                    score_files.append(path)
                    if label.startswith(("random-", "mixed-")):
                        prior = d.read(
                            d.ROOT
                            / config["all_hard_run"]
                            / "scores"
                            / f"{corpus}-{label}-{split}.json"
                        )
                        assert rows == prior
                    if split == "validation":
                        active = [
                            v
                            for v in rows
                            if v["role"] == "verification" and v["score"] is not None
                        ]
                        threshold, _ = s.far_target_threshold(
                            [v["score"] for v in active],
                            [v["is_genuine"] for v in active],
                            0.01,
                        )
                        selection.setdefault(corpus, {})[label] = s.json_threshold(
                            threshold
                        )
                    else:
                        metrics = {
                            role: s.operating(
                                [v for v in rows if v["role"] == role],
                                selection[corpus][label],
                            )
                            for role in sorted({v["role"] for v in rows})
                        }
                        results.setdefault(corpus, {})[label] = metrics
                        if label.startswith(("random-", "mixed-")):
                            assert metrics == old["conditions"][corpus][label]
                    maxdiff = max(
                        maxdiff,
                        independent.independently_score(model, arrays, data, rows),
                    )
                    audited_rows += len(rows)
        if split == "validation":
            d.fixed(
                run / "selection-freeze.json",
                {
                    "thresholds": selection,
                    "models": d.read(run / "all-models-freeze.json")["files"],
                    "validation_files": d.pin(score_files),
                },
            )
    d.fixed(
        run / "results.json",
        {
            "config": config,
            "conditions": results,
            "selection_sha256": d.sha(run / "selection-freeze.json"),
        },
    )
    d.fixed(
        run / "score-audit.json",
        {
            "records": audited_rows,
            "max_difference": maxdiff,
            "files": d.pin(
                score_files + [run / "results.json", run / "selection-freeze.json"]
            ),
        },
    )


def execute(config, run):
    freeze(config, run)
    processes = []
    try:
        for seed in config["encoder_seeds"]:
            log = (run / f"acoustic-{seed}.log").open("a")
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__)), "worker", "--seed", str(seed)],
                cwd=d.ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            processes.append((seed, child, log))
        while processes:
            for item in processes[:]:
                seed, child, log = item
                if child.poll() is not None:
                    log.close()
                    processes.remove(item)
                    assert child.returncode == 0, (
                        f"worker {seed} failed: {child.returncode}"
                    )
            d.state(
                run / "state.json",
                "training",
                active=[seed for seed, _, _ in processes],
            )
            if processes:
                time.sleep(5)
        evaluate(config, run)
        subprocess.run([sys.executable, str(BASE / "audit.py")], check=True, cwd=d.ROOT)
        subprocess.run(
            [sys.executable, str(BASE / "report.py")], check=True, cwd=d.ROOT
        )
        d.state(run / "state.json", "complete")
    except BaseException as error:
        for _, child, log in processes:
            child.terminate()
            child.wait()
            log.close()
        d.state(run / "state.json", "failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "worker"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = d.read(CONFIG)
    run = d.ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    if args.command == "execute":
        execute(config, run)
    else:
        worker(config, run, args.seed)
