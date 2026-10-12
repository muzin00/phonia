"""Paired hard-negative continuation with immutable inputs and shared prefix."""

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
import mining

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
    prior = d.ROOT / config["reference_run"]
    audit = d.read(prior / "complete-audit.json")
    d.check_files(audit["files"])
    files = dict(d.read(prior / "design-freeze.json")["files"])
    files.update(audit["files"])
    files.update(d.pin([CONFIG, *BASE.glob("*.py"), *(BASE / "tests").glob("*.py")]))
    for source in sources(config).values():
        fusion = d.ROOT / source["fusion"]
        files.update(d.read(fusion / "prepared-freeze.json")["files"])
        files.update(
            d.pin(
                [
                    fusion / "design-freeze.json",
                    *(
                        fusion / f"transformer-{config['fusion_seed']}" / n
                        for n in ("last.pt", "fusion.pt", "history.json")
                    ),
                    d.ROOT / source["budget"] / f"schedule-{config['fusion_seed']}.npz",
                    *(
                        fusion / f"{corpus}-{split}{suffix}"
                        for corpus in s.CORPORA
                        for split in ("validation", "test")
                        for suffix in (".npz", "-inputs.json", "-baseline.json")
                    ),
                ]
            )
        )
    d.check_files(files)
    d.fixed(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "sources": {str(k): v for k, v in sources(config).items()},
            "runtime": s.expanded.runtime(),
        },
    )


def prepare(config, run, seed):
    source = sources(config)[seed]
    directory = run / f"acoustic-{seed}"
    directory.mkdir(exist_ok=True)
    if (directory / "schedule-freeze.json").exists():
        d.check_files(d.read(directory / "schedule-freeze.json")["files"])
        return
    fusion = d.ROOT / source["fusion"]
    data = d.read(fusion / "train-inputs.json")
    evaluation_speakers = set()
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            evaluation_speakers.update(
                d.read(fusion / f"{corpus}-{split}-inputs.json")["speakers"]
            )
    assert not evaluation_speakers & set(data["speakers"])
    assert not set(data["enrollment_wav_hashes"]) & set(data["query_wav_hashes"])
    with np.load(fusion / "train.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    with np.load(
        d.ROOT / source["budget"] / f"schedule-{config['fusion_seed']}.npz",
        allow_pickle=False,
    ) as archive:
        original = {k: archive[k].copy() for k in archive.files}
    table, owner = mining.score_table(arrays, data)
    schedule, candidates = mining.make_schedule(original, table, owner, config)
    np.save(directory / "mining-scores.npy", table)
    np.savez_compressed(directory / "mining-candidates.npz", **candidates)
    np.savez(directory / f"schedule-{config['fusion_seed']}.npz", **schedule)
    paths = [
        directory / "mining-scores.npy",
        directory / "mining-candidates.npz",
        directory / f"schedule-{config['fusion_seed']}.npz",
    ]
    d.fixed(
        directory / "design-freeze.json",
        {
            "config": options(config, source),
            "root_design_sha256": d.sha(run / "design-freeze.json"),
            "files": d.pin(paths),
        },
    )
    d.fixed(
        directory / "schedule-freeze.json",
        {
            "files": d.pin(paths),
            "hard_slots": len(candidates["positions"]),
            "changed_negatives": int(
                (original["negative_claims"] != schedule["negative_claims"]).sum()
            ),
            "train_only": True,
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
            ("mixed", run / f"acoustic-{seed}"),
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
    old = d.read(d.ROOT / config["reference_run"] / "results.json")
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
                    if label.startswith("random-"):
                        variant = label.split(f"seed{seed}-")[1]
                        prior = d.read(
                            d.ROOT
                            / config["reference_run"]
                            / "scores"
                            / f"{corpus}-combined420-seed{seed}-e{config['encoder_updates']}-{variant}-{split}.json"
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
                        if label.startswith("random-"):
                            assert (
                                metrics
                                == old["conditions"][corpus][
                                    f"combined420-seed{seed}-e{config['encoder_updates']}-{variant}"
                                ]
                            )
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
