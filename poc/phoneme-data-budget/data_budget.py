"""Paired data-size, acoustic-budget and fusion-budget replication."""

import argparse
import fcntl
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-data-scaling"))
import scaling_study as previous
from expansion_common import atomic, check_files, fixed, pin, read, sha

e, s, budget = previous.e, previous.s, previous.budget

import encoder_continuation as acoustic

CONFIG = BASE / "config.json"


def jobs(config):
    return [(arm, seed) for seed in config["encoder_seeds"] for arm in config["arms"]]


def name(arm, seed):
    return f"{arm}-seed{seed}"


def state(path, status, **details):
    atomic(
        path,
        {
            "status": status,
            "pid": os.getpid(),
            "updated_at_epoch": time.time(),
            **details,
        },
    )


def freeze(config, run):
    oldrun = ROOT / config["data_run"]
    old = previous.verify(oldrun)
    check_files(read(oldrun / "complete-audit.json")["files"])
    files = dict(old["files"])
    for arm in config["arms"]:
        files.update(read(oldrun / arm / "design-freeze.json")["files"])
    prior = previous.model_sources(
        read(oldrun / "design-freeze.json")["config"], oldrun
    )
    for arm in config["arms"]:
        paths = prior[arm]
        files.update(read(paths["fusion"] / "prepared-freeze.json")["files"])
        files.update(
            pin(
                [
                    paths["fusion"] / "design-freeze.json",
                    paths["trial"] / "last.pt",
                    paths["trial"] / "training-freeze.json",
                    paths["trial"] / "training-summary.json",
                    paths["trial"] / "encoder.pt",
                ]
            )
        )
        files.update(
            pin(
                paths["fusion"] / f"transformer-{config['fusion_seed']}" / n
                for n in (
                    "last.pt",
                    "fusion.pt",
                    "history.json",
                    "training-summary.json",
                )
            )
        )
        files.update(
            read(paths["budget"] / f"seed-{config['fusion_seed']}/complete.json")[
                "files"
            ]
        )
    files.update(
        pin(
            [
                CONFIG,
                *BASE.glob("*.py"),
                *(BASE / "tests").glob("*.py"),
                oldrun / "complete-audit.json",
                ROOT / "poc/phoneme-corpus-transformer/audit.py",
                ROOT / config["reference_budget"] / "design-freeze.json",
            ]
        )
    )
    check_files(files)
    fixed(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "runtime": s.expanded.runtime(),
            "phones": old["phones"],
            "reference_design": old["reference_design"],
            "prior_sources": {
                a: {k: str(p.relative_to(ROOT)) for k, p in prior[a].items()}
                for a in config["arms"]
            },
        },
    )
    return read(run / "design-freeze.json")


def verify(run):
    frozen = read(run / "design-freeze.json")
    assert frozen["runtime"] == s.expanded.runtime()
    check_files(frozen["files"])
    return frozen


def link(source, destination):
    if destination.exists():
        assert destination.resolve() == source.resolve()
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def setup_family(config, run, frozen, arm, seed):
    directory = run / name(arm, seed)
    directory.mkdir(exist_ok=True)
    source = ROOT / config["data_run"] / arm
    for filename in (
        "train-segments.jsonl",
        "train-features.f32",
        "preparation-report.json",
    ):
        link(source / filename, directory / filename)
    training = {
        **frozen["reference_design"]["config"],
        "training_seed": seed,
        "run_directory": str(directory.relative_to(ROOT)),
    }
    assert training["maximum_updates"] == config["encoder_budgets"][0]
    fixed(
        directory / "design-freeze.json",
        {
            "config": training,
            "root_design_sha256": sha(run / "design-freeze.json"),
            "files": pin(
                [
                    directory / n
                    for n in (
                        "train-segments.jsonl",
                        "train-features.f32",
                        "preparation-report.json",
                    )
                ]
            ),
        },
    )
    return directory, training


def prepare_fusion(config, frozen, family, trial, fusion, arm):
    fusion.mkdir(exist_ok=True)
    original_path = ROOT / frozen["prior_sources"][arm]["fusion"]
    original = read(original_path / "design-freeze.json")["config"]
    options = {
        **original,
        "seeds": [config["fusion_seed"]],
        "run_directory": str(fusion.relative_to(ROOT)),
        "architectures": ["transformer"],
        "encoder_sha256": sha(trial / "encoder.pt"),
    }
    fixed(
        fusion / "design-freeze.json",
        {
            "config": options,
            "runtime": s.expanded.runtime(),
            "files": pin(
                [
                    family / "design-freeze.json",
                    trial / "encoder.pt",
                    trial / "training-summary.json",
                ]
            ),
            "reference_design": frozen["reference_design"],
            "phones": frozen["phones"],
        },
    )
    if (fusion / "prepared-freeze.json").exists():
        check_files(read(fusion / "prepared-freeze.json")["files"])
        return options
    labels, hashes = set(), set()
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            data, _, manifest = e.evaluation_source(
                config, frozen["reference_design"], corpus, split
            )
            labels.update(data["speakers"])
            hashes.update(v["source_sha256"] for v in s.rows(manifest))
    train, metadata = e.training_inputs(
        options, family / "train-segments.jsonl", frozen["phones"], labels, hashes
    )
    assert train == read(original_path / "train-inputs.json"), (
        "replication changed fusion training inputs"
    )
    fixed(fusion / "train-inputs.json", train)
    features = s.np.memmap(
        family / "train-features.f32",
        dtype="<f4",
        mode="r",
        shape=(train["feature_count"], 128),
    )
    model, _ = s.load_model(trial)
    vectors = s.model_embeddings(model, features)
    s.np.savez(
        fusion / "train.npz", **e.aggregates(train, vectors, metadata, frozen["phones"])
    )
    seed = config["fusion_seed"]
    s.np.savez(fusion / f"schedule-{seed}.npz", **e.make_schedule(options, train, seed))
    for corpus in s.CORPORA:
        previous.eval_arrays(
            config, frozen, model, frozen["phones"], corpus, "validation", fusion
        )
    fixed(
        fusion / "prepared-freeze.json",
        {
            "files": pin(
                [
                    fusion / "train-inputs.json",
                    fusion / "train.npz",
                    fusion / f"schedule-{seed}.npz",
                    *fusion.glob("*-validation*"),
                ]
            )
        },
    )
    return options


def continue_fusion(config, fusion, target):
    target.mkdir(exist_ok=True)
    seed = config["fusion_seed"]
    original = read(fusion / "design-freeze.json")["config"]
    prior = read(ROOT / config["reference_budget"] / "design-freeze.json")["config"]
    options = {
        **prior,
        "reference_run": str(fusion.relative_to(ROOT)),
        "run_directory": str(target.relative_to(ROOT)),
        "seeds": [seed],
        "budgets": [4000, *config["fusion_budgets"]],
        "parallel_workers": 1,
    }
    fixed(
        target / "design-freeze.json",
        {
            "config": options,
            "runtime": s.expanded.runtime(),
            "files": {
                **read(fusion / "prepared-freeze.json")["files"],
                **pin(
                    [
                        fusion / "design-freeze.json",
                        *[
                            fusion / f"transformer-{seed}" / n
                            for n in (
                                "last.pt",
                                "fusion.pt",
                                "history.json",
                                "training-summary.json",
                            )
                        ],
                    ]
                ),
            },
        },
    )
    if not (target / "schedule-freeze.json").exists():
        s.np.savez(
            target / f"schedule-{seed}.npz",
            **budget.build_schedule(
                options, original, read(fusion / "train-inputs.json"), seed
            ),
        )
        fixed(
            target / "schedule-freeze.json",
            {"files": pin([target / f"schedule-{seed}.npz"])},
        )
    check_files(read(target / "schedule-freeze.json")["files"])
    budget.worker(options, target, seed)


def worker(config, run, arm, seed):
    frozen = verify(run)
    family, training = setup_family(config, run, frozen, arm, seed)
    notify = lambda status, **details: state(family / "state.json", status, **details)
    with (family / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            first, last = config["encoder_budgets"]
            for count in (first, last):
                (family / f"encoder-{count}").mkdir(exist_ok=True)
            short = family / f"encoder-{first}/trial"
            long = family / f"encoder-{last}/trial"
            oldseed = frozen["reference_design"]["config"]["training_seed"]
            notify("preparing_encoder_pool")
            pool = s.TrainingPool(training, family)
            if seed == oldseed:
                link(ROOT / frozen["prior_sources"][arm]["trial"], short)
                s.load_model(short)
            else:
                notify("training_encoder", budget=first)
                s.train_trial(training, family, short, frozen["phones"], pool)
            notify("continuing_encoder", update=first)
            acoustic.train(
                config,
                short,
                long,
                pool,
                acoustic.s.ordered(frozen["phones"]),
                seed,
                notify,
            )
            del pool
            gc.collect()
            for count in (first, last):
                directory = family / f"encoder-{count}"
                trial = directory / "trial"
                if count == first and seed == oldseed:
                    fusion = ROOT / frozen["prior_sources"][arm]["fusion"]
                else:
                    fusion = directory / "fusion"
                    notify("preparing_fusion", encoder_updates=count)
                    options = prepare_fusion(config, frozen, family, trial, fusion, arm)
                    e.train(
                        options,
                        fusion,
                        lambda status, encoder_updates=count, **details: notify(
                            status, encoder_updates=encoder_updates, **details
                        ),
                    )
                if count == first and seed == oldseed and arm == "baseline280":
                    continued = ROOT / config["reference_budget"]
                    check_files(
                        read(continued / f"seed-{config['fusion_seed']}/complete.json")[
                            "files"
                        ]
                    )
                else:
                    continued = directory / "continued"
                    notify("continuing_fusion", encoder_updates=count)
                    continue_fusion(config, fusion, continued)
                fixed(
                    directory / "paths.json",
                    {
                        "trial": str(trial.relative_to(ROOT)),
                        "fusion": str(fusion.relative_to(ROOT)),
                        "budget": str(continued.relative_to(ROOT)),
                        "arm": arm,
                        "encoder_seed": seed,
                        "encoder_updates": count,
                    },
                )
            notify("complete")
        except BaseException as error:
            notify("failed", error=str(error))
            raise


def sources(config, run):
    return {
        f"{name(arm, seed)}-e{count}": read(
            run / name(arm, seed) / f"encoder-{count}/paths.json"
        )
        for arm, seed in jobs(config)
        for count in config["encoder_budgets"]
    }


def model_paths(config, paths):
    return {
        f"u{updates}-{mode}": ROOT
        / paths["budget"]
        / f"seed-{config['fusion_seed']}/budget-{updates}/{mode}.pt"
        for updates in config["fusion_budgets"]
        for mode in ("selected", "final")
    }


def models(config, paths):
    original = read(ROOT / paths["fusion"] / "design-freeze.json")["config"]
    values = {}
    for key, path in model_paths(config, paths).items():
        model = e.create_model(original, "transformer")
        model.load_state_dict(s.torch.load(path, weights_only=True)["state_dict"])
        values[key] = model.eval()
    return values


def evaluate(config, run):
    frozen = verify(run)
    all_sources = sources(config, run)
    fixed(
        run / "all-models-freeze.json",
        {
            "files": pin(
                [
                    p
                    for paths in all_sources.values()
                    for p in [
                        ROOT / paths["trial"] / "encoder.pt",
                        *model_paths(config, paths).values(),
                    ]
                ]
            )
        },
    )
    thresholds, validation_files = {}, []
    for key, paths in all_sources.items():
        state(run / "state.json", "calibrating", condition=key)
        fusion = ROOT / paths["fusion"]
        for corpus in s.CORPORA:
            arrays = e.load_arrays(fusion / f"{corpus}-validation.npz")
            baseline = read(fusion / f"{corpus}-validation-baseline.json")
            for variant, model in {"equal": None, **models(config, paths)}.items():
                records = (
                    baseline if model is None else e.score(model, arrays, baseline)
                )
                normal = [
                    v
                    for v in records
                    if v["role"] == "verification" and v["score"] is not None
                ]
                threshold, _ = s.far_target_threshold(
                    [v["score"] for v in normal],
                    [v["is_genuine"] for v in normal],
                    0.01,
                )
                label = f"{key}-{variant}"
                thresholds.setdefault(corpus, {})[label] = s.json_threshold(threshold)
                path = run / "scores" / f"{corpus}-{label}-validation.json"
                fixed(path, records)
                validation_files.append(path)
    fixed(
        run / "selection-freeze.json",
        {
            "models": read(run / "all-models-freeze.json")["files"],
            "thresholds": thresholds,
            "validation_files": pin(validation_files),
        },
    )
    results = {}
    for key, paths in all_sources.items():
        state(run / "state.json", "testing", condition=key)
        fusion = ROOT / paths["fusion"]
        encoder, _ = s.load_model(ROOT / paths["trial"])
        for corpus in s.CORPORA:
            if (
                paths["encoder_seed"] == config["encoder_seeds"][0]
                and paths["encoder_updates"] == config["encoder_budgets"][0]
            ):
                assert (fusion / f"{corpus}-test.npz").exists()
            else:
                previous.eval_arrays(
                    config, frozen, encoder, frozen["phones"], corpus, "test", fusion
                )
            arrays = e.load_arrays(fusion / f"{corpus}-test.npz")
            baseline = read(fusion / f"{corpus}-test-baseline.json")
            for variant, model in {"equal": None, **models(config, paths)}.items():
                records = (
                    baseline if model is None else e.score(model, arrays, baseline)
                )
                label = f"{key}-{variant}"
                fixed(run / "scores" / f"{corpus}-{label}-test.json", records)
                results.setdefault(corpus, {})[label] = {
                    role: s.operating(
                        [v for v in records if v["role"] == role],
                        thresholds[corpus][label],
                    )
                    for role in sorted({v["role"] for v in records})
                }
    fixed(
        run / "results.json",
        {
            "conditions": results,
            "config": config,
            "sources": all_sources,
            "selection_sha256": sha(run / "selection-freeze.json"),
        },
    )
    state(run / "state.json", "evaluated")


def execute(config, run):
    freeze(config, run)
    pending = list(jobs(config))
    active = []
    try:
        while pending or active:
            while pending and len(active) < config["parallel_workers"]:
                arm, seed = pending.pop(0)
                log = (run / f"{name(arm, seed)}.log").open("a")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__)),
                        "worker",
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                active.append((arm, seed, child, log))
            for item in active[:]:
                arm, seed, child, log = item
                code = child.poll()
                if code is not None:
                    log.close()
                    active.remove(item)
                    assert code == 0, f"worker {arm}/{seed} exited {code}"
            state(
                run / "state.json",
                "training",
                active=[name(a, seed) for a, seed, _, _ in active],
                pending=[name(a, seed) for a, seed in pending],
            )
            if active:
                time.sleep(5)
        evaluate(config, run)
        state(run / "state.json", "auditing")
        subprocess.run([sys.executable, str(BASE / "audit.py")], cwd=ROOT, check=True)
        subprocess.run(
            [sys.executable, str(BASE / "report.py")],
            cwd=ROOT,
            check=True,
            env={**os.environ, "MPLCONFIGDIR": "/tmp/phonia-data-budget-mpl"},
        )
        state(run / "state.json", "complete")
    except BaseException as error:
        for _, _, child, _ in active:
            if child.poll() is None:
                child.terminate()
        for _, _, child, log in active:
            child.wait()
            log.close()
        state(run / "state.json", "failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["execute", "worker", "evaluate"])
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        worker(config, run, args.arm, args.seed)
    elif args.phase == "evaluate":
        evaluate(config, run)
    else:
        with (run / "supervisor.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(config, run)


if __name__ == "__main__":
    main()
