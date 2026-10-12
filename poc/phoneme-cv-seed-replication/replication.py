"""Cross fixed acoustic encoders with new Transformer initialization seeds."""

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-cv-hard-negatives"))
import cv_only as h

d, s, e, b = h.d, h.s, h.e, h.b
CONFIG = BASE / "config.json"


def prior(config):
    return d.read(d.ROOT / config["reference_run"] / "design-freeze.json")["config"]


def sources(config):
    return h.sources(prior(config))


def prefix_path(run, acoustic, fusion):
    return run / f"acoustic-{acoustic}" / f"prefix-{fusion}"


def arm_path(run, acoustic, fusion, arm):
    return run / f"acoustic-{acoustic}" / f"fusion-{fusion}" / arm


def schedule_path(config, acoustic, arm):
    old = prior(config)
    root = (
        d.ROOT / sources(config)[acoustic]["budget"]
        if arm == "random"
        else d.ROOT / config["reference_run"] / f"acoustic-{acoustic}"
    )
    return root / f"schedule-{old['fusion_seed']}.npz"


def link(source, destination):
    if destination.exists():
        assert destination.is_symlink() and destination.resolve() == source.resolve()
    else:
        destination.symlink_to(source.resolve())


def freeze(config, run):
    run.mkdir(parents=True, exist_ok=True)
    ref = d.ROOT / config["reference_run"]
    previous = d.read(ref / "design-freeze.json")
    assert d.read(ref / "state.json")["status"] == "complete"
    assert config["encoder_seeds"] == previous["config"]["encoder_seeds"]
    assert config["reference_fusion_seed"] == previous["config"]["fusion_seed"]
    assert len(set(config["new_fusion_seeds"])) == len(config["new_fusion_seeds"])
    assert config["reference_fusion_seed"] not in config["new_fusion_seeds"]
    for key in ("budgets", "validation_every", "continuation_learning_rate"):
        assert config[key] == previous["config"][key]
    files = {**previous["files"], **d.read(ref / "complete-audit.json")["files"]}
    files.update(
        d.pin(
            [
                ref / "complete-audit.json",
                ref / "results.json",
                CONFIG,
                *BASE.glob("*.py"),
                *(BASE / "tests").glob("*.py"),
            ]
        )
    )
    d.check_files(files)
    d.fixed(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "runtime": s.expanded.runtime(),
            "all_test_previously_observed": True,
        },
    )


def prepare_prefix(config, run, acoustic, fusion):
    path = prefix_path(run, acoustic, fusion)
    path.mkdir(parents=True, exist_ok=True)
    source = d.ROOT / sources(config)[acoustic]["fusion"]
    original = d.read(source / "design-freeze.json")["config"]
    options = {**original, "seeds": [fusion], "architectures": ["transformer"]}
    assert options["updates"] == config["budgets"][0]
    d.fixed(
        path / "design-freeze.json",
        {"config": options, "root_design_sha256": d.sha(run / "design-freeze.json")},
    )
    names = ["train.npz", "train-inputs.json"]
    names += [
        f"{c}-validation{suffix}"
        for c in s.CORPORA
        for suffix in (".npz", "-baseline.json")
    ]
    for name in names:
        link(source / name, path / name)
    with s.np.load(schedule_path(config, acoustic, "random"), allow_pickle=False) as a:
        target = path / f"schedule-{fusion}.npz"
        expected = {k: a[k][: options["updates"]].copy() for k in a.files}
        if target.exists():
            with s.np.load(target, allow_pickle=False) as existing:
                for k, value in expected.items():
                    s.np.testing.assert_array_equal(existing[k], value)
        else:
            s.np.savez(target, **expected)
    d.fixed(
        path / "prepared-freeze.json",
        {"files": d.pin([*(path / n for n in names), target])},
    )
    return options, path


def prefix_worker(config, run, acoustic, fusion):
    options, path = prepare_prefix(config, run, acoustic, fusion)
    complete = path / "complete.json"
    if complete.exists():
        d.check_files(d.read(complete)["files"])
        return
    trained = path / f"transformer-{fusion}"
    # The reused short-prefix trainer does not atomically bundle its best model.
    # Never resume a partially written prefix; preserve it for diagnosis instead.
    if (trained / "last.pt").exists() and not (
        trained / "training-summary.json"
    ).exists():
        raise RuntimeError(
            "Incomplete prefix: preserve this run and start a new frozen run"
        )
    e.train(
        options,
        path,
        lambda status, **details: d.state(path / "state.json", status, **details),
    )
    d.fixed(
        complete,
        {
            "files": d.pin(
                [
                    path / "design-freeze.json",
                    path / "prepared-freeze.json",
                    *sorted(trained.glob("*.json")),
                    *sorted(trained.glob("*.pt")),
                ]
            )
        },
    )
    d.state(path / "state.json", "complete", update=options["updates"])


def continuation_options(config, run, acoustic, fusion):
    return {
        **config,
        "reference_run": str(prefix_path(run, acoustic, fusion).relative_to(d.ROOT)),
    }


def arm_worker(config, run, acoustic, fusion, arm):
    prefix = prefix_path(run, acoustic, fusion)
    d.check_files(d.read(prefix / "complete.json")["files"])
    path = arm_path(run, acoustic, fusion, arm)
    path.mkdir(parents=True, exist_ok=True)
    options = continuation_options(config, run, acoustic, fusion)
    d.fixed(
        path / "design-freeze.json",
        {
            "config": options,
            "arm": arm,
            "root_design_sha256": d.sha(run / "design-freeze.json"),
            "prefix_complete_sha256": d.sha(prefix / "complete.json"),
        },
    )
    link(schedule_path(config, acoustic, arm), path / f"schedule-{fusion}.npz")
    b.worker(options, path, fusion)


def model_paths(config, run):
    items = {}
    old = d.ROOT / config["reference_run"]
    for acoustic, source in sources(config).items():
        for fusion in [config["reference_fusion_seed"], *config["new_fusion_seeds"]]:
            for arm in config["arms"]:
                if fusion == config["reference_fusion_seed"]:
                    root = (
                        d.ROOT / source["budget"]
                        if arm == "random"
                        else old / f"acoustic-{acoustic}"
                    )
                else:
                    root = arm_path(run, acoustic, fusion, arm)
                for budget in config["budgets"][1:]:
                    for mode in ("selected", "final"):
                        label = f"{arm}-seed{acoustic}-fusion{fusion}-u{budget}-{mode}"
                        items[label] = {
                            "acoustic": acoustic,
                            "fusion": fusion,
                            "arm": arm,
                            "budget": budget,
                            "mode": mode,
                            "path": root / f"seed-{fusion}/budget-{budget}/{mode}.pt",
                            "prior_label": f"{arm}-seed{acoustic}-u{budget}-{mode}",
                        }
    return items


def evaluate(config, run):
    items = model_paths(config, run)
    d.fixed(
        run / "all-models-freeze.json",
        {"files": d.pin(v["path"] for v in items.values())},
    )
    ref = d.ROOT / config["reference_run"]
    prior_results = d.read(ref / "results.json")["conditions"]
    spec = importlib.util.spec_from_file_location(
        "independent_score_audit", BASE.parent / "phoneme-corpus-transformer/audit.py"
    )
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    thresholds, results, files = {}, {}, []
    records, reused, maxdiff = 0, 0, 0.0
    for split in ("validation", "test"):
        for acoustic, source in sources(config).items():
            fusion_path = d.ROOT / source["fusion"]
            options = d.read(fusion_path / "design-freeze.json")["config"]
            for corpus in s.CORPORA:
                d.state(
                    run / "state.json",
                    "evaluating",
                    split=split,
                    acoustic=acoustic,
                    corpus=corpus,
                )
                arrays = e.load_arrays(fusion_path / f"{corpus}-{split}.npz")
                baseline = d.read(fusion_path / f"{corpus}-{split}-baseline.json")
                data = d.read(fusion_path / f"{corpus}-{split}-inputs.json")
                for label, item in items.items():
                    if item["acoustic"] != acoustic:
                        continue
                    path = run / "scores" / f"{corpus}-{label}-{split}.json"
                    if item["fusion"] == config["reference_fusion_seed"]:
                        link(
                            ref
                            / "scores"
                            / f"{corpus}-{item['prior_label']}-{split}.json",
                            path,
                        )
                        rows = d.read(path)
                        reused += len(rows)
                    else:
                        model = e.create_model(options, "transformer")
                        model.load_state_dict(
                            s.torch.load(item["path"], weights_only=True)["state_dict"]
                        )
                        rows = e.score(model.eval(), arrays, baseline)
                        d.fixed(path, rows)
                        maxdiff = max(
                            maxdiff,
                            independent.independently_score(model, arrays, data, rows),
                        )
                        records += len(rows)
                    files.append(path)
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
                        thresholds.setdefault(corpus, {})[label] = s.json_threshold(
                            threshold
                        )
                    else:
                        metrics = {
                            role: s.operating(
                                [v for v in rows if v["role"] == role],
                                thresholds[corpus][label],
                            )
                            for role in sorted({v["role"] for v in rows})
                        }
                        results.setdefault(corpus, {})[label] = metrics
                        if item["fusion"] == config["reference_fusion_seed"]:
                            assert metrics == prior_results[corpus][item["prior_label"]]
        if split == "validation":
            d.fixed(
                run / "selection-freeze.json",
                {
                    "thresholds": thresholds,
                    "models": d.read(run / "all-models-freeze.json")["files"],
                    "validation_files": d.pin(files),
                },
            )
    d.fixed(run / "results.json", {"config": config, "conditions": results})
    d.fixed(
        run / "score-audit.json",
        {
            "independently_inferred_records": records,
            "previously_audited_records": reused,
            "max_difference": maxdiff,
            "files": d.pin(
                files + [run / "results.json", run / "selection-freeze.json"]
            ),
        },
    )


def pool(config, run, jobs):
    pending, active = list(jobs), []
    try:
        while pending or active:
            while pending and len(active) < config["parallel_workers"]:
                command = pending.pop(0)
                name = "-".join(command)
                log = (run / f"{name}.log").open("a")
                process = subprocess.Popen(
                    [sys.executable, str(BASE / "replication.py"), *command],
                    cwd=d.ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                active.append((name, process, log))
            for item in active[:]:
                name, process, log = item
                if process.poll() is not None:
                    log.close()
                    active.remove(item)
                    assert process.returncode == 0, (
                        f"{name} failed: {process.returncode}"
                    )
            d.state(
                run / "state.json",
                "training",
                active=[v[0] for v in active],
                pending=len(pending),
            )
            if active:
                time.sleep(5)
    finally:
        for _, process, log in active:
            process.terminate()
            process.wait()
            log.close()


def execute(config, run):
    freeze(config, run)
    (run / "scores").mkdir(exist_ok=True)
    try:
        pairs = [
            (a, f) for a in config["encoder_seeds"] for f in config["new_fusion_seeds"]
        ]
        pool(
            config,
            run,
            [["prefix", "--acoustic", str(a), "--fusion", str(f)] for a, f in pairs],
        )
        pool(
            config,
            run,
            [
                ["arm", "--acoustic", str(a), "--fusion", str(f), "--arm", arm]
                for a, f in pairs
                for arm in config["arms"]
            ],
        )
        evaluate(config, run)
        for script in ("audit.py", "report.py"):
            subprocess.run([sys.executable, str(BASE / script)], cwd=d.ROOT, check=True)
        d.state(run / "state.json", "complete")
    except BaseException as error:
        d.state(run / "state.json", "failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "prefix", "arm"))
    parser.add_argument("--acoustic", type=int)
    parser.add_argument("--fusion", type=int)
    parser.add_argument("--arm", choices=("random", "cv_only"))
    args = parser.parse_args()
    config = d.read(CONFIG)
    run = d.ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    if args.command == "execute":
        execute(config, run)
    elif args.command == "prefix":
        prefix_worker(config, run, args.acoustic, args.fusion)
    else:
        arm_worker(config, run, args.acoustic, args.fusion, args.arm)
