"""Controlled data additions with matched encoder and Transformer update budgets."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from expansion_common import BASE, ROOT, atomic, check_files, fixed, pin, read, sha

# Keep this directory's selector out of the legacy runtime's short imports.
sys.path = [p for p in sys.path if Path(p).resolve() != BASE]
sys.path.insert(0, str(BASE.parent / "phoneme-transformer-budget"))
import run as budget

e, s = budget.e, budget.s
CONFIG = BASE / "study-config.json"


def freeze(config, run):
    prepared = ROOT / config["preparation_run"]
    reference = ROOT / config["reference_run"]
    audit = read(prepared / "complete-audit.json")
    assert audit["receipts_verified"] == 11200
    check_files(audit["files"])
    original = s.verify(reference)
    s.verify_prepared(reference)
    old = original["config"]
    assert (
        old["training_seed"] == config["encoder_seed"]
        and old["maximum_updates"] == config["encoder_updates"]
    )
    prior_budget = read(ROOT / config["reference_budget"] / "design-freeze.json")
    assert config["fusion_updates"] in prior_budget["config"]["budgets"]
    assert config["fusion_seeds"] == prior_budget["config"]["seeds"]
    check_files(prior_budget["files"])
    files = {
        **original["files"],
        **read(reference / "prepared-inputs-freeze.json")["files"],
        **audit["files"],
    }
    files.update(
        pin(
            [
                CONFIG,
                BASE / "scaling_study.py",
                BASE / "check_feature_runtime.py",
                BASE / "audit_study.py",
                BASE / "expansion_common.py",
                prepared / "complete-audit.json",
                prepared / "preparation-freeze.json",
                reference / "expanded280/design-freeze.json",
                ROOT / "poc/phoneme-corpus-transformer/experiment.py",
                ROOT / "poc/phoneme-corpus-transformer/inputs.py",
                ROOT / "poc/phoneme-transformer-budget/run.py",
                ROOT / "poc/phoneme-corpus-scaling/scaling_learner.py",
            ]
        )
    )
    for p in [
        reference / "expanded280/trial/encoder.pt",
        reference / "expanded280/trial/training-summary.json",
        ROOT / config["reference_fusion"] / "design-freeze.json",
        ROOT / config["reference_budget"] / "design-freeze.json",
    ]:
        files.update(pin([p]))
    for seed in config["fusion_seeds"]:
        directory = ROOT / config["reference_budget"] / f"seed-{seed}"
        check_files(read(directory / "complete.json")["files"])
        files.update(
            pin(
                directory / "budget-16000" / f"{mode}.pt"
                for mode in ("selected", "final")
            )
        )
    check_files(files)
    fixed(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "reference_design": original,
            "phones": original["phonemes"],
            "runtime": s.expanded.runtime(),
            "all_test_previously_observed": True,
        },
    )
    return read(run / "design-freeze.json")


def verify(run):
    frozen = read(run / "design-freeze.json")
    assert frozen["runtime"] == s.expanded.runtime()
    check_files(frozen["files"])
    return frozen


def prepare_arms(config, run, frozen):
    prepared = ROOT / config["preparation_run"]
    reference = ROOT / config["reference_run"] / "expanded280"
    exclusions = read(prepared / "overlap-audit.json")["excluded_input_ids"]
    report = read(prepared / "preparation-report.json")
    for arm, groups in config["arms"].items():
        directory = run / arm
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "design-freeze.json").exists():
            check_files(read(directory / "design-freeze.json")["files"])
            continue
        manifest = directory / "train-segments.jsonl"
        cache = directory / "train-features.f32"
        count = 0
        speakers = set()
        phones = Counter()
        support = Counter()
        sources = Counter()
        blocks = [
            (
                reference / "train-segments.jsonl",
                reference / "train-features.f32",
                read(reference / "preparation-report.json")["train_feature_count"],
                False,
            )
        ]
        blocks += [
            (
                prepared / f"{g}-segments.jsonl",
                prepared / f"{g}-features.f32",
                report["groups"][g]["segments"],
                True,
            )
            for g in groups
        ]
        with manifest.open("w") as output, cache.open("wb") as features:
            for rows_path, features_path, size, addition in blocks:
                matrix = s.np.memmap(
                    features_path, dtype="<f4", mode="r", shape=(size, 128)
                )
                for index, row in enumerate(s.rows(rows_path)):
                    assert (
                        row["cache_index"] == index
                        and row["split"] == "train"
                        and not row["quality_flags"]
                    )
                    if addition and row["input_id"] in exclusions:
                        continue
                    output.write(
                        json.dumps({**row, "cache_index": count}, ensure_ascii=False)
                        + "\n"
                    )
                    features.write(matrix[index].tobytes())
                    count += 1
                    speakers.add(row["speaker_id"])
                    phones[row["phoneme"]] += 1
                    support[row["speaker_id"], row["phoneme"]] += 1
                    sources[row.get("addition_group", "baseline")] += 1
        planned = read(
            ROOT
            / "artifacts/phoneme-data-scaling/candidates420-20261012-v1/selection-freeze.json"
        )["arms"][arm]["speaker_labels"]
        assert speakers <= set(planned)
        if arm == "baseline280":
            assert sha(cache) == sha(reference / "train-features.f32")
            assert sha(manifest) == sha(reference / "train-segments.jsonl")
        assert all(
            any(p == phone and n >= 2 for (_, p), n in support.items())
            for phone in frozen["phones"]
        )
        fixed(
            directory / "preparation-report.json",
            {
                "training_speakers": sorted(speakers),
                "train_feature_count": count,
                "phone_counts": dict(phones),
                "segment_counts_by_addition_group": dict(sources),
                "planned_labels": len(planned),
                "unsupported_labels": sorted(set(planned) - speakers),
            },
        )
        train_config = {
            **frozen["reference_design"]["config"],
            "run_directory": str(directory.relative_to(ROOT)),
        }
        fixed(
            directory / "design-freeze.json",
            {
                "config": train_config,
                "arm": arm,
                "parent_design_sha256": sha(run / "design-freeze.json"),
                "files": pin([manifest, cache, directory / "preparation-report.json"]),
            },
        )
        print({"arm": arm, "speakers": len(speakers), "segments": count}, flush=True)


def eval_arrays(config, frozen, model, phones, corpus, split, target):
    seal = target / f"{corpus}-{split}-freeze.json"
    if seal.exists():
        check_files(read(seal)["files"])
        return
    data, features, manifest = e.evaluation_source(
        config, frozen["reference_design"], corpus, split
    )
    metadata = s.np.array(
        [
            [(v["end_frame"] - v["start_frame"]) / 24000, v["rms_dbfs"]]
            for v in s.rows(manifest)
        ],
        dtype=s.np.float32,
    )
    vectors = s.model_embeddings(model, features)
    arrays = e.aggregates(data, vectors, metadata, phones)
    baseline = s.score(data, vectors, phones)
    components, masks = e.score_inputs(data, arrays, phones, baseline)
    s.np.savez(
        target / f"{corpus}-{split}.npz", **arrays, components=components, masks=masks
    )
    fixed(target / f"{corpus}-{split}-inputs.json", data)
    fixed(target / f"{corpus}-{split}-baseline.json", baseline)
    fixed(
        seal,
        {
            "files": pin(
                [
                    target / f"{corpus}-{split}.npz",
                    target / f"{corpus}-{split}-inputs.json",
                    target / f"{corpus}-{split}-baseline.json",
                ]
            )
        },
    )


def prepare_fusion(config, run, arm, frozen):
    directory = run / arm
    fusion = directory / "fusion"
    fusion.mkdir(exist_ok=True)
    model, _ = s.load_model(directory / "trial")
    original = read(ROOT / config["reference_fusion"] / "design-freeze.json")["config"]
    fusion_config = {
        **original,
        "run_directory": str(fusion.relative_to(ROOT)),
        "architectures": ["transformer"],
        "encoder_sha256": sha(directory / "trial/encoder.pt"),
    }
    assert fusion_config["seeds"] == config["fusion_seeds"]
    files = pin(
        [
            directory / "design-freeze.json",
            directory / "trial/encoder.pt",
            directory / "trial/training-summary.json",
            run / "design-freeze.json",
        ]
    )
    fixed(
        fusion / "design-freeze.json",
        {
            "config": fusion_config,
            "files": files,
            "runtime": s.expanded.runtime(),
            "reference_design": frozen["reference_design"],
            "phones": frozen["phones"],
        },
    )
    if (fusion / "prepared-freeze.json").exists():
        check_files(read(fusion / "prepared-freeze.json")["files"])
        return fusion_config
    labels, hashes = set(), set()
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            data, _, manifest = e.evaluation_source(
                config, frozen["reference_design"], corpus, split
            )
            labels.update(data["speakers"])
            hashes.update(v["source_sha256"] for v in s.rows(manifest))
    train, metadata = e.training_inputs(
        fusion_config,
        directory / "train-segments.jsonl",
        frozen["phones"],
        labels,
        hashes,
    )
    fixed(fusion / "train-inputs.json", train)
    features = s.np.memmap(
        directory / "train-features.f32",
        dtype="<f4",
        mode="r",
        shape=(train["feature_count"], 128),
    )
    vectors = s.model_embeddings(model, features)
    s.np.savez(
        fusion / "train.npz", **e.aggregates(train, vectors, metadata, frozen["phones"])
    )
    for seed in fusion_config["seeds"]:
        s.np.savez(
            fusion / f"schedule-{seed}.npz",
            **e.make_schedule(fusion_config, train, seed),
        )
    for corpus in s.CORPORA:
        eval_arrays(
            config, frozen, model, frozen["phones"], corpus, "validation", fusion
        )
    paths = [
        fusion / "train-inputs.json",
        fusion / "train.npz",
        *fusion.glob("schedule-*.npz"),
        *fusion.glob("*-validation*"),
    ]
    fixed(
        fusion / "prepared-freeze.json",
        {
            "files": pin(paths),
            "eligible_labels": len(train["speakers"]),
            "queries": len(train["queries"]),
            "excluded_labels": train["excluded_labels"],
        },
    )
    return fusion_config


def continuation(config, run, arm):
    fusion = run / arm / "fusion"
    target = run / arm / "continued16k"
    target.mkdir(exist_ok=True)
    original = read(fusion / "design-freeze.json")["config"]
    prior = read(ROOT / config["reference_budget"] / "design-freeze.json")["config"]
    options = {
        **prior,
        "run_directory": str(target.relative_to(ROOT)),
        "reference_run": str(fusion.relative_to(ROOT)),
        "budgets": [4000, config["fusion_updates"]],
    }
    files = {
        **read(fusion / "prepared-freeze.json")["files"],
        **pin([fusion / "design-freeze.json"]),
    }
    for seed in config["fusion_seeds"]:
        files.update(
            pin(
                fusion / f"transformer-{seed}" / n
                for n in (
                    "fusion.pt",
                    "last.pt",
                    "history.json",
                    "training-summary.json",
                )
            )
        )
    fixed(
        target / "design-freeze.json",
        {"config": options, "files": files, "runtime": s.expanded.runtime()},
    )
    if not (target / "schedule-freeze.json").exists():
        train = read(fusion / "train-inputs.json")
        for seed in config["fusion_seeds"]:
            s.np.savez(
                target / f"schedule-{seed}.npz",
                **budget.build_schedule(options, original, train, seed),
            )
        fixed(
            target / "schedule-freeze.json",
            {
                "files": pin(
                    target / f"schedule-{seed}.npz" for seed in config["fusion_seeds"]
                )
            },
        )
    check_files(read(target / "schedule-freeze.json")["files"])
    for seed in config["fusion_seeds"]:
        budget.worker(options, target, seed)


def worker(config, run, arm):
    frozen = verify(run)
    directory = run / arm
    check_files(read(directory / "design-freeze.json")["files"])
    state = lambda status, **details: atomic(
        directory / "state.json",
        {
            "status": status,
            "pid": os.getpid(),
            "updated_at_epoch": time.time(),
            **details,
        },
    )
    try:
        state("training_encoder")
        train_config = read(directory / "design-freeze.json")["config"]
        pool = s.TrainingPool(train_config, directory)
        s.train_trial(
            train_config, directory, directory / "trial", frozen["phones"], pool
        )
        del pool
        summary = read(directory / "trial/training-summary.json")
        baseline = read(
            ROOT / config["reference_run"] / "expanded280/trial/training-summary.json"
        )
        assert summary["initial_encoder_sha256"] == baseline["initial_encoder_sha256"]
        assert summary["completed_updates"] == config["encoder_updates"]
        state("preparing_fusion")
        fusion_config = prepare_fusion(config, run, arm, frozen)
        e.train(fusion_config, directory / "fusion", state)
        state("continuing_fusion_16k")
        continuation(config, run, arm)
        state("complete")
    except BaseException as error:
        state("failed", error=str(error))
        raise


def model_sources(config, run):
    sources = {}
    for arm in config["arms"]:
        baseline = arm == "baseline280"
        sources[arm] = {
            "fusion": ROOT / config["reference_fusion"]
            if baseline
            else run / arm / "fusion",
            "trial": ROOT / config["reference_run"] / "expanded280/trial"
            if baseline
            else run / arm / "trial",
            "budget": ROOT / config["reference_budget"]
            if baseline
            else run / arm / "continued16k",
        }
    return sources


def evaluate(config, run):
    frozen = verify(run)
    sources = model_sources(config, run)
    models = {}
    for arm, paths in sources.items():
        for seed in config["fusion_seeds"]:
            for mode in ("selected", "final"):
                models[f"{arm}-{mode}-{seed}"] = (
                    paths["budget"] / f"seed-{seed}/budget-16000/{mode}.pt"
                )
    fixed(
        run / "all-models-freeze.json",
        {
            "files": pin(
                [
                    *models.values(),
                    *(p["trial"] / "encoder.pt" for p in sources.values()),
                ]
            ),
            "all_conditions_fixed_before_new_test_scoring": True,
        },
    )
    thresholds = {}
    validation = {}
    for arm, paths in sources.items():
        original = read(paths["fusion"] / "design-freeze.json")["config"]
        for corpus in s.CORPORA:
            arrays = e.load_arrays(paths["fusion"] / f"{corpus}-validation.npz")
            baseline = read(paths["fusion"] / f"{corpus}-validation-baseline.json")
            for variant in [
                "equal",
                *(
                    f"{mode}-{seed}"
                    for mode in ("selected", "final")
                    for seed in config["fusion_seeds"]
                ),
            ]:
                key = f"{arm}-{variant}"
                if variant == "equal":
                    records = baseline
                else:
                    model = e.create_model(original, "transformer")
                    model.load_state_dict(
                        s.torch.load(models[key], weights_only=True)["state_dict"]
                    )
                    model.eval()
                    records = e.score(model, arrays, baseline)
                normal = [
                    v
                    for v in records
                    if v["role"] == "verification" and v["score"] is not None
                ]
                point, _ = s.far_target_threshold(
                    [v["score"] for v in normal],
                    [v["is_genuine"] for v in normal],
                    0.01,
                )
                thresholds.setdefault(corpus, {})[key] = s.json_threshold(point)
                path = run / "scores" / f"{corpus}-{key}-validation.json"
                fixed(path, records)
                validation[str(path.relative_to(ROOT))] = sha(path)
    fixed(
        run / "selection-freeze.json",
        {
            "models": read(run / "all-models-freeze.json")["files"],
            "thresholds": thresholds,
            "validation_files": validation,
        },
    )
    results = {}
    for arm, paths in sources.items():
        original = read(paths["fusion"] / "design-freeze.json")["config"]
        model, _ = s.load_model(paths["trial"])
        for corpus in s.CORPORA:
            if arm != "baseline280":
                eval_arrays(
                    config,
                    frozen,
                    model,
                    frozen["phones"],
                    corpus,
                    "test",
                    paths["fusion"],
                )
            arrays = e.load_arrays(paths["fusion"] / f"{corpus}-test.npz")
            baseline = read(paths["fusion"] / f"{corpus}-test-baseline.json")
            for variant in [
                "equal",
                *(
                    f"{mode}-{seed}"
                    for mode in ("selected", "final")
                    for seed in config["fusion_seeds"]
                ),
            ]:
                key = f"{arm}-{variant}"
                if variant == "equal":
                    records = baseline
                else:
                    fusion_model = e.create_model(original, "transformer")
                    fusion_model.load_state_dict(
                        s.torch.load(models[key], weights_only=True)["state_dict"]
                    )
                    fusion_model.eval()
                    records = e.score(fusion_model, arrays, baseline)
                fixed(run / "scores" / f"{corpus}-{key}-test.json", records)
                results.setdefault(corpus, {})[key] = {
                    role: s.operating(
                        [v for v in records if v["role"] == role],
                        thresholds[corpus][key],
                    )
                    for role in sorted({v["role"] for v in records})
                }
    fixed(
        run / "results.json",
        {
            "conditions": results,
            "config": config,
            "selection_sha256": sha(run / "selection-freeze.json"),
        },
    )
    lines = [
        "# 話者・発話数の拡張比較",
        "",
        "同じ更新数・音響encoder 1 seed、Transformer 3 seed。EERは採点可能分。全test既観測。",
        "",
        "| corpus | condition | EER | FAR | FRR |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus, values in results.items():
        for arm in config["arms"]:
            for mode in ("equal", "selected", "final"):
                keys = (
                    [f"{arm}-equal"]
                    if mode == "equal"
                    else [f"{arm}-{mode}-{seed}" for seed in config["fusion_seeds"]]
                )
                metrics = [
                    100 * s.np.mean([values[k]["verification"][m] for k in keys])
                    for m in ("eer", "far", "frr")
                ]
                lines.append(
                    f"| {corpus} | {arm}/{mode} | "
                    + " | ".join(f"{v:.3f}%" for v in metrics)
                    + " |"
                )
    (run / "results.md").write_text("\n".join(lines) + "\n")


def execute(config, run):
    frozen = freeze(config, run)
    subprocess.run(
        [sys.executable, str(BASE / "check_feature_runtime.py")], cwd=ROOT, check=True
    )
    prepare_arms(config, run, frozen)
    children = []
    streams = []
    try:
        for arm in config["arms"]:
            if arm == "baseline280":
                continue
            stream = (run / (arm + ".log")).open("a")
            streams.append(stream)
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(BASE / "scaling_study.py"),
                    "worker",
                    "--arm",
                    arm,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            children.append((arm, child))
        while any(p.poll() is None for _, p in children):
            failed = [
                (arm, p.returncode) for arm, p in children if p.poll() not in (None, 0)
            ]
            if failed:
                raise RuntimeError(str(failed))
            time.sleep(5)
        assert all(p.returncode == 0 for _, p in children)
        evaluate(config, run)
        atomic(
            run / "state.json", {"status": "complete", "finished_at_epoch": time.time()}
        )
    except BaseException as error:
        for _, p in children:
            if p.poll() is None:
                p.terminate()
        for _, p in children:
            p.wait()
        atomic(
            run / "state.json",
            {"status": "failed", "error": str(error), "updated_at_epoch": time.time()},
        )
        raise
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["execute", "worker", "evaluate"])
    parser.add_argument("--arm")
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "worker":
        worker(config, run, args.arm)
    elif args.phase == "evaluate":
        evaluate(config, run)
    else:
        with (run / "supervisor.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execute(config, run)


if __name__ == "__main__":
    main()
