"""Matched 420/490-label encoder training and CV-only Transformer comparison."""

import argparse
import csv
import fcntl
import gc
import hashlib
import heapq
import os
import subprocess
import sys
import time
import wave
from collections import Counter
from pathlib import Path

from shared import (
    BASE,
    CONFIG,
    ROOT,
    b,
    check_files,
    corpus,
    cv_schedule,
    d,
    e,
    fixed,
    install_corpus_adapter,
    mining,
    module,
    old_sources,
    pin,
    read,
    s,
    sha,
)


def freeze(config, run):
    if (run / "design-freeze.json").exists():
        result = read(run / "design-freeze.json")
        assert result["config"] == config and result["runtime"] == s.expanded.runtime()
        check_files(result["files"])
        return result
    prepared = read(run / "prepared-freeze.json")
    assert prepared["runtime"] == s.expanded.runtime()
    prior = read(ROOT / config["baseline_run"] / "design-freeze.json")
    old_cv = ROOT / config["baseline_cv_only"]
    old_design = read(old_cv / "design-freeze.json")
    for key in (
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
        assert config[key] == old_design["config"][key]
    files = {
        **prepared["files"],
        **read(run / "source-freeze.json")["files"],
        **old_design["files"],
        **read(old_cv / "complete-audit.json")["files"],
    }
    files.update(
        pin(
            [
                CONFIG,
                *BASE.glob("*.py"),
                *(BASE / "tests").glob("*.py"),
                old_cv / "results.json",
                ROOT / config["baseline_run"] / "results.json",
                run / "prepared-freeze.json",
                old_cv / "complete-audit.json",
            ]
        )
    )
    # Pin every imported repository Python implementation, including adapters.
    paths = set()
    for imported in list(sys.modules.values()):
        path = getattr(imported, "__file__", None)
        if path:
            path = Path(path).resolve()
            if (
                path.is_relative_to(ROOT / "poc")
                and path.suffix == ".py"
                and ".venv" not in path.parts
            ):
                paths.add(path)
    files.update(pin(paths))
    for source in old_sources(config).values():
        fusion = ROOT / source["fusion"]
        files.update(read(fusion / "prepared-freeze.json")["files"])
        files.update(
            pin(
                [
                    fusion / "design-freeze.json",
                    ROOT / source["trial"] / "encoder.pt",
                    ROOT / source["trial"] / "training-summary.json",
                ]
            )
        )
    check_files(files)
    frozen = {
        "config": config,
        "files": files,
        "runtime": s.expanded.runtime(),
        "reference_design": prior["reference_design"],
        "phones": prior["phones"],
        "baseline_sources": {str(k): v for k, v in old_sources(config).items()},
        "test_previously_observed": True,
    }
    fixed(run / "design-freeze.json", frozen)
    return frozen


def prepare_fusion(config, run, frozen, family):
    fusion = family / "fusion"
    fusion.mkdir(exist_ok=True)
    first = next(iter(old_sources(config).values()))
    original = read(ROOT / first["fusion"] / "design-freeze.json")["config"]
    options = {
        **original,
        "run_directory": str(fusion.relative_to(ROOT)),
        "seeds": [config["fusion_seed"]],
        "architectures": ["transformer"],
        "encoder_sha256": sha(family / "trial/encoder.pt"),
    }
    fixed(
        fusion / "design-freeze.json",
        {
            "config": options,
            "files": pin([family / "design-freeze.json", family / "trial/encoder.pt"]),
            "phones": frozen["phones"],
            "runtime": s.expanded.runtime(),
        },
    )
    if (fusion / "prepared-freeze.json").exists():
        check_files(read(fusion / "prepared-freeze.json")["files"])
        return options
    labels, hashes = set(), set()
    for c in s.CORPORA:
        for split in ("validation", "test"):
            data, _, manifest = e.evaluation_source(
                config, frozen["reference_design"], c, split
            )
            labels.update(data["speakers"])
            hashes.update(v["source_sha256"] for v in s.rows(manifest))
    train, metadata = e.training_inputs(
        options, run / "train-segments.jsonl", frozen["phones"], labels, hashes
    )
    train["training_label_counts"] = dict(Counter(corpus(v) for v in train["speakers"]))
    assert train["encoder_training_labels"] == 490
    assert train["training_label_counts"]["src4vc"] >= 8
    assert sum(train["training_label_counts"].values()) == len(train["speakers"])
    fixed(fusion / "train-inputs.json", train)
    matrix = s.np.memmap(
        run / "train-features.f32",
        dtype="<f4",
        mode="r",
        shape=(train["feature_count"], 128),
    )
    model, _ = s.load_model(family / "trial")
    vectors = s.model_embeddings(model, matrix)
    s.np.savez(
        fusion / "train.npz", **e.aggregates(train, vectors, metadata, frozen["phones"])
    )
    del matrix, vectors, metadata
    seed = config["fusion_seed"]
    s.np.savez(fusion / f"schedule-{seed}.npz", **e.make_schedule(options, train, seed))
    for c in s.CORPORA:
        d.previous.eval_arrays(
            config, frozen, model, frozen["phones"], c, "validation", fusion
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


def make_cv_schedule(config, original, arrays, data):
    table, owner = mining.score_table(arrays, data)
    all_hard, candidates = mining.make_schedule(original, table, owner, config)
    is_cv = s.np.array(
        [corpus(q["speaker_id"]) == "common_voice" for q in data["queries"]]
    )
    result = cv_schedule.restrict_to_cv(original, all_hard, is_cv, config["budgets"][0])
    query_owner = owner[result["queries"]]
    sources = s.np.array([corpus(v) for v in data["speakers"]])
    assert (sources[query_owner] == sources[result["negative_claims"]]).all()
    assert (query_owner != result["negative_claims"]).all()
    noncv = ~is_cv[result["queries"]]
    assert s.np.array_equal(
        result["negative_claims"][noncv], original["negative_claims"][noncv]
    )
    return result, table, candidates


def continue_fusion(config, family):
    fusion, target = family / "fusion", family / "continued"
    target.mkdir(exist_ok=True)
    seed = config["fusion_seed"]
    original = read(fusion / "design-freeze.json")["config"]
    options = {**config, "reference_run": str(fusion.relative_to(ROOT))}
    if not (target / "schedule-freeze.json").exists():
        data = read(fusion / "train-inputs.json")
        random = b.build_schedule(options, original, data, seed)
        with s.np.load(fusion / "train.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        schedule, table, candidates = make_cv_schedule(config, random, arrays, data)
        s.np.savez(target / "random-schedule.npz", **random)
        s.np.savez(target / f"schedule-{seed}.npz", **schedule)
        s.np.save(target / "mining-scores.npy", table)
        s.np.savez_compressed(target / "mining-candidates.npz", **candidates)
        fixed(
            target / "schedule-freeze.json",
            {
                "files": pin(
                    list(target.glob("*.npz")) + [target / "mining-scores.npy"]
                ),
                "changed_cv_slots": int(
                    (schedule["negative_claims"] != random["negative_claims"]).sum()
                ),
                "noncv_unchanged": True,
                "train_only": True,
            },
        )
    check_files(read(target / "schedule-freeze.json")["files"])
    fixed(
        target / "design-freeze.json",
        {
            "config": options,
            "files": {
                **read(target / "schedule-freeze.json")["files"],
                **read(fusion / "prepared-freeze.json")["files"],
                **pin(
                    [
                        fusion / "design-freeze.json",
                        *(
                            fusion / f"transformer-{seed}" / n
                            for n in (
                                "last.pt",
                                "fusion.pt",
                                "history.json",
                                "training-summary.json",
                            )
                        ),
                    ]
                ),
            },
        },
    )
    b.worker(options, target, seed)


def worker(config, run, seed):
    s.torch.set_num_threads(1)
    install_corpus_adapter()
    frozen = freeze(config, run)
    family = run / f"acoustic-{seed}"
    family.mkdir(exist_ok=True)

    def state(status, **details):
        d.state(family / "state.json", status, **details)

    try:
        training = {
            **frozen["reference_design"]["config"],
            "training_seed": seed,
            "maximum_updates": config["encoder_updates"],
            "run_directory": str(family.relative_to(ROOT)),
        }
        for name in (
            "train-segments.jsonl",
            "train-features.f32",
            "preparation-report.json",
        ):
            d.link(run / name, family / name)
        fixed(
            family / "design-freeze.json",
            {
                "config": training,
                "root_sha256": sha(run / "design-freeze.json"),
                "files": pin(
                    [
                        family / n
                        for n in (
                            "train-segments.jsonl",
                            "train-features.f32",
                            "preparation-report.json",
                        )
                    ]
                ),
            },
        )
        state("training_encoder")
        pool = s.TrainingPool(training, family)
        s.train_trial(training, family, family / "trial", frozen["phones"], pool)
        del pool
        gc.collect()
        new = read(family / "trial/training-summary.json")
        old = read(ROOT / old_sources(config)[seed]["trial"] / "training-summary.json")
        assert new["initial_encoder_sha256"] == old["initial_encoder_sha256"]
        assert (
            new["completed_updates"]
            == old["completed_updates"]
            == config["encoder_updates"]
        )
        state("preparing_fusion")
        options = prepare_fusion(config, run, frozen, family)
        state("training_fusion_prefix")
        e.train(options, family / "fusion", state)
        state("continuing_cv_only_fusion")
        continue_fusion(config, family)
        state("complete")
    except BaseException as error:
        state("failed", error=str(error))
        raise


def sources(config, run):
    result = {}
    for seed, old in old_sources(config).items():
        result["baseline420", seed] = {
            "trial": ROOT / old["trial"],
            "fusion": ROOT / old["fusion"],
            "budget": ROOT / config["baseline_cv_only"] / f"acoustic-{seed}",
        }
        family = run / f"acoustic-{seed}"
        result["src490", seed] = {
            "trial": family / "trial",
            "fusion": family / "fusion",
            "budget": family / "continued",
        }
    return result


def evaluate(config, run, frozen):
    independent = module(
        "src490_score_audit", BASE.parent / "phoneme-corpus-transformer/audit.py"
    )
    all_sources = sources(config, run)
    models = []
    for paths in all_sources.values():
        models.append(paths["trial"] / "encoder.pt")
        models.extend(
            paths["budget"] / f"seed-{config['fusion_seed']}/budget-{budget}/{mode}.pt"
            for budget in config["budgets"][1:]
            for mode in ("selected", "final")
        )
    fixed(run / "all-models-freeze.json", {"files": pin(models)})
    points, results, score_files = {}, {}, []
    maxdiff, audited = 0.0, 0
    for split in ("validation", "test"):
        for (arm, seed), paths in all_sources.items():
            fusion = paths["fusion"]
            original = read(fusion / "design-freeze.json")["config"]
            enc, _ = s.load_model(paths["trial"])
            variants = {"equal": None}
            for budget in config["budgets"][1:]:
                for mode in ("selected", "final"):
                    model = e.create_model(original, "transformer")
                    path = (
                        paths["budget"]
                        / f"seed-{config['fusion_seed']}/budget-{budget}/{mode}.pt"
                    )
                    model.load_state_dict(
                        s.torch.load(path, weights_only=True)["state_dict"]
                    )
                    variants[f"u{budget}-{mode}"] = model.eval()
            for c in s.CORPORA:
                d.state(
                    run / "state.json",
                    "evaluating",
                    split=split,
                    arm=arm,
                    seed=seed,
                    corpus=c,
                )
                if arm == "src490" and split == "test":
                    assert (run / "selection-freeze.json").exists()
                    d.previous.eval_arrays(
                        config, frozen, enc, frozen["phones"], c, split, fusion
                    )
                arrays = e.load_arrays(fusion / f"{c}-{split}.npz")
                baseline = read(fusion / f"{c}-{split}-baseline.json")
                data = read(fusion / f"{c}-{split}-inputs.json")
                reference_data = read(
                    ROOT
                    / old_sources(config)[seed]["fusion"]
                    / f"{c}-{split}-inputs.json"
                )
                assert data == reference_data
                for variant, model in variants.items():
                    label = f"{arm}-seed{seed}-{variant}"
                    rows = (
                        baseline if model is None else e.score(model, arrays, baseline)
                    )
                    if arm == "baseline420" and model is not None:
                        old_label = f"cv_only-seed{seed}-{variant}"
                        prior = read(
                            ROOT
                            / config["baseline_cv_only"]
                            / "scores"
                            / f"{c}-{old_label}-{split}.json"
                        )
                        assert rows == prior
                    if model is not None:
                        maxdiff = max(
                            maxdiff,
                            independent.independently_score(model, arrays, data, rows),
                        )
                    path = run / "scores" / f"{c}-{label}-{split}.json"
                    fixed(path, rows)
                    score_files.append(path)
                    audited += len(rows)
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
                        points.setdefault(c, {})[label] = s.json_threshold(threshold)
                    else:
                        metrics = {
                            role: s.operating(
                                [v for v in rows if v["role"] == role], points[c][label]
                            )
                            for role in sorted({v["role"] for v in rows})
                        }
                        results.setdefault(c, {})[label] = metrics
                        if arm == "baseline420" and model is not None:
                            assert (
                                metrics
                                == read(
                                    ROOT / config["baseline_cv_only"] / "results.json"
                                )["conditions"][c][old_label]
                            )
        if split == "validation":
            fixed(
                run / "selection-freeze.json",
                {
                    "thresholds": points,
                    "models": read(run / "all-models-freeze.json")["files"],
                    "validation_files": pin(score_files),
                },
            )
    fixed(
        run / "results.json",
        {
            "config": config,
            "conditions": results,
            "selection_sha256": sha(run / "selection-freeze.json"),
        },
    )
    fixed(
        run / "score-audit.json",
        {
            "records": audited,
            "maximum_fusion_score_difference": maxdiff,
            "files": pin(
                score_files + [run / "results.json", run / "selection-freeze.json"]
            ),
        },
    )


def audit(config, run):
    frozen = freeze(config, run)
    report = read(run / "preparation-report.json")
    baseline = ROOT / config["baseline_data"]
    for filename, size_key in (
        ("train-segments.jsonl", "manifest"),
        ("train-features.f32", "features"),
    ):
        size = report["baseline_prefix_bytes"][size_key]
        h = hashlib.sha256()
        with (run / filename).open("rb") as f:
            while size:
                data = f.read(min(size, 1048576))
                assert data
                h.update(data)
                size -= len(data)
        assert h.hexdigest() == sha(baseline / filename)
    phones = frozen["phones"]
    src_rows = [
        v
        for v in s.rows(run / "train-segments.jsonl")
        if corpus(v["speaker_id"]) == "src4vc"
    ]
    cache = s.np.memmap(
        run / "train-features.f32",
        dtype="<f4",
        mode="r",
        shape=(report["train_feature_count"], 128),
    )
    pipe = s.pipeline(read(run / "source-freeze.json")["extraction"])
    max_feature = 0.0
    for row in heapq.nsmallest(
        200, src_rows, key=lambda v: s.rank(20261013, v["segment_id"])
    ):
        with wave.open(str(ROOT / row["source_file"])) as wav:
            wav.setpos(row["start_frame"])
            pcm = (
                s.np.frombuffer(
                    wav.readframes(row["end_frame"] - row["start_frame"]), dtype="<i2"
                ).astype(s.np.float32)
                / 32768
            )
        actual = s.pool_features(pipe, pcm, row["segment_id"])
        max_feature = max(
            max_feature, float(s.np.abs(actual - cache[row["cache_index"]]).max())
        )
    assert max_feature <= 2e-6
    input_audit = {}
    for seed in config["encoder_seeds"]:
        family = run / f"acoustic-{seed}"
        summary = read(family / "trial/training-summary.json")
        assert (
            summary["completed_updates"] == config["encoder_updates"]
            and summary["encoder_parameters"] == 65920
        )
        fusion, target = family / "fusion", family / "continued"
        data = read(fusion / "train-inputs.json")
        enrollment_hashes = set(data["enrollment_wav_hashes"])
        assert not enrollment_hashes & set(data["query_wav_hashes"])
        # Every profile and query index must resolve to the same source and phone.
        identities = [
            (v["speaker_id"], v["source_sha256"], v["phoneme"])
            for v in s.rows(run / "train-segments.jsonl")
        ]
        for speaker, profile in data["profiles"].items():
            for phone, ids in profile.items():
                for i in ids:
                    owner, digest, p = identities[i]
                    assert (
                        owner == speaker and p == phone and digest in enrollment_hashes
                    )
        for query in data["queries"]:
            for phone, ids in query["groups"].items():
                for i in ids:
                    owner, digest, p = identities[i]
                    assert (
                        owner == query["speaker_id"]
                        and p == phone
                        and digest == query["query_id"]
                    )
        del identities
        options = {**config, "reference_run": str(fusion.relative_to(ROOT))}
        original = read(fusion / "design-freeze.json")["config"]
        random = b.build_schedule(options, original, data, config["fusion_seed"])
        with s.np.load(target / "random-schedule.npz", allow_pickle=False) as saved:
            for key in random:
                s.np.testing.assert_array_equal(random[key], saved[key])
        with s.np.load(fusion / "train.npz", allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        schedule, _, _ = make_cv_schedule(config, random, arrays, data)
        with s.np.load(
            target / f"schedule-{config['fusion_seed']}.npz", allow_pickle=False
        ) as saved:
            for key in schedule:
                s.np.testing.assert_array_equal(schedule[key], saved[key])
        directory = target / f"seed-{config['fusion_seed']}"
        history = read(directory / "history.json")
        assert [v["update"] for v in history] == list(
            range(0, max(config["budgets"]) + 1, config["validation_every"])
        )
        for budget in config["budgets"][1:]:
            selected = b.select_best([v for v in history if v["update"] <= budget])
            result = read(directory / f"budget-{budget}/summary.json")
            assert result["selected_update"] == selected["update"]
            assert result["encoder_sha256"] == sha(family / "trial/encoder.pt")
            assert (
                s.torch.load(
                    directory / f"budget-{budget}/final.pt", weights_only=True
                )["update"]
                == budget
            )
        input_audit[str(seed)] = {
            "labels": data["training_label_counts"],
            "queries": len(data["queries"]),
            "encoder_sha256": summary["encoder_sha256"],
            "phones": len(phones),
        }
    files = pin(
        [
            run / "results.json",
            run / "score-audit.json",
            run / "selection-freeze.json",
            run / "all-models-freeze.json",
        ]
    )
    files.update(read(run / "score-audit.json")["files"])
    files.update(read(run / "all-models-freeze.json")["files"])
    for seed in config["encoder_seeds"]:
        family = run / f"acoustic-{seed}"
        files.update(read(family / "fusion/prepared-freeze.json")["files"])
        files.update(read(family / "continued/schedule-freeze.json")["files"])
        files.update(
            read(family / f"continued/seed-{config['fusion_seed']}/complete.json")[
                "files"
            ]
        )
        files.update(
            pin(
                [
                    family / "trial/training-summary.json",
                    family / "trial/history.jsonl",
                    family / "trial/training-freeze.json",
                    family / "fusion/design-freeze.json",
                ]
            )
        )
    check_files(files)
    fixed(
        run / "complete-audit.json",
        {
            "files": files,
            "feature_probes": 200,
            "maximum_feature_difference": max_feature,
            "inputs": input_audit,
            "baseline_prefix_preserved": True,
            "training_labels": 490,
            "reserved_src4vc_speakers_used": False,
        },
    )


def report(config, run):
    results = read(run / "results.json")["conditions"]
    summary, table = {}, []
    for variant in [
        "equal",
        *(f"u{n}-{m}" for n in config["budgets"][1:] for m in ("selected", "final")),
    ]:
        summary[variant] = {}
        for c in [*s.CORPORA, "macro"]:
            values = {}
            for arm in ("baseline420", "src490"):
                values[arm] = {}
                for metric in ("eer", "far", "frr", "coverage"):
                    samples = [
                        float(
                            s.np.mean(
                                [
                                    results[k][f"{arm}-seed{seed}-{variant}"][
                                        "verification"
                                    ][metric]
                                    * 100
                                    for k in (s.CORPORA if c == "macro" else [c])
                                ]
                            )
                        )
                        for seed in config["encoder_seeds"]
                    ]
                    values[arm][metric] = {
                        "mean_percent": float(s.np.mean(samples)),
                        "sd_percent": float(s.np.std(samples, ddof=1)),
                        "seeds_percent": samples,
                    }
            delta = (
                s.np.array(values["src490"]["eer"]["seeds_percent"])
                - values["baseline420"]["eer"]["seeds_percent"]
            )
            item = {
                "arms": values,
                "eer_delta_pp": delta.tolist(),
                "mean_delta_pp": float(delta.mean()),
                "improved": int((delta < -1e-10).sum()),
                "worsened": int((delta > 1e-10).sum()),
                "tied": int((abs(delta) <= 1e-10).sum()),
            }
            summary[variant][c] = item
            table.append(
                {
                    "variant": variant,
                    "corpus": c,
                    "baseline_eer_percent": values["baseline420"]["eer"][
                        "mean_percent"
                    ],
                    "src490_eer_percent": values["src490"]["eer"]["mean_percent"],
                    "delta_pp": item["mean_delta_pp"],
                    "improved_seeds": item["improved"],
                    "worsened_seeds": item["worsened"],
                }
            )
    fixed(run / "summary.json", summary)
    with (run / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(
        {
            "primary": summary[f"u{config['primary_budget']}-selected"],
            "encoder": summary["equal"],
        },
        flush=True,
    )


def execute(config, run):
    frozen = freeze(config, run)
    d.state(run / "state.json", "training", seeds=config["encoder_seeds"])
    children = []
    try:
        for seed in config["encoder_seeds"]:
            log = (run / f"worker-{seed}.log").open("a")
            process = subprocess.Popen(
                [sys.executable, str(BASE / "study.py"), "worker", "--seed", str(seed)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            log.close()
            children.append(process)
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("A training worker failed; inspect worker logs")
            time.sleep(5)
        assert all(p.returncode == 0 for p in children)
        evaluate(config, run, frozen)
        audit(config, run)
        report(config, run)
        d.state(run / "state.json", "complete")
    except BaseException as error:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            process.wait()
        d.state(run / "state.json", "failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("execute", "worker", "evaluate", "audit", "report", "status")
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    install_corpus_adapter()
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    if args.phase == "worker":
        assert args.seed in config["encoder_seeds"]
        with (run / f"worker-{args.seed}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(config, run, args.seed)
    elif args.phase == "status":
        for path in [
            run / "state.json",
            run / "preparation-state.json",
            *run.glob("acoustic-*/state.json"),
            *run.glob("acoustic-*/continued/seed-*/state.json"),
        ]:
            if path.exists():
                print(str(path.relative_to(run)), read(path))
    else:
        with (run / "execution.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.phase == "execute":
                execute(config, run)
            elif args.phase == "evaluate":
                evaluate(config, run, freeze(config, run))
            elif args.phase == "audit":
                audit(config, run)
            else:
                report(config, run)


if __name__ == "__main__":
    main()
