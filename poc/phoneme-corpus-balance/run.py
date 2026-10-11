"""Compare two corpus sampling policies on the same frozen 280-label pools."""

from __future__ import annotations

import argparse
import fcntl
import os
import statistics
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-corpus-seed-replication"))
import replicate as r
import sampler
import scaling_learner as learner

s = r.s
CONFIG = BASE / "config.json"
CONTEXT = ("baseline140", "expanded210", "expanded280")


def reference(config, seed):
    return s.ROOT / (
        config["reference_run"]
        if seed == config["seeds"][0]
        else f"{config['replication_run']}/seed-{seed}"
    )


def freeze(config, run):
    replication = s.ROOT / config["replication_run"]
    previous = s.read(replication / "design-freeze.json")
    assert config["seeds"] == r.seeds(previous["config"])
    assert s.read(replication / "execution-state.json")["status"] == "complete"
    assert s.read(replication / "completion-audit.json")["status"] == "passed"
    assert previous["runtime"] == s.expanded.runtime()
    files = {
        **previous["files"],
        **s.read(replication / "all-models-freeze.json")["files"],
    }
    paths = [
        CONFIG,
        BASE / "run.py",
        BASE / "sampler.py",
        *sorted((BASE / "tests").glob("*.py")),
    ]
    paths += [
        replication / name
        for name in (
            "design-freeze.json",
            "all-models-freeze.json",
            "results.json",
            "completion-audit.json",
        )
    ]
    for seed in config["seeds"]:
        source = reference(config, seed)
        paths += [
            source / name
            for name in ("design-freeze.json", "results.json", "evaluation-freeze.json")
        ]
        paths += list((source / "scores").glob("*.json"))
        paths += list((source / "evaluation-inputs").glob("*.json"))
    files.update(r.pin(paths))
    r.checked_files(files)
    frozen = {
        "config": config,
        "files": files,
        "runtime": s.expanded.runtime(),
        "reference_design": previous["reference_design"],
        "all_test_previously_observed": True,
    }
    s.fixed(run / "design-freeze.json", frozen)
    return frozen


def prepare(config, run, frozen, seed):
    directory = run / f"seed-{seed}"
    original = frozen["reference_design"]
    initial = s.read(
        reference(config, seed) / "expanded280/trial/training-summary.json"
    )["initial_encoder_sha256"]
    current = {
        **original["config"],
        "training_seed": seed,
        "run_directory": str(directory.relative_to(s.ROOT)),
        "interpretation": config["interpretation"],
    }
    s.fixed(
        directory / "design-freeze.json",
        {
            **original,
            "config": current,
            "files": frozen["files"],
            "previously_observed_test": dict.fromkeys(s.CORPORA, True),
            "source_baseline_initial_encoder_sha256": initial,
        },
    )
    s.fixed(
        directory / "corpus-overlap-audit.json",
        s.read(s.ROOT / config["reference_run"] / "corpus-overlap-audit.json"),
    )
    source = s.ROOT / config["reference_run"] / "expanded280"
    for arm, cycle in config["conditions"].items():
        destination = directory / arm
        for name in r.POOL_FILES:
            r.link_checked(source / name, destination / name)
        s.fixed(
            destination / "design-freeze.json",
            {
                "config": current,
                "sampling_condition": arm,
                "sampling_cycle": cycle,
                "parent_sha256": s.sha(run / "design-freeze.json"),
                "files": r.pin(destination / name for name in r.POOL_FILES),
            },
        )
    return current, directory, initial


def train(config, run, frozen, state):
    phones = frozen["reference_design"]["phonemes"]
    original_schedule = learner.schedule
    count = 0
    for seed in config["seeds"]:
        current, directory, initial = prepare(config, run, frozen, seed)
        source = s.ROOT / config["reference_run"] / "expanded280"
        pool = s.TrainingPool(current, source)
        control_path = directory / "original-exposure.json"
        if not control_path.exists():
            state("measuring_original_sampling", seed=seed, completed_models=count)
            planned, labels, valid = original_schedule(
                pool.groups,
                phones,
                current["maximum_updates"],
                seed,
                pool.speaker_to_class,
            )
            s.fixed(
                control_path, sampler.exposure(labels, valid, phones, pool.speakers)
            )
            del planned, labels, valid
        control = s.read(control_path)
        for arm, cycle in config["conditions"].items():
            state("training", seed=seed, arm=arm, completed_models=count)
            destination = directory / arm

            def scheduled(
                groups,
                selected,
                updates,
                schedule_seed,
                speaker_to_class,
                *,
                cycle=cycle,
                speakers=pool.speakers,
                control=control,
                destination=destination,
            ):
                planned, labels, valid = sampler.schedule(
                    groups,
                    selected,
                    updates,
                    schedule_seed,
                    speaker_to_class,
                    cycle=cycle,
                    rank=s.rank,
                )
                report = sampler.exposure(labels, valid, selected, speakers)
                assert report["valid_examples"] == control["valid_examples"]
                for phone in selected:
                    assert sum(report["phone_corpus_examples"][phone].values()) == sum(
                        control["phone_corpus_examples"][phone].values()
                    )
                s.fixed(destination / "sampling-exposure.json", report)
                return planned, labels, valid

            # Scoped dependency substitution: only the sampler changes. Frozen learner
            # bytes, loss, optimizer, class count, seed, data and update budget remain.
            learner.schedule = scheduled
            try:
                s.train_trial(current, destination, destination / "trial", phones, pool)
            finally:
                learner.schedule = original_schedule
            r.validate_trial(current, destination, initial)
            assert (
                s.read(destination / "trial/training-summary.json")["training_examples"]
                == control["valid_examples"]
            )
            count += 1
        del pool
    s.fixed(
        run / "all-models-freeze.json",
        {
            "files": r.pin(
                run / f"seed-{seed}" / arm / name
                for seed in config["seeds"]
                for arm in config["conditions"]
                for name in (
                    "trial/encoder.pt",
                    "trial/training-summary.json",
                    "trial/training-freeze.json",
                    "sampling-exposure.json",
                )
            ),
            "all_training_complete_before_calibration": True,
        },
    )


def data(config, run, seed, corpus, split, phones):
    directory = run / f"seed-{seed}"
    current = s.read(directory / "design-freeze.json")["config"]
    result, features = s.evaluation_data(current, directory, corpus, split, phones)
    name = f"{corpus}-{split}.json"
    assert s.sha(directory / "evaluation-inputs" / name) == s.sha(
        reference(config, seed) / "evaluation-inputs" / name
    )
    return result, features


def evaluate(config, run, frozen, state):
    r.checked_files(frozen["files"])
    r.checked_files(s.read(run / "all-models-freeze.json")["files"])
    phones = frozen["reference_design"]["phonemes"]
    calibration = {}
    for seed in config["seeds"]:
        state("calibrating", seed=seed, completed_models=10)
        directory = run / f"seed-{seed}"
        source = reference(config, seed)
        old = s.read(source / "evaluation-freeze.json")
        models = {
            arm: s.load_model(directory / arm / "trial")[0]
            for arm in config["conditions"]
        }
        calibration[str(seed)] = {
            "thresholds": {},
            "models": {arm: old["models"][arm] for arm in CONTEXT},
        }
        calibration[str(seed)]["models"].update(
            {arm: s.sha(directory / arm / "trial/encoder.pt") for arm in models}
        )
        for corpus in s.CORPORA:
            inputs, features = data(config, run, seed, corpus, "validation", phones)
            points = {arm: old["thresholds"][corpus][arm] for arm in CONTEXT}
            for arm in CONTEXT:
                for split in ("validation", "test"):
                    name = f"{corpus}-{arm}-{split}.json"
                    r.link_checked(
                        source / "scores" / name, directory / "scores" / name
                    )
            for arm, model in models.items():
                records = s.score(inputs, s.model_embeddings(model, features), phones)
                rows = [
                    v
                    for v in records
                    if v["role"] == "verification" and v["score"] is not None
                ]
                point, _ = s.far_target_threshold(
                    s.np.array([v["score"] for v in rows]),
                    s.np.array([v["is_genuine"] for v in rows]),
                    0.01,
                )
                points[arm] = s.json_threshold(point)
                s.fixed(
                    directory / "scores" / f"{corpus}-{arm}-validation.json", records
                )
            calibration[str(seed)]["thresholds"][corpus] = points
        del models
    s.fixed(
        run / "evaluation-freeze.json",
        {
            "seeds": calibration,
            "new_test_used_for_selection": False,
            "validation_files": r.pin(
                p
                for seed in config["seeds"]
                for p in (run / f"seed-{seed}" / "scores").glob("*-validation.json")
            ),
        },
    )
    for seed in config["seeds"]:
        state("evaluating", seed=seed, completed_models=10)
        directory = run / f"seed-{seed}"
        original = s.read(reference(config, seed) / "results.json")
        models = {
            arm: s.load_model(directory / arm / "trial")[0]
            for arm in config["conditions"]
        }
        result = {}
        for corpus in s.CORPORA:
            inputs, features = data(config, run, seed, corpus, "test", phones)
            result[corpus] = {
                arm: original["conditions"][corpus][arm] for arm in CONTEXT
            }
            for arm, model in models.items():
                records = s.score(inputs, s.model_embeddings(model, features), phones)
                s.fixed(directory / "scores" / f"{corpus}-{arm}-test.json", records)
                point = calibration[str(seed)]["thresholds"][corpus][arm]
                result[corpus][arm] = {
                    role: s.operating([v for v in records if v["role"] == role], point)
                    for role in sorted({v["role"] for v in records})
                }
        s.fixed(
            directory / "results.json",
            {
                "seed": seed,
                "conditions": result,
                "evaluation_freeze_sha256": s.sha(run / "evaluation-freeze.json"),
            },
        )


def aggregate(config, run):
    results = [
        s.read(run / f"seed-{seed}" / "results.json") for seed in config["seeds"]
    ]
    summary = {}
    for corpus in s.CORPORA:
        summary[corpus] = {}
        for arm in (*CONTEXT, *config["conditions"]):
            summary[corpus][arm] = {}
            for metric in ("eer", "far", "frr", "coverage"):
                values = [
                    v["conditions"][corpus][arm]["verification"][metric]
                    for v in results
                ]
                delta = [
                    value
                    - v["conditions"][corpus]["expanded280"]["verification"][metric]
                    for value, v in zip(values, results, strict=True)
                ]
                summary[corpus][arm][metric] = {
                    "values": values,
                    "mean": statistics.mean(values),
                    "sd": statistics.stdev(values),
                    "paired_delta_vs_expanded280": delta,
                    "paired_delta_mean": statistics.mean(delta),
                    "decreased_seeds": sum(d < 0 for d in delta),
                    "increased_seeds": sum(d > 0 for d in delta),
                }
    report = {
        "seeds": config["seeds"],
        "aggregate": summary,
        "primary_control": "expanded280",
        "interpretation": config["interpretation"],
        "source_results": r.pin(
            run / f"seed-{seed}" / "results.json" for seed in config["seeds"]
        ),
    }
    s.fixed(run / "results.json", report)
    lines = [
        "# 学習投入比率の比較（固定5seed）",
        "",
        "平均±標本SD。差は同seedの元の280話者条件に対して計算。",
        "",
        "| コーパス | 条件 | EER | FAR | FRR | EER差(pt) | FRR差(pt) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, arms in summary.items():
        for arm, metrics in arms.items():
            rates = [
                f"{metrics[m]['mean'] * 100:.3f}±{metrics[m]['sd'] * 100:.3f}%"
                for m in ("eer", "far", "frr")
            ]
            lines.append(
                f"| {corpus} | {arm} | {' | '.join(rates)} | {metrics['eer']['paired_delta_mean'] * 100:+.3f} | {metrics['frr']['paired_delta_mean'] * 100:+.3f} |"
            )
    lines += [
        "",
        "全testは既観測。新しい人物・録音条件への汎化は未検証。未採点は本人拒否に含む。希少音素では容量に応じて割当を再配分し、実投入比率を記録した。",
    ]
    (run / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "execute", "status"))
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = s.read(CONFIG)
    run = s.ROOT / config["run_directory"]
    if args.phase == "status":
        print(s.read(run / "execution-state.json"))
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def state(status, **details):
            s.atomic(
                run / "execution-state.json",
                {
                    "status": status,
                    "pid": os.getpid(),
                    "updated_at_epoch": time.time(),
                    "total_models": 10,
                    **details,
                },
            )

        try:
            frozen = freeze(config, run)
            if args.phase == "execute":
                train(config, run, frozen, state)
                evaluate(config, run, frozen, state)
                aggregate(config, run)
                state(
                    "complete",
                    completed_models=10,
                    results=str((run / "results.md").relative_to(s.ROOT)),
                )
        except BaseException as error:
            state("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
