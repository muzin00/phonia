"""Repeat the frozen four-arm study without modifying its code or artifacts."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "phoneme-corpus-scaling"))
import study as s
from common import create_encoder

CONFIG = BASE / "config.json"
POOL_FILES = ("train-segments.jsonl", "train-features.f32", "preparation-report.json")


def checked_files(files):
    for name, digest in files.items():
        assert s.sha(s.ROOT / name) == digest, name


def pin(paths):
    return {str(p.relative_to(s.ROOT)): s.sha(p) for p in paths}


def seeds(config):
    result = [config["reference_seed"], *config["additional_seeds"]]
    assert len(result) == 5 and len(set(result)) == 5
    return result


def initialize(config, run):
    seeds(config)
    source = s.ROOT / config["reference_run"]
    original = s.verify(source)
    s.verify_prepared(source)
    assert s.read(source / "execution-state.json")["status"] == "complete"
    assert original["config"]["training_seed"] == config["reference_seed"]
    paths = [CONFIG, Path(__file__), *sorted((BASE / "tests").glob("*.py"))]
    paths += [
        source / name
        for name in (
            "design-freeze.json",
            "prepared-inputs-freeze.json",
            "results.json",
            "evaluation-freeze.json",
            "corpus-overlap-audit.json",
            "feature-runtime-check.json",
        )
    ]
    paths += list((source / "evaluation-inputs").glob("*.json"))
    corpus = s.ROOT / original["config"]["corpus_run"]
    paths += [
        corpus / name
        for name in ("inputs.json", "preparation-report.json", "complete-audit.json")
    ]
    files = {
        **original["files"],
        **s.read(source / "prepared-inputs-freeze.json")["files"],
    }
    for arm in s.ARMS:
        directory = source / arm
        arm_freeze = s.read(directory / "design-freeze.json")
        checked_files(arm_freeze["files"])
        paths += [
            directory / "design-freeze.json",
            *(directory / name for name in POOL_FILES),
        ]
        paths += [
            directory / "trial" / name
            for name in ("training-summary.json", "encoder.pt")
        ]
        s.load_model(directory / "trial")
    files.update(pin(paths))
    checked_files(files)
    frozen = {
        "config": config,
        "files": files,
        "runtime": s.expanded.runtime(),
        "previously_observed_test": dict.fromkeys(s.CORPORA, True),
        "reference_design": original,
    }
    s.fixed(run / "design-freeze.json", frozen)
    return frozen


def link_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.link(source, destination)
    assert s.sha(source) == s.sha(destination)


def prepare_seed(config, run, frozen, seed):
    reference = s.ROOT / config["reference_run"]
    directory = run / f"seed-{seed}"
    original = frozen["reference_design"]
    current = {
        **original["config"],
        "training_seed": seed,
        "run_directory": str(directory.relative_to(s.ROOT)),
        "baseline_policy": "retrain same baseline pool from this seed, matched to every arm",
        "interpretation": config["interpretation"],
    }
    s.expanded.seed_everything(seed)
    initial = s.expanded.model_sha256(create_encoder(current["encoder"]))
    s.fixed(
        directory / "design-freeze.json",
        {
            **original,
            "config": current,
            "files": frozen["files"],
            "source_baseline_initial_encoder_sha256": initial,
            "previously_observed_test": dict.fromkeys(s.CORPORA, True),
            "replication_design_sha256": s.sha(run / "design-freeze.json"),
        },
    )
    for name in ("prepared-inputs-freeze.json", "corpus-overlap-audit.json"):
        s.fixed(directory / name, s.read(reference / name))
    for arm in s.ARMS:
        destination = directory / arm
        for name in POOL_FILES:
            link_checked(reference / arm / name, destination / name)
        s.fixed(
            destination / "design-freeze.json",
            {
                "config": current,
                "arm": arm,
                "parent_design_sha256": s.sha(directory / "design-freeze.json"),
                "files": pin(destination / name for name in POOL_FILES),
            },
        )
    return current, directory, initial


def validate_trial(config, directory, initial):
    trial = directory / "trial"
    summary = s.read(trial / "training-summary.json")
    training = s.read(trial / "training-freeze.json")
    checked_files(s.read(directory / "design-freeze.json")["files"])
    assert summary["completed_updates"] == config["maximum_updates"]
    assert summary["initial_encoder_sha256"] == initial
    assert training["settings"]["seed"] == config["training_seed"]
    assert training["settings"]["cohort"] == len(
        s.read(directory / "preparation-report.json")["training_speakers"]
    )
    assert training["design_freeze_sha256"] == s.sha(directory / "design-freeze.json")
    assert summary["training_freeze_sha256"] == s.sha(trial / "training-freeze.json")
    assert training["all_pool_manifest_sha256"] == s.sha(
        directory / "train-segments.jsonl"
    )
    assert training["feature_cache_sha256"] == s.sha(directory / "train-features.f32")
    assert summary["history_sha256"] == s.sha(trial / "history.jsonl")
    s.load_model(trial)
    return summary


def summarize(results):
    """Pair within seed; never compare a new arm to an old-seed baseline."""
    output = {}
    for corpus in s.CORPORA:
        output[corpus] = {}
        for arm in s.ARMS:
            metrics = {}
            for metric in ("eer", "far", "frr", "coverage"):
                values = s.np.array(
                    [
                        r["conditions"][corpus][arm]["verification"][metric]
                        for r in results
                    ]
                )
                base = s.np.array(
                    [
                        r["conditions"][corpus]["baseline140"]["verification"][metric]
                        for r in results
                    ]
                )
                assert s.np.isfinite(values).all()
                delta = values - base
                metrics[metric] = {
                    "values": values.tolist(),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "paired_delta_values": delta.tolist(),
                    "paired_delta_mean": float(delta.mean()),
                    "paired_delta_min": float(delta.min()),
                    "paired_delta_max": float(delta.max()),
                    "decreased_seeds": int((delta < 0).sum()),
                    "increased_seeds": int((delta > 0).sum()),
                    "unchanged_seeds": int((delta == 0).sum()),
                }
            output[corpus][arm] = metrics
    return output


def write_results(config, run):
    sources = [
        s.ROOT / config["reference_run"],
        *(run / f"seed-{seed}" for seed in config["additional_seeds"]),
    ]
    results = [s.read(path / "results.json") for path in sources]
    for seed, result in zip(seeds(config), results, strict=True):
        assert result["training_seeds"] == [seed]
    aggregate = summarize(results)
    s.fixed(
        run / "results.json",
        {
            "seeds": seeds(config),
            "aggregate": aggregate,
            "source_results": pin(path / "results.json" for path in sources),
            "interpretation": config["interpretation"],
            "statistics": "sample SD/range across fixed seeds; paired deltas use each seed's own baseline; no independent-data CI",
        },
    )
    lines = [
        "# 固定5seedでのコーパス拡張比較",
        "",
        "同じデータ・216,000更新。平均±標本SD。差は各seedの基準モデルとの差を平均。",
        "",
        "| コーパス | 条件 | EER | FAR | FRR | EER差(pt) | FRR差(pt) | EER低下seed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for corpus, arms in aggregate.items():
        for arm, metrics in arms.items():
            rates = [
                f"{metrics[m]['mean'] * 100:.3f}±{metrics[m]['sd'] * 100:.3f}%"
                for m in ("eer", "far", "frr")
            ]
            lines.append(
                f"| {corpus} | {arm} | {' | '.join(rates)} | {metrics['eer']['paired_delta_mean'] * 100:+.3f} | {metrics['frr']['paired_delta_mean'] * 100:+.3f} | {metrics['eer']['decreased_seeds']}/5 |"
            )
    lines += [
        "",
        "全testは既観測。5seedは初期化・学習順への感度を調べるもので、独立データの追加検証や人物母集団の信頼区間ではない。",
        "採点不能は本人拒否に含める。話者数以外に音声量・録音環境・ラベル品質も異なる。初回結果も含め全seedを報告し、最良seedを選ばない。",
    ]
    (run / "results.md").write_text("\n".join(lines) + "\n")


def execute(config, run, frozen):
    def state(status, **details):
        s.atomic(
            run / "execution-state.json",
            {
                "status": status,
                "pid": os.getpid(),
                "updated_at_epoch": time.time(),
                **details,
            },
        )

    prepared = {}
    for seed in seeds(config):
        prepared[seed] = prepare_seed(config, run, frozen, seed)
    # Replay the original baseline once to prove the reused learner/pool is equivalent.
    initial_config, replay, initial = prepared[config["reference_seed"]]
    state(
        "training",
        seed=config["reference_seed"],
        arm="baseline140",
        purpose="baseline_replay",
        completed_models=0,
        total_models=17,
    )
    pool = s.TrainingPool(initial_config, replay / "baseline140")
    s.train_trial(
        initial_config,
        replay / "baseline140",
        replay / "baseline140/trial",
        frozen["reference_design"]["phonemes"],
        pool,
    )
    del pool
    summary = validate_trial(initial_config, replay / "baseline140", initial)
    reference = s.ROOT / config["reference_run"]
    expected = s.read(reference / "baseline140/trial/training-summary.json")
    assert summary["final_encoder_sha256"] == expected["final_encoder_sha256"], (
        "Baseline replay differs"
    )
    assert summary["schedule_sha256"] == expected["schedule_sha256"], (
        "Baseline schedule differs"
    )
    s.fixed(
        run / "baseline-replay.json",
        {
            "final_encoder_sha256": summary["final_encoder_sha256"],
            "schedule_sha256": summary["schedule_sha256"],
            "identical_to_reference": True,
        },
    )
    count = 1
    model_paths = []
    for seed in config["additional_seeds"]:
        current, directory, initial = prepared[seed]
        for arm in s.ARMS:
            state(
                "training", seed=seed, arm=arm, completed_models=count, total_models=17
            )
            pool = s.TrainingPool(current, directory / arm)
            s.train_trial(
                current,
                directory / arm,
                directory / arm / "trial",
                frozen["reference_design"]["phonemes"],
                pool,
            )
            del pool
            validate_trial(current, directory / arm, initial)
            model_paths += [
                directory / arm / "trial" / name
                for name in (
                    "encoder.pt",
                    "training-summary.json",
                    "training-freeze.json",
                )
            ]
            count += 1
    s.fixed(
        run / "all-models-freeze.json",
        {"files": pin(model_paths), "training_complete_before_new_evaluation": True},
    )
    for seed in config["additional_seeds"]:
        current, directory, _ = prepared[seed]
        state("evaluating", seed=seed, completed_models=count, total_models=17)
        checked_files(s.read(run / "all-models-freeze.json")["files"])
        s.evaluate(current, directory)
        for corpus in s.CORPORA:
            for split in ("validation", "test"):
                name = f"{corpus}-{split}.json"
                assert s.sha(directory / "evaluation-inputs" / name) == s.sha(
                    reference / "evaluation-inputs" / name
                )
    write_results(config, run)
    state(
        "complete",
        completed_models=count,
        total_models=17,
        results=str((run / "results.md").relative_to(s.ROOT)),
    )


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
        try:
            frozen = initialize(config, run)
            if args.phase == "execute":
                execute(config, run, frozen)
        except BaseException as error:
            s.atomic(
                run / "execution-state.json",
                {
                    "status": "failed",
                    "error": str(error),
                    "updated_at_epoch": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
