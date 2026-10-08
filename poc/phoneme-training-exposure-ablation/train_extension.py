"""Continue the exact expanded trajectory to a predetermined exposure target."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-training-data-expansion"))
import torch
import training as source

CONFIG = BASE / "config/protocol.json"
read_json, checked, sha256_file, write_json = (
    source.read_json,
    source.checked,
    source.sha256_file,
    source.write_json,
)


def validate_protocol(config):
    if (
        config["resume_update"] != 15000
        or config["target_update"] != 30000
        or config["new_training_conditions"] != 1
        or config["checkpoint_selection"]
        != "fixed_target_update_selected_before_new_training"
        or config["early_stopping"] != "disabled_to_reach_predeclared_exposure_target"
        or config["optimizer_scheduler_rng"]
        != "restore_source_checkpoint_without_changes"
        or any(
            config[k]
            for k in (
                "change_training_data",
                "change_feature_statistics",
                "change_encoder",
                "test_threshold_recalibration",
            )
        )
        or config["seed"] != 20260926
        or config["torch_threads"] != 1
        or config["enrollment_count_per_vowel"] != 10
        or config["minimum_vowels"] != 5
    ):
        raise ValueError("unsupported fixed exposure protocol")


def verify_exposure(current, baseline):
    jvs = {s: n for s, n in current.items() if s.startswith("jvs")}
    cv = {s: n for s, n in current.items() if s.startswith("cv17_")}
    if len(jvs) != 70 or len(cv) != 70 or set(jvs) != set(baseline):
        raise ValueError("training speaker populations changed")
    if any(abs(jvs[s] - baseline[s]) > 1 for s in jvs):
        raise ValueError("JVS speaker exposure does not match baseline")
    if max(current.values()) - min(current.values()) > 1:
        raise ValueError("unbalanced speaker exposure")
    return {
        "jvs_label_batch_count_min": min(jvs.values()),
        "jvs_label_batch_count_max": max(jvs.values()),
        "jvs_label_batch_count_total": sum(jvs.values()),
        "cv_label_batch_count_total": sum(cv.values()),
        "maximum_jvs_difference_from_baseline": max(
            abs(jvs[s] - baseline[s]) for s in jvs
        ),
        "baseline_label_batch_count_min": min(baseline.values()),
        "baseline_label_batch_count_max": max(baseline.values()),
    }


def prepare(config, run):
    validate_protocol(config)
    if run.exists() or not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("unused artifact directory required")
    prior = ROOT / config["source_training_run"]
    checked(prior / "training-freeze.json", config["source_training_freeze_sha256"])
    checked(
        prior / "expanded/training/summary.json",
        config["source_training_summary_sha256"],
    )
    freeze = read_json(prior / "training-freeze.json")
    summary = read_json(prior / "expanded/training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["selected_update"] != config["resume_update"]
    ):
        raise ValueError("requires original selected checkpoint")
    files = dict(freeze["files"])
    for name, checksum in summary["outputs_sha256"].items():
        checked(prior / "expanded" / name, checksum)
        files[str((prior / "expanded" / name).relative_to(ROOT))] = checksum
    baseline = ROOT / freeze["config"]["baseline_run"] / "checkpoints/best.pt"
    paths = [
        CONFIG,
        Path(__file__),
        BASE / "tests/test_extension.py",
        prior / "training-freeze.json",
        prior / "expanded/training/summary.json",
        baseline,
    ]
    for path in paths:
        files[str(path.relative_to(ROOT))] = sha256_file(path)
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    state = torch.load(
        prior / config["resume_checkpoint"], map_location="cpu", weights_only=False
    )
    reference = torch.load(baseline, map_location="cpu", weights_only=False)
    if state["update"] != 15000 or reference["update"] != 15000:
        raise ValueError("source update changed")
    # Predict the exact balanced selection count range before any new training.
    target_selections = config["target_update"] * 10
    lower, remainder = divmod(target_selections, 140)
    if min(reference["sampler"]["speaker_counts"].values()) != lower or max(
        reference["sampler"]["speaker_counts"].values()
    ) != lower + bool(remainder):
        raise ValueError("target does not match baseline per-speaker exposure")
    run.mkdir(parents=True)
    write_json(
        run / "training-freeze.json",
        {
            "status": "frozen_before_training",
            "config": config,
            "files": files,
            "runtime": source.runtime(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_settings": freeze["config"],
            "baseline_sampler_counts": reference["sampler"]["speaker_counts"],
            "source_sampler_counts": state["sampler"]["speaker_counts"],
            "source_validation_selected_before_this_study": True,
            "new_checkpoint_selection_uses_test": False,
            "test_previously_observed": True,
        },
    )
    print(
        "training frozen: resume15000 -> fixed30000; unchanged data, optimizer and schedule",
        flush=True,
    )


def continue_fixed(trainer, target, *, checkpoint_dir, on_update):
    if target <= trainer.update:
        raise ValueError("extension must increase update count")
    # Keep the Phase 3 training loop, with no early-stopping callback. Diagnostics
    # run in on_update, while the target checkpoint remains predetermined.
    trainer.train_until(
        target, on_validation=None, checkpoint_dir=checkpoint_dir, on_update=on_update
    )


def train(config, run):
    freeze = read_json(run / "training-freeze.json")
    if (
        freeze["status"] != "frozen_before_training"
        or freeze["config"] != config
        or freeze["runtime"] != source.runtime()
    ):
        raise ValueError("training design/runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    output = run / "expanded"
    completion = output / "training/summary.json"
    if completion.exists():
        summary = read_json(completion)
        if (
            summary["status"] != "completed"
            or summary["selected_update"] != config["target_update"]
        ):
            raise ValueError("invalid completed extension")
        for name, checksum in summary["outputs_sha256"].items():
            checked(output / name, checksum)
        print(
            "extension already completed; frozen inputs and outputs verified",
            flush=True,
        )
        return
    prior = ROOT / config["source_training_run"]
    trainer, pipeline = source.make_trainer(freeze["source_settings"], prior)
    output.mkdir(exist_ok=True)
    descriptor = {
        "settings": asdict(trainer.settings),
        "configuration_sha256": trainer.settings.sha256,
        "freeze_sha256": sha256_file(run / "training-freeze.json"),
        "initialization": "resume_same_optimizer_scheduler_rng_from_selected15000",
        "encoder_parameters": 65920,
        "training_head_parameters": 17920,
        "fixed_target_update": config["target_update"],
        "test_used": False,
    }
    if (output / "run.json").exists() and read_json(output / "run.json") != descriptor:
        raise ValueError("existing run design changed")
    write_json(output / "run.json", descriptor)
    checkpoint = output / "checkpoints/last.pt"
    trainer.load_checkpoint(
        checkpoint if checkpoint.exists() else prior / config["resume_checkpoint"]
    )
    history = output / "training/history.jsonl"
    history.parent.mkdir(exist_ok=True)
    if history.exists():
        source._trim_history_to_checkpoint(history, trainer.update)
    old_history = {
        r["update"]: r
        for r in source.read_rows(prior / "expanded/training/history.jsonl")
    }
    started = perf_counter()
    with history.open("a", encoding="utf-8") as stream:

        def record(value):
            u = value["update"]
            if u in old_history and value != old_history[u]:
                raise ValueError(f"continuation differs from original training at {u}")
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            if u % 100 == 0:
                print(json.dumps(value, allow_nan=False), flush=True)
            if u % trainer.settings.validation_interval == 0:
                source._compact_validation_outputs(output, config["target_update"])
                metrics = source.evaluate_validation(
                    trainer.model,
                    pipeline,
                    root=ROOT,
                    manifest=ROOT / freeze["source_settings"]["evaluation_manifest"],
                    selection_root=ROOT
                    / freeze["source_settings"]["validation_selections"],
                    output_root=output / f"validation/update-{u:06d}",
                    device="cpu",
                    batch_size=32,
                )
                print(
                    f"diagnostic validation {u}: macro EER {100 * metrics['macro_eer']:.3f}%; fixed selection remains30000",
                    flush=True,
                )

        if trainer.update < config["target_update"]:
            continue_fixed(
                trainer,
                config["target_update"],
                checkpoint_dir=output / "checkpoints",
                on_update=record,
            )
    if trainer.update != config["target_update"]:
        raise ValueError("exposure target not reached")
    trainer.save_checkpoint(checkpoint)
    exposure = verify_exposure(
        trainer.sampler.state_at(trainer.update)["speaker_counts"],
        freeze["baseline_sampler_counts"],
    )
    bundle = output / "bundle"
    bundle.mkdir(exist_ok=True)
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "encoder": "statistics_mlp",
            "embedding_dimension": 128,
            "checkpoint_update": trainer.update,
            "run_seed": trainer.settings.seed,
            "encoder_parameters": 65920,
            "configuration_sha256": trainer.settings.sha256,
        },
        bundle / "encoder.pt",
    )
    shutil.copyfile(
        prior / "feature-statistics.json", bundle / "feature-statistics.json"
    )
    trainer.model.eval()
    restored = source.create_encoder("statistics_mlp").eval()
    restored.load_state_dict(
        torch.load(bundle / "encoder.pt", map_location="cpu", weights_only=True)[
            "model"
        ],
        strict=True,
    )
    selected = sorted(
        source.load_segments(
            ROOT / freeze["source_settings"]["evaluation_manifest"],
            split="validation",
            role="verification",
        ),
        key=lambda s: s.segment_id,
    )[:8]
    probe = source.SegmentDataset(
        ROOT, selected, pipeline, kind="log_mel", mode="center"
    )
    batch = source.collate_segments([probe[i] for i in range(8)])
    with torch.inference_mode():
        before = trainer.model(batch["input"], batch["mask"])
        after = restored(batch["input"], batch["mask"])
    if (
        before.shape != (8, 128)
        or not torch.equal(before, after)
        or not torch.isfinite(after).all()
    ):
        raise ValueError("export reload differs")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    metrics_path = (
        output / f"validation/update-{trainer.update:06d}/metrics/validation.json"
    )
    metrics = read_json(metrics_path)
    if metrics["partial"]:
        raise ValueError("requires full validation")
    records = list(source.read_rows(history))
    if [r["update"] for r in records] != list(
        range(config["resume_update"] + 1, config["target_update"] + 1)
    ):
        raise ValueError("incomplete extension history")
    paths = [
        checkpoint,
        history,
        metrics_path,
        bundle / "encoder.pt",
        bundle / "feature-statistics.json",
        output / "run.json",
    ]
    summary = {
        "status": "completed",
        "selected_update": trainer.update,
        "completed_updates": trainer.update,
        "resume_update": config["resume_update"],
        "additional_updates": len(records),
        "validation_macro_eer": metrics["macro_eer"],
        "selection_rule": config["checkpoint_selection"],
        "encoder_parameters": 65920,
        "training_head_parameters": 17920,
        "export_reload_bitwise_equal_validation_segments": 8,
        "test_used": False,
        "original_trajectory_bitwise_equal_history_updates": sum(
            r["update"] in old_history for r in records
        ),
        "exposure": exposure,
        "elapsed_this_invocation_seconds": perf_counter() - started,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs_sha256": {str(p.relative_to(output)): sha256_file(p) for p in paths},
    }
    write_json(completion, summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    torch.set_num_threads(config["torch_threads"])
    {"prepare": prepare, "train": train}[args.stage](
        config, ROOT / config["training_run"]
    )


if __name__ == "__main__":
    main()
