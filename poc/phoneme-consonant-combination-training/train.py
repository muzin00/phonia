#!/usr/bin/env python3
"""Prepare, freeze and train one JVS + CV five-vowel /m/ /n/ /s/ encoder."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PHASE3 = ROOT / "poc/phoneme-speaker-encoder"
EXPANSION = ROOT / "poc/phoneme-training-data-expansion"
sys.path.insert(0, str(PHASE3))
sys.path.insert(0, str(PHASE3 / "scripts"))
sys.path.insert(0, str(EXPANSION))

import torch
import training as expanded
from phase3_data.artifacts import compute_feature_statistics, write_json
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import (
    Segment,
    ids_checksum,
    json_sha256,
    load_segments,
    sha256_file,
)
from phase3_train.evaluation import evaluate_validation
from phase3_train.losses import AAMSoftmax
from phase3_train.models import create_encoder
from phase3_train.training import Trainer, _restore_rng, _rng_state, seed_everything
from run_phase3 import _trim_history_to_checkpoint
from sampling import PHONEMES, PhonemeSampler

CONFIG = BASE / "config/protocol.json"
read_json = expanded.read_json
checked = expanded.checked
read_rows = expanded.read_rows


def relative(path):
    return str(path.relative_to(ROOT))


def segment(row):
    return Segment(
        segment_id=row["vowel_interval_id"],
        speaker_id=row["speaker_id"],
        vowel=row["normalized_phoneme"],
        split="train",
        role="training",
        cohorts=(140,),
        source_file=row["source_file"],
        start_frame=row["start_frame"],
        end_frame=row["end_frame"],
        source_sha256=row["source_sha256"],
        quality_flags=tuple(row["quality_flags"]),
    )


def load_training(path):
    rows = list(read_rows(path))
    seen = set()
    for row in rows:
        if (
            row["split"] != "train"
            or row["evaluation_role"] != "training"
            or row["normalized_phoneme"] not in PHONEMES
            or row["learning_curve_cohorts"] != [140]
            or row["start_frame"] < 0
            or row["end_frame"] - row["start_frame"] < 720
            or row["quality_flags"]
            or row["vowel_interval_id"] in seen
            or not (ROOT / row["source_file"]).resolve().is_relative_to(ROOT)
        ):
            raise ValueError("invalid train-only phoneme segment")
        seen.add(row["vowel_interval_id"])
    result = [segment(row) for row in rows]
    split = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    speakers = {s.speaker_id for s in result}
    if (
        len(speakers) != 140
        or {s for s in speakers if not s.startswith("cv17_")} != set(split["train"])
        or speakers & (set(split["validation"]) | set(split["test"]))
    ):
        raise ValueError("speaker split mismatch")
    PhonemeSampler(result, 0)  # Require coverage before freezing inputs.
    return result


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(config, run):
    if run.exists():
        raise ValueError("preparation requires an unused run directory")
    prior = ROOT / config["prior_training_run"]
    review = ROOT / config["s_review_run"]
    checked(prior / "training-freeze.json", config["prior_training_freeze_sha256"])
    checked(review / "review-freeze.json", config["s_review_freeze_sha256"])
    inputs = {}
    for freeze_path in (prior / "training-freeze.json", review / "review-freeze.json"):
        for name, checksum in read_json(freeze_path)["files"].items():
            if inputs.setdefault(name, checksum) != checksum:
                raise ValueError("conflicting frozen input")
        inputs[relative(freeze_path)] = sha256_file(freeze_path)
    for path in (prior / "training-segments.jsonl", prior / "feature-statistics.json"):
        inputs[relative(path)] = sha256_file(path)
    for name, checksum in inputs.items():
        checked(ROOT / name, checksum)
    run.mkdir(parents=True)
    sources = [review / corpus / "eligible.jsonl" for corpus in ("jvs", "common-voice")]
    with (run / "s-eligible.jsonl").open("wb") as output:
        for path in sources:
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    s_rows = list(read_rows(run / "s-eligible.jsonl"))
    if any(
        r["normalized_phoneme"] != "s"
        or r["end_frame"] - r["start_frame"] > 6000
        or r["rms_dbfs"] < -50
        for r in s_rows
    ):
        raise ValueError("unexpected frozen /s/ candidate")
    with (run / "training-segments.jsonl").open("wb") as output:
        for path in (prior / "training-segments.jsonl", run / "s-eligible.jsonl"):
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    all_train = load_training(run / "training-segments.jsonl")
    coverage = Counter(
        f"{'CV' if s.speaker_id.startswith('cv17_') else 'JVS'}/{s.vowel}"
        for s in all_train
    )
    groups = Counter((s.speaker_id, s.vowel) for s in all_train)
    report = {
        "status": "prepared",
        "training_labels": 140,
        "training_segments": len(all_train),
        "additional_s_segments": len(s_rows),
        "coverage": dict(coverage),
        "minimum_segments_per_speaker_phoneme": min(groups.values()),
        "source_inputs": inputs,
        "test_used": False,
        "s_input_policy": "exact_frozen_post_qc_eligible_manifests; no_per_item_selection",
        "cross_corpus_person_identity_overlap": "unknown",
    }
    write_json(run / "preparation-report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_inputs"}),
        flush=True,
    )


def freeze(config, run):
    if (run / "training-freeze.json").exists():
        raise ValueError("training inputs are already frozen")
    prior = ROOT / config["prior_training_run"]
    original = read_json(prior / "feature-statistics.json")
    prior_ids = [
        r["vowel_interval_id"] for r in read_rows(prior / "training-segments.jsonl")
    ]
    if original["segment_ids_sha256"] != ids_checksum(prior_ids):
        raise ValueError("reused seven-phone statistics population mismatch")
    pipeline = InputPipeline(expanded.BASELINE["input"], rms_enabled=False)
    new_segments = [segment(r) for r in read_rows(run / "s-eligible.jsonl")]
    print(f"train-only /s/ feature statistics: {len(new_segments):,}", flush=True)
    s_stats = compute_feature_statistics(
        ROOT,
        new_segments,
        pipeline,
        cohort=140,
        manifest_sha256=sha256_file(run / "s-eligible.jsonl"),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        calculation_code_sha256=sha256_file(Path(__file__)),
    )
    write_json(run / "s-feature-statistics.json", s_stats)
    all_train = load_training(run / "training-segments.jsonl")
    stats = {
        **s_stats,
        **expanded.union_statistics(original, s_stats),
        "manifest_sha256": sha256_file(run / "training-segments.jsonl"),
        "segment_ids_sha256": ids_checksum(s.segment_id for s in all_train),
        "segment_count": len(all_train),
        "calculation": "merge_unchanged_seven_phone_and_new_s_train_only_moments",
        "source_statistics_sha256": [
            sha256_file(prior / "feature-statistics.json"),
            sha256_file(run / "s-feature-statistics.json"),
        ],
    }
    stats.pop("sha256")
    stats["sha256"] = json_sha256(stats)
    write_json(run / "feature-statistics.json", stats)
    files = read_json(run / "preparation-report.json")["source_inputs"]
    paths = {
        CONFIG,
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        EXPANSION / "training.py",
        *PHASE3.glob("phase3_data/*.py"),
        *PHASE3.glob("phase3_train/*.py"),
        *PHASE3.glob("config/*.json"),
        PHASE3 / "scripts/run_phase3.py",
        *(ROOT / config["validation_selections"]).rglob("*.json"),
        *(ROOT / config["validation_selections"]).rglob("*.jsonl"),
        *(
            run / name
            for name in (
                "preparation-report.json",
                "s-eligible.jsonl",
                "training-segments.jsonl",
                "feature-statistics.json",
                "s-feature-statistics.json",
            )
        ),
        ROOT / config["evaluation_manifest"],
    }
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).is_absolute():
            path = Path(filename).resolve()
            if (
                path.suffix == ".py"
                and path.is_file()
                and path.is_relative_to(ROOT)
                and ".venv" not in path.parts
            ):
                paths.add(path)
    for path in sorted(paths):
        name, checksum = relative(path), sha256_file(path)
        if files.setdefault(name, checksum) != checksum:
            raise ValueError("frozen source changed")
    for s in [
        *all_train,
        *load_segments(ROOT / config["evaluation_manifest"], split="validation"),
    ]:
        if files.setdefault(s.source_file, s.source_sha256) != s.source_sha256:
            raise ValueError("source checksum conflict")
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    write_json(
        run / "training-freeze.json",
        {
            "status": "frozen_before_training",
            "config": config,
            "files": files,
            "runtime": expanded.runtime(),
            "test_used": False,
            "cross_corpus_person_identity_overlap": "unknown; no reidentification attempted",
        },
    )
    print("all eight-phone inputs frozen", flush=True)


def make_trainer(config, run):
    seed_everything(config["seed"])
    segments = load_training(run / "training-segments.jsonl")
    pipeline = InputPipeline(
        expanded.BASELINE["input"],
        rms_enabled=False,
        statistics=read_json(run / "feature-statistics.json"),
    )
    settings = expanded.ExpansionSettings(
        encoder="statistics_mlp",
        cohort=140,
        rms_enabled=False,
        supcon_enabled=True,
        seed=config["seed"],
        maximum_updates=config["maximum_updates"],
        warmup_updates=config["warmup_updates"],
        validation_interval=config["validation_interval"],
    )
    model = create_encoder(settings.encoder)
    dataset = SegmentDataset(
        ROOT, segments, pipeline, run_seed=settings.seed, mode="random"
    )
    trainer = Trainer(
        model,
        AAMSoftmax(140),
        settings,
        dataset,
        PhonemeSampler(segments, settings.seed),
        manifest_sha256=sha256_file(run / "training-segments.jsonl"),
        feature_statistics_sha256=sha256_file(run / "feature-statistics.json"),
    )
    return trainer, pipeline


def train(config, run):
    frozen = read_json(run / "training-freeze.json")
    if frozen["config"] != config or frozen["runtime"] != expanded.runtime():
        raise ValueError("frozen protocol/runtime mismatch")
    for path, sha in frozen["files"].items():
        checked(ROOT / path, sha)
    output = run / "training"
    output.mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        summary = read_json(output / "summary.json")
        for name, sha in summary["outputs_sha256"].items():
            checked(run / name, sha)
        print("completed training outputs verified", flush=True)
        return
    trainer, pipeline = make_trainer(config, run)
    initial_sha = expanded.model_sha256(trainer.model)
    checkpoint = output / "last.pt"
    history = output / "history.jsonl"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
        _trim_history_to_checkpoint(history, trainer.update)
    elif history.exists() and history.stat().st_size:
        raise ValueError("history without checkpoint")
    start = time.perf_counter()
    diagnostics = []
    with history.open("a", encoding="utf-8") as stream:

        def record(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if row["update"] % 1000 == 0:
                print(json.dumps(row, allow_nan=False), flush=True)

        for target in range(
            config["validation_interval"],
            config["maximum_updates"] + 1,
            config["validation_interval"],
        ):
            if trainer.update > target:
                previous = read_json(
                    run / f"validation/update-{target:06d}/metrics/validation.json"
                )
                diagnostics.append(
                    {"update": target, "vowel_only_macro_eer": previous["macro_eer"]}
                )
                continue
            if trainer.update < target:
                trainer.train_until(target, checkpoint_dir=output, on_update=record)
            state = _rng_state()
            try:
                metrics = evaluate_validation(
                    trainer.model,
                    pipeline,
                    root=ROOT,
                    manifest=ROOT / config["evaluation_manifest"],
                    selection_root=ROOT / config["validation_selections"],
                    output_root=run / f"validation/update-{target:06d}",
                    device="cpu",
                    batch_size=32,
                )
            finally:
                _restore_rng(state)
            diagnostics.append(
                {"update": target, "vowel_only_macro_eer": metrics["macro_eer"]}
            )
            print(
                f"vowel-only diagnostic {target}: {100 * metrics['macro_eer']:.3f}% EER",
                flush=True,
            )
    if trainer.update != config["maximum_updates"]:
        raise ValueError("fixed training budget not completed")
    trainer.save_checkpoint(checkpoint)
    trainer.model.eval()
    bundle = run / "bundle"
    bundle.mkdir(exist_ok=True)
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "encoder": "statistics_mlp",
            "phonemes": PHONEMES,
            "embedding_dimension": 128,
            "checkpoint_update": trainer.update,
            "run_seed": config["seed"],
        },
        bundle / "encoder.pt",
    )
    shutil.copyfile(run / "feature-statistics.json", bundle / "feature-statistics.json")
    restored = create_encoder("statistics_mlp").eval()
    restored.load_state_dict(
        torch.load(bundle / "encoder.pt", weights_only=True)["model"], strict=True
    )
    probes = [
        next(s for s in trainer.dataset.segments if s.vowel == p) for p in PHONEMES
    ]
    probe = SegmentDataset(ROOT, probes, pipeline, mode="center")
    batch = collate_segments([probe[i] for i in range(len(probes))])
    with torch.inference_mode():
        before = trainer.model(batch["input"], batch["mask"])
        after = restored(batch["input"], batch["mask"])
    if (
        after.shape != (8, 128)
        or not torch.equal(before, after)
        or not torch.isfinite(after).all()
    ):
        raise ValueError("eight-phone export reload mismatch")
    for path, sha in frozen["files"].items():
        checked(ROOT / path, sha)
    phone_counts = Counter(
        p for u in range(trainer.update) for p in PhonemeSampler.phones_at(u)
    )
    summary = {
        "status": "completed",
        "settings": asdict(trainer.settings),
        "completed_updates": trainer.update,
        "training_examples": trainer.update * 100,
        "encoder_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "initial_encoder_sha256": initial_sha,
        "final_encoder_sha256": expanded.model_sha256(trainer.model),
        "selected_update": trainer.update,
        "selection": "fixed_final_update",
        "phoneme_microbatch_counts": dict(phone_counts),
        "speaker_selection_counts": trainer.sampler.state_at(trainer.update)[
            "speaker_counts"
        ],
        "validation_diagnostics": diagnostics,
        "export_reload_bitwise_equal_phonemes": list(PHONEMES),
        "elapsed_this_invocation_seconds": time.perf_counter() - start,
        "test_used": False,
        "comparison_completed": False,
        "freeze_sha256": sha256_file(run / "training-freeze.json"),
        "outputs_sha256": {
            relative(p).removeprefix(relative(run) + "/"): sha256_file(p)
            for p in (
                checkpoint,
                history,
                bundle / "encoder.pt",
                bundle / "feature-statistics.json",
                *(run / "validation").rglob("validation.json"),
            )
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "train"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    if (
        config["phonemes"] != list(PHONEMES)
        or config["corpora"] != ["JVS", "Common Voice"]
        or config["encoder"] != "statistics_mlp"
        or config["training_speakers"] != 140
        or config["batch_size"] != 100
        or config["phoneme_groups_per_update"] != 5
        or config["rms_enabled"]
        or not config["supcon_enabled"]
        or config["test_used"]
        or config["early_stopping"]
        or config["device"] != "cpu"
        or config["workers"] != 0
        or config["minimum_frames"] != 720
        or config["maximum_updates"] % config["validation_interval"]
    ):
        raise ValueError("unsupported training protocol")
    torch.set_num_threads(config["torch_threads"])
    {"prepare": prepare, "freeze": freeze, "train": train}[args.stage](
        config, ROOT / config["run_directory"]
    )


if __name__ == "__main__":
    main()
