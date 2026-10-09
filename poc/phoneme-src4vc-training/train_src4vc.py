"""Train one fixed SRC4VC corpus replacement with the frozen Phase 3 machinery."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE_BASE = BASE.parent / "phoneme-training-data-expansion"
sys.path.insert(0, str(SOURCE_BASE))
import torch
from training import (
    BASELINE,
    PHASE3,
    AAMSoftmax,
    BalancedSampler,
    ExpansionSettings,
    InputPipeline,
    Segment,
    SegmentDataset,
    Trainer,
    _compact_validation_outputs,
    _trim_history_to_checkpoint,
    checked,
    collate_segments,
    compute_feature_statistics,
    create_encoder,
    evaluate_validation,
    ids_checksum,
    json_sha256,
    load_feature_statistics,
    load_segments,
    model_sha256,
    read_json,
    read_rows,
    runtime,
    seed_everything,
    sha256_file,
    union_statistics,
    write_json,
)

CONFIG = BASE / "config/protocol.json"


def validate_protocol(config):
    if (
        config["maximum_updates"] != 30000
        or config["checkpoint_selection"] != "fixed_30000_selected_before_training"
        or config["early_stopping"] != "disabled"
        or config["seed"] != 20260926
        or config["warmup_updates"] != 1000
        or config["new_training_conditions"] != 1
        or config["test_used"]
        or config["train_reserved_src4vc_speakers"]
        or config["rms_enabled"]
        or not config["supcon_enabled"]
        or config["torch_threads"] != 1
        or config["initialization"] != "from_scratch_same_seed_as_CV_reference"
        or config["additional_speaker_prefix"] != "ext_src4vc"
        or config["encoder"] != "statistics_mlp"
        or config["workers"] != 0
        or config["device"] != "cpu"
        or config["evaluation_batch_size"] != 32
        or config["validation_interval"] != 1000
    ):
        raise ValueError("unsupported fixed single-condition protocol")


def verify_exposure(current, baseline):
    jvs = {s: n for s, n in current.items() if s.startswith("jvs")}
    additional = {s: n for s, n in current.items() if s.startswith("ext_src4vc")}
    if (
        len(jvs) != 70
        or len(additional) != 70
        or set(jvs) != set(baseline)
        or len(current) != 140
    ):
        raise ValueError("training speaker populations changed")
    if any(abs(jvs[s] - baseline[s]) > 1 for s in jvs):
        raise ValueError("JVS speaker exposure differs from baseline")
    if max(current.values()) - min(current.values()) > 1:
        raise ValueError("unbalanced speaker exposure")
    return {
        "jvs_label_batch_count_min": min(jvs.values()),
        "jvs_label_batch_count_max": max(jvs.values()),
        "jvs_label_batch_count_total": sum(jvs.values()),
        "src4vc_label_batch_count_total": sum(additional.values()),
        "maximum_jvs_difference_from_baseline": max(
            abs(jvs[s] - baseline[s]) for s in jvs
        ),
    }


def load_training(path):
    result, seen = [], set()
    for row in read_rows(path):
        if (
            row["split"] != "train"
            or row["evaluation_role"] != "training"
            or row["learning_curve_cohorts"] != [140]
            or row["normalized_phoneme"] not in "aiueo"
            or row["end_frame"] - row["start_frame"] < 720
            or row["start_frame"] < 0
            or set(row["quality_flags"]) & {"silent", "near_silent"}
            or row["vowel_interval_id"] in seen
            or not (ROOT / row["source_file"]).resolve().is_relative_to(ROOT)
            or (row["sample_rate_hz"], row["channels"], row["sample_width_bytes"])
            != (24000, 1, 2)
        ):
            raise ValueError("invalid expanded train-only segment")
        seen.add(row["vowel_interval_id"])
        result.append(
            Segment(
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
                is_devoiced=row.get("is_devoiced"),
                is_long=row.get("is_long"),
                quality_flags=tuple(row["quality_flags"]),
            )
        )
    speakers = {s.speaker_id for s in result}
    if len(speakers) != 140 or sum(s.startswith("ext_src4vc") for s in speakers) != 70:
        raise ValueError("expected JVS70 plus SRC4VC70")
    split = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    reserved = read_json(path.parent / "speaker-splits.json")["splits"]
    expected_src = {"ext_src4vc" + s[6:] for s in reserved["train"]}
    if speakers != set(split["train"]) | expected_src:
        raise ValueError("train speaker identity/split mismatch")
    return result


def statistics(config, run):
    validate_protocol(config)
    report = read_json(run / "preparation-report.json")
    if report["status"] != "completed":
        raise ValueError("complete data preparation required")
    for name, checksum in report["outputs_sha256"].items():
        checked(run / name, checksum)
    canonical = ROOT / config["evaluation_manifest"]
    checked(canonical, config["evaluation_manifest_sha256"])
    destination = run / "training-segments.jsonl"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in read_rows(canonical):
            if row["split"] != "train":
                continue
            row["learning_curve_cohorts"] = [140]
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        for row in read_rows(run / "additional-segments.jsonl"):
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if destination.exists():
        checked(temporary, sha256_file(destination))
    temporary.replace(destination)
    all_train = load_training(destination)
    split = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    jvs = {s.speaker_id for s in all_train if not s.speaker_id.startswith("ext_src4vc")}
    if jvs != set(split["train"]) or jvs & (
        set(split["validation"]) | set(split["test"])
    ):
        raise ValueError("JVS training/evaluation speaker leak")
    original = ROOT / config["baseline_statistics"]
    checked(original, config["baseline_statistics_sha256"])
    jvs_statistics = read_json(original)
    jvs_train = [s for s in all_train if s.speaker_id in jvs]
    if jvs_statistics["segment_ids_sha256"] != ids_checksum(
        s.segment_id for s in jvs_train
    ) or jvs_statistics["speaker_ids_sha256"] != ids_checksum(jvs):
        raise ValueError("baseline train statistics population changed")
    additional = [
        replace(s, cohorts=(70,))
        for s in all_train
        if s.speaker_id.startswith("ext_src4vc")
    ]
    unnormalized = InputPipeline(BASELINE["input"], rms_enabled=False)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    additional_path = run / "additional-feature-statistics.json"
    if additional_path.exists():
        additional_statistics = load_feature_statistics(
            additional_path,
            manifest_sha256=sha256_file(run / "additional-segments.jsonl"),
            cohort=70,
            speaker_ids_sha256=ids_checksum(s.speaker_id for s in additional),
            segment_ids_sha256=ids_checksum(s.segment_id for s in additional),
            preprocessing_sha256=unnormalized.preprocessing_sha256,
        )
    else:
        print(
            f"train-only feature statistics: {len(additional):,} additional segments",
            flush=True,
        )
        additional_statistics = compute_feature_statistics(
            ROOT,
            additional,
            unnormalized,
            cohort=70,
            manifest_sha256=sha256_file(run / "additional-segments.jsonl"),
            git_commit=git_commit,
            calculation_code_sha256=sha256_file(Path(__file__)),
        )
        write_json(additional_path, additional_statistics)
    joint = {
        "schema_version": 1,
        "design_version": "2.0.0",
        "cohort": 140,
        "rms_enabled": False,
        "manifest_sha256": sha256_file(destination),
        "speaker_ids_sha256": ids_checksum(s.speaker_id for s in all_train),
        "segment_ids_sha256": ids_checksum(s.segment_id for s in all_train),
        "preprocessing_sha256": unnormalized.preprocessing_sha256,
        "segment_count": len(all_train),
        "speaker_count": 140,
        **union_statistics(jvs_statistics, additional_statistics),
        "calculation_git_commit": git_commit,
        "calculation_code_sha256": sha256_file(Path(__file__)),
        "calculation": "merge_disjoint_JVS70_and_SRC4VC70_train_frame_population_moments",
        "source_statistics_sha256": [
            sha256_file(original),
            sha256_file(additional_path),
        ],
    }
    joint["sha256"] = json_sha256(joint)
    write_json(run / "feature-statistics.json", joint)
    # Pin code, protocol, selections and every train WAV before training starts.
    paths = {
        CONFIG,
        destination,
        run / "feature-statistics.json",
        run / "preparation-report.json",
        run / "preparation-freeze.json",
        original,
        canonical,
        ROOT / config["baseline_run"] / "run.json",
        ROOT / config["baseline_run"] / "training/summary.json",
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        *SOURCE_BASE.glob("*.py"),
        *(ROOT / "poc/phoneme-speaker-dataset/scripts").glob("*.py"),
        ROOT / "poc/phoneme-speaker-dataset/config/vowel-dataset.json",
        run / "additional-feature-statistics.json",
        ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json",
        run / "selection.json",
        run / "speaker-splits.json",
        run / "source/manifest.json",
        run / "source/SRC4VC_ver1.zip",
        ROOT / config["reference_training_summary"],
        ROOT / config["baseline_run"] / "checkpoints/best.pt",
        ROOT / config["reference_initialization"],
        *PHASE3.glob("phase3_data/*.py"),
        *PHASE3.glob("phase3_train/*.py"),
        *PHASE3.glob("config/*.json"),
        PHASE3 / "scripts/run_phase3.py",
        *(ROOT / config["validation_selections"]).rglob("*.json"),
        *(ROOT / config["validation_selections"]).rglob("*.jsonl"),
    }
    reference = ROOT / config["reference_training_summary"]
    checked(reference, config["reference_training_summary_sha256"])
    state = torch.load(
        ROOT / config["baseline_run"] / "checkpoints/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if state["update"] != 15000:
        raise ValueError("baseline exposure reference changed")
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}
    for segment in all_train:
        old = files.setdefault(segment.source_file, segment.source_sha256)
        if old != segment.source_sha256:
            raise ValueError("conflicting training source checksum")
    # Evaluation audio is pinned too, without reading test waveforms.
    for segment in load_segments(canonical, split="validation"):
        old = files.setdefault(segment.source_file, segment.source_sha256)
        if old != segment.source_sha256:
            raise ValueError("conflicting validation source checksum")
    for name, checksum in files.items():
        checked(ROOT / name, checksum)
    write_json(
        run / "training-freeze.json",
        {
            "status": "frozen_before_training",
            "config": config,
            "files": files,
            "speaker_count": 140,
            "training_segments": len(all_train),
            "runtime": runtime(),
            "baseline_sampler_counts": state["sampler"]["speaker_counts"],
            "test_used": False,
            "validation_speakers_used_for_training": False,
            "cross_corpus_person_identity_overlap": "unknown; pseudonymous_labels_are_not_verified_person_identities",
        },
    )
    print(
        f"training inputs frozen: 140 labels, {len(all_train):,} segments", flush=True
    )


def make_trainer(config, run):
    seed_everything(config["seed"])
    segments = load_training(run / "training-segments.jsonl")
    pipeline = InputPipeline(
        BASELINE["input"],
        rms_enabled=False,
        statistics=read_json(run / "feature-statistics.json"),
    )
    settings = ExpansionSettings(
        encoder=config["encoder"],
        cohort=140,
        rms_enabled=False,
        supcon_enabled=True,
        seed=config["seed"],
        maximum_updates=config["maximum_updates"],
        warmup_updates=config["warmup_updates"],
        validation_interval=config["validation_interval"],
        workers=0,
        device="cpu",
    )
    model = create_encoder(settings.encoder)
    dataset = SegmentDataset(
        ROOT,
        segments,
        pipeline,
        kind=model.input_kind,
        run_seed=settings.seed,
        mode="random",
    )
    trainer = Trainer(
        model,
        AAMSoftmax(140),
        settings,
        dataset,
        BalancedSampler(segments, settings.seed),
        manifest_sha256=sha256_file(run / "training-segments.jsonl"),
        feature_statistics_sha256=sha256_file(run / "feature-statistics.json"),
    )
    return trainer, pipeline


def train(config, run):
    validate_protocol(config)
    freeze = read_json(run / "training-freeze.json")
    if (
        freeze["status"] != "frozen_before_training"
        or freeze["config"] != config
        or freeze["runtime"] != runtime()
    ):
        raise ValueError("frozen training design/runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    output = run / "expanded"
    completion = output / "training/summary.json"
    if completion.exists():
        summary = read_json(completion)
        if (
            summary["status"] != "completed"
            or summary["selected_update"] != config["maximum_updates"]
        ):
            raise ValueError("invalid existing training completion")
        for name, checksum in summary["outputs_sha256"].items():
            checked(output / name, checksum)
        print(
            "training already completed; frozen inputs and outputs verified", flush=True
        )
        return
    trainer, pipeline = make_trainer(config, run)
    initial_hash = model_sha256(trainer.model)
    reference = read_json(ROOT / config["reference_initialization"])
    if (
        initial_hash != reference["initial_encoder_sha256"]
        or asdict(trainer.settings) != reference["settings"]
    ):
        raise ValueError("initialization/settings differ from CV reference")
    expected_jvs_classes = {
        s: i + 70 for i, s in enumerate(sorted(freeze["baseline_sampler_counts"]))
    }
    if any(trainer.speaker_to_class[s] != i for s, i in expected_jvs_classes.items()):
        raise ValueError("JVS classifier row ordering differs from CV reference")
    output.mkdir(exist_ok=True)
    descriptor = {
        "settings": asdict(trainer.settings),
        "configuration_sha256": trainer.settings.sha256,
        "freeze_sha256": sha256_file(run / "training-freeze.json"),
        "initialization": config["initialization"],
        "initial_encoder_sha256": initial_hash,
        "encoder_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "training_head_parameters": trainer.head.weight.numel(),
        "test_used": False,
    }
    if (
        descriptor["encoder_parameters"] != 65920
        or descriptor["training_head_parameters"] != 17920
    ):
        raise ValueError("encoder/head size changed")
    if (output / "run.json").exists() and read_json(output / "run.json") != descriptor:
        raise ValueError("existing run belongs to different frozen inputs")
    write_json(output / "run.json", descriptor)
    history = output / "training/history.jsonl"
    history.parent.mkdir(exist_ok=True)
    checkpoint = output / "checkpoints/last.pt"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
        _trim_history_to_checkpoint(history, trainer.update)
    elif history.exists() and history.stat().st_size:
        raise ValueError("training history exists without resumable checkpoint")
    started = perf_counter()
    with history.open("a", encoding="utf-8") as stream:

        def record(value):
            u = value["update"]
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            if u % 100 == 0:
                print(json.dumps(value, allow_nan=False), flush=True)
            if u % trainer.settings.validation_interval == 0:
                _compact_validation_outputs(output, config["maximum_updates"])
                metrics = evaluate_validation(
                    trainer.model,
                    pipeline,
                    root=ROOT,
                    manifest=ROOT / config["evaluation_manifest"],
                    selection_root=ROOT / config["validation_selections"],
                    output_root=output / f"validation/update-{u:06d}",
                    device="cpu",
                    batch_size=32,
                )
                print(
                    f"diagnostic validation {u}: macro EER {100 * metrics['macro_eer']:.3f}%; fixed selection remains30000",
                    flush=True,
                )

        if trainer.update < config["maximum_updates"]:
            trainer.train_until(
                config["maximum_updates"],
                on_validation=None,
                checkpoint_dir=output / "checkpoints",
                on_update=record,
            )
    if trainer.update != config["maximum_updates"]:
        raise ValueError("fixed training budget not reached")
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
    shutil.copyfile(run / "feature-statistics.json", bundle / "feature-statistics.json")
    trainer.model.eval()
    restored = create_encoder("statistics_mlp").eval()
    restored.load_state_dict(
        torch.load(bundle / "encoder.pt", map_location="cpu", weights_only=True)[
            "model"
        ],
        strict=True,
    )
    selected = sorted(
        load_segments(
            ROOT / config["evaluation_manifest"],
            split="validation",
            role="verification",
        ),
        key=lambda s: s.segment_id,
    )[:8]
    probe = SegmentDataset(ROOT, selected, pipeline, kind="log_mel", mode="center")
    batch = collate_segments([probe[i] for i in range(8)])
    with torch.inference_mode():
        before = trainer.model(batch["input"], batch["mask"])
        after = restored(batch["input"], batch["mask"])
    if (
        before.shape != (8, 128)
        or not torch.equal(before, after)
        or not torch.isfinite(after).all()
    ):
        raise ValueError("export/reload inference differs")
    if not all(
        torch.isfinite(v).all()
        for module in (trainer.model, trainer.head)
        for v in module.state_dict().values()
    ):
        raise ValueError("nonfinite final checkpoint")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    metrics_path = (
        output / f"validation/update-{trainer.update:06d}/metrics/validation.json"
    )
    metrics = read_json(metrics_path)
    if metrics["partial"]:
        raise ValueError("requires full JVS validation")
    records = list(read_rows(history))
    if [r["update"] for r in records] != list(range(1, config["maximum_updates"] + 1)):
        raise ValueError("incomplete training history")
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
        "validation_macro_eer": metrics["macro_eer"],
        "selection_rule": config["checkpoint_selection"],
        "encoder_parameters": 65920,
        "training_head_parameters": 17920,
        "export_reload_bitwise_equal_validation_segments": 8,
        "test_used": False,
        "new_training_conditions": 1,
        "exposure": exposure,
        "elapsed_this_invocation_seconds": perf_counter() - started,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs_sha256": {str(p.relative_to(output)): sha256_file(p) for p in paths},
    }
    write_json(completion, summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("statistics", "train"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    torch.set_num_threads(config["torch_threads"])
    {"statistics": statistics, "train": train}[args.stage](
        config, ROOT / config["run_directory"]
    )


if __name__ == "__main__":
    main()
