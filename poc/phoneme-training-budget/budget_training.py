"""Two from-scratch 60k trajectories with immutable 30k/45k/60k snapshots."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-src4vc-training"))
import torch
import train_src4vc as src
import training as cv
from phase3_train.training import _restore_rng, _rng_state

CONFIG = BASE / "config/protocol.json"
read_json, checked, sha256_file, write_json = (
    cv.read_json,
    cv.checked,
    cv.sha256_file,
    cv.write_json,
)


def validate_protocol(config):
    expected = {
        "corpora": ["cv", "src"],
        "snapshot_updates": [30000, 45000, 60000],
        "maximum_updates": 60000,
        "warmup_updates": 1000,
        "validation_interval": 1000,
        "seed": 20260926,
        "torch_threads": 1,
        "device": "cpu",
        "workers": 0,
        "encoder": "statistics_mlp",
        "cohort": 140,
        "early_stopping": False,
        "initialization": "from_scratch_same_initial_encoder_and_head_seed",
        "schedule": "linear_warmup_cosine_to_60000",
        "feature_statistics": "reuse_each_corpus_frozen_train_only_statistics",
        "data_selection": "reuse_each_corpus_frozen_training_segments",
        "test_checkpoint_selection": "all_three_updates_preregistered_before_training_no_test_selection",
        "test_threshold_recalibration": False,
        "test_used_for_training": False,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported fixed 60k training trajectory protocol")


def derived_config(config, source_freeze):
    result = dict(source_freeze["config"])
    result["maximum_updates"] = config["maximum_updates"]
    # make_trainer uses only model/optimizer/input settings; source selection
    # metadata remains historical, while the new protocol owns adoption rules.
    return result


def validate_settings(actual, reference):
    expected = {**reference, "maximum_updates": 60000}
    if actual != expected:
        raise ValueError("only maximum_updates may differ from source training")


@contextmanager
def preserve_rng():
    state = _rng_state()
    try:
        yield
    finally:
        _restore_rng(state)


def merge_files(destination, additions):
    for name, digest in additions.items():
        if name in destination and destination[name] != digest:
            raise ValueError(f"conflicting source checksum: {name}")
        destination[name] = digest


def check_files(files):
    for name, digest in files.items():
        checked(ROOT / name, digest)


def prepare(config, run):
    validate_protocol(config)
    if not run.resolve().is_relative_to(ROOT / "artifacts"):
        raise ValueError("training artifacts must stay under artifacts")
    freeze_path = run / "training-freeze.json"
    if freeze_path.exists():
        frozen(config, run)
        print("existing frozen training inputs verified", flush=True)
        return
    if run.exists():
        raise ValueError("unused training output directory required")
    files, sources = {}, {}
    split = read_json(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    for corpus in config["corpora"]:
        source = ROOT / config["source_runs"][corpus]
        checked(source / "training-freeze.json", config["source_freeze_sha256"][corpus])
        freeze = read_json(source / "training-freeze.json")
        if (
            freeze["status"] != "frozen_before_training"
            or freeze["runtime"] != cv.runtime()
            or freeze["test_used"]
        ):
            raise ValueError("source training freeze/runtime changed")
        merge_files(files, freeze["files"])
        module = cv if corpus == "cv" else src
        segments = module.load_training(source / "training-segments.jsonl")
        speakers = {s.speaker_id for s in segments}
        jvs = {s for s in speakers if s.startswith("jvs")}
        if jvs != set(split["train"]) or speakers & (
            set(split["validation"]) | set(split["test"])
        ):
            raise ValueError("training/evaluation speaker overlap")
        descriptor = read_json(source / "expanded/run.json")
        sources[corpus] = {
            "source_run": str(source.relative_to(ROOT)),
            "segment_count": len(segments),
            "speaker_count": len(speakers),
            "initial_encoder_sha256": descriptor["initial_encoder_sha256"],
            "reference_settings": descriptor["settings"],
            "train_speakers_sha256": cv.ids_checksum(speakers),
            "train_segments_sha256": cv.ids_checksum(s.segment_id for s in segments),
            "feature_statistics_sha256": sha256_file(
                source / "feature-statistics.json"
            ),
            "training_manifest_sha256": sha256_file(source / "training-segments.jsonl"),
        }
        for p in (source / "training-freeze.json", source / "expanded/run.json"):
            files[str(p.relative_to(ROOT))] = sha256_file(p)
    if (
        sources["cv"]["initial_encoder_sha256"]
        != sources["src"]["initial_encoder_sha256"]
    ):
        raise ValueError("corpus initial encoders differ")
    for p in (CONFIG, Path(__file__), *BASE.glob("tests/test_training.py")):
        files[str(p.relative_to(ROOT))] = sha256_file(p)
    check_files(files)
    run.mkdir(parents=True)
    write_json(
        freeze_path,
        {
            "status": "frozen_before_training",
            "config": config,
            "runtime": cv.runtime(),
            "sources": sources,
            "files": files,
            "test_used": False,
            "validation_speakers_used_for_training": False,
        },
    )
    print(f"two 60k trajectories frozen; {len(files)} input checksums", flush=True)


def frozen(config, run):
    freeze = read_json(run / "training-freeze.json")
    if (
        freeze["status"] != "frozen_before_training"
        or freeze["config"] != config
        or freeze["runtime"] != cv.runtime()
    ):
        raise ValueError("frozen training protocol/runtime changed")
    check_files(freeze["files"])
    return freeze


def export_snapshot(trainer, pipeline, config, output, source, update):
    destination = output / f"snapshots/update-{update:06d}"
    if (destination / "snapshot.json").exists():
        record = read_json(destination / "snapshot.json")
        for name, digest in record["outputs_sha256"].items():
            checked(destination / name, digest)
        if record["encoder_tensor_sha256"] != cv.model_sha256(trainer.model):
            raise ValueError("existing snapshot weights differ from current trajectory")
        return
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / "bundle"
    bundle.mkdir(exist_ok=True)
    trainer.save_checkpoint(destination / "checkpoint.pt")
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "encoder": "statistics_mlp",
            "embedding_dimension": 128,
            "checkpoint_update": update,
            "run_seed": trainer.settings.seed,
            "encoder_parameters": 65920,
            "configuration_sha256": trainer.settings.sha256,
        },
        bundle / "encoder.pt",
    )
    shutil.copyfile(
        source / "feature-statistics.json", bundle / "feature-statistics.json"
    )
    # Export verification must not consume randomness from subsequent updates.
    with preserve_rng():
        restored = cv.create_encoder("statistics_mlp").eval()
        restored.load_state_dict(
            torch.load(bundle / "encoder.pt", map_location="cpu", weights_only=True)[
                "model"
            ]
        )
        selected = sorted(
            cv.load_segments(
                ROOT / config["evaluation_manifest"],
                split="validation",
                role="verification",
            ),
            key=lambda s: s.segment_id,
        )[:8]
        probe = cv.SegmentDataset(
            ROOT, selected, pipeline, kind="log_mel", mode="center"
        )
        batch = cv.collate_segments([probe[i] for i in range(8)])
        trainer.model.eval()
        with torch.inference_mode():
            before = trainer.model(batch["input"], batch["mask"])
            after = restored(batch["input"], batch["mask"])
        if (
            before.shape != (8, 128)
            or not torch.equal(before, after)
            or not torch.isfinite(after).all()
        ):
            raise ValueError("snapshot export/reload inference differs")
    counts = trainer.sampler.state_at(update)["speaker_counts"]
    if len(counts) != 140 or max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("unbalanced training exposure")
    files = [destination / "checkpoint.pt", *bundle.glob("*")]
    metrics = output / f"validation/update-{update:06d}/metrics/validation.json"
    write_json(
        destination / "snapshot.json",
        {
            "status": "completed",
            "update": update,
            "maximum_updates": 60000,
            "validation_macro_eer": read_json(metrics)["macro_eer"],
            "encoder_tensor_sha256": cv.model_sha256(trainer.model),
            "export_reload_bitwise_equal_validation_segments": 8,
            "speaker_counts": counts,
            "test_used": False,
            "outputs_sha256": {
                str(p.relative_to(destination)): sha256_file(p) for p in files
            },
        },
    )


def train(config, run, corpus):
    validate_protocol(config)
    freeze = frozen(config, run)
    output = run / corpus
    completion = output / "training/summary.json"
    if completion.exists():
        summary = read_json(completion)
        if summary["status"] != "completed" or summary["completed_updates"] != 60000:
            raise ValueError("invalid existing completion")
        for name, digest in summary["outputs_sha256"].items():
            checked(output / name, digest)
        print(f"{corpus}: complete training output verified", flush=True)
        return
    source = ROOT / config["source_runs"][corpus]
    source_freeze = read_json(source / "training-freeze.json")
    settings_config = derived_config(config, source_freeze)
    module = cv if corpus == "cv" else src
    trainer, pipeline = module.make_trainer(settings_config, source)
    initial_hash = cv.model_sha256(trainer.model)
    initial_head = cv.model_sha256(trainer.head)
    validate_settings(
        asdict(trainer.settings), freeze["sources"][corpus]["reference_settings"]
    )
    if initial_hash != freeze["sources"][corpus]["initial_encoder_sha256"]:
        raise ValueError("initial encoder differs from source")
    jvs = sorted(s for s in trainer.speaker_to_class if s.startswith("jvs"))
    if any(trainer.speaker_to_class[s] != i + 70 for i, s in enumerate(jvs)):
        raise ValueError("JVS classifier row ordering changed")
    descriptor = {
        "corpus": corpus,
        "settings": asdict(trainer.settings),
        "configuration_sha256": trainer.settings.sha256,
        "freeze_sha256": sha256_file(run / "training-freeze.json"),
        "initial_encoder_sha256": initial_hash,
        "initial_head_sha256": initial_head,
        "encoder_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "training_head_parameters": trainer.head.weight.numel(),
        "test_used": False,
    }
    if (
        descriptor["encoder_parameters"] != 65920
        or descriptor["training_head_parameters"] != 17920
    ):
        raise ValueError("encoder or classifier size changed")
    output.mkdir(exist_ok=True)
    if (output / "run.json").exists() and read_json(output / "run.json") != descriptor:
        raise ValueError("existing trajectory belongs to different inputs")
    write_json(output / "run.json", descriptor)
    history = output / "training/history.jsonl"
    history.parent.mkdir(exist_ok=True)
    checkpoint = output / "checkpoints/last.pt"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
        cv._trim_history_to_checkpoint(history, trainer.update)
    elif history.exists() and history.stat().st_size:
        raise ValueError("history exists without resumable checkpoint")
    started = perf_counter()
    with history.open("a", encoding="utf-8") as stream:

        def record(value):
            update = value["update"]
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            if update % 1000 == 0:
                print(f"{corpus}: {json.dumps(value, allow_nan=False)}", flush=True)
                cv._compact_validation_outputs(output, 60000)
                metrics = cv.evaluate_validation(
                    trainer.model,
                    pipeline,
                    root=ROOT,
                    manifest=ROOT / settings_config["evaluation_manifest"],
                    selection_root=ROOT / settings_config["validation_selections"],
                    output_root=output / f"validation/update-{update:06d}",
                    device="cpu",
                    batch_size=32,
                )
                print(
                    f"{corpus}: validation {update} EER {100 * metrics['macro_eer']:.4f}%",
                    flush=True,
                )
                if update in config["snapshot_updates"]:
                    export_snapshot(
                        trainer, pipeline, settings_config, output, source, update
                    )

        trainer.train_until(
            60000,
            on_validation=None,
            checkpoint_dir=output / "checkpoints",
            on_update=record,
        )
    if trainer.update != 60000:
        raise ValueError("full fixed training budget not reached")
    check_files(freeze["files"])
    records = list(cv.read_rows(history))
    if [r["update"] for r in records] != list(range(1, 60001)):
        raise ValueError("incomplete training update history")
    paths = [checkpoint, history, output / "run.json"]
    paths += [p for p in (output / "snapshots").rglob("*") if p.is_file()]
    paths += list((output / "validation").glob("*/metrics/validation.json"))
    summary = {
        "status": "completed",
        "corpus": corpus,
        "completed_updates": 60000,
        "snapshot_updates": config["snapshot_updates"],
        "test_used": False,
        "elapsed_this_invocation_seconds": perf_counter() - started,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs_sha256": {str(p.relative_to(output)): sha256_file(p) for p in paths},
    }
    write_json(completion, summary)
    print(f"{corpus}: 60k training and all three snapshots complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--corpus", choices=("cv", "src"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    torch.set_num_threads(config["torch_threads"])
    run = ROOT / config["run_directory"]
    if args.stage == "prepare":
        prepare(config, run)
    elif args.corpus:
        train(config, run, args.corpus)
    else:
        parser.error("train requires --corpus")


if __name__ == "__main__":
    main()
