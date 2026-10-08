"""One from-scratch expanded training run with the unchanged Phase 3 encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PHASE3 = ROOT / "poc/phoneme-speaker-encoder"
sys.path.insert(0, str(PHASE3))
sys.path.insert(0, str(PHASE3 / "scripts"))

import numpy as np
import torch
from phase3_data.artifacts import compute_feature_statistics, write_json
from phase3_data.input import (
    InputPipeline,
    SegmentDataset,
    collate_segments,
    load_feature_statistics,
)
from phase3_data.manifest import (
    Segment,
    ids_checksum,
    json_sha256,
    load_segments,
    sha256_file,
)
from phase3_data.sampling import BalancedSampler
from phase3_train.evaluation import evaluate_validation
from phase3_train.losses import AAMSoftmax
from phase3_train.models import create_encoder
from phase3_train.training import Trainer, TrainSettings, seed_everything
from run_phase3 import _compact_validation_outputs, _trim_history_to_checkpoint

CONFIG = BASE / "config/protocol.json"
BASELINE = json.loads((PHASE3 / "config/baseline-log-mel.json").read_text())


@dataclass(frozen=True)
class ExpansionSettings(TrainSettings):
    """Allow this experiment's 140 labels without changing frozen Phase 3 code."""

    def __post_init__(self):
        if self.cohort != 140 or self.overfit:
            raise ValueError("this experiment requires 140 training speaker labels")
        # Reuse all original optimizer/budget validation. Actual settings and
        # checkpoint configuration hashes retain the expanded cohort of 140.
        TrainSettings(**{**asdict(self), "cohort": 70})


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked(path, checksum):
    if sha256_file(path) != checksum:
        raise ValueError(f"checksum mismatch: {path}")


def runtime():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def model_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def union_statistics(first, second):
    """Combine disjoint train-frame population moments, including mean shift."""
    n, m = first["frame_count"], second["frame_count"]
    if n <= 0 or m <= 0:
        raise ValueError("both frame populations must be nonempty")
    a, b = (
        np.array(first["mean"], dtype=np.float64),
        np.array(second["mean"], dtype=np.float64),
    )
    va = np.array(first["population_variance"], dtype=np.float64)
    vb = np.array(second["population_variance"], dtype=np.float64)
    if (
        a.shape != (64,)
        or b.shape != (64,)
        or va.shape != (64,)
        or vb.shape != (64,)
        or not all(np.isfinite(x).all() for x in (a, b, va, vb))
        or np.any(va < 0)
        or np.any(vb < 0)
    ):
        raise ValueError("invalid frame moments")
    delta = b - a
    mean = a + delta * (m / (n + m))
    variance = (n * va + m * vb + delta**2 * (n * m / (n + m))) / (n + m)
    return {
        "frame_count": n + m,
        "mean": mean.tolist(),
        "population_variance": variance.tolist(),
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
            or "near_silent" in row["quality_flags"]
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
    if len(speakers) != 140 or sum(s.startswith("cv17_") for s in speakers) != 70:
        raise ValueError("expected JVS70 plus CV70")
    return result


def statistics(config, run):
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
    jvs = {s.speaker_id for s in all_train if not s.speaker_id.startswith("cv17_")}
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
        replace(s, cohorts=(70,)) for s in all_train if s.speaker_id.startswith("cv17_")
    ]
    unnormalized = InputPipeline(BASELINE["input"], rms_enabled=False)
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    cv_path = run / "additional-feature-statistics.json"
    if cv_path.exists():
        cv_statistics = load_feature_statistics(
            cv_path,
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
        cv_statistics = compute_feature_statistics(
            ROOT,
            additional,
            unnormalized,
            cohort=70,
            manifest_sha256=sha256_file(run / "additional-segments.jsonl"),
            git_commit=git_commit,
            calculation_code_sha256=sha256_file(Path(__file__)),
        )
        write_json(cv_path, cv_statistics)
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
        **union_statistics(jvs_statistics, cv_statistics),
        "calculation_git_commit": git_commit,
        "calculation_code_sha256": sha256_file(Path(__file__)),
        "calculation": "merge_disjoint_JVS70_and_CV70_train_frame_population_moments",
        "source_statistics_sha256": [sha256_file(original), sha256_file(cv_path)],
    }
    joint["sha256"] = json_sha256(joint)
    write_json(run / "feature-statistics.json", joint)
    # Pin code, protocol, selections and every train WAV before training starts.
    paths = {
        CONFIG,
        destination,
        run / "feature-statistics.json",
        run / "preparation-report.json",
        original,
        canonical,
        ROOT / config["baseline_run"] / "run.json",
        ROOT / config["baseline_run"] / "training/summary.json",
        *BASE.glob("*.py"),
        *PHASE3.glob("phase3_data/*.py"),
        *PHASE3.glob("phase3_train/*.py"),
        *PHASE3.glob("config/*.json"),
        PHASE3 / "scripts/run_phase3.py",
        *(ROOT / config["validation_selections"]).rglob("*.json"),
        *(ROOT / config["validation_selections"]).rglob("*.jsonl"),
    }
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
    freeze = read_json(run / "training-freeze.json")
    if (
        freeze["status"] != "frozen_before_training"
        or freeze["config"] != config
        or freeze["runtime"] != runtime()
    ):
        raise ValueError("training design changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    completed = run / "expanded/training/summary.json"
    if completed.exists():
        summary = read_json(completed)
        if summary["status"] != "completed":
            raise ValueError("invalid existing training completion record")
        for name, checksum in summary["outputs_sha256"].items():
            checked(run / "expanded" / name, checksum)
        print(
            "training already completed; frozen inputs and outputs verified", flush=True
        )
        return
    trainer, pipeline = make_trainer(config, run)
    output = run / "expanded"
    output.mkdir(exist_ok=True)
    descriptor = {
        "settings": asdict(trainer.settings),
        "configuration_sha256": trainer.settings.sha256,
        "freeze_sha256": sha256_file(run / "training-freeze.json"),
        "initialization": "from_scratch",
        "initial_encoder_sha256": model_sha256(trainer.model),
        "encoder_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "training_head_parameters": trainer.head.weight.numel(),
        "test_used": False,
    }
    if (output / "run.json").exists() and read_json(output / "run.json") != descriptor:
        raise ValueError("existing training belongs to different frozen inputs")
    write_json(output / "run.json", descriptor)
    history = output / "training/history.jsonl"
    history.parent.mkdir(exist_ok=True)
    checkpoint = output / "checkpoints/last.pt"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
        _trim_history_to_checkpoint(history, trainer.update)
    elif history.exists() and history.stat().st_size:
        raise ValueError("history exists without a resumable checkpoint")
    started = time.perf_counter()
    with history.open("a", encoding="utf-8") as stream:

        def record(value):
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            if value["update"] % 100 == 0:
                print(json.dumps(value, allow_nan=False), flush=True)

        def validate(active):
            _compact_validation_outputs(output, active.best["update"])
            metrics = evaluate_validation(
                active.model,
                pipeline,
                root=ROOT,
                manifest=ROOT / config["evaluation_manifest"],
                selection_root=ROOT / config["validation_selections"],
                output_root=output / f"validation/update-{active.update:06d}",
                device="cpu",
                batch_size=32,
            )
            print(
                f"validation {active.update}: macro EER {100 * metrics['macro_eer']:.3f}%",
                flush=True,
            )
            return metrics["macro_eer"]

        if (
            trainer.early_stopping["patience"] < 8
            and trainer.update < trainer.settings.maximum_updates
        ):
            trainer.train_until(
                trainer.settings.maximum_updates,
                on_validation=validate,
                checkpoint_dir=output / "checkpoints",
                on_update=record,
            )
        trainer.save_checkpoint(checkpoint)
        _compact_validation_outputs(output, trainer.best["update"])
    if trainer.best["update"] is None:
        raise ValueError("no full validation-selected checkpoint")
    if (
        trainer.update < trainer.settings.maximum_updates
        and trainer.early_stopping["patience"] < 8
    ):
        raise ValueError("training budget not completed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    best = output / "checkpoints/best.pt"
    state = torch.load(best, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.model.eval()
    trainer.model.requires_grad_(False)
    bundle = output / "bundle"
    bundle.mkdir(exist_ok=True)
    torch.save(
        {
            "model": state["model"],
            "encoder": "statistics_mlp",
            "embedding_dimension": 128,
            "checkpoint_update": state["update"],
            "run_seed": config["seed"],
            "encoder_parameters": descriptor["encoder_parameters"],
            "configuration_sha256": trainer.settings.sha256,
        },
        bundle / "encoder.pt",
    )
    shutil.copyfile(run / "feature-statistics.json", bundle / "feature-statistics.json")
    restored = create_encoder("statistics_mlp").eval()
    exported = torch.load(bundle / "encoder.pt", map_location="cpu", weights_only=True)
    restored.load_state_dict(exported["model"], strict=True)
    selected = sorted(
        load_segments(
            ROOT / config["evaluation_manifest"],
            split="validation",
            role="verification",
        ),
        key=lambda s: s.segment_id,
    )[:8]
    probe = SegmentDataset(ROOT, selected, pipeline, kind="log_mel", mode="center")
    batch = collate_segments([probe[i] for i in range(len(selected))])
    with torch.inference_mode():
        before = trainer.model(batch["input"], batch["mask"])
        after = restored(batch["input"], batch["mask"])
    if (
        before.shape != (8, 128)
        or not torch.equal(before, after)
        or not torch.isfinite(after).all()
    ):
        raise ValueError("exported encoder inference differs")
    metrics_path = (
        output / f"validation/update-{state['update']:06d}/metrics/validation.json"
    )
    metrics = read_json(metrics_path)
    if metrics["partial"] or metrics["macro_eer"] != trainer.best["eer"]:
        raise ValueError("selected validation metrics/checkpoint disagree")
    for name in ("model", "head"):
        if not all(torch.isfinite(v).all() for v in state[name].values()):
            raise ValueError("nonfinite selected checkpoint")
    summary = {
        "status": "completed",
        "conditions": config["conditions"],
        "new_training_runs": 1,
        "completed_updates": trainer.update,
        "maximum_updates": trainer.settings.maximum_updates,
        "early_stopped": trainer.early_stopping["patience"] >= 8,
        "selected_update": state["update"],
        "validation_macro_eer": metrics["macro_eer"],
        "best": trainer.best,
        "encoder_parameters": descriptor["encoder_parameters"],
        "export_reload_bitwise_equal_validation_segments": len(selected),
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_head_parameters": descriptor["training_head_parameters"],
        "elapsed_this_invocation_seconds": time.perf_counter() - started,
        "test_used": False,
        "outputs_sha256": {
            str(p.relative_to(output)): sha256_file(p)
            for p in (
                best,
                checkpoint,
                history,
                metrics_path,
                bundle / "encoder.pt",
                bundle / "feature-statistics.json",
                output / "run.json",
            )
        },
    }
    write_json(output / "training/summary.json", summary)
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
