"""Train an encoder and evaluate a frozen checkpoint on fixed validation trials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import InputPipeline, load_segments, sha256_file
from phase3_data.artifacts import write_json
from phase3_data.input import SegmentDataset, collate_segments, load_feature_statistics
from phase3_data.manifest import ids_checksum, json_sha256
from phase3_data.sampling import BalancedSampler
from phase3_train.evaluation import evaluate_validation
from phase3_train.losses import AAMSoftmax
from phase3_train.models import create_encoder
from phase3_train.training import Trainer, TrainSettings, seed_everything

BASELINE = json.loads(
    (BASE / "config/baseline-log-mel.json").read_text(encoding="utf-8")
)
MANIFEST = ROOT / BASELINE["data"]["manifest"]
ARTIFACTS = ROOT / BASELINE["artifacts_root"]
SELECTIONS = ARTIFACTS / "fixed-validation"
ENCODERS = (
    "statistics_mlp",
    "tdnn",
    "framewise_cnn",
    "waveform_cnn_k80",
    "waveform_cnn_k240",
    "waveform_cnn_k240_context27",
)


def _code_sha256() -> str:
    paths = [
        *sorted((BASE / "phase3_data").glob("*.py")),
        *sorted((BASE / "phase3_train").glob("*.py")),
        Path(__file__),
    ]
    return json_sha256(
        {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    )


def _inputs(settings: TrainSettings, manifest: Path, statistics_path: Path | None):
    manifest_sha = sha256_file(manifest)
    if (
        manifest.resolve() == MANIFEST.resolve()
        and manifest_sha != BASELINE["data"]["manifest_sha256"]
    ):
        raise ValueError("canonical manifest checksum mismatch")
    segments = load_segments(
        manifest, split="train", role="training", cohort=settings.cohort
    )
    pipeline = InputPipeline(BASELINE["input"], rms_enabled=settings.rms_enabled)
    statistics_sha = None
    if settings.encoder in ENCODERS[:3]:
        if statistics_path is None:
            raise ValueError("log-Mel encoder requires --statistics")
        statistics = load_feature_statistics(
            statistics_path,
            manifest_sha256=manifest_sha,
            cohort=settings.cohort,
            speaker_ids_sha256=ids_checksum(
                {segment.speaker_id for segment in segments}
            ),
            segment_ids_sha256=ids_checksum(segment.segment_id for segment in segments),
            preprocessing_sha256=pipeline.preprocessing_sha256,
        )
        statistics_sha = sha256_file(statistics_path)
        pipeline = InputPipeline(
            BASELINE["input"], rms_enabled=settings.rms_enabled, statistics=statistics
        )
    return segments, pipeline, manifest_sha, statistics_sha


def _trainer(settings: TrainSettings, manifest: Path, statistics_path: Path | None):
    seed_everything(settings.seed)
    segments, pipeline, manifest_sha, statistics_sha = _inputs(
        settings, manifest, statistics_path
    )
    if settings.overfit:
        selected_speakers = sorted({segment.speaker_id for segment in segments})[:4]
        selected = []
        for vowel in "aiue":
            for speaker in selected_speakers:
                selected.extend(
                    sorted(
                        (
                            segment
                            for segment in segments
                            if segment.speaker_id == speaker and segment.vowel == vowel
                        ),
                        key=lambda segment: segment.segment_id,
                    )[:2]
                )
        if len(selected) != 32:
            raise ValueError(
                "overfit fixture requires four speakers x four vowels x two segments"
            )
        segments = selected
    model = create_encoder(
        settings.encoder, dropout_override=0.0 if settings.overfit else None
    )
    dataset = SegmentDataset(
        ROOT,
        segments,
        pipeline,
        kind=model.input_kind,
        run_seed=settings.seed,
        mode="center" if settings.overfit else "random",
    )
    sampler = None if settings.overfit else BalancedSampler(segments, settings.seed)
    trainer = Trainer(
        model,
        AAMSoftmax(len({segment.speaker_id for segment in segments})),
        settings,
        dataset,
        sampler,
        manifest_sha256=manifest_sha,
        feature_statistics_sha256=statistics_sha,
    )
    return trainer, pipeline, selected if settings.overfit else None


def _settings(args: argparse.Namespace) -> TrainSettings:
    overfit = args.command == "overfit"
    return TrainSettings(
        encoder=args.encoder,
        cohort=args.cohort,
        rms_enabled=args.rms == "on",
        supcon_enabled=True if overfit else args.supcon == "on",
        seed=20260926 if overfit else args.seed,
        maximum_updates=500 if overfit else args.maximum_updates,
        warmup_updates=0 if overfit else args.warmup_updates,
        validation_interval=500 if overfit else args.validation_interval,
        weight_decay=0.0 if overfit else 1e-4,
        workers=0 if overfit else args.workers,
        device=args.device,
        overfit=overfit,
    )


def _train(args: argparse.Namespace) -> None:
    settings = _settings(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "run.json"
    run = {
        "schema_version": 1,
        "settings": asdict(settings),
        "manifest": str(args.manifest.resolve()),
        "statistics": str(args.statistics.resolve()) if args.statistics else None,
        "selections": str(args.selections.resolve()),
        "configuration_sha256": settings.sha256,
        "encoder_config_sha256": json_sha256(
            {
                name: sha256_file(BASE / "config" / name)
                for name in ("log-mel-encoders.json", "waveform-encoders.json")
            }
        ),
        "code_sha256": _code_sha256(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if (
        settings_path.exists()
        and json.loads(settings_path.read_text(encoding="utf-8")) != run
    ):
        raise ValueError("output directory belongs to a different run")
    write_json(settings_path, run)
    trainer, pipeline, fixed = _trainer(settings, args.manifest, args.statistics)
    if args.resume:
        trainer.load_checkpoint(output / "checkpoints/last.pt")
    start = time.monotonic()
    history_path = output / "training/history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a" if args.resume else "w", encoding="utf-8") as history:

        def record(value):
            history.write(
                json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
            )
            history.flush()
            print(json.dumps(value, allow_nan=False), flush=True)

        if settings.overfit:
            write_json(
                output / "training/overfit-segments.json",
                {"segment_ids": [s.segment_id for s in fixed]},
            )
            batches = [
                collate_segments(
                    [trainer.dataset[i] for i in range(offset, offset + 8)]
                )
                for offset in range(0, 32, 8)
            ]
            value = None
            while trainer.update < settings.maximum_updates:
                value = trainer.train_update(batches)
                record(value)
                if value["accuracy"] >= 0.95:
                    break
            trainer.save_checkpoint(output / "checkpoints/last.pt")
        else:

            def validate(active):
                metric = evaluate_validation(
                    active.model,
                    pipeline,
                    root=ROOT,
                    manifest=args.manifest,
                    selection_root=args.selections,
                    output_root=output / f"validation/update-{active.update:06d}",
                    device=settings.device,
                    batch_size=args.eval_batch_size,
                )
                if metric["macro_eer"] is None:
                    raise ValueError("validation macro EER is undefined")
                return metric["macro_eer"]

            target = (
                settings.maximum_updates
                if args.target_update is None
                else args.target_update
            )
            trainer.train_until(
                target,
                on_validation=validate if not args.skip_validation else None,
                checkpoint_dir=output / "checkpoints",
                on_update=record,
            )
            trainer.save_checkpoint(output / "checkpoints/last.pt")
    summary = {
        "completed_updates": trainer.update,
        "elapsed_seconds": time.monotonic() - start,
        "best": trainer.best,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "checkpoint_sha256": sha256_file(output / "checkpoints/last.pt"),
    }
    if settings.overfit:
        summary["reached_95pct_train_accuracy"] = (
            value is not None and value["accuracy"] >= 0.95
        )
    write_json(output / "training/summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    if args.evaluate:
        if settings.overfit:
            raise ValueError("overfit diagnostic has no validation evaluation")
        _evaluate(output, args.eval_batch_size, args.max_eval_queries_per_vowel)


def _evaluate(
    run_dir: Path, batch_size: int, max_queries_per_vowel: int | None
) -> None:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("code_sha256") is not None and run["code_sha256"] != _code_sha256():
        raise ValueError("run code checksum differs from current implementation")
    settings = TrainSettings(**run["settings"])
    if settings.overfit:
        raise ValueError("overfit checkpoints cannot be used for validation")
    manifest = Path(run["manifest"])
    trainer, pipeline, _ = _trainer(
        settings, manifest, Path(run["statistics"]) if run["statistics"] else None
    )
    best = run_dir / "checkpoints/best.pt"
    checkpoint = best if best.exists() else run_dir / "checkpoints/last.pt"
    trainer.load_checkpoint(checkpoint)
    output = (
        run_dir
        / "validation"
        / (
            f"update-{trainer.update:06d}-partial"
            if max_queries_per_vowel
            else f"update-{trainer.update:06d}"
        )
    )
    metrics = evaluate_validation(
        trainer.model,
        pipeline,
        root=ROOT,
        manifest=manifest,
        selection_root=Path(run["selections"]),
        output_root=output,
        device=settings.device,
        batch_size=batch_size,
        max_queries_per_vowel=max_queries_per_vowel,
    )
    metrics.update(
        {
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_update": trainer.update,
        }
    )
    write_json(output / "metrics/validation.json", metrics)
    print(
        json.dumps(
            {
                "metrics": str(output / "metrics/validation.json"),
                "macro_eer": metrics["macro_eer"],
                "query_count": metrics["query_count"],
                "trial_count": metrics["trial_count"],
                "partial": metrics["partial"],
            },
            allow_nan=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "overfit"):
        command = commands.add_parser(name)
        command.add_argument("--encoder", choices=ENCODERS, required=True)
        command.add_argument("--cohort", type=int, choices=(10, 25, 50, 70), default=10)
        command.add_argument("--rms", choices=("on", "off"), default="on")
        command.add_argument("--supcon", choices=("on", "off"), default="on")
        command.add_argument("--seed", type=int, default=20260926)
        command.add_argument("--maximum-updates", type=int, default=2000)
        command.add_argument("--target-update", type=int)
        command.add_argument("--warmup-updates", type=int, default=100)
        command.add_argument("--validation-interval", type=int, default=250)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--device", default="cpu")
        command.add_argument("--manifest", type=Path, default=MANIFEST)
        command.add_argument("--statistics", type=Path)
        command.add_argument("--selections", type=Path, default=SELECTIONS)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
        command.add_argument("--evaluate", action="store_true")
        command.add_argument("--skip-validation", action="store_true")
        command.add_argument("--eval-batch-size", type=int, default=32)
        command.add_argument("--max-eval-queries-per-vowel", type=int)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--max-queries-per-vowel", type=int)
    args = parser.parse_args()
    if args.command == "evaluate":
        _evaluate(args.run_dir.resolve(), args.batch_size, args.max_queries_per_vowel)
    else:
        _train(args)


if __name__ == "__main__":
    main()
