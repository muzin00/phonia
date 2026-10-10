#!/usr/bin/env python3
"""Prepare, freeze and train one JVS + CV five-vowel /m/ /n/ encoder."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import wave
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

import numpy as np
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
    run.mkdir(parents=True)
    cv = ROOT / config["vowel_source"]
    checked(cv / "training-segments.jsonl", config["vowel_manifest_sha256"])
    checked(cv / "feature-statistics.json", config["vowel_statistics_sha256"])
    checked(ROOT / config["evaluation_manifest"], config["evaluation_manifest_sha256"])
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    split = read_json(split_path)["speaker_splits"]
    jvs_meta = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = {
        r["utterance_id"]: r for r in read_rows(jvs_meta) if r["split"] == "train"
    }
    cv_meta = list(read_rows(cv / "additional-utterances.jsonl"))
    selection = read_json(cv / "selection.json")
    cv_speakers = {r["speaker_id"] for r in selection}
    if len(cv_speakers) != 70 or any(
        r["split"] != "train"
        or r["source_split"] != "train"
        or r["speaker_id"] not in cv_speakers
        for r in cv_meta
    ):
        raise ValueError("CV training metadata mismatch")
    metadata.update({r["utterance_id"]: r for r in cv_meta})
    inputs = {
        relative(p): sha256_file(p)
        for p in (
            split_path,
            jvs_meta,
            cv / "selection.json",
            cv / "additional-utterances.jsonl",
        )
    }
    raw_rows, accepted, exclusions = [], [], Counter()
    counts = Counter()
    sources = (
        (
            "JVS",
            ROOT / "poc/phoneme-speaker-dataset/data/generated/alignments/raw",
            split["train"],
        ),
        ("Common Voice", cv / "alignments/raw", sorted(cv_speakers)),
    )
    for corpus, directory, speakers in sources:
        for speaker in sorted(speakers):
            for path in sorted((directory / speaker).glob("*.json")):
                alignment = read_json(path)
                meta = metadata[alignment["utterance_id"]]
                for key in ("speaker_id", "source_file", "source_sha256"):
                    if (
                        alignment[key] != meta[key]
                        or alignment["speaker_id"] != speaker
                    ):
                        raise ValueError("alignment metadata mismatch")
                if (
                    "raw_phonemes" in meta
                    and alignment["raw_phonemes"] != meta["raw_phonemes"]
                ):
                    raise ValueError("alignment text phoneme mismatch")
                inputs[relative(path)] = sha256_file(path)
                source = ROOT / alignment["source_file"]
                checked(source, alignment["source_sha256"])
                inputs[alignment["source_file"]] = alignment["source_sha256"]
                with wave.open(str(source), "rb") as audio:
                    if (
                        audio.getframerate(),
                        audio.getnchannels(),
                        audio.getsampwidth(),
                        audio.getcomptype(),
                    ) != (24000, 1, 2, "NONE"):
                        raise ValueError("invalid source WAV")
                    pcm = (
                        np.frombuffer(
                            audio.readframes(audio.getnframes()), dtype="<i2"
                        ).astype(np.float64)
                        / 32768
                    )
                for index, interval in enumerate(alignment["intervals"]):
                    phone = interval["phoneme"]
                    if phone not in ("m", "n"):
                        continue
                    first, last = (
                        round(interval["start_sec"] * 24000),
                        round(interval["end_sec"] * 24000),
                    )
                    if not 0 <= first < last <= len(pcm):
                        raise ValueError("invalid nasal boundary")
                    signal = pcm[first:last]
                    rms = float(np.sqrt(np.mean(signal**2)))
                    dbfs = 20 * math.log10(rms) if rms else None
                    reasons = []
                    if last - first < config["minimum_frames"]:
                        reasons.append("too_short")
                    if dbfs is None or dbfs < config["minimum_rms_dbfs"]:
                        reasons.append("near_silent")
                    row = {
                        "schema_version": 1,
                        "design_version": "phase8-vowels-mn-v1",
                        "vowel_interval_id": f"{alignment['utterance_id']}--phone-{index:03d}-{phone}",
                        "utterance_id": alignment["utterance_id"],
                        "speaker_id": speaker,
                        "corpus": corpus,
                        "normalized_phoneme": phone,
                        "raw_phoneme": phone,
                        "split": "train",
                        "evaluation_role": "training",
                        "learning_curve_cohorts": [140],
                        "source_file": alignment["source_file"],
                        "source_sha256": alignment["source_sha256"],
                        "raw_alignment_file": relative(path),
                        "raw_alignment_sha256": inputs[relative(path)],
                        "interval_index": index,
                        "transcript": meta["transcript"],
                        "start_frame": first,
                        "end_frame": last,
                        "frame_count": last - first,
                        "start_sec": first / 24000,
                        "end_sec": last / 24000,
                        "rms_dbfs": dbfs,
                        "quality_flags": reasons,
                        "sample_rate_hz": 24000,
                        "channels": 1,
                        "sample_width_bytes": 2,
                        "storage_mode": "source_slice",
                    }
                    raw_rows.append(row)
                    counts[f"{corpus}/{phone}/raw"] += 1
                    if reasons:
                        for reason in reasons:
                            exclusions[f"{corpus}/{phone}/{reason}"] += 1
                    else:
                        accepted.append(row)
                        counts[f"{corpus}/{phone}/eligible"] += 1
        print(f"prepared {corpus}: {dict(counts)}", flush=True)
    write_rows(run / "nasal-raw.jsonl", raw_rows)
    write_rows(
        run / "nasal-excluded.jsonl", [r for r in raw_rows if r["quality_flags"]]
    )
    write_rows(run / "nasal-eligible.jsonl", accepted)
    with (run / "training-segments.jsonl").open("wb") as output:
        with (cv / "training-segments.jsonl").open("rb") as original:
            shutil.copyfileobj(original, output)
        with (run / "nasal-eligible.jsonl").open("rb") as additional:
            shutil.copyfileobj(additional, output)
    all_train = load_training(run / "training-segments.jsonl")
    review_checks = {}
    accepted_ids = {r["vowel_interval_id"] for r in accepted}
    for name, corpus in (("review", "JVS"), ("common-voice-review", "Common Voice")):
        path = ROOT / "poc/phoneme-consonant-review/data" / name / "sample.jsonl"
        selected = [r for r in read_rows(path) if r["corpus"] == corpus]
        inputs[relative(path)] = sha256_file(path)
        review_checks[corpus] = {
            "prepared_sample_count": len(selected),
            "retained_by_automatic_qc": sum(
                r["item_id"] in accepted_ids for r in selected
            ),
            "individual_answers_saved": False,
            "qualitative_user_feedback": "品質はとても良い"
            if corpus == "JVS"
            else "Common VoiceはOK",
        }
    adoption = {
        "date": "2026-10-10",
        "decision_by": "user",
        "authorization": '"/m /n"の学習作業進めて',
        "adopted": ["JVS", "Common Voice"],
        "excluded_from_this_run": ["SRC4VC"],
        "review_scope": "prior qualitative train-only m/n listening; vowels reused from existing adopted run",
        "review_details": review_checks,
        "procedure_deviations": [
            "User explicitly authorized training after qualitative preliminary review; no new post-filter 100-per-corpus review or per-item answer aggregation.",
            "No quantitative human label-accuracy or usability rate is available.",
        ],
    }
    write_json(run / "adoption.json", adoption)
    coverage = Counter(
        f"{'CV' if s.speaker_id.startswith('cv17_') else 'JVS'}/{s.vowel}"
        for s in all_train
    )
    group_counts = Counter((s.speaker_id, s.vowel) for s in all_train)
    report = {
        "status": "prepared",
        "training_labels": 140,
        "training_segments": len(all_train),
        "nasal_counts": dict(counts),
        "exclusions": dict(exclusions),
        "coverage": dict(coverage),
        "minimum_segments_per_speaker_phoneme": min(group_counts.values()),
        "source_inputs": inputs,
        "test_used": False,
    }
    write_json(run / "preparation-report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_inputs"}),
        flush=True,
    )


def freeze(config, run):
    cv = ROOT / config["vowel_source"]
    original = read_json(cv / "feature-statistics.json")
    vowels = expanded.load_training(cv / "training-segments.jsonl")
    if original["segment_ids_sha256"] != ids_checksum(s.segment_id for s in vowels):
        raise ValueError("reused vowel statistics population mismatch")
    pipeline = InputPipeline(expanded.BASELINE["input"], rms_enabled=False)
    nasals = [segment(r) for r in read_rows(run / "nasal-eligible.jsonl")]
    print(f"train-only nasal feature statistics: {len(nasals):,}", flush=True)
    nasal_stats = compute_feature_statistics(
        ROOT,
        nasals,
        pipeline,
        cohort=140,
        manifest_sha256=sha256_file(run / "nasal-eligible.jsonl"),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        calculation_code_sha256=sha256_file(Path(__file__)),
    )
    write_json(run / "nasal-feature-statistics.json", nasal_stats)
    all_train = load_training(run / "training-segments.jsonl")
    stats = {
        **nasal_stats,
        **expanded.union_statistics(original, nasal_stats),
        "manifest_sha256": sha256_file(run / "training-segments.jsonl"),
        "segment_ids_sha256": ids_checksum(s.segment_id for s in all_train),
        "segment_count": len(all_train),
        "calculation": "merge_unchanged_vowel_and_new_nasal_train_only_moments",
        "source_statistics_sha256": [
            sha256_file(cv / "feature-statistics.json"),
            sha256_file(run / "nasal-feature-statistics.json"),
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
                "adoption.json",
                "preparation-report.json",
                "nasal-raw.jsonl",
                "nasal-excluded.jsonl",
                "nasal-eligible.jsonl",
                "training-segments.jsonl",
                "feature-statistics.json",
                "nasal-feature-statistics.json",
            )
        ),
        cv / "training-segments.jsonl",
        cv / "feature-statistics.json",
        ROOT / config["evaluation_manifest"],
    }
    # Include every imported repository module, including helper dependencies.
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
        files[relative(path)] = sha256_file(path)
    for s in [
        *all_train,
        *load_segments(ROOT / config["evaluation_manifest"], split="validation"),
    ]:
        if files.setdefault(s.source_file, s.source_sha256) != s.source_sha256:
            raise ValueError("source checksum conflict")
    for name, sha in files.items():
        checked(ROOT / name, sha)
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
    print("all seven-phone inputs frozen", flush=True)


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
        after.shape != (7, 128)
        or not torch.equal(before, after)
        or not torch.isfinite(after).all()
    ):
        raise ValueError("seven-phone export reload mismatch")
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
