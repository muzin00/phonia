"""Prepare train-only, post-QC /s/ review samples and unchanged source audio."""

from __future__ import annotations

import json
import math
import statistics
import sys
import wave
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-consonant-review"))
import numpy as np
import sample_review as shared

CONFIG = BASE / "config/protocol.json"
read_json, sha256, relative, write_json = (
    shared.read_json,
    shared.sha256,
    shared.relative,
    shared.write_json,
)


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def collect(config, run, corpus, directory, speakers, metadata, files):
    raw_rows, eligible, rejected = [], [], Counter()
    alignment_count = 0
    for speaker in sorted(speakers):
        paths = sorted((directory / speaker).glob("*.json"))
        if not paths:
            raise ValueError(f"no alignments for {speaker}")
        for path in paths:
            alignment = read_json(path)
            meta = metadata[alignment["utterance_id"]]
            if meta["split"] != "train" or meta["speaker_id"] != speaker:
                raise ValueError("non-training source metadata")
            if any(
                alignment[k] != meta[k]
                for k in ("utterance_id", "speaker_id", "source_file", "source_sha256")
            ):
                raise ValueError("alignment/source mismatch")
            if (
                "raw_phonemes" in meta
                and alignment["raw_phonemes"] != meta["raw_phonemes"]
            ):
                raise ValueError("alignment/transcript phone mismatch")
            files[relative(path)] = sha256(path)
            alignment_count += 1
            intervals = [
                (i, p)
                for i, p in enumerate(alignment["intervals"])
                if p["phoneme"] == config["phoneme"]
            ]
            if not intervals:
                continue
            source = (ROOT / alignment["source_file"]).resolve()
            source.relative_to(ROOT)
            if sha256(source) != meta["source_sha256"]:
                raise ValueError("source WAV checksum mismatch")
            files[relative(source)] = meta["source_sha256"]
            with wave.open(str(source), "rb") as audio:
                if (
                    audio.getframerate(),
                    audio.getnchannels(),
                    audio.getsampwidth(),
                    audio.getcomptype(),
                ) != (24000, 1, 2, "NONE"):
                    raise ValueError("expected unchanged 24kHz mono PCM16")
                samples = (
                    np.frombuffer(
                        audio.readframes(audio.getnframes()), dtype="<i2"
                    ).astype(np.float64)
                    / 32768
                )
            for index, interval in intervals:
                original_first = round(interval["start_sec"] * 24000)
                original_last = round(interval["end_sec"] * 24000)
                if not 0 <= original_first < original_last <= len(samples):
                    raise ValueError("invalid alignment bounds")
                size = min(config["maximum_frames"], original_last - original_first)
                first = original_first + (original_last - original_first - size) // 2
                last = first + size
                rms = float(np.sqrt(np.mean(samples[first:last] ** 2)))
                dbfs = 20 * math.log10(rms) if rms else None
                reasons = []
                if size < config["minimum_frames"]:
                    reasons.append("short_interval")
                if rms < 10 ** (config["minimum_rms_dbfs"] / 20):
                    reasons.append("low_rms")
                origin = f"{meta['utterance_id']}--phone-{index:03d}-s"
                identifier = f"{origin}--center-{first}-{size}"
                row = {
                    "item_id": identifier,
                    "vowel_interval_id": identifier,
                    "original_segment_id": origin,
                    "corpus": corpus,
                    "speaker_id": speaker,
                    "split": "train",
                    "evaluation_role": "training",
                    "learning_curve_cohorts": [140],
                    "utterance_id": meta["utterance_id"],
                    "phoneme": "s",
                    "normalized_phoneme": "s",
                    "raw_phoneme": "s",
                    "interval_index": index,
                    "original_start_frame": original_first,
                    "original_end_frame": original_last,
                    "start_frame": first,
                    "end_frame": last,
                    "first_frame": first,
                    "last_frame": last,
                    "start_sec": first / 24000,
                    "end_sec": last / 24000,
                    "frame_count": size,
                    "duration_sec": size / 24000,
                    "rms_dbfs": dbfs,
                    "quality_flags": reasons,
                    "source_file": meta["source_file"],
                    "source_sha256": meta["source_sha256"],
                    "source_subset": meta["source_subset"],
                    "transcript": meta["transcript"],
                    "raw_alignment_file": relative(path),
                    "raw_alignment_sha256": files[relative(path)],
                    "previous_phoneme": alignment["intervals"][index - 1]["phoneme"]
                    if index
                    else None,
                    "next_phoneme": alignment["intervals"][index + 1]["phoneme"]
                    if index + 1 < len(alignment["intervals"])
                    else None,
                    "sample_rate_hz": 24000,
                    "channels": 1,
                    "sample_width_bytes": 2,
                    "storage_mode": "source_slice",
                }
                raw_rows.append(row)
                if reasons:
                    rejected.update(reasons)
                else:
                    eligible.append(row)
    if len({r["item_id"] for r in raw_rows}) != len(raw_rows):
        raise ValueError("duplicate source intervals")
    name = "jvs" if corpus == "JVS" else "common-voice"
    output = run / name
    output.mkdir()
    write_rows(output / "raw.jsonl", raw_rows)
    write_rows(output / "eligible.jsonl", eligible)
    write_rows(output / "excluded.jsonl", [r for r in raw_rows if r["quality_flags"]])
    groups = Counter(r["speaker_id"] for r in eligible)
    if set(groups) != set(speakers) or min(groups.values()) < 2:
        raise ValueError("insufficient post-QC speaker coverage")
    report = {
        "training_speakers": len(speakers),
        "alignments": alignment_count,
        "raw_candidates": len(raw_rows),
        "eligible_candidates": len(eligible),
        "excluded_candidates": len(raw_rows) - len(eligible),
        "exclusion_reasons": dict(rejected),
        "minimum_eligible_per_speaker": min(groups.values()),
        "center_cropped_candidates": sum(
            r["original_end_frame"] - r["original_start_frame"] > 6000 for r in eligible
        ),
    }
    print(f"{corpus}: {report}", flush=True)
    return eligible, report


def main():
    config = read_json(CONFIG)
    if (
        config["phoneme"] != "s"
        or config["split"] != "train"
        or config["minimum_frames"] != 720
        or config["maximum_frames"] != 6000
        or config["samples_per_corpus"] != 100
    ):
        raise ValueError("unsupported review protocol")
    run = ROOT / config["run_directory"]
    outputs = {
        "JVS": BASE / "data/jvs-review",
        "Common Voice": BASE / "data/common-voice-review",
    }
    if run.exists() or any(p.exists() for p in outputs.values()):
        raise ValueError(
            "unused output directories required; preserve existing samples and answers"
        )
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    split = read_json(split_path)["speaker_splits"]
    cv = ROOT / config["common_voice_run"]
    cv_selection = read_json(cv / "selection.json")
    cv_speakers = {r["speaker_id"] for r in cv_selection}
    if (
        len(split["train"]) != 70
        or len(cv_speakers) != 70
        or any(r["locale"] != "ja" for r in cv_selection)
    ):
        raise ValueError("unexpected existing speaker selection")
    jvs_meta = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    cv_meta = cv / "additional-utterances.jsonl"
    metadata = {}
    for path in (jvs_meta, cv_meta):
        with path.open() as stream:
            for row in map(json.loads, stream):
                if row["split"] != "train":
                    continue
                if path == cv_meta and (
                    row["speaker_id"] not in cv_speakers
                    or row["source_split"] != "train"
                    or row["source_subset"] != "common_voice17_ja_train"
                ):
                    raise ValueError("unexpected Common Voice training metadata")
                if row["utterance_id"] in metadata:
                    raise ValueError("duplicate source utterance")
                metadata[row["utterance_id"]] = row
    if {
        r["speaker_id"] for r in metadata.values() if r["speaker_id"].startswith("jvs")
    } != set(split["train"]):
        raise ValueError("JVS train labels changed")
    files = {
        relative(p): sha256(p)
        for p in (
            CONFIG,
            split_path,
            jvs_meta,
            cv_meta,
            cv / "selection.json",
            Path(__file__),
            Path(shared.__file__),
        )
    }
    run.mkdir(parents=True)
    summary = {
        "protocol": config,
        "status": "prepared_for_human_review",
        "corpora": {},
        "training_started": False,
        "human_review_complete": False,
    }
    sources = (
        (
            "JVS",
            ROOT / "poc/phoneme-speaker-dataset/data/generated/alignments/raw",
            split["train"],
        ),
        ("Common Voice", cv / "alignments/raw", cv_speakers),
    )
    for corpus, directory, speakers in sources:
        pool, report = collect(
            config, run, corpus, directory, speakers, metadata, files
        )
        selected = shared.select(
            {(corpus, "s"): pool}, config["seed"], config["samples_per_corpus"]
        )
        media = run / ("jvs" if corpus == "JVS" else "common-voice") / "media"
        shared.stage_audio(selected, metadata, media)
        dataset_id = (
            "jvs-s-post-qc-training-review"
            if corpus == "JVS"
            else "common-voice17-ja-s-post-qc-training-review"
        )
        dataset = shared.review_dataset(
            selected, config["seed"], dataset_id=dataset_id, corpus_title=corpus
        )
        dataset["title"] = f"{corpus} /s/ 学習前レビュー（自動チェック後100区間）"
        dataset["datasetVersion"] = f"20261010-v1-center250-seed{config['seed']}-n100"
        dataset["playback"]["contextPaddingSec"] = config["context_padding_sec"]
        output = outputs[corpus]
        output.mkdir(parents=True)
        write_json(output / "review-dataset.json", dataset)
        write_rows(output / "sample.jsonl", selected)
        durations = [r["duration_sec"] for r in selected]
        report.update(
            dataset_id=dataset_id,
            dataset_version=dataset["datasetVersion"],
            selected_count=len(selected),
            selected_speakers=len({r["speaker_id"] for r in selected}),
            selected_sources=len({r["source_file"] for r in selected}),
            selected_duration_sec={
                "minimum": min(durations),
                "median": statistics.median(durations),
                "maximum": max(durations),
            },
            media_root=relative(media),
            review_dataset_sha256=sha256(output / "review-dataset.json"),
            sample_sha256=sha256(output / "sample.jsonl"),
            human_review_complete=False,
            adoption_status="pending_human_review",
            answers_file=relative(output / "review-records.jsonl"),
        )
        write_json(output / "sampling-summary.json", {"protocol": config, **report})
        summary["corpora"][corpus] = report
        for path in output.glob("*"):
            files[relative(path)] = sha256(path)
        for path in media.rglob("*.wav"):
            files[relative(path)] = sha256(path)
        for path in media.parent.glob("*.jsonl"):
            files[relative(path)] = sha256(path)
    write_json(run / "preparation-report.json", summary)
    write_json(
        run / "review-freeze.json",
        {
            "status": "samples_frozen_before_human_review",
            "config": config,
            "files": files,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
