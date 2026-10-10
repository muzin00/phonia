"""Prepare a small, unsaved listening check of additional training phones."""

from __future__ import annotations

import json
import math
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-consonant-review"))
import sample_review as shared

CONFIG = BASE / "config/review.json"
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


def crop_interval(samples, interval, config):
    rate = config["sample_rate_hz"]
    original_first = round(interval["start_sec"] * rate)
    original_last = round(interval["end_sec"] * rate)
    if not 0 <= original_first < original_last <= len(samples):
        raise ValueError("invalid alignment bounds")
    size = min(config["maximum_frames"], original_last - original_first)
    first = original_first + (original_last - original_first - size) // 2
    last = first + size
    rms = float(np.sqrt(np.mean(samples[first:last] ** 2)))
    reasons = []
    if size < config["minimum_frames"]:
        reasons.append("short_interval")
    if rms < 10 ** (config["minimum_rms_dbfs"] / 20):
        reasons.append("low_rms")
    return {
        "original_start_frame": original_first,
        "original_end_frame": original_last,
        "first_frame": first,
        "last_frame": last,
        "start_frame": first,
        "end_frame": last,
        "start_sec": first / rate,
        "end_sec": last / rate,
        "frame_count": size,
        "duration_sec": size / rate,
        "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "quality_flags": reasons,
    }


def collect(config, run, corpus, directory, speakers, metadata, files):
    phones = config["review_phonemes"]
    raw_rows, eligible = [], []
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
                alignment[key] != meta[key]
                for key in (
                    "utterance_id",
                    "speaker_id",
                    "source_file",
                    "source_sha256",
                )
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
                (index, interval)
                for index, interval in enumerate(alignment["intervals"])
                if interval["phoneme"] in phones
            ]
            if not intervals:
                continue
            source = (ROOT / meta["source_file"]).resolve()
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
                ) != (config["sample_rate_hz"], 1, 2, "NONE"):
                    raise ValueError("expected unchanged 24kHz mono PCM16")
                samples = (
                    np.frombuffer(
                        audio.readframes(audio.getnframes()), dtype="<i2"
                    ).astype(np.float64)
                    / 32768
                )
            for index, interval in intervals:
                phone = interval["phoneme"]
                crop = crop_interval(samples, interval, config)
                origin = f"{meta['utterance_id']}--phone-{index:03d}-{phone}"
                identifier = (
                    f"{origin}--center-{crop['first_frame']}-{crop['frame_count']}"
                )
                row = {
                    "item_id": identifier,
                    "vowel_interval_id": identifier,
                    "original_segment_id": origin,
                    "corpus": corpus,
                    "speaker_id": speaker,
                    "split": "train",
                    "evaluation_role": "training",
                    "utterance_id": meta["utterance_id"],
                    "phoneme": phone,
                    "normalized_phoneme": phone,
                    "raw_phoneme": phone,
                    "interval_index": index,
                    **crop,
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
                    "sample_rate_hz": config["sample_rate_hz"],
                    "channels": 1,
                    "sample_width_bytes": 2,
                    "storage_mode": "source_slice",
                }
                raw_rows.append(row)
                if not crop["quality_flags"]:
                    eligible.append(row)
    if len({row["item_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("duplicate source intervals")
    output = run / ("jvs" if corpus == "JVS" else "common-voice")
    output.mkdir()
    write_rows(output / "raw.jsonl", raw_rows)
    write_rows(output / "eligible.jsonl", eligible)
    write_rows(output / "excluded.jsonl", [r for r in raw_rows if r["quality_flags"]])
    reports = {}
    for phone in phones:
        raw = [r for r in raw_rows if r["phoneme"] == phone]
        accepted = [r for r in eligible if r["phoneme"] == phone]
        coverage = Counter(r["speaker_id"] for r in accepted)
        if set(coverage) != set(speakers) or min(coverage.values()) < 2:
            raise ValueError(f"insufficient post-QC speaker coverage: {corpus}/{phone}")
        reports[phone] = {
            "raw_candidates": len(raw),
            "eligible_candidates": len(accepted),
            "excluded_candidates": len(raw) - len(accepted),
            "exclusion_reasons": dict(
                Counter(reason for r in raw for reason in r["quality_flags"])
            ),
            "minimum_eligible_per_speaker": min(coverage.values()),
        }
    report = {
        "training_speakers": len(speakers),
        "alignments": alignment_count,
        "raw_candidates": len(raw_rows),
        "eligible_candidates": len(eligible),
        "phones": reports,
    }
    print(f"{corpus}: {json.dumps(report, ensure_ascii=False)}", flush=True)
    for path in output.glob("*.jsonl"):
        files[relative(path)] = sha256(path)
    return eligible, report


def main():
    config = read_json(CONFIG)
    if (
        config["split"] != "train"
        or config["corpora"] != ["JVS", "Common Voice"]
        or config["review_phonemes"] != ["t", "d", "k", "g", "z", "h"]
        or config["sample_rate_hz"] != 24000
        or not set(config["review_phonemes"]) <= set(config["model_phonemes"])
    ):
        raise ValueError("unsupported listening protocol")
    run = ROOT / config["run_directory"]
    if run.exists():
        raise ValueError("unused output directory required; preserve existing samples")
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    split = read_json(split_path)["speaker_splits"]
    cv = ROOT / config["common_voice_run"]
    cv_selection = read_json(cv / "selection.json")
    cv_speakers = {r["speaker_id"] for r in cv_selection}
    if (
        len(split["train"]) != 70
        or len(cv_speakers) != 70
        or any(r["locale"] != "ja" for r in cv_selection)
        or set(split["train"]) & (set(split["validation"]) | set(split["test"]))
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
        relative(path): sha256(path)
        for path in (
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
    pools, reports = {}, {}
    sources = (
        (
            "JVS",
            ROOT / "poc/phoneme-speaker-dataset/data/generated/alignments/raw",
            split["train"],
        ),
        ("Common Voice", cv / "alignments/raw", cv_speakers),
    )
    for corpus, directory, speakers in sources:
        eligible, reports[corpus] = collect(
            config, run, corpus, directory, speakers, metadata, files
        )
        for phone in config["review_phonemes"]:
            pools[corpus, phone] = [r for r in eligible if r["phoneme"] == phone]
    selected = shared.select(
        pools, config["seed"], config["samples_per_corpus_and_phoneme"]
    )
    selected.sort(
        key=lambda r: (
            config["corpora"].index(r["corpus"]),
            config["review_phonemes"].index(r["phoneme"]),
            shared.priority(config["seed"] + 1, r["item_id"]),
        )
    )
    if len({r["item_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate listening samples")
    shared.stage_audio(selected, metadata, run / "media")
    dataset = shared.review_dataset(
        selected,
        config["seed"],
        dataset_id="jvs-cv-voicing-listening",
        corpus_title="JVS・Common Voice",
    )
    dataset["title"] = "追加6音素の試聴（保存なし・60区間）"
    dataset["datasetVersion"] = (
        f"20261010-v1-center250-seed{config['seed']}-n{len(selected)}"
    )
    dataset["playback"]["contextPaddingSec"] = config["context_padding_sec"]
    for item, row in zip(dataset["items"], selected, strict=True):
        item["target"]["tags"].extend(
            [
                f"話者 {row['speaker_id']}",
                f"前 /{row['previous_phoneme']}/ → /{row['phoneme']}/ → 後 /{row['next_phoneme']}/",
            ]
        )
    write_json(run / "review-dataset.json", dataset)
    write_rows(run / "sample.jsonl", selected)
    sample_counts = {}
    for corpus in config["corpora"]:
        sample_counts[corpus] = {}
        for phone in config["review_phonemes"]:
            rows = [
                r for r in selected if r["corpus"] == corpus and r["phoneme"] == phone
            ]
            sample_counts[corpus][phone] = {
                "intervals": len(rows),
                "speakers": len({r["speaker_id"] for r in rows}),
                "minimum_duration_ms": min(r["duration_sec"] for r in rows) * 1000,
                "maximum_duration_ms": max(r["duration_sec"] for r in rows) * 1000,
            }
    report = {
        "status": "listening_samples_prepared",
        "protocol": config,
        "corpora": reports,
        "samples": sample_counts,
        "sample_count": len(selected),
        "review_answers_saved": False,
        "formal_review_required": False,
        "training_started": False,
        "context_is_listening_aid_only": True,
    }
    write_json(run / "preparation-report.json", report)
    for path in (
        run / "review-dataset.json",
        run / "sample.jsonl",
        run / "preparation-report.json",
    ):
        files[relative(path)] = sha256(path)
    for path in (run / "media").rglob("*.wav"):
        files[relative(path)] = sha256(path)
    write_json(
        run / "sample-freeze.json",
        {"status": "listening_inputs_frozen", "files": files},
    )
    print(
        json.dumps(
            {
                "run_directory": relative(run),
                "sample_count": len(selected),
                "samples": sample_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
