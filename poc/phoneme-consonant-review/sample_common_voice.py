#!/usr/bin/env python3
"""Sample 100 train-only Common Voice nasal intervals, without volume filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import sample_review as shared

CV = shared.ROOT / "artifacts/phoneme-training-data-expansion/training-data-20261008-v1"
DEFAULT_OUTPUT = shared.BASE / "data/common-voice-review"
DEFAULT_MEDIA = (
    shared.ROOT / "artifacts/phoneme-consonant-review/cv-mn-100-20261010-v1/media"
)
CORPUS = "Common Voice"


def load_pools() -> tuple[dict, dict, dict, set]:
    selection = shared.read_json(CV / "selection.json")
    training_speakers = {row["speaker_id"] for row in selection}
    if len(training_speakers) != 70 or any(row["locale"] != "ja" for row in selection):
        raise ValueError("Expected the existing 70-speaker Japanese training selection")
    metadata = {}
    with (CV / "additional-utterances.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if (
                row["speaker_id"] not in training_speakers
                or row["split"] != "train"
                or row["source_split"] != "train"
                or row["source_subset"] != "common_voice17_ja_train"
            ):
                raise ValueError("Unexpected Common Voice training metadata")
            if row["utterance_id"] in metadata:
                raise ValueError("Duplicate training utterance ID")
            metadata[row["utterance_id"]] = row
    if len(metadata) != len(selection):
        raise ValueError("Training metadata differs from frozen selection")

    pools = {(CORPUS, phone): [] for phone in shared.PHONEMES}
    excluded_short = Counter()
    inventory = hashlib.sha256()
    alignment_count = 0
    directory = CV / "alignments/raw"
    for speaker in sorted(training_speakers):
        paths = sorted((directory / speaker).glob("*.json"))
        if not paths:
            raise ValueError(f"No training alignments for {speaker}")
        for path in paths:
            raw_bytes = path.read_bytes()
            alignment = json.loads(raw_bytes)
            row = metadata[alignment["utterance_id"]]
            for field in ("speaker_id", "source_file", "source_sha256"):
                if alignment[field] != row[field]:
                    raise ValueError(f"Alignment/metadata mismatch: {path}/{field}")
            if alignment["speaker_id"] != speaker:
                raise ValueError(f"Wrong speaker in {path}")
            alignment_count += 1
            inventory.update(shared.relative(path).encode() + b"\0")
            inventory.update(hashlib.sha256(raw_bytes).digest())
            for index, interval in enumerate(alignment["intervals"]):
                phone = interval["phoneme"]
                # The moraic nasal /N/ and palatalized /my/, /ny/ stay separate.
                if phone not in shared.PHONEMES:
                    continue
                start, end = interval["start_sec"], interval["end_sec"]
                first, last = round(start * shared.RATE), round(end * shared.RATE)
                if first < 0 or last <= first:
                    raise ValueError(f"Invalid boundary in {path}")
                if last - first < shared.MINIMUM_FRAMES:
                    excluded_short[phone] += 1
                    continue
                pools[(CORPUS, phone)].append(
                    {
                        "item_id": f"{alignment['utterance_id']}--phone-{index:03d}-{phone}",
                        "corpus": CORPUS,
                        "speaker_id": speaker,
                        "split": "train",
                        "utterance_id": alignment["utterance_id"],
                        "phoneme": phone,
                        "interval_index": index,
                        "start_sec": start,
                        "end_sec": end,
                        "first_frame": first,
                        "last_frame": last,
                        "source_file": alignment["source_file"],
                        "source_sha256": alignment["source_sha256"],
                        "raw_alignment_file": shared.relative(path),
                        "raw_alignment_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "previous_phoneme": alignment["intervals"][index - 1]["phoneme"]
                        if index > 0
                        else None,
                        "next_phoneme": alignment["intervals"][index + 1]["phoneme"]
                        if index + 1 < len(alignment["intervals"])
                        else None,
                    }
                )
    summary = {
        "training_speaker_count": len(training_speakers),
        "training_utterance_count": len(metadata),
        "training_alignment_count": alignment_count,
        "training_alignment_inventory_sha256": inventory.hexdigest(),
        "training_selection_sha256": shared.sha256(CV / "selection.json"),
        "training_metadata_sha256": shared.sha256(CV / "additional-utterances.jsonl"),
        "eligible_interval_counts": {
            phone: len(pools[(CORPUS, phone)]) for phone in shared.PHONEMES
        },
        "excluded_short_interval_counts": dict(excluded_short),
    }
    return pools, metadata, summary, training_speakers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--per-phoneme", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA)
    args = parser.parse_args()
    if args.per_phoneme < 1:
        parser.error("--per-phoneme must be positive")
    pools, metadata, summary, training = load_pools()
    selected = shared.select(pools, args.seed, args.per_phoneme)
    if len({row["item_id"] for row in selected}) != args.per_phoneme * 2:
        raise ValueError("Review IDs must be unique")
    if any(row["speaker_id"] not in training for row in selected):
        raise ValueError("Non-training speaker in sample")
    shared.stage_audio(selected, metadata, args.media_root)
    dataset = shared.review_dataset(
        selected,
        args.seed,
        dataset_id="common-voice17-ja-mn-training-review",
        corpus_title="Common Voice日本語",
    )
    shared.write_json(args.output_dir / "review-dataset.json", dataset)
    (args.output_dir / "sample.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    durations = [row["duration_sec"] for row in selected]
    summary.update(
        {
            "schema_version": 1,
            "dataset_id": dataset["datasetId"],
            "dataset_version": dataset["datasetVersion"],
            "seed": args.seed,
            "sampling": "smallest_seeded_sha256_without_replacement_within_phoneme",
            "presentation_order": "independent_seeded_sha256",
            "minimum_interval_frames": shared.MINIMUM_FRAMES,
            "sample_rate_hz": shared.RATE,
            "audio_volume_filter": False,
            "reviewed": False,
            "selected_count": len(selected),
            "selected_phoneme_counts": dict(
                Counter(row["phoneme"] for row in selected)
            ),
            "selected_speaker_count": len({row["speaker_id"] for row in selected}),
            "selected_source_file_count": len({row["source_file"] for row in selected}),
            "duration_sec": {
                "minimum": min(durations),
                "median": statistics.median(durations),
                "maximum": max(durations),
            },
            "near_silent_or_silent_count": sum(
                row["rms_dbfs"] is None or row["rms_dbfs"] < -50 for row in selected
            ),
            "media_root": shared.relative(args.media_root),
            "review_dataset_sha256": shared.sha256(
                args.output_dir / "review-dataset.json"
            ),
            "sample_sha256": shared.sha256(args.output_dir / "sample.jsonl"),
            "human_answers_used_for_training": False,
        }
    )
    shared.write_json(args.output_dir / "sampling-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
