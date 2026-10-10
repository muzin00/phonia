#!/usr/bin/env python3
"""Sample train-only /m/ and /n/ boundaries for the existing local review UI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import wave
from array import array
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
JVS = ROOT / "poc/phoneme-speaker-dataset"
SRC = ROOT / "artifacts/phoneme-src4vc-training/src4vc-20261009-v1"
DEFAULT_OUTPUT = BASE / "data/review"
DEFAULT_MEDIA = ROOT / "artifacts/phoneme-consonant-review/mn-100-20261010-v1/media"
PHONEMES = ("m", "n")
RATE = 24000
MINIMUM_FRAMES = 720


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def priority(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def load_pools() -> tuple[dict, dict, dict, dict]:
    """Open alignments only in the two previously frozen training splits."""
    jvs_train = set(
        read_json(JVS / "config/dataset-split.json")["speaker_splits"]["train"]
    )
    src_train = {
        "ext_" + speaker.lower()
        for speaker in read_json(SRC / "speaker-splits.json")["splits"]["train"]
    }
    if len(jvs_train) != 70 or len(src_train) != 70:
        raise ValueError("Expected the existing 70 + 70 training speakers")
    transcripts = {}
    with (JVS / "data/generated/expected-phonemes.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            if record["speaker_id"] in jvs_train:
                if record["split"] != "train":
                    raise ValueError("Training-speaker metadata has another split")
                transcripts[record["utterance_id"]] = record

    pools = {(corpus, phone): [] for corpus in ("JVS", "SRC4VC") for phone in PHONEMES}
    alignment_counts = Counter()
    excluded_short = Counter()
    inventories = {}
    sources = (
        ("JVS", JVS / "data/generated/alignments/raw", jvs_train),
        ("SRC4VC", SRC / "alignments/raw", src_train),
    )
    for corpus, directory, training_speakers in sources:
        inventory = hashlib.sha256()
        for speaker in sorted(training_speakers):
            paths = sorted((directory / speaker).glob("*.json"))
            if not paths:
                raise ValueError(f"No training alignments for {speaker}")
            for path in paths:
                raw_bytes = path.read_bytes()
                alignment = json.loads(raw_bytes)
                if alignment["speaker_id"] != speaker:
                    raise ValueError(f"Wrong speaker in {path}")
                alignment_counts[corpus] += 1
                inventory.update(relative(path).encode() + b"\0")
                inventory.update(hashlib.sha256(raw_bytes).digest())
                for index, interval in enumerate(alignment["intervals"]):
                    phone = interval["phoneme"]
                    # Case sensitive: the moraic nasal /N/ is not /n/.
                    if phone not in PHONEMES:
                        continue
                    start, end = interval["start_sec"], interval["end_sec"]
                    first, last = round(start * RATE), round(end * RATE)
                    if first < 0 or last <= first:
                        raise ValueError(f"Invalid boundary in {path}")
                    if last - first < MINIMUM_FRAMES:
                        excluded_short[f"{corpus}/{phone}"] += 1
                        continue
                    pools[(corpus, phone)].append(
                        {
                            "item_id": f"{alignment['utterance_id']}--phone-{index:03d}-{phone}",
                            "corpus": corpus,
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
                            "raw_alignment_file": relative(path),
                            "raw_alignment_sha256": hashlib.sha256(
                                raw_bytes
                            ).hexdigest(),
                            "previous_phoneme": alignment["intervals"][index - 1][
                                "phoneme"
                            ]
                            if index > 0
                            else None,
                            "next_phoneme": alignment["intervals"][index + 1]["phoneme"]
                            if index + 1 < len(alignment["intervals"])
                            else None,
                        }
                    )
        inventories[corpus] = inventory.hexdigest()
    pool_summary = {
        "training_speaker_counts": {"JVS": len(jvs_train), "SRC4VC": len(src_train)},
        "training_alignment_counts": dict(alignment_counts),
        "training_alignment_inventory_sha256": inventories,
        "eligible_interval_counts": {
            f"{c}/{p}": len(rows) for (c, p), rows in pools.items()
        },
        "excluded_short_interval_counts": dict(sorted(excluded_short.items())),
    }
    return pools, transcripts, pool_summary, {"JVS": jvs_train, "SRC4VC": src_train}


def select(pools: dict, seed: int, per_stratum: int) -> list[dict]:
    selected = []
    for stratum, pool in sorted(pools.items()):
        if len(pool) < per_stratum:
            raise ValueError(f"Insufficient intervals in {stratum}")
        ordered = sorted(
            pool, key=lambda row: (priority(seed, row["item_id"]), row["item_id"])
        )
        selected.extend(dict(row) for row in ordered[:per_stratum])
    # Shuffle the presentation independently of selection, interleaving strata.
    return sorted(selected, key=lambda row: priority(seed + 1, row["item_id"]))


def stage_audio(selected: list[dict], transcripts: dict, media: Path) -> None:
    source_cache = {}
    for row in selected:
        source_path = (ROOT / row["source_file"]).resolve()
        source_path.relative_to(ROOT)
        utterance_id = row["utterance_id"]
        if utterance_id not in source_cache:
            digest = sha256(source_path)
            if digest != row["source_sha256"]:
                raise ValueError(f"Source checksum changed: {source_path}")
            if row["corpus"] in ("JVS", "Common Voice"):
                metadata = transcripts[utterance_id]
            elif row["corpus"] == "SRC4VC":
                metadata = read_json(SRC / "prepared" / f"{utterance_id}.json")[
                    "utterance"
                ]
            else:
                raise ValueError(f"Unsupported corpus: {row['corpus']}")
            for field in ("speaker_id", "source_file", "source_sha256", "split"):
                if metadata[field] != row[field]:
                    raise ValueError(
                        f"Source metadata mismatch: {utterance_id}/{field}"
                    )
            original = media / "utterances" / f"{utterance_id}.wav"
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, original)
            if sha256(original) != digest:
                raise ValueError("Review copy differs from training waveform")
            source_cache[utterance_id] = metadata
        metadata = source_cache[utterance_id]
        row["transcript"] = metadata["transcript"]
        row["source_subset"] = metadata["source_subset"]
        row["audio_url"] = f"/media/utterances/{utterance_id}.wav"
        with wave.open(str(source_path), "rb") as source:
            if (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                source.getcomptype(),
            ) != (RATE, 1, 2, "NONE"):
                raise ValueError("Expected unchanged 24 kHz mono PCM16 waveform")
            if row["last_frame"] > source.getnframes():
                raise ValueError(f"Boundary exceeds waveform: {row['item_id']}")
            row["source_frame_count"] = source.getnframes()
            source.setpos(row["first_frame"])
            pcm = source.readframes(row["last_frame"] - row["first_frame"])
        samples = array("h", pcm)
        rms = (
            math.sqrt(sum(float(value) ** 2 for value in samples) / len(samples))
            / 32768
        )
        row["rms_dbfs"] = 20 * math.log10(rms) if rms else None
        row["duration_sec"] = len(samples) / RATE
        clip = media / "segments" / f"{row['item_id']}.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(clip), "wb") as destination:
            destination.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            destination.writeframes(pcm)
        row["segment_file"] = relative(clip)
        row["segment_sha256"] = sha256(clip)


def review_dataset(
    selected: list[dict],
    seed: int,
    *,
    dataset_id: str = "jvs-src4vc-mn-training-review",
    corpus_title: str = "JVS・SRC4VC",
) -> dict:
    return {
        "schemaVersion": 2,
        "datasetId": dataset_id,
        "datasetVersion": f"seed{seed}-n{len(selected)}",
        "title": f"{corpus_title} /m/・/n/ 学習前レビュー（{len(selected)}区間）",
        "protocol": {"id": "phoneme-audibility-training-review", "version": "1"},
        "playback": {"contextPaddingSec": 0.1},
        "form": {
            "candidateQuestions": [
                {
                    "id": "perceived_content",
                    "prompt": "「候補Aだけ再生」で、表示された音素は聞こえますか？",
                    "choices": [
                        {
                            "value": "target_phoneme",
                            "label": "表示された音素が聞こえる",
                        },
                        {
                            "value": "target_with_other",
                            "label": "表示された音素と隣接する音が聞こえる",
                        },
                        {"value": "other_only", "label": "別の音だけが聞こえる"},
                        {"value": "near_silence", "label": "ほぼ音が聞こえない"},
                        {"value": "uncertain", "label": "判断できない"},
                    ],
                }
            ],
            "reviewStatus": {
                "prompt": "切り出し位置も確認して、この区間を表示された音素の学習データとして利用できますか？",
                "choices": [
                    {"value": "accepted", "label": "利用できる"},
                    {"value": "rejected", "label": "利用できない"},
                    {"value": "uncertain", "label": "判断できない"},
                ],
            },
        },
        "items": [
            {
                "id": row["item_id"],
                "utterance": {
                    "id": row["utterance_id"],
                    "text": row["transcript"],
                    "audioUrl": row["audio_url"],
                },
                "target": {
                    "label": row["phoneme"],
                    "index": row["interval_index"],
                    "unitCount": 1,
                    "tags": [
                        row["corpus"],
                        "train",
                        f"区間 {row['duration_sec'] * 1000:.0f}ms",
                    ],
                },
                "candidates": [
                    {
                        "id": "A",
                        "status": "available",
                        "segment": {
                            "startSec": row["start_sec"],
                            "endSec": row["end_sec"],
                        },
                    }
                ],
            }
            for row in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--per-stratum", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA)
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    pools, transcripts, summary, training = load_pools()
    selected = select(pools, args.seed, args.per_stratum)
    if len({row["item_id"] for row in selected}) != args.per_stratum * 4:
        raise ValueError("Review IDs must be unique")
    if any(row["speaker_id"] not in training[row["corpus"]] for row in selected):
        raise ValueError("Non-training speaker in sample")
    stage_audio(selected, transcripts, args.media_root)
    dataset = review_dataset(selected, args.seed)
    write_json(args.output_dir / "review-dataset.json", dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            "seed": args.seed,
            "dataset_id": dataset["datasetId"],
            "dataset_version": dataset["datasetVersion"],
            "sampling": "smallest_seeded_sha256_without_replacement_within_corpus_and_phoneme",
            "presentation_order": "independent_seeded_sha256",
            "phonemes": list(PHONEMES),
            "minimum_interval_frames": MINIMUM_FRAMES,
            "sample_rate_hz": RATE,
            "audio_volume_filter": False,
            "reviewed": False,
            "selected_count": len(selected),
            "selected_stratum_counts": dict(
                sorted(
                    Counter(f"{r['corpus']}/{r['phoneme']}" for r in selected).items()
                )
            ),
            "selected_speaker_count": len({row["speaker_id"] for row in selected}),
            "selected_speakers_by_corpus": {
                corpus: len(
                    {r["speaker_id"] for r in selected if r["corpus"] == corpus}
                )
                for corpus in ("JVS", "SRC4VC")
            },
            "selected_source_subset_counts": dict(
                sorted(Counter(r["source_subset"] for r in selected).items())
            ),
            "selected_source_file_count": len({row["source_file"] for row in selected}),
            "duration_sec": {
                "minimum": min(durations),
                "median": statistics.median(durations),
                "maximum": max(durations),
            },
            "near_silent_or_silent_count": sum(
                r["rms_dbfs"] is None or r["rms_dbfs"] < -50 for r in selected
            ),
            "media_root": relative(args.media_root),
            "review_dataset_sha256": sha256(args.output_dir / "review-dataset.json"),
            "sample_sha256": sha256(args.output_dir / "sample.jsonl"),
            "human_answers_used_for_training": False,
        }
    )
    write_json(args.output_dir / "sampling-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
