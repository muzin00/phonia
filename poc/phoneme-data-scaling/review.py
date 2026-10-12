"""Prepare a reproducible, train-only human review; never train a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from collections import Counter, defaultdict

from selection import BASE, ROOT, rank, read, sha
from selection import fixed as write


def fixed_write(path, value):
    if path.exists():
        assert read(path) == value, f"Frozen output changed: {path}"
    else:
        write(path, value)


def select_utterances(rows, seed):
    grouped = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            grouped[row["speaker_id"]].append(row)
    return [
        min(grouped[s], key=lambda v: (rank(seed, v["path"]), v["path"]))
        for s in sorted(grouped)
    ]


def sample_segments(rows, count, seed, display_seed):
    assert all(r["source_split"] == "train" and not r["quality_flags"] for r in rows)
    assert len({r["segment_id"] for r in rows}) == len(rows)
    if len(rows) < count:
        raise ValueError("Insufficient QC-passing segments; do not replace failures")
    groups = defaultdict(list)
    for row in rows:
        groups[row["phoneme"]].append(row)
    for phone in groups:
        groups[phone].sort(key=lambda r: (rank(seed, r["segment_id"]), r["segment_id"]))
    phones = sorted(groups, key=lambda p: (rank(seed, p), p))
    selected, depth = [], 0
    while len(selected) < count:
        for phone in phones:
            if depth < len(groups[phone]):
                selected.append(groups[phone][depth])
                if len(selected) == count:
                    break
        depth += 1
    return sorted(
        selected, key=lambda r: (rank(display_seed, r["segment_id"]), r["segment_id"])
    )


def extract(config, run):
    sys.path.insert(0, str(BASE.parent / "phoneme-live-recording"))
    import runtime as r

    engine = r.Runtime(read(ROOT / config["runtime_freeze"])["config"])
    frozen = {
        "asr_sha256": sha(run / "asr.json"),
        "extract_config": engine.extract_config,
        "phones": list(engine.phones),
    }
    fixed_write(run / "extraction-freeze.json", frozen)
    raw_rows, accepted, inventory = [], [], []
    clips = read(run / "asr.json")
    for index, row in enumerate(clips):
        assert row["split"] == "train"
        source = ROOT / row["local_path"]
        assert sha(source) == row["sha256"]
        if row["status"] != "transcribed":
            inventory.append({"source": row["local_path"], "status": "asr_failed"})
            continue
        output = run / "extraction" / source.stem
        output.mkdir(parents=True, exist_ok=True)
        destination = output / "processed"
        try:
            if (destination / "result.json").exists():
                result = read(destination / "result.json")
                for name, digest in result["files"].items():
                    assert sha(ROOT / name) == digest, name
            else:
                wav = output / "input24k.wav"
                subprocess.run(
                    [
                        "/usr/bin/afconvert",
                        "-f",
                        "WAVE",
                        "-d",
                        "LEI16@24000",
                        "-c",
                        "1",
                        str(source),
                        str(wav),
                    ],
                    check=True,
                    capture_output=True,
                )
                with wave.open(str(wav)) as f:
                    assert (f.getframerate(), f.getsampwidth(), f.getnchannels()) == (
                        24000,
                        2,
                        1,
                    )
                    pcm = r.np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
                result = engine.process(
                    pcm,
                    destination,
                    {
                        "utterance_id": row["corpus"] + "_" + row["sha256"][:24],
                        "speaker_id": row["speaker_id"].lower(),
                        "transcript": row["asr"]["text"].strip(),
                        "origin": row["corpus"] + "_train_expansion_review",
                        "role": "training",
                        "session_id": None,
                        "original_audio_sha256": row["sha256"],
                        "original_audio_path": row["path"],
                    },
                )
            expected = read(destination / "expected.json")
            raw_path = (
                destination
                / "alignment/raw"
                / expected["speaker_id"]
                / (expected["utterance_id"] + ".json")
            )
            extracted = (
                r.t.h.primitives.extract(
                    raw_path, expected, engine.extract_config, {}, Counter()
                )
                if raw_path.exists()
                else []
            )
            clip_accepted = []
            for segment, _ in extracted:
                segment = {
                    **segment,
                    "source_split": "train",
                    "review_group": row["review_group"],
                    "corpus": row["corpus"],
                    "transcript_provenance": row["transcript_provenance"],
                    "original_mp3": row["local_path"],
                    "original_mp3_sha256": row["sha256"],
                    "transcript": expected["transcript"],
                    "unit_count": len(expected["raw_phonemes"]),
                }
                segment["exclusion_reasons"] = segment["quality_flags"] + (
                    []
                    if segment["phoneme"] in engine.phones
                    else ["outside_model_phone_inventory"]
                )
                raw_rows.append(segment)
                if not segment["exclusion_reasons"]:
                    clip_accepted.append(segment)
            saved = [
                json.loads(line)
                for line in (destination / "segments.jsonl").read_text().splitlines()
            ]
            assert {v["segment_id"] for v in clip_accepted} == {
                v["segment_id"] for v in saved
            }
            accepted.extend(clip_accepted)
            inventory.append(
                {
                    "source": row["local_path"],
                    "status": result["status"],
                    "candidates": len(extracted),
                    "accepted": len(saved),
                    "result_sha256": sha(destination / "result.json"),
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve the fixed denominator.
            inventory.append(
                {"source": row["local_path"], "status": "failed", "error": str(error)}
            )
        print(
            f"Extraction {index + 1}/{len(clips)} {inventory[-1]['status']}", flush=True
        )
    fixed_write(run / "pre-qc.json", raw_rows)
    fixed_write(run / "post-qc.json", accepted)
    fixed_write(run / "excluded.json", [v for v in raw_rows if v["exclusion_reasons"]])
    fixed_write(run / "extraction-inventory.json", inventory)


def build(config, run):
    pool = read(run / "post-qc.json")
    chosen = []
    for group in config["groups"]:
        chosen.extend(
            sample_segments(
                [v for v in pool if v["review_group"] == group],
                config["segments_per_group"],
                config["segment_seed"],
                config["display_seed"],
            )
        )
    chosen.sort(key=lambda v: rank(config["display_seed"], v["segment_id"]))
    fixed_write(run / "selected-segments.json", chosen)
    dataset = read(ROOT / "tools/phoneme-review-ui/public/examples/review-dataset.json")
    dataset.update(
        {
            "datasetId": run.name,
            "datasetVersion": "1",
            "title": "話者・発話拡張レビュー100区間（4集合 × 25件）",
            "protocol": {"id": "data-expansion-phone-quality", "version": "1"},
            "items": [],
        }
    )
    dataset["form"] = {
        "candidateQuestions": [
            {
                "id": "perceived_content",
                "prompt": "表示された音素と切り出された音は合っていますか？ JTubeの書き起こしは自動生成です。",
                "choices": [
                    {"value": "target_phone", "label": "対象の音素が聞こえる"},
                    {"value": "mixed", "label": "対象音素に隣の音が混ざる"},
                    {"value": "wrong_phone", "label": "別の音が聞こえる"},
                    {"value": "noise_or_silence", "label": "雑音・無音が主"},
                    {"value": "uncertain", "label": "判断できない"},
                ],
            }
        ],
        "reviewStatus": {
            "prompt": "この区間を、表示された音素の学習データとして利用できますか？",
            "choices": [
                {"value": "accepted", "label": "利用できる"},
                {"value": "rejected", "label": "利用できない"},
                {"value": "uncertain", "label": "判断できない"},
            ],
        },
    }
    audit = []
    for row in chosen:
        source = ROOT / row["source_file"]
        assert sha(source) == row["source_sha256"]
        target = run / "media" / (row["utterance_id"] + ".wav")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha(target) == row["source_sha256"]
        with wave.open(str(target)) as f:
            assert (f.getframerate(), f.getsampwidth(), f.getnchannels()) == (
                24000,
                2,
                1,
            )
            assert 0 <= row["start_frame"] < row["end_frame"] <= f.getnframes()
            f.setpos(row["start_frame"])
            size = row["end_frame"] - row["start_frame"]
            assert 720 <= size <= 6000 and row["rms_dbfs"] >= -50
            crop_sha = hashlib.sha256(f.readframes(size)).hexdigest()
        dataset["items"].append(
            {
                "id": row["segment_id"],
                "utterance": {
                    "id": row["utterance_id"],
                    "text": row["transcript"],
                    "audioUrl": "/media/" + target.name,
                },
                "target": {
                    "label": row["phoneme"],
                    "index": row["interval_index"] - 1,
                    "unitCount": 1,
                    "tags": [
                        {
                            "existing_cv": "Common Voice・既存話者の発話追加",
                            "new_cv": "Common Voice・新規話者",
                            "existing_jtube": "JTube・既存話者の発話追加",
                            "new_jtube": "JTube・新規話者",
                        }[row["review_group"]],
                        "書き起こし：自動生成・未確認"
                        if row["corpus"] == "jtube"
                        else "テキスト：Common Voice配布文",
                    ],
                },
                "candidates": [
                    {
                        "id": "A",
                        "status": "available",
                        "segment": {
                            "startSec": row["start_frame"] / 24000,
                            "endSec": row["end_frame"] / 24000,
                        },
                    }
                ],
            }
        )
        audit.append(
            {
                "item_id": row["segment_id"],
                "media_sha256": sha(target),
                "crop_pcm_sha256": crop_sha,
            }
        )
    fixed_write(run / "review-dataset.json", dataset)
    inventory = read(run / "extraction-inventory.json")
    summary = {
        "sampling_frame": config["sampling_frame"],
        "selected_utterances": len(read(run / "selected-utterances.json")),
        "extraction_status": dict(Counter(v["status"] for v in inventory)),
        "pre_qc_count": len(read(run / "pre-qc.json")),
        "post_qc_count": len(pool),
        "review_count": len(chosen),
        "review_speakers": len({r["speaker_id"] for r in chosen}),
        "pool_phone_counts": dict(sorted(Counter(r["phoneme"] for r in pool).items())),
        "sample_phone_counts": dict(
            sorted(Counter(r["phoneme"] for r in chosen).items())
        ),
        "dataset_sha256": sha(run / "review-dataset.json"),
        "audit": audit,
        "training_started": False,
        "group_counts": dict(Counter(v["review_group"] for v in chosen)),
        "unreviewed_phones": sorted(
            set(read(run / "extraction-freeze.json")["phones"])
            - {v["phoneme"] for v in chosen}
        ),
    }
    fixed_write(run / "review-audit.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "audit"},
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["extract", "build"])
    args = parser.parse_args()
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    for name, digest in read(run / "freeze.json")["files"].items():
        assert sha(ROOT / name) == digest, name
    if args.phase == "build":
        build(config, run)
        return
    from pathlib import Path

    original_run = subprocess.run

    def bounded_run(command, *args, **kwargs):
        if isinstance(command, list) and Path(command[0]).name == "julius":
            if "input" in kwargs:
                with wave.open(kwargs["input"].strip()) as wav:
                    seconds = wav.getnframes() / wav.getframerate()
                if seconds > config["maximum_utterance_seconds"]:
                    raise ValueError(f"alignment_duration_limit: {seconds:.3f}s")
            kwargs["timeout"] = config["julius_timeout_seconds"]
        return original_run(command, *args, **kwargs)

    subprocess.run = bounded_run
    try:
        extract(config, run)
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    main()
