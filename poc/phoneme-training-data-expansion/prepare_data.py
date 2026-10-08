"""Prepare one checksum-pinned, train-only Japanese data expansion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import threading
import urllib.request
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "poc/phoneme-speaker-dataset/scripts"))
import build_vowel_manifest as vowels
import generate_expected_phonemes as expected
import run_julius_dataset as alignment

CONFIG = BASE / "config/protocol.json"
G2P_LOCK = threading.Lock()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    alignment.atomic_write_json(path, value)


def write_rows(path, rows):
    alignment.atomic_write_jsonl(path, rows)


def stable_hash(*values):
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def checked(path, checksum):
    if alignment.sha256(path) != checksum:
        raise ValueError(f"checksum mismatch: {path}")


def download_file(item, output, repository, revision):
    path = output / item["path"]
    checksum = item.get("lfs", {}).get("oid")
    if path.exists():
        if path.stat().st_size != item["size"]:
            raise ValueError(f"existing download has wrong size: {path}")
        if checksum:
            checked(path, checksum)
        return path
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{item['path']}"
    request = urllib.request.Request(url, headers={"User-Agent": "phonia-research/1.0"})
    temporary = path.with_suffix(path.suffix + ".part")
    with (
        urllib.request.urlopen(request, timeout=120) as source,
        temporary.open("wb") as target,
    ):
        while data := source.read(1024 * 1024):
            target.write(data)
    if temporary.stat().st_size != item["size"]:
        raise ValueError(f"download size mismatch: {path}")
    if checksum:
        checked(temporary, checksum)
    temporary.replace(path)
    return path


def download(config, run):
    output = run / "source"
    output.mkdir(parents=True, exist_ok=True)
    index = output / "index.json"
    repository, revision = config["source_repository"], config["source_revision"]
    if index.exists():
        files = read_json(index)
    else:
        url = f"https://huggingface.co/api/datasets/{repository}/tree/{revision}?recursive=true"
        with urllib.request.urlopen(url, timeout=60) as source:
            files = json.load(source)
        files = [
            item
            for item in files
            if item["type"] == "file"
            and (
                item["path"].startswith("train_")
                and item["path"].endswith(".parquet")
                or item["path"] in ("README.md", "dataset_info.json")
            )
        ]
        if len(files) != 39:
            raise ValueError("unexpected pinned train file list")
        write_json(index, files)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(download_file, item, output, repository, revision): item
            for item in files
        }
        for count, future in enumerate(as_completed(futures), 1):
            path = future.result()
            print(f"download {count}/{len(files)}: {path.name}", flush=True)
    write_json(
        output / "manifest.json",
        {
            "repository": repository,
            "revision": revision,
            "source_split": "train",
            "files": {
                item["path"]: alignment.sha256(output / item["path"]) for item in files
            },
        },
    )


def select_speakers(rows, config):
    grouped = defaultdict(list)
    for row in rows:
        if (
            not row.get("client_id")
            or not row.get("sentence")
            or row.get("locale") != "ja"
        ):
            raise ValueError("expected Japanese rows with client_id and sentence")
        grouped[row["client_id"]].append(row)
    candidates = [
        s
        for s, clips in grouped.items()
        if len(clips) >= config["minimum_candidate_clips"]
    ]
    order = sorted(
        candidates,
        key=lambda s: (
            -len(grouped[s]),
            stable_hash("speaker", config["selection_seed"], s),
        ),
    )
    chosen = order[: config["additional_speakers"]]
    if len(chosen) != config["additional_speakers"]:
        raise ValueError(f"not enough candidate speakers: {len(chosen)}")
    result = []
    for client_id in chosen:
        clips = sorted(
            grouped[client_id],
            key=lambda r: stable_hash(
                "clip", config["selection_seed"], r["audio_path"], r["sentence"]
            ),
        )
        clips = clips[: config["maximum_clips_per_additional_speaker"]]
        result.extend(
            {**r, "speaker_id": "cv17_" + stable_hash("speaker_id", client_id)[:16]}
            for r in clips
        )
    return result


def prepare_clip(row, run, version, dictionary):
    identifier = (
        "cv17_" + stable_hash(row["client_id"], row["audio_path"], row["sentence"])[:24]
    )
    saved = run / "prepared" / f"{identifier}.json"
    if saved.exists():
        value = read_json(saved)
        if value.get("utterance"):
            checked(
                ROOT / value["utterance"]["source_file"],
                value["utterance"]["source_sha256"],
            )
        return value
    source = run / "audio" / row["speaker_id"] / f"{identifier}.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phonia-cv-convert-") as folder:
        compressed = Path(folder) / "clip.mp3"
        compressed.write_bytes(row["audio_bytes"])
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@24000",
                "-c",
                "1",
                str(compressed),
                str(source),
            ],
            check=True,
            capture_output=True,
        )
    with wave.open(str(source)) as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError("conversion is not 24 kHz mono PCM")
        frames = wav.getnframes()
    record = {
        "schema_version": 1,
        "utterance_id": identifier,
        "speaker_id": row["speaker_id"],
        "source_file": str(source.relative_to(ROOT)),
        "source_sha256": alignment.sha256(source),
        "source_subset": "common_voice17_ja_train",
        "transcript": row["sentence"],
        "split": "train",
        "evaluation_role": "training",
        "learning_curve_cohorts": [140],
        "session_id": None,
        "duration_sec": frames / 24000,
        "sample_rate_hz": 24000,
        "channels": 1,
        "sample_width_bytes": 2,
        "frame_count": frames,
        "original_audio_sha256": hashlib.sha256(row["audio_bytes"]).hexdigest(),
        "original_audio_path": row["audio_path"],
        "source_split": "train",
    }
    try:
        with G2P_LOCK:
            phonemes = expected.convert_record(record, version, dictionary)
        results = alignment.process_records(
            [phonemes],
            run / "alignments",
            lambda item, raw, fail: alignment.align_record(
                item,
                raw,
                fail,
                alignment.DEFAULT_JULIUS,
                alignment.DEFAULT_SEGMENTATION_KIT
                / "models/hmmdefs_monof_mix16_gid.binhmm",
            ),
        )
        segments, failure = vowels.process_utterance(
            phonemes, results[0], 0.03, -50.0, True
        )
        excluded = [
            s
            for s in segments
            if s["frame_count"] < 720
            or any(f in s["quality_flags"] for f in ("silent", "near_silent"))
        ]
        eligible = [s for s in segments if s not in excluded]
        value = {
            "utterance": record,
            "expected": phonemes,
            "alignment": results[0],
            "eligible": eligible,
            "excluded": excluded,
            "failure": failure,
        }
    except Exception as error:  # noqa: BLE001 - persist each failed corpus utterance.
        value = {
            "utterance": record,
            "eligible": [],
            "excluded": [],
            "failure": {
                "utterance_id": identifier,
                "stage": "g2p_or_alignment",
                "type": type(error).__name__,
                "reason": str(error),
            },
        }
    write_json(saved, value)
    return value


def prepare(config, run):
    import pyopenjtalk
    from pyarrow import parquet

    checked(ROOT / config["evaluation_manifest"], config["evaluation_manifest_sha256"])
    model = alignment.DEFAULT_SEGMENTATION_KIT / "models/hmmdefs_monof_mix16_gid.binhmm"
    alignment_config = read_json(
        ROOT / "poc/phoneme-speaker-dataset/config/vowel-dataset.json"
    )["alignment"]
    checked(model, alignment_config["model_sha256"])
    if alignment.executable_version(alignment.DEFAULT_JULIUS) != "4.6":
        raise ValueError("Julius version differs")
    manifest = read_json(run / "source/manifest.json")
    rows = []
    for name, checksum in manifest["files"].items():
        checked(run / "source" / name, checksum)
        if not name.endswith(".parquet"):
            continue
        for index, row in enumerate(
            parquet.read_table(
                run / "source" / name,
                columns=["client_id", "sentence", "locale", "audio"],
            ).to_pylist()
        ):
            audio = row.pop("audio")
            rows.append(
                {
                    **row,
                    "audio_path": audio["path"],
                    "audio_bytes": audio["bytes"],
                    "shard": name,
                    "shard_row": index,
                }
            )
    selected = select_speakers(rows, config)
    del rows
    selection = [
        {k: v for k, v in row.items() if k != "audio_bytes"} for row in selected
    ]
    path = run / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("data selection changed")
    write_json(path, selection)
    dictionary = pyopenjtalk.OPEN_JTALK_DICT_DIR
    if isinstance(dictionary, bytes):
        dictionary = dictionary.decode()
    version = importlib.metadata.version("pyopenjtalk")
    prepared = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(prepare_clip, row, run, version, dictionary) for row in selected
        ]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                prepared.append(future.result())
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            if index % 100 == 0 or index == len(futures):
                success = sum(p["failure"] is None for p in prepared)
                print(f"prepare {index}/{len(futures)}; aligned {success}", flush=True)
    prepared.sort(key=lambda p: p["utterance"]["utterance_id"])
    segments = sorted(
        [s for p in prepared for s in p["eligible"]],
        key=lambda s: s["vowel_interval_id"],
    )
    counts = Counter((s["speaker_id"], s["normalized_phoneme"]) for s in segments)
    speakers = sorted({r["speaker_id"] for r in selected})
    if len(speakers) != 70 or any(counts[s, v] < 2 for s in speakers for v in "aiueo"):
        raise ValueError("additional speakers lack required vowel coverage")
    sources = [p["utterance"] for p in prepared]
    if len({s["original_audio_sha256"] for s in sources}) != len(sources):
        raise ValueError("duplicate original audio in selected training inputs")
    write_rows(run / "additional-segments.jsonl", segments)
    write_rows(run / "additional-utterances.jsonl", sources)
    write_json(
        run / "preparation-report.json",
        {
            "status": "completed",
            "speakers": len(speakers),
            "selected_clips": len(selected),
            "aligned_clips": sum(p["failure"] is None for p in prepared),
            "eligible_segments": len(segments),
            "excluded_segments": sum(len(p["excluded"]) for p in prepared),
            "failures": [p["failure"] for p in prepared if p["failure"] is not None],
            "by_vowel": dict(Counter(s["normalized_phoneme"] for s in segments)),
            "source_hours": sum(r["duration_sec"] for r in sources) / 3600,
            "eligible_vowel_hours": sum(s["frame_count"] for s in segments)
            / 24000
            / 3600,
            "g2p_version": version,
            "pyarrow_version": importlib.metadata.version("pyarrow"),
            "source_split": "train",
            "selection_without_evaluation_metrics": True,
            "outputs_sha256": {
                p.name: alignment.sha256(p)
                for p in (
                    run / "additional-segments.jsonl",
                    run / "additional-utterances.jsonl",
                    run / "selection.json",
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "prepare"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    {"download": download, "prepare": prepare}[args.stage](config, run)


if __name__ == "__main__":
    main()
