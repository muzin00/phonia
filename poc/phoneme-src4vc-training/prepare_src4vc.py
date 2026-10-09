"""Select and align original spoken SRC4VC audio, without reading held-out WAVs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import re
import subprocess
import sys
import tempfile
import threading
import wave
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-training-data-expansion"))
import prepare_data as source

CONFIG = BASE / "config/protocol.json"
read_json, write_json, write_rows, checked = (
    source.read_json,
    source.write_json,
    source.write_rows,
    source.checked,
)
ARCHIVES = threading.local()
G2P_LOCK = threading.Lock()


def spoken_subset(stem):
    if re.fullmatch(r"RECITATION_\d+", stem):
        return "RECITATION"
    if re.fullmatch(r"JVNV(anger|disgust|fear|happiness|sadness|surprise)_\d+", stem):
        return "JVNV"
    if re.fullmatch(r"(CALLS|STUDIES)_\d+", stem):
        return stem.split("_")[0]
    return None


def select_inventory(names, config):
    if (
        config["include_singing"]
        or config["include_restored_audio"]
        or config["maximum_spoken_clips_per_speaker"] != 50
        or config["included_subsets"] != ["RECITATION", "JVNV", "CALLS", "STUDIES"]
    ):
        raise ValueError("unsupported spoken-only inventory policy")
    grouped = defaultdict(list)
    speakers = set()
    for name in names:
        metadata = re.fullmatch(r"SRC4VC_ver1/(SRC4VC\d{3})/speaker_metadata.yml", name)
        if metadata:
            speakers.add(metadata[1])
        match = re.fullmatch(r"SRC4VC_ver1/(SRC4VC\d{3})/wav/([^/]+)\.wav", name)
        if match and (subset := spoken_subset(match[2])):
            transcript = name.replace("/wav/", "/txt/").removesuffix(".wav") + ".txt"
            if transcript not in names:
                raise ValueError(f"missing transcript: {name}")
            grouped[match[1]].append(
                {"raw_member": name, "transcript_member": transcript, "subset": subset}
            )
    if speakers != {f"SRC4VC{i:03d}" for i in range(1, 101)}:
        raise ValueError("expected official 100-speaker inventory")
    expected_counts = {"RECITATION": 10, "JVNV": 30, "CALLS": 5, "STUDIES": 5}
    for speaker in speakers:
        counts = Counter(r["subset"] for r in grouped[speaker])
        if counts != expected_counts:
            raise ValueError(f"unexpected spoken inventory for {speaker}: {counts}")
    order = sorted(
        speakers,
        key=lambda s: source.stable_hash("src4vc-speaker", config["selection_seed"], s),
    )
    counts = config["speaker_split_counts"]
    if counts != {"train": 70, "validation_reserved": 15, "test_reserved": 15}:
        raise ValueError("expected fixed 70/15/15 split")
    splits = {
        "train": order[:70],
        "validation_reserved": order[70:85],
        "test_reserved": order[85:],
    }
    selected = []
    for speaker in splits["train"]:
        for item in sorted(grouped[speaker], key=lambda r: r["raw_member"]):
            selected.append(
                {
                    **item,
                    "source_speaker_id": speaker,
                    "speaker_id": config["additional_speaker_prefix"] + speaker[6:],
                    "utterance_id": speaker.lower()
                    + "__"
                    + Path(item["raw_member"]).stem.lower(),
                }
            )
    return splits, selected


def archive_for(path):
    if getattr(ARCHIVES, "path", None) != path:
        if getattr(ARCHIVES, "archive", None):
            ARCHIVES.archive.close()
        ARCHIVES.archive, ARCHIVES.path = zipfile.ZipFile(path), path
    return ARCHIVES.archive


def prepare_clip(row, archive, run, version, dictionary):
    identifier = row["utterance_id"]
    saved = run / "prepared" / f"{identifier}.json"
    if saved.exists():
        value = read_json(saved)
        checked(
            ROOT / value["utterance"]["source_file"],
            value["utterance"]["source_sha256"],
        )
        if (
            value["utterance"]["original_audio_path"] != row["raw_member"]
            or value["utterance"]["transcript"] != row["sentence"]
        ):
            raise ValueError("prepared clip selection changed")
        return value
    raw = archive_for(archive).read(row["raw_member"])
    audio = run / "audio" / row["speaker_id"] / f"{identifier}.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phonia-src4vc-") as folder:
        original = Path(folder) / "original.wav"
        original.write_bytes(raw)
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@24000",
                "-c",
                "1",
                str(original),
                str(audio),
            ],
            check=True,
            capture_output=True,
        )
    with wave.open(str(audio)) as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError("expected 24 kHz mono PCM")
        frames = wav.getnframes()
    record = {
        "schema_version": 1,
        "utterance_id": identifier,
        "speaker_id": row["speaker_id"],
        "source_file": str(audio.relative_to(ROOT)),
        "source_sha256": source.alignment.sha256(audio),
        "source_subset": "src4vc_ver1_" + row["subset"],
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
        "original_audio_sha256": hashlib.sha256(raw).hexdigest(),
        "original_audio_path": row["raw_member"],
        "source_split": "train",
    }
    try:
        with G2P_LOCK:
            phonemes = source.expected.convert_record(record, version, dictionary)
        results = source.alignment.process_records(
            [phonemes],
            run / "alignments",
            lambda item, raw, fail: source.alignment.align_record(
                item,
                raw,
                fail,
                source.alignment.DEFAULT_JULIUS,
                source.alignment.DEFAULT_SEGMENTATION_KIT
                / "models/hmmdefs_monof_mix16_gid.binhmm",
            ),
        )
        segments, failure = source.vowels.process_utterance(
            phonemes, results[0], 0.03, -50.0, True
        )
        excluded, eligible = [], []
        for segment in segments:
            target = (
                excluded
                if segment["frame_count"] < 720
                or set(segment["quality_flags"]) & {"silent", "near_silent"}
                else eligible
            )
            target.append(segment)
        value = {
            "utterance": record,
            "expected": phonemes,
            "alignment": results[0],
            "eligible": eligible,
            "excluded": excluded,
            "failure": failure,
        }
    except Exception as error:  # noqa: BLE001 - retain failures without selecting replacement data.
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


def pinned_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"frozen selection/provenance changed: {path}")
    write_json(path, value)


def prepare(config, run):
    import pyopenjtalk
    import yaml

    archive = run / "source/SRC4VC_ver1.zip"
    if archive.stat().st_size != config["source_archive_size_bytes"]:
        raise ValueError("archive size mismatch")
    digest = source.alignment.sha256(archive)
    pinned_json(
        run / "source/manifest.json",
        {
            "archive": str(archive.relative_to(ROOT)),
            "sha256": digest,
            "bytes": archive.stat().st_size,
            "url": config["source_archive_url"],
            "terms": config["source_terms"],
            "project_url": config["source_project_url"],
            "checksum_provenance": "locally_computed_before_selection; official_checksum_not_provided",
        },
    )
    checked(ROOT / config["evaluation_manifest"], config["evaluation_manifest_sha256"])
    model = (
        source.alignment.DEFAULT_SEGMENTATION_KIT
        / "models/hmmdefs_monof_mix16_gid.binhmm"
    )
    alignment_config = read_json(
        ROOT / "poc/phoneme-speaker-dataset/config/vowel-dataset.json"
    )["alignment"]
    checked(model, alignment_config["model_sha256"])
    if source.alignment.executable_version(source.alignment.DEFAULT_JULIUS) != "4.6":
        raise ValueError("Julius version changed")
    with zipfile.ZipFile(archive) as zipped:
        splits, selected = select_inventory(set(zipped.namelist()), config)
        for row in selected:
            row["sentence"] = (
                zipped.read(row["transcript_member"]).decode("utf-8-sig").strip()
            )
            if not row["sentence"]:
                raise ValueError("empty transcript")
        metadata = {
            speaker: yaml.safe_load(
                zipped.read(f"SRC4VC_ver1/{speaker}/speaker_metadata.yml").decode(
                    "utf-8-sig"
                )
            )
            for population in splits.values()
            for speaker in population
        }
    pinned_json(
        run / "speaker-splits.json",
        {
            "seed": config["selection_seed"],
            "splits": splits,
            "metadata": metadata,
            "reserved_audio_read": False,
        },
    )
    pinned_json(run / "selection.json", selected)
    dictionary = pyopenjtalk.OPEN_JTALK_DICT_DIR
    if isinstance(dictionary, bytes):
        dictionary = dictionary.decode()
    version = importlib.metadata.version("pyopenjtalk")
    code_paths = [
        Path(__file__),
        Path(source.__file__),
        Path(source.expected.__file__),
        Path(source.alignment.__file__),
        Path(source.vowels.__file__),
    ]
    pinned_json(
        run / "preparation-freeze.json",
        {
            "config": config,
            "archive_sha256": digest,
            "selection_sha256": source.alignment.sha256(run / "selection.json"),
            "g2p_version": version,
            "julius_version": "4.6",
            "alignment_model_sha256": source.alignment.sha256(model),
            "code_sha256": {
                str(p.relative_to(ROOT)): source.alignment.sha256(p) for p in code_paths
            },
        },
    )
    prepared = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(prepare_clip, row, archive, run, version, dictionary)
            for row in selected
        ]
        for index, future in enumerate(as_completed(futures), 1):
            prepared.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(
                    f"prepare {index}/{len(futures)}; aligned {sum(p['failure'] is None for p in prepared)}",
                    flush=True,
                )
    prepared.sort(key=lambda p: p["utterance"]["utterance_id"])
    segments = sorted(
        [s for p in prepared for s in p["eligible"]],
        key=lambda s: s["vowel_interval_id"],
    )
    counts = Counter((s["speaker_id"], s["normalized_phoneme"]) for s in segments)
    speakers = {r["speaker_id"] for r in selected}
    if len(speakers) != 70 or any(counts[s, v] < 2 for s in speakers for v in "aiueo"):
        raise ValueError("required 70-speaker vowel coverage unavailable")
    utterances = [p["utterance"] for p in prepared]
    if len({r["original_audio_sha256"] for r in utterances}) != len(utterances):
        raise ValueError("duplicate selected raw audio")
    write_rows(run / "additional-segments.jsonl", segments)
    write_rows(run / "additional-utterances.jsonl", utterances)
    genders = Counter(
        str(metadata[s].get("gender", "unspecified")) for s in splits["train"]
    )
    ages = [
        float(metadata[s]["age"])
        for s in splits["train"]
        if metadata[s].get("age") is not None
    ]
    devices = Counter(
        str(metadata[s].get("device_normalized_name", "unspecified"))
        for s in splits["train"]
    )
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
            "subset_clips": dict(Counter(r["subset"] for r in selected)),
            "source_hours": sum(r["duration_sec"] for r in utterances) / 3600,
            "eligible_vowel_hours": sum(s["frame_count"] for s in segments)
            / 24000
            / 3600,
            "g2p_version": version,
            "source_split": "train",
            "reserved_audio_read": False,
            "selection_without_evaluation_metrics": True,
            "train_gender_labels": dict(genders),
            "train_age_years": {
                "count": len(ages),
                "min": min(ages),
                "max": max(ages),
                "mean": sum(ages) / len(ages),
            },
            "train_device_labels": dict(devices),
            "outputs_sha256": {
                str(p.relative_to(run)): source.alignment.sha256(p)
                for p in (
                    run / "additional-segments.jsonl",
                    run / "additional-utterances.jsonl",
                    run / "selection.json",
                    run / "speaker-splits.json",
                    run / "source/manifest.json",
                )
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = read_json(CONFIG)
    prepare(config, ROOT / config["run_directory"])


if __name__ == "__main__":
    main()
