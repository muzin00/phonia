#!/usr/bin/env python3
"""Prepare Julius inputs and align every utterance in the sample manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
import warnings
import wave
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "phonemes" / "expected.jsonl"
DEFAULT_JULIUS_DIR = PROJECT_DIR / "data" / "alignments" / "julius"
DEFAULT_INPUT_DIR = DEFAULT_JULIUS_DIR / "input"
DEFAULT_RAW_DIR = DEFAULT_JULIUS_DIR / "raw"
DEFAULT_RUN_METADATA = DEFAULT_JULIUS_DIR / "run.json"
DEFAULT_RUNTIME = Path.home() / ".local" / "share" / "phonia" / "julius"
DEFAULT_JULIUS = DEFAULT_RUNTIME / "4.6" / "bin" / "julius"
DEFAULT_SEGMENTATION_KIT = DEFAULT_RUNTIME / "segmentation-kit"
JULIUS_VERSION = "4.6"
SEGMENTATION_KIT_COMMIT = "e0e8bbaf98e27d19dfc6fe8312be607ad03592ad"
MODEL_ID = "segmentation-kit-monophone"
MODEL_SHA256 = "58952ccfe60f283c7efb1c22f9e012c7d297429700b7eeb4b60f7651a6e65940"
TARGET_SAMPLE_RATE_HZ = 16_000
FRAME_SHIFT_SEC = 0.01
WINDOW_SEC = 0.025
ALIGNMENT_OFFSET_SEC = WINDOW_SEC / 2
ALIGNMENT_LINE = re.compile(
    r"^\[\s*(\d+)\s+(\d+)\]\s+([-+0-9.eE]+)\s+(.+?)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Julius inputs and align all samples in the manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--raw-directory", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument("--julius-executable", type=Path, default=DEFAULT_JULIUS)
    parser.add_argument(
        "--segmentation-kit", type=Path, default=DEFAULT_SEGMENTATION_KIT
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("utterance_id"):
                raise ValueError(f"{path}:{line_number}: utterance_id is required")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def resolve_source_file(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_phonemes_by_utterance(path: Path) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        phonemes = record.get("raw_phonemes")
        if not isinstance(phonemes, list) or not phonemes:
            raise ValueError(f"{path}: raw_phonemes is required")
        expected[record["utterance_id"]] = [str(phone) for phone in phonemes]
    return expected


def to_julius_phonemes(phonemes: list[str]) -> list[str]:
    converted = []
    for phone in phonemes:
        if phone == "pau":
            converted.append("sp")
        elif phone == "cl":
            converted.append("q")
        elif phone in {"A", "I", "U", "E", "O"}:
            # The bundled monophone model has no separate devoiced-vowel states.
            converted.append(phone.lower())
        else:
            converted.append(phone)
    return converted


def resample_wave(source_path: Path, destination_path: Path) -> dict[str, Any]:
    with wave.open(str(source_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"{source_path}: compressed WAV is not supported")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{source_path}: expected mono 16-bit PCM WAV")
        source_rate = source.getframerate()
        source_frames = source.getnframes()
        pcm = source.readframes(source_frames)

    if source_rate == TARGET_SAMPLE_RATE_HZ:
        converted = pcm
    else:
        converted, _ = audioop.ratecv(
            pcm, 2, 1, source_rate, TARGET_SAMPLE_RATE_HZ, None
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination_path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(TARGET_SAMPLE_RATE_HZ)
        destination.writeframes(converted)

    return {
        "source_sample_rate_hz": source_rate,
        "source_frame_count": source_frames,
        "derived_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "derived_frame_count": len(converted) // 2,
        "derived_sha256": sha256(destination_path),
    }


def write_alignment_inputs(stem: Path, phonemes: list[str]) -> tuple[Path, Path]:
    words = ["silB", " ".join(phonemes), "silE"]
    dfa_path = stem.with_suffix(".dfa")
    dictionary_path = stem.with_suffix(".dict")
    final_state = len(words)
    dfa_lines = []
    for index in range(len(words)):
        accept = 1 if index == 0 else 0
        dfa_lines.append(
            f"{index} {final_state - index - 1} {index + 1} 0 {accept}"
        )
    dfa_lines.append(f"{final_state} -1 -1 1 0")
    dfa_path.write_text("\n".join(dfa_lines) + "\n", encoding="utf-8")
    dictionary_path.write_text(
        "".join(f"{index} [w_{index}] {word}\n" for index, word in enumerate(words)),
        encoding="utf-8",
    )
    return dfa_path, dictionary_path


def parse_alignment_log(log: str) -> list[dict[str, Any]]:
    in_alignment = False
    completed = False
    intervals: list[dict[str, Any]] = []
    for line in log.splitlines():
        if "begin forced alignment" in line:
            in_alignment = True
            continue
        if "end forced alignment" in line:
            completed = True
            in_alignment = False
            continue
        if not in_alignment:
            continue
        match = ALIGNMENT_LINE.match(line)
        if match is None:
            continue
        begin_frame = int(match.group(1))
        end_frame = int(match.group(2))
        start_sec = begin_frame * FRAME_SHIFT_SEC
        if begin_frame != 0:
            start_sec += ALIGNMENT_OFFSET_SEC
        end_sec = (end_frame + 1) * FRAME_SHIFT_SEC + ALIGNMENT_OFFSET_SEC
        intervals.append(
            {
                "begin_frame": begin_frame,
                "end_frame": end_frame,
                "score": float(match.group(3)),
                "phoneme": match.group(4),
                "start_sec": round(start_sec, 7),
                "end_sec": round(end_sec, 7),
            }
        )
    if not completed:
        raise RuntimeError("Julius log does not contain a completed forced alignment")
    if not intervals:
        raise RuntimeError("Julius did not output phoneme intervals")
    return intervals


def validate_intervals(
    intervals: list[dict[str, Any]], phonemes: list[str], duration_sec: float
) -> None:
    expected = ["silB", *phonemes, "silE"]
    actual = [interval["phoneme"] for interval in intervals]
    if actual != expected:
        raise RuntimeError(
            f"Julius phonemes differ from input\nexpected={expected}\nactual={actual}"
        )
    previous_end = 0.0
    for interval in intervals:
        start_sec = float(interval["start_sec"])
        end_sec = float(interval["end_sec"])
        if start_sec < previous_end - 1e-9 or end_sec <= start_sec:
            raise RuntimeError(f"Invalid Julius interval: {interval}")
        previous_end = end_sec
    if previous_end > duration_sec + WINDOW_SEC:
        raise RuntimeError(
            f"Julius alignment ends at {previous_end}, beyond audio {duration_sec}"
        )


def julius_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "-version"], capture_output=True, text=True
    )
    match = re.search(r"JuliusLib rev\.([0-9.]+)", result.stdout + result.stderr)
    if match is None:
        raise RuntimeError("Could not determine Julius version")
    return match.group(1)


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def align_record(
    record: dict[str, Any],
    phonemes: list[str],
    input_directory: Path,
    raw_directory: Path,
    executable: Path,
    model: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = resolve_source_file(str(record["source_file"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    stem = source_path.stem
    derived_wave = input_directory / f"{stem}.wav"
    conversion = resample_wave(source_path, derived_wave)
    expected_duration = conversion["derived_frame_count"] / TARGET_SAMPLE_RATE_HZ
    input_stem = input_directory / stem
    (input_directory / f"{stem}.txt").write_text(
        " ".join(phonemes) + "\n", encoding="utf-8"
    )
    dfa_path, dictionary_path = write_alignment_inputs(input_stem, phonemes)
    command = [
        str(executable),
        "-h",
        str(model),
        "-dfa",
        str(dfa_path),
        "-v",
        str(dictionary_path),
        "-palign",
        "-input",
        "file",
    ]
    started_at = time.perf_counter()
    result = subprocess.run(
        command,
        input=f"{derived_wave}\n",
        capture_output=True,
        text=True,
    )
    duration_sec = time.perf_counter() - started_at
    combined_log = result.stdout + result.stderr
    raw_directory.mkdir(parents=True, exist_ok=True)
    log_path = raw_directory / f"{stem}.log"
    log_path.write_text(combined_log, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Julius failed for {stem}; see {log_path}")
    intervals = parse_alignment_log(combined_log)
    validate_intervals(intervals, phonemes, expected_duration)
    alignment = {
        "utterance_id": record["utterance_id"],
        "source_file": record["source_file"],
        "derived_audio_file": str(derived_wave.relative_to(PROJECT_DIR)),
        "aligner": "julius",
        "aligner_version": JULIUS_VERSION,
        "model_id": MODEL_ID,
        "model_version": SEGMENTATION_KIT_COMMIT,
        "frame_shift_sec": FRAME_SHIFT_SEC,
        "window_sec": WINDOW_SEC,
        "alignment_offset_sec": ALIGNMENT_OFFSET_SEC,
        "intervals": intervals,
    }
    alignment_path = raw_directory / f"{stem}.json"
    alignment_path.write_text(
        json.dumps(alignment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return (
        {
            "utterance_id": record["utterance_id"],
            "duration_sec": round(duration_sec, 3),
            "phone_interval_count": len(intervals),
            **conversion,
        },
        alignment,
    )


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.expected = args.expected.resolve()
    args.input_directory = args.input_directory.resolve()
    args.raw_directory = args.raw_directory.resolve()
    args.run_metadata = args.run_metadata.resolve()
    executable = args.julius_executable.expanduser().resolve()
    segmentation_kit = args.segmentation_kit.expanduser().resolve()
    model = segmentation_kit / "models" / "hmmdefs_monof_mix16_gid.binhmm"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not model.is_file():
        raise FileNotFoundError(model)
    version = julius_version(executable)
    if version != JULIUS_VERSION:
        raise RuntimeError(f"Expected Julius {JULIUS_VERSION}, found {version}")
    if sha256(model) != MODEL_SHA256:
        raise RuntimeError("Julius monophone model checksum does not match")
    kit_commit = git_commit(segmentation_kit)
    if kit_commit != SEGMENTATION_KIT_COMMIT:
        raise RuntimeError(
            f"Expected segmentation-kit {SEGMENTATION_KIT_COMMIT}, found {kit_commit}"
        )

    manifest = read_jsonl(args.manifest)
    expected = expected_phonemes_by_utterance(args.expected)
    args.input_directory.mkdir(parents=True, exist_ok=True)
    args.raw_directory.mkdir(parents=True, exist_ok=True)
    utterance_runs = []
    started_at = time.perf_counter()
    for record in manifest:
        utterance_id = record["utterance_id"]
        if utterance_id not in expected:
            raise ValueError(f"No expected phonemes for {utterance_id}")
        phonemes = to_julius_phonemes(expected[utterance_id])
        run, _ = align_record(
            record,
            phonemes,
            args.input_directory,
            args.raw_directory,
            executable,
            model,
        )
        utterance_runs.append(run)
        print(f"{utterance_id}: {run['phone_interval_count']} phone intervals")
    total_duration = time.perf_counter() - started_at

    metadata = {
        "aligner": "julius",
        "aligner_version": version,
        "aligner_commit": "3b7174d0d4091f5e6ebb917769822032d079996f",
        "architecture": platform.machine(),
        "model_id": MODEL_ID,
        "model_version": kit_commit,
        "model_sha256": MODEL_SHA256,
        "julius_executable_sha256": sha256(executable),
        "manifest": str(args.manifest.relative_to(PROJECT_DIR)),
        "manifest_sha256": sha256(args.manifest),
        "expected_phonemes": str(args.expected.relative_to(PROJECT_DIR)),
        "expected_phonemes_sha256": sha256(args.expected),
        "utterance_count": len(utterance_runs),
        "duration_sec": round(total_duration, 3),
        "input_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "input_channels": 1,
        "input_sample_width_bytes": 2,
        "output_format": "json+log",
        "utterances": utterance_runs,
    }
    args.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.run_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Aligned {len(utterance_runs)} utterances in {total_duration:.3f}s; "
        f"wrote results to {args.raw_directory}"
    )


if __name__ == "__main__":
    main()
