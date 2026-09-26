#!/usr/bin/env python3
"""Run resumable Julius forced alignment for the Phase 2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import time
import warnings
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "vowel-dataset.json"
DEFAULT_INPUT = PROJECT_DIR / "data" / "generated" / "expected-phonemes.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "generated" / "alignments"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.jsonl"
DEFAULT_RUN_METADATA = DEFAULT_OUTPUT_DIR / "run.json"
DEFAULT_RUNTIME = Path.home() / ".local" / "share" / "phonia" / "julius"
DEFAULT_JULIUS = DEFAULT_RUNTIME / "4.6" / "bin" / "julius"
DEFAULT_SEGMENTATION_KIT = DEFAULT_RUNTIME / "segmentation-kit"

SCHEMA_VERSION = 1
DESIGN_VERSION = "1.0.0"
JULIUS_VERSION = "4.6"
JULIUS_COMMIT = "3b7174d0d4091f5e6ebb917769822032d079996f"
SEGMENTATION_KIT_COMMIT = "e0e8bbaf98e27d19dfc6fe8312be607ad03592ad"
MODEL_ID = "segmentation-kit-monophone"
MODEL_SHA256 = "58952ccfe60f283c7efb1c22f9e012c7d297429700b7eeb4b60f7651a6e65940"
TARGET_SAMPLE_RATE_HZ = 16_000
FRAME_SHIFT_SEC = 0.01
WINDOW_SEC = 0.025
ALIGNMENT_OFFSET_SEC = WINDOW_SEC / 2
PHONE_MAPPING_VERSION = 1
UNSUPPORTED_PHONE_MAPPING = {"v": "b", "ty": "ch"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALIGNMENT_LINE = re.compile(r"^\[\s*(\d+)\s+(\d+)\]\s+([-+0-9.eE]+)\s+(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable Julius alignment for every Phase 2 utterance."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument("--julius-executable", type=Path, default=DEFAULT_JULIUS)
    parser.add_argument(
        "--segmentation-kit", type=Path, default=DEFAULT_SEGMENTATION_KIT
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N utterances for a development smoke test.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore compatible successful raw alignment checkpoints.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            utterance_id = str(record.get("utterance_id", ""))
            speaker_id = str(record.get("speaker_id", ""))
            if not SAFE_ID.fullmatch(utterance_id):
                raise ValueError(
                    f"{path}:{line_number}: unsafe utterance_id {utterance_id!r}"
                )
            if not SAFE_ID.fullmatch(speaker_id):
                raise ValueError(
                    f"{path}:{line_number}: unsafe speaker_id {speaker_id!r}"
                )
            if utterance_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate utterance_id {utterance_id}"
                )
            if not isinstance(record.get("raw_phonemes"), list):
                raise TypeError(f"{path}:{line_number}: raw_phonemes must be a list")
            seen.add(utterance_id)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
    temporary.replace(path)


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_DIR))
    except ValueError:
        return str(resolved)


def artifact_paths(output_directory: Path, record: dict[str, Any]) -> dict[str, Path]:
    speaker_id = str(record["speaker_id"])
    utterance_id = str(record["utterance_id"])
    if not SAFE_ID.fullmatch(speaker_id) or not SAFE_ID.fullmatch(utterance_id):
        raise ValueError("speaker_id and utterance_id must be safe path components")
    return {
        "raw": output_directory / "raw" / speaker_id / f"{utterance_id}.json",
        "failure_log": (
            output_directory / "failures" / speaker_id / f"{utterance_id}.log"
        ),
    }


def resolve_source_file(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def to_julius_phonemes(phonemes: list[str]) -> list[str]:
    converted: list[str] = []
    for phone in phonemes:
        if phone == "pau":
            converted.append("sp")
        elif phone == "cl":
            converted.append("q")
        elif phone in {"A", "I", "U", "E", "O"}:
            converted.append(phone.lower())
        else:
            converted.append(UNSUPPORTED_PHONE_MAPPING.get(phone, phone))
    return converted


def alignment_input_sha256(record: dict[str, Any]) -> str:
    return stable_sha256(
        {
            "utterance_id": record["utterance_id"],
            "source_sha256": record["source_sha256"],
            "raw_phonemes": record["raw_phonemes"],
            "julius_phonemes": to_julius_phonemes(
                [str(phone) for phone in record["raw_phonemes"]]
            ),
            "phone_mapping_version": PHONE_MAPPING_VERSION,
            "g2p_engine": record["g2p_engine"],
            "g2p_version": record["g2p_version"],
            "aligner": "julius",
            "aligner_version": JULIUS_VERSION,
            "model_id": MODEL_ID,
            "model_version": SEGMENTATION_KIT_COMMIT,
        }
    )


def resample_wave(source_path: Path, destination_path: Path) -> dict[str, Any]:
    with wave.open(str(source_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"{source_path}: compressed WAV is not supported")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{source_path}: expected mono 16-bit PCM WAV")
        source_rate = source.getframerate()
        source_frames = source.getnframes()
        pcm = source.readframes(source_frames)
    converted = pcm
    if source_rate != TARGET_SAMPLE_RATE_HZ:
        converted, _ = audioop.ratecv(
            pcm, 2, 1, source_rate, TARGET_SAMPLE_RATE_HZ, None
        )
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
    }


def write_alignment_inputs(stem: Path, phonemes: list[str]) -> tuple[Path, Path]:
    words = ["silB", " ".join(phonemes), "silE"]
    dfa_path = stem.with_suffix(".dfa")
    dictionary_path = stem.with_suffix(".dict")
    final_state = len(words)
    dfa_lines = []
    for index in range(len(words)):
        accept = 1 if index == 0 else 0
        dfa_lines.append(f"{index} {final_state - index - 1} {index + 1} 0 {accept}")
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
    if "<search failed>" in log:
        raise RuntimeError("Julius search failed: no alignment candidate")
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
            f"Julius phonemes differ from input: expected={expected}, actual={actual}"
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


def reusable_alignment(
    path: Path, fingerprint: str, record: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    intervals = value.get("intervals")
    if not isinstance(intervals, list) or not intervals:
        return None
    if value.get("alignment_input_sha256") == fingerprint:
        return value
    # Migrate a checkpoint made before the phone mapping was fingerprinted when
    # its actual Julius input and every other immutable input are still equal.
    legacy_compatible = (
        value.get("utterance_id") == record["utterance_id"]
        and value.get("source_sha256") == record["source_sha256"]
        and value.get("raw_phonemes") == record["raw_phonemes"]
        and value.get("julius_phonemes")
        == to_julius_phonemes([str(phone) for phone in record["raw_phonemes"]])
        and value.get("aligner_version") == JULIUS_VERSION
        and value.get("model_id") == MODEL_ID
        and value.get("model_version") == SEGMENTATION_KIT_COMMIT
    )
    if not legacy_compatible:
        return None
    value["alignment_input_sha256"] = fingerprint
    value["design_version"] = DESIGN_VERSION
    value["phone_mapping_version"] = PHONE_MAPPING_VERSION
    value["unsupported_phone_mapping"] = UNSUPPORTED_PHONE_MAPPING
    atomic_write_json(path, value)
    return value


def align_record(
    record: dict[str, Any],
    raw_path: Path,
    failure_log_path: Path,
    executable: Path,
    model: Path,
) -> dict[str, Any]:
    source_path = resolve_source_file(str(record["source_file"]))
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if sha256(source_path) != record["source_sha256"]:
        raise ValueError(f"{source_path}: checksum differs from utterance manifest")
    phonemes = to_julius_phonemes([str(phone) for phone in record["raw_phonemes"]])
    fingerprint = alignment_input_sha256(record)
    with tempfile.TemporaryDirectory(prefix="phonia-julius-") as temporary_directory:
        work = Path(temporary_directory)
        derived_wave = work / "input.wav"
        conversion = resample_wave(source_path, derived_wave)
        dfa_path, dictionary_path = write_alignment_inputs(work / "input", phonemes)
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
            check=False,
        )
        elapsed = time.perf_counter() - started_at
        combined_log = result.stdout + result.stderr
        if result.returncode != 0:
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            failure_log_path.write_text(combined_log, encoding="utf-8")
            raise RuntimeError(f"Julius exited with status {result.returncode}")
        try:
            intervals = parse_alignment_log(combined_log)
            duration_sec = conversion["derived_frame_count"] / TARGET_SAMPLE_RATE_HZ
            validate_intervals(intervals, phonemes, duration_sec)
        except Exception:
            failure_log_path.parent.mkdir(parents=True, exist_ok=True)
            failure_log_path.write_text(combined_log, encoding="utf-8")
            raise

    raw = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "utterance_id": record["utterance_id"],
        "speaker_id": record["speaker_id"],
        "source_file": record["source_file"],
        "source_sha256": record["source_sha256"],
        "alignment_input_sha256": fingerprint,
        "aligner": "julius",
        "aligner_version": JULIUS_VERSION,
        "model_id": MODEL_ID,
        "model_version": SEGMENTATION_KIT_COMMIT,
        "phone_mapping_version": PHONE_MAPPING_VERSION,
        "unsupported_phone_mapping": UNSUPPORTED_PHONE_MAPPING,
        "frame_shift_sec": FRAME_SHIFT_SEC,
        "window_sec": WINDOW_SEC,
        "alignment_offset_sec": ALIGNMENT_OFFSET_SEC,
        "raw_phonemes": record["raw_phonemes"],
        "julius_phonemes": phonemes,
        "intervals": intervals,
    }
    atomic_write_json(raw_path, raw)
    return {
        "duration_sec": round(elapsed, 3),
        "phone_interval_count": len(intervals),
        "raw_alignment_file": repository_relative(raw_path),
        "reused": False,
    }


def base_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "utterance_id": record["utterance_id"],
        "speaker_id": record["speaker_id"],
        "source_file": record["source_file"],
        "source_sha256": record["source_sha256"],
        "source_subset": record["source_subset"],
        "split": record["split"],
        "evaluation_role": record["evaluation_role"],
        "learning_curve_cohorts": record["learning_curve_cohorts"],
        "session_id": record.get("session_id"),
        "g2p_engine": record["g2p_engine"],
        "g2p_version": record["g2p_version"],
        "g2p_dictionary": record["g2p_dictionary"],
        "aligner": "julius",
        "aligner_version": JULIUS_VERSION,
        "model_id": MODEL_ID,
        "model_version": SEGMENTATION_KIT_COMMIT,
        "phone_mapping_version": PHONE_MAPPING_VERSION,
        "alignment_input_sha256": alignment_input_sha256(record),
    }


def process_records(
    records: list[dict[str, Any]],
    output_directory: Path,
    aligner: Callable[[dict[str, Any], Path, Path], dict[str, Any]],
    resume: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        paths = artifact_paths(output_directory, record)
        base = base_result(record)
        fingerprint = base["alignment_input_sha256"]
        checkpoint = (
            reusable_alignment(paths["raw"], fingerprint, record) if resume else None
        )
        if checkpoint is not None:
            results.append(
                {
                    **base,
                    "status": "success",
                    "duration_sec": 0.0,
                    "phone_interval_count": len(checkpoint["intervals"]),
                    "raw_alignment_file": repository_relative(paths["raw"]),
                    "reused": True,
                }
            )
            continue
        try:
            details = aligner(record, paths["raw"], paths["failure_log"])
            results.append({**base, "status": "success", **details})
        # A bad utterance must be recorded without stopping the dataset run.
        except Exception as error:  # noqa: BLE001
            failure: dict[str, Any] = {
                **base,
                "status": "failure",
                "stage": "julius_alignment",
                "error_type": type(error).__name__,
                "reason": str(error),
            }
            if paths["failure_log"].is_file():
                failure["log_file"] = repository_relative(paths["failure_log"])
            results.append(failure)
    return results


def executable_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "-version"], capture_output=True, text=True, check=False
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


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    alignment_config = config.get("alignment", {})
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dataset config schema_version does not match the runner")
    if config.get("design_version") != DESIGN_VERSION:
        raise ValueError("dataset config design_version does not match the runner")
    if alignment_config.get("aligner_version") != JULIUS_VERSION:
        raise ValueError("dataset config Julius version does not match the runner")
    if alignment_config.get("model_sha256") != MODEL_SHA256:
        raise ValueError("dataset config model checksum does not match the runner")
    if alignment_config.get("phone_mapping_version") != PHONE_MAPPING_VERSION:
        raise ValueError("dataset config phone mapping version does not match")
    if alignment_config.get("unsupported_phone_mapping") != UNSUPPORTED_PHONE_MAPPING:
        raise ValueError("dataset config unsupported phone mapping does not match")
    executable = args.julius_executable.expanduser().resolve()
    segmentation_kit = args.segmentation_kit.expanduser().resolve()
    model = segmentation_kit / "models" / "hmmdefs_monof_mix16_gid.binhmm"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if not model.is_file():
        raise FileNotFoundError(model)
    version = executable_version(executable)
    if version != JULIUS_VERSION:
        raise RuntimeError(f"Expected Julius {JULIUS_VERSION}, found {version}")
    if sha256(model) != MODEL_SHA256:
        raise RuntimeError("Julius monophone model checksum does not match")
    kit_commit = git_commit(segmentation_kit)
    if kit_commit != SEGMENTATION_KIT_COMMIT:
        raise RuntimeError(
            f"Expected segmentation-kit {SEGMENTATION_KIT_COMMIT}, found {kit_commit}"
        )

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    output_directory = args.output_directory.resolve()

    def run_aligner(
        record: dict[str, Any], raw_path: Path, failure_log_path: Path
    ) -> dict[str, Any]:
        return align_record(record, raw_path, failure_log_path, executable, model)

    started_at = time.perf_counter()
    results = process_records(
        records, output_directory, run_aligner, resume=not args.no_resume
    )
    elapsed = time.perf_counter() - started_at
    atomic_write_jsonl(args.manifest, results)
    success_count = sum(record["status"] == "success" for record in results)
    failure_count = len(results) - success_count
    reused_count = sum(record.get("reused", False) for record in results)
    run = {
        "schema_version": SCHEMA_VERSION,
        "design_version": DESIGN_VERSION,
        "stage": "julius_alignment",
        "status": "pass" if not failure_count else "completed_with_failures",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "aligner": "julius",
        "aligner_version": version,
        "aligner_commit": JULIUS_COMMIT,
        "aligner_executable_sha256": sha256(executable),
        "architecture": platform.machine(),
        "model_id": MODEL_ID,
        "model_version": kit_commit,
        "model_sha256": MODEL_SHA256,
        "phone_mapping_version": PHONE_MAPPING_VERSION,
        "unsupported_phone_mapping": UNSUPPORTED_PHONE_MAPPING,
        "input_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "reused_count": reused_count,
        "duration_sec": round(elapsed, 3),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
    }
    atomic_write_json(args.run_metadata, run)
    print(
        f"Aligned {success_count}/{len(results)} utterances in {elapsed:.3f}s; "
        f"reused={reused_count}, failures={failure_count}"
    )


if __name__ == "__main__":
    main()
