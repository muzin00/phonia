"""Inspect a local JVS corpus without modifying or redistributing its contents."""

from __future__ import annotations

import argparse
import json
import re
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = PROJECT_DIR / "data" / "source" / "jvs_ver1"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "jvs-inventory.json"
SPEAKER_PATTERN = re.compile(r"jvs\d{3}")


@dataclass(frozen=True)
class SubsetDefinition:
    name: str
    directory_names: tuple[str, ...]
    expected_utterances: int
    expected_transcripts: int
    expected_labels: bool
    allow_unrecorded_transcripts: bool = False


SUBSETS = (
    SubsetDefinition("parallel100", ("parallel100",), 100, 100, True),
    SubsetDefinition(
        "nonpara30", ("nonpara30",), 30, 50, True, allow_unrecorded_transcripts=True
    ),
    SubsetDefinition("whisper10", ("whisper10",), 10, 10, False),
    SubsetDefinition("falset10", ("falset10", "falsetto10"), 10, 10, False),
)
EXPECTED_SPEAKERS = tuple(f"jvs{number:03d}" for number in range(1, 101))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local JVS corpus.")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a failure when the corpus differs from its documented design.",
    )
    return parser.parse_args()


def resolve_corpus_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if any((path / speaker).is_dir() for speaker in EXPECTED_SPEAKERS):
        return path
    nested = path / "jvs_ver1"
    if any((nested / speaker).is_dir() for speaker in EXPECTED_SPEAKERS):
        return nested
    raise ValueError(f"JVS speaker directories were not found below {path}")


def find_subset_directory(
    speaker_directory: Path, definition: SubsetDefinition
) -> Path | None:
    matches = [
        speaker_directory / name
        for name in definition.directory_names
        if (speaker_directory / name).is_dir()
    ]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"{speaker_directory}: multiple directories for {definition.name}: {names}"
        )
    return matches[0] if matches else None


def find_transcript_file(subset_directory: Path) -> Path | None:
    candidates = (
        subset_directory / "transcripts_utf8.txt",
        subset_directory / "transcript_utf8.txt",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"{subset_directory}: multiple transcript files: {names}")
    return matches[0] if matches else None


def parse_transcripts(path: Path) -> tuple[dict[str, str], list[str]]:
    transcripts: dict[str, str] = {}
    errors: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            errors.append(f"line {line_number}: missing ':' separator")
            continue
        utterance_id, transcript = (part.strip() for part in line.split(":", 1))
        if not utterance_id or not transcript:
            errors.append(f"line {line_number}: empty utterance ID or transcript")
            continue
        if utterance_id in transcripts:
            errors.append(f"line {line_number}: duplicate utterance ID {utterance_id}")
            continue
        transcripts[utterance_id] = transcript
    return transcripts, errors


def inspect_audio(path: Path) -> tuple[str | None, float | None, str | None]:
    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width_bits = audio.getsampwidth() * 8
            compression = audio.getcomptype()
            frame_count = audio.getnframes()
        duration = frame_count / sample_rate if sample_rate else 0.0
        audio_format = (
            f"{sample_rate}Hz/{channels}ch/{sample_width_bits}bit/{compression}"
        )
        return audio_format, duration, None
    except (EOFError, OSError, wave.Error) as error:
        return None, None, str(error)


def inspect_subset(
    corpus_root: Path,
    speaker_directory: Path,
    definition: SubsetDefinition,
) -> dict[str, Any]:
    subset_directory = find_subset_directory(speaker_directory, definition)
    if subset_directory is None:
        return {
            "name": definition.name,
            "present": False,
            "expected_utterances": definition.expected_utterances,
            "expected_transcripts": definition.expected_transcripts,
        }

    audio_directory = subset_directory / "wav24kHz16bit"
    wav_paths = (
        sorted(audio_directory.glob("*.wav")) if audio_directory.is_dir() else []
    )
    wav_ids = {path.stem for path in wav_paths}

    transcript_path = find_transcript_file(subset_directory)
    transcripts: dict[str, str] = {}
    transcript_errors: list[str] = []
    if transcript_path is not None:
        transcripts, transcript_errors = parse_transcripts(transcript_path)
    transcript_ids = set(transcripts)

    formats: Counter[str] = Counter()
    duration_sec = 0.0
    audio_errors: list[dict[str, str]] = []
    for wav_path in wav_paths:
        audio_format, duration, error = inspect_audio(wav_path)
        if error is not None:
            audio_errors.append(
                {"path": wav_path.relative_to(corpus_root).as_posix(), "error": error}
            )
            continue
        formats[str(audio_format)] += 1
        duration_sec += float(duration)

    label_files = (
        sorted((subset_directory / "lab").rglob("*.lab"))
        if (subset_directory / "lab").is_dir()
        else []
    )
    label_counts: Counter[str] = Counter()
    for label_path in label_files:
        relative_parts = label_path.relative_to(subset_directory / "lab").parts
        label_kind = relative_parts[0] if len(relative_parts) > 1 else "root"
        label_counts[label_kind] += 1

    return {
        "name": definition.name,
        "directory_name": subset_directory.name,
        "present": True,
        "expected_utterances": definition.expected_utterances,
        "expected_transcripts": definition.expected_transcripts,
        "allow_unrecorded_transcripts": definition.allow_unrecorded_transcripts,
        "wav_count": len(wav_paths),
        "transcript_file": (
            transcript_path.relative_to(corpus_root).as_posix()
            if transcript_path is not None
            else None
        ),
        "transcript_count": len(transcripts),
        "paired_wav_transcript_count": len(wav_ids & transcript_ids),
        "wav_without_transcript": sorted(wav_ids - transcript_ids),
        "transcript_without_wav": sorted(transcript_ids - wav_ids),
        "transcript_errors": transcript_errors,
        "label_file_count": len(label_files),
        "label_counts": dict(sorted(label_counts.items())),
        "labels_expected": definition.expected_labels,
        "duration_sec": round(duration_sec, 6),
        "audio_formats": dict(sorted(formats.items())),
        "audio_errors": audio_errors,
    }


def subset_issue_messages(speaker: str, subset: dict[str, Any]) -> list[str]:
    name = str(subset["name"])
    if not subset["present"]:
        return [f"{speaker}/{name}: subset directory is missing"]

    issues: list[str] = []
    expected_utterances = int(subset["expected_utterances"])
    if subset["wav_count"] != expected_utterances:
        issues.append(
            f"{speaker}/{name}: expected {expected_utterances} WAV files, "
            f"found {subset['wav_count']}"
        )
    expected_transcripts = int(subset["expected_transcripts"])
    if subset["transcript_count"] != expected_transcripts:
        issues.append(
            f"{speaker}/{name}: expected {expected_transcripts} transcripts, "
            f"found {subset['transcript_count']}"
        )
    if subset["wav_without_transcript"]:
        issues.append(
            f"{speaker}/{name}: {len(subset['wav_without_transcript'])} WAV files "
            "have no transcript"
        )
    if subset["transcript_without_wav"] and not subset["allow_unrecorded_transcripts"]:
        issues.append(
            f"{speaker}/{name}: {len(subset['transcript_without_wav'])} transcripts "
            "have no WAV file"
        )
    if subset["transcript_errors"]:
        issues.append(
            f"{speaker}/{name}: {len(subset['transcript_errors'])} transcript parse errors"
        )
    if subset["audio_errors"]:
        issues.append(
            f"{speaker}/{name}: {len(subset['audio_errors'])} unreadable WAV files"
        )
    unexpected_formats = set(subset["audio_formats"]) - {"24000Hz/1ch/16bit/NONE"}
    if unexpected_formats:
        issues.append(
            f"{speaker}/{name}: unexpected audio formats: "
            f"{', '.join(sorted(unexpected_formats))}"
        )
    return issues


def inspect_corpus(path: Path) -> dict[str, Any]:
    corpus_root = resolve_corpus_root(path)
    actual_speakers = sorted(
        child.name
        for child in corpus_root.iterdir()
        if child.is_dir() and SPEAKER_PATTERN.fullmatch(child.name)
    )
    missing_speakers = sorted(set(EXPECTED_SPEAKERS) - set(actual_speakers))
    unexpected_speakers = sorted(set(actual_speakers) - set(EXPECTED_SPEAKERS))

    speaker_records: list[dict[str, Any]] = []
    issues = [
        f"{speaker}: speaker directory is missing" for speaker in missing_speakers
    ]
    subset_totals: dict[str, dict[str, Any]] = {
        definition.name: {
            "expected_utterances_per_speaker": definition.expected_utterances,
            "expected_transcripts_per_speaker": definition.expected_transcripts,
            "speakers_present": 0,
            "wav_count": 0,
            "transcript_count": 0,
            "paired_wav_transcript_count": 0,
            "label_file_count": 0,
            "label_counts": Counter(),
            "duration_sec": 0.0,
            "audio_formats": Counter(),
        }
        for definition in SUBSETS
    }

    for speaker in actual_speakers:
        speaker_directory = corpus_root / speaker
        subsets = [
            inspect_subset(corpus_root, speaker_directory, definition)
            for definition in SUBSETS
        ]
        speaker_records.append({"speaker_id": speaker, "subsets": subsets})
        for subset in subsets:
            issues.extend(subset_issue_messages(speaker, subset))
            if not subset["present"]:
                continue
            total = subset_totals[str(subset["name"])]
            total["speakers_present"] += 1
            for field in (
                "wav_count",
                "transcript_count",
                "paired_wav_transcript_count",
                "label_file_count",
            ):
                total[field] += int(subset[field])
            total["duration_sec"] += float(subset["duration_sec"])
            total["audio_formats"].update(subset["audio_formats"])
            total["label_counts"].update(subset["label_counts"])

    serialized_totals: dict[str, dict[str, Any]] = {}
    for name, total in subset_totals.items():
        serialized_totals[name] = {
            **total,
            "duration_sec": round(float(total["duration_sec"]), 6),
            "audio_formats": dict(sorted(total["audio_formats"].items())),
            "label_counts": dict(sorted(total["label_counts"].items())),
        }

    return {
        "schema_version": 1,
        "corpus": "JVS",
        "corpus_root_name": corpus_root.name,
        "root_files": sorted(
            path.name for path in corpus_root.iterdir() if path.is_file()
        ),
        "expected_speaker_count": len(EXPECTED_SPEAKERS),
        "speaker_count": len(actual_speakers),
        "missing_speakers": missing_speakers,
        "unexpected_speakers": unexpected_speakers,
        "subsets": serialized_totals,
        "issue_count": len(issues),
        "issues": issues,
        "speakers": speaker_records,
    }


def main() -> None:
    args = parse_args()
    inventory = inspect_corpus(args.corpus_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Inspected {inventory['speaker_count']} speakers; "
        f"found {inventory['issue_count']} issues; wrote {args.output}"
    )
    if args.strict and inventory["issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
