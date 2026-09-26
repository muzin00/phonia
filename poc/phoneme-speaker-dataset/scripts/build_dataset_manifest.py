"""Build and validate the Phase 2 JVS utterance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import wave
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scripts.inspect_jvs_corpus import (
    DEFAULT_CORPUS_ROOT,
    EXPECTED_SPEAKERS,
    find_transcript_file,
    parse_transcripts,
    resolve_corpus_root,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config" / "dataset-split.json"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "utterance-manifest.jsonl"
DEFAULT_VALIDATION = PROJECT_DIR / "data" / "dataset-validation.json"
PARALLEL_ID_PATTERN = re.compile(r"VOICEACTRESS100_(\d{3})$")
KNOWN_MISSING_PARALLEL = {
    "jvs030/VOICEACTRESS100_045",
    "jvs074/VOICEACTRESS100_094",
    "jvs089/VOICEACTRESS100_019",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the Phase 2 JVS utterance manifest."
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path, repository_dir: Path = REPOSITORY_DIR) -> str:
    return path.resolve().relative_to(repository_dir.resolve()).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")


def validate_config(
    config: dict[str, Any], expected_speakers: Iterable[str] = EXPECTED_SPEAKERS
) -> None:
    required_splits = {"train", "validation", "test"}
    speaker_splits = config.get("speaker_splits")
    if not isinstance(speaker_splits, dict) or set(speaker_splits) != required_splits:
        raise ValueError("speaker_splits must contain train, validation, and test")

    split_sets: dict[str, set[str]] = {}
    for name in sorted(required_splits):
        speakers = speaker_splits[name]
        if not isinstance(speakers, list) or not all(
            isinstance(speaker, str) for speaker in speakers
        ):
            raise ValueError(f"speaker_splits.{name} must be a list of strings")
        if len(speakers) != len(set(speakers)):
            raise ValueError(f"speaker_splits.{name} contains duplicate speakers")
        split_sets[name] = set(speakers)

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            raise ValueError(f"speaker overlap between {left} and {right}: {overlap}")

    actual = set().union(*split_sets.values())
    expected = set(expected_speakers)
    if actual != expected:
        raise ValueError(
            f"speaker coverage differs: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    randomization = config.get("randomization")
    if not isinstance(randomization, dict):
        raise TypeError("randomization must be a JSON object")
    if randomization.get("method") != "python_random_shuffle":
        raise ValueError("unsupported randomization method")
    seed = randomization.get("seed")
    if not isinstance(seed, int):
        raise TypeError("randomization.seed must be an integer")
    expected_order = sorted(expected)
    random.Random(seed).shuffle(expected_order)
    train_count = len(speaker_splits["train"])
    validation_count = len(speaker_splits["validation"])
    if speaker_splits["train"] != expected_order[:train_count]:
        raise ValueError("train speaker order does not match randomization settings")
    expected_validation = set(
        expected_order[train_count : train_count + validation_count]
    )
    expected_test = set(expected_order[train_count + validation_count :])
    if split_sets["validation"] != expected_validation:
        raise ValueError("validation speakers do not match randomization settings")
    if split_sets["test"] != expected_test:
        raise ValueError("test speakers do not match randomization settings")

    sizes = config.get("learning_curve_sizes")
    if (
        not isinstance(sizes, list)
        or not sizes
        or not all(isinstance(size, int) and size > 0 for size in sizes)
    ):
        raise ValueError("learning_curve_sizes must be a non-empty list of integers")
    if sizes != sorted(set(sizes)):
        raise ValueError("learning_curve_sizes must be sorted and unique")
    if sizes[-1] != len(speaker_splits["train"]):
        raise ValueError(
            "largest learning curve size must equal the train speaker count"
        )


def split_by_speaker(config: dict[str, Any]) -> dict[str, str]:
    return {
        speaker: split
        for split, speakers in config["speaker_splits"].items()
        for speaker in speakers
    }


def learning_curve_cohorts(speaker: str, config: dict[str, Any]) -> list[int]:
    train_order = config["speaker_splits"]["train"]
    if speaker not in train_order:
        return []
    position = train_order.index(speaker)
    return [size for size in config["learning_curve_sizes"] if position < int(size)]


def evaluation_role(
    split: str, subset: str, utterance_stem: str, config: dict[str, Any]
) -> str:
    if split == "train":
        return "training"
    if subset == "nonpara30":
        return str(config["evaluation_roles"]["nonpara30"])
    if subset != "parallel100":
        raise ValueError(f"unsupported evaluation subset: {subset}")

    match = PARALLEL_ID_PATTERN.fullmatch(utterance_stem)
    if match is None:
        raise ValueError(f"unexpected parallel100 utterance ID: {utterance_stem}")
    utterance_number = int(match.group(1))
    for role in ("enrollment", "verification"):
        rule = config["evaluation_roles"]["parallel100"][role]
        if (
            int(rule["utterance_number_start"])
            <= utterance_number
            <= int(rule["utterance_number_end"])
        ):
            return role
    raise ValueError(f"parallel100 utterance is outside role ranges: {utterance_stem}")


def inspect_wave(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        compression_type = audio.getcomptype()
    if (sample_rate, channels, sample_width, compression_type) != (
        24_000,
        1,
        2,
        "NONE",
    ):
        raise ValueError(
            f"{path}: expected 24000 Hz mono 16-bit PCM, found "
            f"{sample_rate} Hz/{channels} ch/{sample_width * 8} bit/{compression_type}"
        )
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_sec": round(frame_count / sample_rate, 6),
    }


def collect_records(
    corpus_root: Path,
    config: dict[str, Any],
    repository_dir: Path = REPOSITORY_DIR,
) -> list[dict[str, Any]]:
    corpus_root = resolve_corpus_root(corpus_root)
    speaker_splits = split_by_speaker(config)
    records: list[dict[str, Any]] = []

    for speaker in sorted(speaker_splits):
        speaker_directory = corpus_root / speaker
        if not speaker_directory.is_dir():
            raise ValueError(f"speaker directory is missing: {speaker_directory}")
        split = speaker_splits[speaker]
        for subset in config["included_subsets"]:
            subset_directory = speaker_directory / subset
            transcript_path = find_transcript_file(subset_directory)
            if transcript_path is None:
                raise ValueError(f"transcript file is missing: {subset_directory}")
            transcripts, errors = parse_transcripts(transcript_path)
            if errors:
                raise ValueError(f"{transcript_path}: {errors}")

            audio_directory = subset_directory / "wav24kHz16bit"
            for wav_path in sorted(audio_directory.glob("*.wav")):
                transcript = transcripts.get(wav_path.stem)
                if not transcript:
                    raise ValueError(f"transcript is missing: {wav_path}")
                role = evaluation_role(split, subset, wav_path.stem, config)
                records.append(
                    {
                        "schema_version": 1,
                        "utterance_id": f"{speaker}_{subset}_{wav_path.stem}".lower(),
                        "speaker_id": speaker,
                        "source_corpus": config["corpus"],
                        "source_corpus_version": config["corpus_version"],
                        "source_subset": subset,
                        "source_utterance_id": wav_path.stem,
                        "source_file": repository_relative(wav_path, repository_dir),
                        "source_sha256": sha256_file(wav_path),
                        "transcript": transcript,
                        "split": split,
                        "evaluation_role": role,
                        "learning_curve_cohorts": learning_curve_cohorts(
                            speaker, config
                        ),
                        "session_id": None,
                        **inspect_wave(wav_path),
                    }
                )
    return sorted(
        records,
        key=lambda record: (
            record["speaker_id"],
            record["source_subset"],
            record["source_utterance_id"],
        ),
    )


def find_missing_parallel(corpus_root: Path) -> list[str]:
    corpus_root = resolve_corpus_root(corpus_root)
    missing: list[str] = []
    for speaker in EXPECTED_SPEAKERS:
        audio_directory = corpus_root / speaker / "parallel100" / "wav24kHz16bit"
        for number in range(1, 101):
            utterance = f"VOICEACTRESS100_{number:03d}"
            if not (audio_directory / f"{utterance}.wav").is_file():
                missing.append(f"{speaker}/{utterance}")
    return missing


def duplicate_values(records: list[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(str(record[field]) for record in records)
    return sorted(value for value, count in counts.items() if count > 1)


def summarize_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    missing_parallel: list[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    split_counts: dict[str, dict[str, Any]] = {}
    role_counts: Counter[tuple[str, str]] = Counter()
    speaker_role_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for field in ("utterance_id", "source_file", "source_sha256"):
        duplicates = duplicate_values(records, field)
        if duplicates:
            errors.append(f"duplicate {field}: {duplicates[:10]}")

    for record in records:
        role_counts[(str(record["split"]), str(record["evaluation_role"]))] += 1
        speaker_role_counts[str(record["speaker_id"])][
            str(record["evaluation_role"])
        ] += 1

    for split, configured_speakers in config["speaker_splits"].items():
        split_records = [record for record in records if record["split"] == split]
        actual_speakers = sorted(
            {str(record["speaker_id"]) for record in split_records}
        )
        if set(actual_speakers) != set(configured_speakers):
            errors.append(f"{split}: manifest speaker set differs from config")
        subset_counts = Counter(
            str(record["source_subset"]) for record in split_records
        )
        split_counts[split] = {
            "speaker_count": len(actual_speakers),
            "utterance_count": len(split_records),
            "duration_hours": round(
                sum(float(record["duration_sec"]) for record in split_records) / 3600,
                6,
            ),
            "subset_utterance_counts": dict(sorted(subset_counts.items())),
        }

    minimums = config["minimum_evaluation_utterances_per_speaker"]
    for split in ("validation", "test"):
        for speaker in config["speaker_splits"][split]:
            for role, minimum in minimums.items():
                actual = speaker_role_counts[speaker][role]
                if actual < int(minimum):
                    errors.append(
                        f"{speaker}/{role}: expected at least {minimum}, found {actual}"
                    )

    missing_set = set(missing_parallel)
    if missing_set == KNOWN_MISSING_PARALLEL:
        warnings.append(
            "parallel100 has the three known missing WAV files: "
            + ", ".join(sorted(missing_set))
        )
    else:
        errors.append(
            "parallel100 missing WAV set differs from the documented corpus: "
            f"{sorted(missing_set)}"
        )

    train_order = config["speaker_splits"]["train"]
    cohorts: dict[str, dict[str, int]] = {}
    previous: set[str] = set()
    for size in config["learning_curve_sizes"]:
        speakers = set(train_order[: int(size)])
        if not previous.issubset(speakers):
            errors.append(f"learning curve cohort {size} is not nested")
        cohort_records = [
            record
            for record in records
            if record["speaker_id"] in speakers and record["split"] == "train"
        ]
        cohorts[str(size)] = {
            "speaker_count": len(speakers),
            "utterance_count": len(cohort_records),
        }
        previous = speakers

    return {
        "schema_version": 1,
        "design_version": config["design_version"],
        "status": "pass" if not errors else "fail",
        "corpus": config["corpus"],
        "corpus_version": config["corpus_version"],
        "utterance_count": len(records),
        "speaker_count": len({record["speaker_id"] for record in records}),
        "duration_hours": round(
            sum(float(record["duration_sec"]) for record in records) / 3600, 6
        ),
        "splits": split_counts,
        "evaluation_role_counts": {
            f"{split}/{role}": count
            for (split, role), count in sorted(role_counts.items())
        },
        "learning_curve_cohorts": cohorts,
        "missing_parallel_wav": sorted(missing_parallel),
        "error_count": len(errors),
        "errors": errors,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    records = collect_records(args.corpus_root, config)
    missing_parallel = find_missing_parallel(args.corpus_root)
    validation = summarize_records(records, config, missing_parallel)

    write_jsonl(args.manifest, records)
    validation["config"] = repository_relative(args.config)
    validation["config_sha256"] = sha256_file(args.config)
    validation["manifest"] = repository_relative(args.manifest)
    validation["manifest_sha256"] = sha256_file(args.manifest)
    write_json(args.validation, validation)

    print(
        f"Built {len(records)} utterances from {validation['speaker_count']} speakers; "
        f"validation={validation['status']}; wrote {args.manifest} and {args.validation}"
    )
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
