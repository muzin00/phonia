#!/usr/bin/env python3
"""Prepare the MFA corpus and align every utterance in the sample manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_AUDIO_DIR = PROJECT_DIR / "data" / "samples"
DEFAULT_MFA_DIR = PROJECT_DIR / "data" / "alignments" / "mfa"
DEFAULT_CORPUS_DIR = DEFAULT_MFA_DIR / "input"
DEFAULT_OUTPUT_DIR = DEFAULT_MFA_DIR / "raw"
DEFAULT_RUN_METADATA = DEFAULT_MFA_DIR / "run.json"
DEFAULT_ENVIRONMENT = "phonia-mfa"
MFA_MODEL = "japanese_mfa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MFA labels and align all samples in the manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-directory", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--corpus-directory", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--conda-executable", type=Path)
    parser.add_argument("--num-jobs", type=int, default=3)
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Enable MFA multiprocessing (disabled by default for macOS stability).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("utterance_id", "speaker_id", "source_file", "transcript"):
                if not record.get(field):
                    raise ValueError(f"{path}:{line_number}: {field} is required")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def resolve_source_file(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def prepare_corpus(records: list[dict[str, Any]], corpus_directory: Path) -> list[str]:
    corpus_directory.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    for record in records:
        audio_path = resolve_source_file(record["source_file"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not audio_path.stem.startswith(record["speaker_id"]):
            raise ValueError(
                f"{audio_path.name}: filename must start with {record['speaker_id']}"
            )
        label_path = corpus_directory / f"{audio_path.stem}.lab"
        label_path.write_text(f"{record['transcript'].strip()}\n", encoding="utf-8")
        stems.append(audio_path.stem)
    return stems


def find_conda(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.expanduser()
    elif executable := shutil.which("conda"):
        candidate = Path(executable)
    else:
        candidate = Path.home() / "miniforge3" / "bin" / "conda"
    if not candidate.is_file():
        raise FileNotFoundError(
            "Conda was not found. Pass --conda-executable or install Miniforge."
        )
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conda_command(conda: Path, environment: str, *command: str) -> list[str]:
    return [str(conda), "run", "--no-capture-output", "-n", environment, *command]


def installed_versions(conda: Path, environment: str) -> dict[str, str]:
    result = subprocess.run(
        [str(conda), "list", "-n", environment, "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    requested = {
        "python",
        "montreal-forced-aligner",
        "kalpy",
        "kaldi",
        "spacy",
        "sudachipy",
        "sudachidict-core",
    }
    return {
        package["name"]: package["version"]
        for package in json.loads(result.stdout)
        if package["name"] in requested
    }


def conda_platform(conda: Path) -> str:
    result = subprocess.run(
        [str(conda), "info", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["platform"]


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.manifest)
    stems = prepare_corpus(records, args.corpus_directory)
    conda = find_conda(args.conda_executable)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    speaker_lengths = {len(record["speaker_id"]) for record in records}
    if len(speaker_lengths) != 1:
        raise ValueError("All speaker IDs must have the same length for MFA")
    speaker_characters = speaker_lengths.pop()

    command = conda_command(
        conda,
        args.environment,
        "mfa",
        "align",
        str(args.corpus_directory),
        MFA_MODEL,
        MFA_MODEL,
        str(args.output_directory),
        "--audio_directory",
        str(args.audio_directory),
        "--output_format",
        "json",
        "--include_original_text",
        "--speaker_characters",
        str(speaker_characters),
        "--num_jobs",
        str(args.num_jobs),
        "--clean",
        "--overwrite",
    )
    if not args.use_multiprocessing:
        command.append("--no_use_mp")
    started_at = time.time()
    subprocess.run(command, check=True)
    duration_sec = time.time() - started_at

    expected_outputs = {f"{stem}.json" for stem in stems}
    actual_outputs = {path.name for path in args.output_directory.glob("*.json")}
    missing_outputs = sorted(expected_outputs - actual_outputs)
    if missing_outputs:
        raise RuntimeError(f"MFA did not produce: {', '.join(missing_outputs)}")

    metadata = {
        "aligner": "mfa",
        "environment": args.environment,
        "platform": conda_platform(conda),
        "acoustic_model": MFA_MODEL,
        "acoustic_model_version": "3.0.0",
        "dictionary": MFA_MODEL,
        "dictionary_version": "3.0.0",
        "packages": installed_versions(conda, args.environment),
        "manifest": str(args.manifest.relative_to(PROJECT_DIR)),
        "manifest_sha256": sha256(args.manifest),
        "utterance_count": len(records),
        "duration_sec": round(duration_sec, 3),
        "num_jobs": args.num_jobs,
        "use_multiprocessing": args.use_multiprocessing,
        "output_format": "json",
    }
    args.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.run_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Aligned {len(records)} utterances in {duration_sec:.3f}s; "
        f"wrote results to {args.output_directory}"
    )


if __name__ == "__main__":
    main()
