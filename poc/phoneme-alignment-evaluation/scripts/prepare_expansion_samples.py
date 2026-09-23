#!/usr/bin/env python3
"""Download a deterministic 40-utterance JVS expansion set without redistributing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import wave
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "expansion" / "samples"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "expansion" / "manifest.jsonl"
UPSTREAM_REPOSITORY = "MontrealCorpusTools/japanese-jvs-demo"
UPSTREAM_REVISION = "727c7cf2d52fd4dd154624a62a32ffd1db9b214e"
SPEAKERS = ("jvs001", "jvs002", "jvs003", "jvs004")
UTTERANCES_PER_SPEAKER = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the deterministic 40-utterance JVS expansion set."
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "phonia-evaluation"})
    with urllib.request.urlopen(request) as response:  # noqa: S310
        return response.read()


def upstream_files(speaker: str) -> list[dict[str, Any]]:
    url = (
        "https://api.github.com/repos/"
        f"{UPSTREAM_REPOSITORY}/contents/{speaker}?ref={UPSTREAM_REVISION}"
    )
    return json.loads(request_bytes(url))


def select_stems(files: list[dict[str, Any]], count: int) -> list[str]:
    names = {str(item["name"]) for item in files if item.get("type") == "file"}
    stems = sorted(
        Path(name).stem
        for name in names
        if name.endswith(".wav") and f"{Path(name).stem}.lab" in names
    )
    if len(stems) < count:
        raise ValueError(f"Only {len(stems)} paired utterances are available")
    return stems[:count]


def download_file(speaker: str, name: str, destination: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/"
        f"{UPSTREAM_REPOSITORY}/{UPSTREAM_REVISION}/{speaker}/{name}"
    )
    payload = request_bytes(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metadata(path: Path) -> tuple[int, int, str, float]:
    with wave.open(str(path), "rb") as audio:
        sample_width = audio.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {sample_width * 8}-bit")
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        duration = audio.getnframes() / sample_rate
    return sample_rate, channels, "pcm_s16le", duration


def relative_to_repository(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_DIR.resolve()).as_posix()


def build_record(speaker: str, stem: str, output_directory: Path) -> dict[str, Any]:
    wav_path = output_directory / f"{stem}.wav"
    lab_path = output_directory / f"{stem}.lab"
    if not wav_path.exists():
        download_file(speaker, wav_path.name, wav_path)
    if not lab_path.exists():
        download_file(speaker, lab_path.name, lab_path)

    transcript = lab_path.read_text(encoding="utf-8").strip()
    if not transcript:
        raise ValueError(f"{lab_path}: transcript is empty")
    sample_rate, channels, sample_format, duration = audio_metadata(wav_path)
    subset = "parallel100" if "_parallel100_" in stem else "nonpara30"
    return {
        "utterance_id": stem.lower(),
        "speaker_id": speaker,
        "session_id": "mfa_jvs_demo",
        "source_file": relative_to_repository(wav_path),
        "source_corpus": "JVS",
        "source_subset": subset,
        "transcript": transcript,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_format": sample_format,
        "duration_sec": round(duration, 6),
        "sha256": sha256(wav_path),
        "source_repository": UPSTREAM_REPOSITORY,
        "source_revision": UPSTREAM_REVISION,
    }


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        files = upstream_files(speaker)
        for stem in select_stems(files, UTTERANCES_PER_SPEAKER):
            records.append(build_record(speaker, stem, args.output_directory))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as destination:
        for record in records:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
    print(f"Wrote {len(records)} records to {args.manifest}")


if __name__ == "__main__":
    main()
