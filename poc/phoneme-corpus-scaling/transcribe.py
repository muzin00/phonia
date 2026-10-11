"""Local ASR preflight, separated from the frozen encoder environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import time
import wave
from pathlib import Path

import mlx_whisper
import numpy as np
from acquire import BASE, ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = read(BASE / "config.json")
    run = ROOT / config["run_directory"]
    model = read(ROOT / "artifacts/phoneme-corpus-scaling/asr-model.json")
    manifest = run / ("pilot-download.json" if args.pilot else "download.json")
    inputs = read(manifest)
    destination = run / ("pilot-asr" if args.pilot else "asr")
    model_files = {
        str(p.relative_to(ROOT)): sha(p)
        for p in Path(model["path"]).rglob("*")
        if p.is_file()
    }
    frozen = {
        "model": model,
        "model_files": model_files,
        "manifest_sha256": sha(manifest),
        "source_sha256": sha(Path(__file__)),
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "packages": {
            n: importlib.metadata.version(n) for n in ("mlx-whisper", "mlx", "numpy")
        },
        "transcript_provenance": "machine_generated_not_human_reference",
    }
    path = destination / "freeze.json"
    if path.exists():
        assert read(path) == frozen, "ASR configuration changed"
    else:
        write(path, frozen)
    results = []
    for index, row in enumerate(inputs):
        source = ROOT / row["local_path"]
        assert sha(source) == row["sha256"]
        saved = destination / (source.stem + ".json")
        if saved.exists():
            item = read(saved)
            assert item["sha256"] == row["sha256"]
        else:
            wav = source.with_suffix(".wav")
            subprocess.run(
                [
                    "/usr/bin/afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    "LEI16@16000",
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
                    16000,
                    2,
                    1,
                )
                audio = (
                    np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(
                        np.float32
                    )
                    / 32768
                )
            start = time.monotonic()
            output = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model["path"],
                language="ja",
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=None,
            )
            item = {
                **row,
                "asr": output,
                "asr_seconds": time.monotonic() - start,
                "duration_seconds": len(audio) / 16000,
                "pcm16k_sha256": sha(wav),
                "freeze_sha256": sha(path),
            }
            write(saved, item)
        results.append(item)
        print(
            json.dumps(
                {
                    "processed": index + 1,
                    "total": len(inputs),
                    "audio_seconds": item["duration_seconds"],
                    "asr_seconds": item["asr_seconds"],
                }
            ),
            flush=True,
        )
    write(run / ("pilot-asr.json" if args.pilot else "asr.json"), results)


if __name__ == "__main__":
    main()
