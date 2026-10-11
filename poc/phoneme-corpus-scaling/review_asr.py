"""Transcribe only the fixed review sample in the separate MLX environment."""

import importlib.metadata
import subprocess
import time
import wave
from pathlib import Path

import mlx_whisper
import numpy as np
from acquire import BASE, ROOT, read, sha, write


def main():
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    selection = read(run / "freeze.json")
    for name, digest in selection["files"].items():
        assert sha(ROOT / name) == digest, name
    model = read(ROOT / "artifacts/phoneme-corpus-scaling/asr-model.json")
    frozen = {
        "selection_sha256": sha(
            run / "selection-asr-freeze.json"
            if (run / "selection-asr-freeze.json").exists()
            else run / "freeze.json"
        ),
        "model": model,
        "model_files": {
            str(p.relative_to(ROOT)): sha(p)
            for p in Path(model["path"]).rglob("*")
            if p.is_file()
        },
        "packages": {
            n: importlib.metadata.version(n) for n in ("mlx-whisper", "mlx", "numpy")
        },
        "language": "ja",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "transcript_provenance": "machine_generated_not_human_reference",
    }
    freeze_path = run / "asr/freeze.json"
    if freeze_path.exists():
        assert read(freeze_path) == frozen
    else:
        write(freeze_path, frozen)
    results = []
    clips = read(run / "selected-utterances.json")
    for index, row in enumerate(clips):
        assert row["split"] == "train"
        source = ROOT / row["local_path"]
        assert sha(source) == row["sha256"]
        saved = run / "asr" / (source.stem + ".json")
        if saved.exists():
            item = read(saved)
            assert item["sha256"] == row["sha256"]
            assert item["freeze_sha256"] == sha(freeze_path)
        else:
            wav = run / "asr" / (source.stem + ".wav")
            try:
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
                    "status": "transcribed",
                    "asr_seconds": time.monotonic() - start,
                    "duration_seconds": len(audio) / 16000,
                    "pcm16k_sha256": sha(wav),
                }
            except Exception as error:  # noqa: BLE001 - keep failures without replacement.
                item = {**row, "status": "failed", "error": str(error)}
            item["freeze_sha256"] = sha(freeze_path)
            write(saved, item)
        results.append(item)
        print(f"ASR {index + 1}/{len(clips)} {item['status']}", flush=True)
    write(run / "asr.json", results)


if __name__ == "__main__":
    main()
