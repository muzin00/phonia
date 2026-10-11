"""Produce atomic per-clip ASR results in the isolated MLX environment."""

import importlib.metadata
import subprocess
import time
import wave
from collections import Counter
from pathlib import Path

import mlx_whisper
import numpy as np
from acquire import ROOT, read, sha
from corpus import CONFIG, atomic, fixed, progress, verify


def main():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    verify(run)
    model = read(ROOT / config["asr_model_metadata"])
    freeze = {
        "preparation_sha256": sha(run / "preparation-freeze.json"),
        "model": model,
        "model_files": {
            str(p.relative_to(ROOT)): sha(p)
            for p in Path(model["path"]).rglob("*")
            if p.is_file()
        },
        "packages": {
            n: importlib.metadata.version(n) for n in ("mlx-whisper", "mlx", "numpy")
        },
        "language": config["language"],
        "temperature": config["temperature"],
        "condition_on_previous_text": config["condition_on_previous_text"],
        "transcript_provenance": "machine_generated_not_human_reference",
    }
    fixed(run / "asr-freeze.json", freeze)
    freeze_sha = sha(run / "asr-freeze.json")
    rows, counts, started = read(run / "inputs.json"), Counter(), time.monotonic()
    atomic(run / "asr-state.json", {"status": "running"})
    try:
        for index, row in enumerate(rows):
            source = ROOT / row["local_path"]
            assert sha(source) == row["sha256"]
            saved = run / "asr" / (row["sha256"] + ".json")
            if saved.exists():
                item = read(saved)
                assert (
                    item["input_sha256"] == row["sha256"]
                    and item["freeze_sha256"] == freeze_sha
                )
            else:
                saved.parent.mkdir(parents=True, exist_ok=True)
                wav = saved.with_suffix(".wav")
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
                    timeout=60,
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
                seconds = len(audio) / 16000
                item = {
                    "input_sha256": row["sha256"],
                    "freeze_sha256": freeze_sha,
                    "duration_seconds": seconds,
                    "pcm16k_sha256": sha(wav),
                }
                if seconds > config["maximum_utterance_seconds"]:
                    item.update(
                        status="duration_limit",
                        reason="utterance exceeds frozen 30s resource limit; no replacement",
                    )
                else:
                    before = time.monotonic()
                    output = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=model["path"],
                        language=config["language"],
                        temperature=config["temperature"],
                        condition_on_previous_text=config["condition_on_previous_text"],
                        verbose=None,
                    )
                    item.update(
                        status="transcribed"
                        if output["text"].strip()
                        else "empty_transcript",
                        asr=output,
                        asr_seconds=time.monotonic() - before,
                    )
                atomic(saved, item)
            counts[item["status"]] += 1
            progress(run, "asr", index + 1, len(rows), counts, started)
        atomic(run / "asr-state.json", {"status": "complete", "processed": len(rows)})
    except BaseException as error:
        atomic(run / "asr-state.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
