"""Check ASR-derived phone extraction on fixed train-only pilot clips."""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

from acquire import BASE, ROOT, read, sha, write

sys.path.insert(0, str(BASE.parent / "phoneme-live-recording"))
import runtime as r


def main():
    config = read(BASE / "config.json")
    run = ROOT / config["run_directory"]
    clips = read(run / "pilot-asr.json")
    frozen = read(
        ROOT / "artifacts/phoneme-live-recording/pilot-20261011-v2/protocol-freeze.json"
    )
    engine = r.Runtime(frozen["config"])
    inventory, phones = [], Counter()
    for row in clips:
        assert row["split"] == "train" and row["pilot"]
        source = ROOT / row["local_path"]
        assert sha(source) == row["sha256"]
        output = run / "pilot-extraction" / source.stem
        output.mkdir(parents=True, exist_ok=True)
        wav = output / "input24k.wav"
        subprocess.run(
            [
                "/usr/bin/afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@24000",
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
                24000,
                2,
                1,
            )
            pcm = r.np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        metadata = {
            "utterance_id": "jtube_" + row["sha256"][:24],
            "speaker_id": row["speaker_id"].lower(),
            "transcript": row["asr"]["text"].strip(),
            "origin": "jtubespeech_asv_train_preflight",
            "role": "training",
            "session_id": None,
            "original_audio_sha256": row["sha256"],
            "original_audio_path": row["path"],
        }
        try:
            result = engine.process(pcm, output / "processed", metadata)
            phones.update(result["counts"])
            item = {
                "source": row["local_path"],
                "speaker_id": row["speaker_id"],
                "duration_seconds": row["duration_seconds"],
                "transcript": metadata["transcript"],
                "asr_seconds": row["asr_seconds"],
                "status": result["status"],
                "counts": result["counts"],
                "qc": result["qc"],
                "result_sha256": sha(output / "processed/result.json"),
            }
        except Exception as error:  # noqa: BLE001 - retain failed clips in the preflight denominator.
            item = {
                "source": row["local_path"],
                "speaker_id": row["speaker_id"],
                "status": "failed",
                "error": str(error),
            }
        inventory.append(item)
    summary = {
        "clips": len(inventory),
        "speakers": len({v["speaker_id"] for v in inventory}),
        "status_counts": dict(Counter(v["status"] for v in inventory)),
        "phone_counts": dict(phones),
        "input_manifest_sha256": sha(run / "pilot-asr.json"),
        "source_sha256": sha(Path(__file__)),
        "machine_transcripts_not_human_verified": True,
        "alignment_success_is_not_label_correctness": True,
        "details": inventory,
    }
    write(run / "pilot-quality.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
