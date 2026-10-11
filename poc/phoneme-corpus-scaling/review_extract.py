"""Bound Julius resource use while retaining failures in the fixed sample."""

import subprocess
import wave
from pathlib import Path

import review
from acquire import BASE, ROOT, read, sha


def main():
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    review.fixed_write(
        run / "alignment-resource-policy.json",
        {
            "maximum_utterance_seconds": 30,
            "julius_timeout_seconds": 10,
            "source_sha256": sha(Path(__file__)),
            "reason": "Initial unbounded run terminated with exit 137 during a 50.508s utterance; keep long/timeout failures without replacement",
            "initial_completed_utterances": 4,
        },
    )
    original_run = subprocess.run

    def bounded_run(command, *args, **kwargs):
        if isinstance(command, list) and Path(command[0]).name == "julius":
            if "input" in kwargs:
                with wave.open(kwargs["input"].strip()) as wav:
                    seconds = wav.getnframes() / wav.getframerate()
                if seconds > 30:
                    raise ValueError(
                        f"alignment_duration_limit: {seconds:.3f}s exceeds 30s"
                    )
            kwargs["timeout"] = 10
        return original_run(command, *args, **kwargs)

    subprocess.run = bounded_run
    try:
        for name, digest in read(run / "freeze.json")["files"].items():
            assert sha(ROOT / name) == digest, name
        review.extract(config, run)
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    main()
