"""Recover only verified empty-G2P clips, then resume the frozen comparison."""

from __future__ import annotations

import fcntl
import importlib.metadata
import os
import subprocess
import sys
import time
from pathlib import Path

import pyopenjtalk
from acquire import BASE, ROOT, read, sha
from corpus import CONFIG, atomic, fixed

EMPTY_G2P = "G2P produced an empty phoneme sequence"


def recover(run):
    state = read(run / "pipeline-state.json")
    if state.get("status") != "failed" or state.get("error") != EMPTY_G2P:
        return False
    rows = read(run / "inputs.json")
    processed = read(run / "extraction-progress.json")["processed"]
    if not 0 <= processed < len(rows):
        raise RuntimeError("Empty-G2P recovery has no unfinished input")
    row = rows[processed]
    receipt_path = run / "extraction" / row["sha256"] / "receipt.json"
    event_path = run / "recovery-events" / (row["sha256"] + ".json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        event = read(event_path)
        assert receipt["status"] == "g2p_unusable"
        assert event["original_failure_state"] == state
        assert receipt["input_sha256"] == row["sha256"]
        assert receipt["files"][str(event_path.relative_to(ROOT))] == sha(event_path)
        return True
    asr_path = run / "asr" / (row["sha256"] + ".json")
    asr = read(asr_path)
    assert asr["input_sha256"] == row["sha256"] == sha(ROOT / row["local_path"])
    assert asr["status"] == "transcribed"
    transcript = asr["asr"]["text"].strip()
    # Reproduce the exact failed operation; never classify arbitrary failures as empty.
    assert not pyopenjtalk.g2p(transcript).split(), "Failure cannot be reproduced"
    event = {
        "source": row["local_path"],
        "source_sha256": row["sha256"],
        "transcript": transcript,
        "reason": EMPTY_G2P,
        "original_failure_state": state,
        "pyopenjtalk_version": importlib.metadata.version("pyopenjtalk"),
        "recovery_source_sha256": sha(Path(__file__)),
        "replacement_selected": False,
    }
    fixed(event_path, event)
    receipt = {
        "input_sha256": row["sha256"],
        "asr_sha256": sha(asr_path),
        "split": row["split"],
        "speaker_id": row["speaker_id"],
        "status": "g2p_unusable",
        "reason": EMPTY_G2P,
        "files": {str(event_path.relative_to(ROOT)): sha(event_path)},
    }
    fixed(receipt_path, receipt)
    print(f"Recorded empty-G2P failure without replacement: {row['path']}", flush=True)
    return True


def main():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    study_config = read(BASE / "study-config.json")
    comparison = ROOT / study_config["run_directory"]
    state_path = comparison / "execution-state.json"
    with (run / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed(
            run / "recovery-policy.json",
            {
                "recoverable_error": EMPTY_G2P,
                "rule": "reproduce empty pyopenjtalk sequence; record unusable receipt, never replace or edit ASR",
                "source_sha256": sha(Path(__file__)),
                "original_preparation_freeze_sha256": sha(
                    run / "preparation-freeze.json"
                ),
                "training_design_sha256": sha(comparison / "design-freeze.json"),
            },
        )
        try:
            recover(run)
            while read(run / "pipeline-state.json")["status"] != "complete":
                atomic(
                    state_path,
                    {
                        "status": "waiting_for_corpus",
                        "pid": os.getpid(),
                        "supervisor": "resume_pipeline",
                        "updated_at_epoch": time.time(),
                    },
                )
                with (run / "pipeline.log").open("a") as log:
                    process = subprocess.run(
                        [sys.executable, str(BASE / "corpus.py"), "pipeline"],
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode == 0:
                    break
                if not recover(run):
                    raise RuntimeError(
                        f"Nonrecoverable preparation failure: {read(run / 'pipeline-state.json')}"
                    )
            with (comparison / "execution.log").open("a") as log:
                process = subprocess.run(
                    [
                        str(
                            ROOT
                            / "poc/phoneme-verification-evaluation/.venv/bin/python"
                        ),
                        str(BASE / "study.py"),
                        "execute",
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode:
                raise RuntimeError(f"Comparison stopped: {read(state_path)}")
        except BaseException as error:
            atomic(
                state_path,
                {
                    "status": "failed",
                    "error": str(error),
                    "updated_at_epoch": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
