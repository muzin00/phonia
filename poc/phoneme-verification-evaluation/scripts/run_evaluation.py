"""Run prepared validation, freeze conditions, and evaluate fixed test trials."""

import argparse
import datetime
import sys
import uuid
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import torch

from evaluation import pipeline
from evaluation.storage import save_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "validation", "freeze", "test", "report", "all")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stages = (
        ("prepare", "validation", "freeze", "test", "report")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        can_record_failure = stage != "prepare" or not output.exists()
        try:
            if stage == "prepare":
                pipeline.prepare(output, BASE / "config/evaluation-protocol.json")
            else:
                getattr(pipeline, stage)(output)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            if can_record_failure and output.is_dir():
                save_document(
                    output / f"failures/{stage}-{uuid.uuid4().hex}.json",
                    {
                        "stage": stage,
                        "failed_at": datetime.datetime.now(datetime.UTC).isoformat(),
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            print(f"evaluation error ({stage}): {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
