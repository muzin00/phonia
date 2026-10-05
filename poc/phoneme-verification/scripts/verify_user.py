"""Prepare validation queries, compare a saved profile and inspect result files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [
    str(BASE),
    str(BASE.parent / "phoneme-user-registration"),
    str(BASE.parent / "phoneme-speaker-encoder"),
]

import torch
from registration import FrozenEncoder, load_profile

from verification import (
    IncompleteVerification,
    load_result,
    load_verification_input,
    save_result,
    verify,
)
from verification.jvs import prepare_jvs_input
from verification.scoring import write_document

DEFAULT_BUNDLE = (
    ROOT
    / "artifacts/phoneme-speaker-encoder/comparisons/phase3-full-v2-70spk-3seed-20260927-r2/selection-evaluation/selected-bundle/20260926"
)
DEFAULT_MANIFEST = (
    ROOT / "poc/phoneme-speaker-dataset/data/generated/phase3-vowel-segments.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-jvs", help="select queries from validation only")
    prepare.add_argument("--speaker", required=True)
    prepare.add_argument(
        "--role",
        choices=("verification", "cross_text_verification"),
        default="verification",
    )
    prepare.add_argument("--segments-per-vowel", type=int, default=5)
    prepare.add_argument("--seed", type=int, default=20261005)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("verify", help="compare a query to a saved user profile")
    compare.add_argument("--profile", type=Path, required=True)
    compare.add_argument("--input", type=Path, required=True)
    compare.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    compare.add_argument("--audio-root", type=Path, default=ROOT)
    compare.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser(
        "inspect", help="validate and summarize an existing result"
    )
    inspect.add_argument("--result", type=Path, required=True)
    inspect.add_argument("--bundle", type=Path, help="also check encoder compatibility")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        if args.command in ("prepare-jvs", "verify") and args.output.exists():
            raise FileExistsError(args.output)
        if args.command == "prepare-jvs":
            document = prepare_jvs_input(
                args.manifest,
                args.speaker,
                role=args.role,
                count=args.segments_per_vowel,
                seed=args.seed,
            )
            write_document(args.output, document)
            print(
                json.dumps(
                    {
                        "input": str(args.output),
                        "query_id": document["query_id"],
                        "segment_count": len(document["segments"]),
                    }
                )
            )
            return 0
        if args.command == "verify":
            encoder = FrozenEncoder.from_bundle(args.bundle)
            profile = load_profile(args.profile, encoder=encoder)
            result = verify(
                profile,
                load_verification_input(args.input),
                encoder,
                audio_root=args.audio_root,
            )
            save_result(result, args.output)
            result = load_result(args.output, encoder=encoder)
        else:
            encoder = FrozenEncoder.from_bundle(args.bundle) if args.bundle else None
            result = load_result(args.result, encoder=encoder)
        print(
            json.dumps(
                {
                    "query_id": result.data["query_id"],
                    "user_id": result.data["user_id"],
                    "verification_score": result.score,
                    "vowels": result.data["vowels"],
                    "excluded_segments": sum(
                        row["status"] == "excluded" for row in result.data["segments"]
                    ),
                    "result_sha256": result.checksum,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except IncompleteVerification as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "query_id": exc.query_id,
                    "segment_counts": exc.counts,
                    "missing_vowels": exc.missing_vowels,
                    "segments": exc.records,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"verification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
