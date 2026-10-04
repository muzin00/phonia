"""Create/read Phase 4 profiles, and prepare a validation-only JVS enrollment input."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE), str(BASE.parent / "phoneme-speaker-encoder")]

import torch
from phase3_data import load_segments, make_enrollment
from phase3_data.artifacts import write_json

from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    IncompleteEnrollment,
    load_profile,
    load_registration_input,
    register_user,
    save_profile,
)

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
    prepare = sub.add_parser(
        "prepare-jvs", help="prepare fixed validation enrollment for one speaker"
    )
    prepare.add_argument("--speaker", required=True)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--output", type=Path, required=True)
    register = sub.add_parser(
        "register", help="create a complete profile; never overwrite"
    )
    register.add_argument("--input", type=Path, required=True)
    register.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    register.add_argument("--audio-root", type=Path, default=ROOT)
    register.add_argument("--minimum-segments", type=int, default=10)
    register.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser("inspect", help="validate and summarize a saved profile")
    inspect.add_argument("--profile", type=Path, required=True)
    inspect.add_argument(
        "--bundle", type=Path, help="also check encoder/preprocessing compatibility"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        if args.command == "prepare-jvs":
            if args.output.exists():
                raise FileExistsError(args.output)
            segments = [
                s
                for s in load_segments(
                    args.manifest, split="validation", role="enrollment"
                )
                if s.speaker_id == args.speaker
            ]
            selection = make_enrollment(segments, split="validation", count=10)
            by_id = {s.segment_id: s for s in segments}
            rows = []
            for row in selection:
                segment = by_id[row["segment_id"]]
                rows.append(
                    asdict(
                        EnrollmentSegment(
                            segment.segment_id,
                            segment.vowel,
                            segment.source_file,
                            segment.start_frame,
                            segment.end_frame,
                            segment.source_sha256,
                        )
                    )
                )
            write_json(
                args.output,
                {"schema_version": 1, "user_id": args.speaker, "segments": rows},
            )
            print(
                json.dumps(
                    {
                        "input": str(args.output),
                        "user_id": args.speaker,
                        "segment_count": len(rows),
                    }
                )
            )
            return 0
        if args.command == "register":
            if args.output.exists():
                raise FileExistsError(args.output)
            encoder = FrozenEncoder.from_bundle(args.bundle)
            user_id, segments = load_registration_input(args.input)
            profile = register_user(
                user_id,
                segments,
                encoder,
                audio_root=args.audio_root,
                minimum_segments=args.minimum_segments,
            )
            save_profile(profile, args.output)
            profile = load_profile(args.output, encoder=encoder)
        else:
            encoder = FrozenEncoder.from_bundle(args.bundle) if args.bundle else None
            profile = load_profile(args.profile, encoder=encoder)
        print(
            json.dumps(
                {
                    "user_id": profile.user_id,
                    "profile_sha256": profile.checksum,
                    "segment_counts": {
                        vowel: item["segment_count"]
                        for vowel, item in profile.data["vowels"].items()
                    },
                    "excluded_segments": sum(
                        row["status"] == "excluded" for row in profile.data["segments"]
                    ),
                    "encoder_seed": profile.data["encoder"]["seed"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except IncompleteEnrollment as exc:
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "segment_counts": exc.counts,
                    "segments": exc.segments,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"registration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
