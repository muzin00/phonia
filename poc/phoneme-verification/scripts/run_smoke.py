"""Reproduce validation-only genuine/impostor checks without fitting a threshold."""

from __future__ import annotations

import argparse
import copy
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
from phase3_data.manifest import sha256_file
from registration import FrozenEncoder, load_profile
from verify_user import DEFAULT_BUNDLE, DEFAULT_MANIFEST

from verification import (
    VerificationInput,
    load_result,
    load_verification_input,
    save_result,
    verify,
)
from verification.jvs import prepare_jvs_input
from verification.scoring import write_document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--impostor-speaker", default="jvs016")
    parser.add_argument("--segments-per-vowel", type=int, default=5)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        before_bundle = {
            name: sha256_file(args.bundle / name)
            for name in ("best.pt", "run.json", "feature-statistics.json")
        }
        before_profile = sha256_file(args.profile)
        encoder = FrozenEncoder.from_bundle(args.bundle)
        profile = load_profile(args.profile, encoder=encoder)
        if profile.user_id == args.impostor_speaker:
            raise ValueError("impostor speaker must differ from the registered user")
        before_parameters = {
            name: value.clone() for name, value in encoder.model.state_dict().items()
        }
        before_statistics = copy.deepcopy(encoder.pipeline.statistics)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        cases = {}
        for name, speaker, role in (
            ("genuine", profile.user_id, "verification"),
            ("impostor", args.impostor_speaker, "verification"),
            ("cross-text-genuine", profile.user_id, "cross_text_verification"),
        ):
            document = prepare_jvs_input(
                args.manifest,
                speaker,
                role=role,
                count=args.segments_per_vowel,
            )
            input_path = args.output_dir / f"{name}-input.json"
            write_document(input_path, document)
            query = load_verification_input(input_path)
            result = verify(profile, query, encoder, audio_root=args.audio_root)
            result_path = args.output_dir / f"{name}-result.json"
            save_result(result, result_path)
            load_result(result_path, encoder=encoder)
            repeated = verify(
                profile,
                VerificationInput(query.query_id, tuple(reversed(query.segments))),
                encoder,
                audio_root=args.audio_root,
            )
            repeat_path = args.output_dir / f"{name}-result-repeat.json"
            save_result(repeated, repeat_path)
            cases[name] = {
                "query_speaker_id": speaker,
                "query_role": role,
                "is_genuine": speaker == profile.user_id,
                "verification_score": result.score,
                "vowels": result.data["vowels"],
                "excluded_segments": sum(
                    row["status"] == "excluded" for row in result.data["segments"]
                ),
                "query_source_wav_count": len(
                    {row["source_sha256"] for row in result.data["segments"]}
                ),
                "result_sha256": result.checksum,
                "repeat_result_bytes_equal": result_path.read_bytes()
                == repeat_path.read_bytes(),
                "enrollment_query_checksums_disjoint": {
                    row["source_sha256"] for row in result.data["segments"]
                }.isdisjoint(
                    {row["source_sha256"] for row in profile.data["segments"]}
                ),
            }
        invariants = {
            "profile_file_unchanged": before_profile == sha256_file(args.profile),
            "parameters_unchanged": all(
                torch.equal(value, before_parameters[name])
                for name, value in encoder.model.state_dict().items()
            ),
            "statistics_unchanged": before_statistics == encoder.pipeline.statistics,
            "gradients_disabled": all(
                not value.requires_grad and value.grad is None
                for value in encoder.model.parameters()
            ),
            **{
                f"bundle_{name}_unchanged": checksum == sha256_file(args.bundle / name)
                for name, checksum in before_bundle.items()
            },
        }
        if not all(invariants.values()) or not all(
            case["repeat_result_bytes_equal"]
            and case["enrollment_query_checksums_disjoint"]
            for case in cases.values()
        ):
            raise ValueError("smoke verification invariants failed")
        report = {
            "schema_version": 1,
            "purpose": "functional_smoke_not_authentication_performance",
            "split": "validation",
            "user_id": profile.user_id,
            "profile_sha256": profile.checksum,
            "selection_seed": 20261005,
            "query_segments_per_vowel": args.segments_per_vowel,
            "manifest_sha256": document["origin"]["manifest_sha256"],
            "cases": cases,
            "invariants": invariants,
        }
        write_document(args.output_dir / "verification.json", report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"smoke verification error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
