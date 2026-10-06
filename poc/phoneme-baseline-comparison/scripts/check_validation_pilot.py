"""Compare two pilot runs, then check every exact trial using the uncached API."""

import argparse
import platform
import sys
from pathlib import Path
from time import perf_counter

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
for project in (
    "phoneme-speaker-encoder",
    "phoneme-user-registration",
    "phoneme-verification",
    "phoneme-verification-evaluation",
):
    sys.path.insert(0, str(ROOT / "poc" / project))
sys.path.insert(0, str(BASE))

import numpy as np
import torch
from evaluation.inference import model_snapshot, require_unchanged
from registration import EnrollmentSegment, FrozenEncoder, load_profile, register_user
from verification import IncompleteVerification, VerificationInput, verify

from pilot import load_inputs, read_rows
from smoke import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--recheck-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    inputs = load_inputs(args.recheck_run)
    load_inputs(args.reference_run)
    if str(torch.__version__) != inputs["protocol"]["models"]["vowel"]["torch"]:
        raise ValueError("uncached parity checks require the Phase 6 environment")
    files = ["inputs.json", "preflight.json", "scores.jsonl"]
    for worker, dimension in (("vowels", 128), ("ecapa", 192)):
        files.extend(
            f"{worker}/{name}"
            for name in (
                "embeddings.npy",
                "embedding-inputs.jsonl",
                "profiles.jsonl",
                "scores.jsonl",
            )
        )
        first = np.load(
            args.reference_run / worker / "embeddings.npy", allow_pickle=False
        )
        second = np.load(
            args.recheck_run / worker / "embeddings.npy", allow_pickle=False
        )
        if (
            first.shape[1] != dimension
            or not np.array_equal(first, second)
            or not np.isfinite(first).all()
        ):
            raise ValueError("cross-process embedding parity failed")
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, rtol=0, atol=1e-12)
        for profile in read_rows(args.recheck_run / worker / "profiles.jsonl"):
            vectors = (
                profile["vectors"].values()
                if "vectors" in profile
                else [profile["vector"]]
            )
            for vector in vectors:
                if len(vector) != dimension or not np.isfinite(vector).all():
                    raise ValueError("invalid profile vector")
                np.testing.assert_allclose(
                    np.linalg.norm(vector), 1, rtol=0, atol=1e-12
                )
    files.extend(
        str(p.relative_to(args.reference_run))
        for p in sorted((args.reference_run / "vowels/native-profiles").rglob("*.json"))
    )
    identical = {name: sha256_file(args.reference_run / name) for name in files}
    if any(
        sha256_file(args.recheck_run / name) != value
        for name, value in identical.items()
    ):
        raise ValueError("cross-process profile/input/score parity failed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder = FrozenEncoder.from_bundle(
        ROOT / inputs["protocol"]["models"]["vowel"]["bundle"]
    )
    before = model_snapshot(encoder)
    profiles = {}
    for item in inputs["enrollment"]:
        speaker, count = item["user_id"], item["enrollment_count"]
        native = register_user(
            speaker,
            [EnrollmentSegment(**s) for s in item["segments"]],
            encoder,
            audio_root=ROOT,
            minimum_segments=count,
        )
        saved = load_profile(
            args.recheck_run / f"vowels/native-profiles/{count}/{speaker}.json",
            encoder=encoder,
        )
        if native.checksum != saved.checksum:
            raise ValueError("uncached public registration parity failed")
        profiles[speaker, count] = native
    scores = {
        (r["condition_id"], r["trial_id"]): r
        for r in read_rows(args.recheck_run / "vowels/scores.jsonl")
        if r["method_id"] == "vowel_exact"
    }
    queries = {q["query_id"]: q for q in inputs["queries"]}
    checks = 0
    for window in inputs["windows"]:
        q = queries[window["query_id"]]
        by_id = {s["segment_id"]: s for s in q["segments"]}
        query = VerificationInput(
            q["query_id"],
            tuple(
                EnrollmentSegment(**by_id[a["segment_id"]]) for a in window["anchors"]
            ),
        )
        for trial in inputs["trials"]:
            if trial["query_id"] != q["query_id"]:
                continue
            saved = scores[window["condition_id"], trial["trial_id"]]
            try:
                result = verify(
                    profiles[trial["claimed_speaker_id"], trial["enrollment_count"]],
                    query,
                    encoder,
                    audio_root=ROOT,
                )
                values = {
                    "fused": result.score,
                    **{v: i["score"] for v, i in result.data["vowels"].items()},
                }
                if saved["status"] != "scored" or values != saved["scores"]:
                    raise ValueError("uncached public verification parity failed")
            except IncompleteVerification as exc:
                if (
                    saved["status"] != "no_score"
                    or saved["scores"] is not None
                    or (set(saved["missing_vowels"]) != set(exc.missing_vowels))
                ):
                    raise ValueError("uncached public no-score parity failed")
            checks += 1
    if checks != len(scores):
        raise ValueError("uncached check omitted score slots")
    require_unchanged(encoder, before)
    load_inputs(args.recheck_run)
    report = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "cross_process_and_uncached_public_api_pilot_parity",
        "reference_run": str(args.reference_run),
        "recheck_run": str(args.recheck_run),
        "identical_files_sha256": identical,
        "all_embedding_arrays_bitwise_equal": True,
        "all_profile_and_embedding_vectors_finite_and_unit": True,
        "timing_reports_excluded": True,
        "uncached_native_profile_checksum_checks": len(profiles),
        "uncached_score_status_checks": checks,
        "maximum_absolute_score_difference": 0.0,
        "parameters_and_statistics_unchanged": True,
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__))
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        f"{len(identical)} identical files; {len(profiles)} uncached profiles; {checks} uncached score/status checks passed"
    )


if __name__ == "__main__":
    main()
