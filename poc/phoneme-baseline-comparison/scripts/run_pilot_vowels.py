"""Run exact/context inputs under the checksum-pinned Phase 6 environment."""

import argparse
import platform
import sys
from collections import defaultdict
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
from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    load_profile,
    register_user,
    save_profile,
)
from verification import IncompleteVerification, VerificationInput, verify
from verification.inputs import prepare_query

from comparison_inputs import effective_window
from pilot import (
    VOWELS,
    AudioCache,
    EmbeddingCache,
    load_inputs,
    mean_profile,
    read_rows,
    save_worker,
    score_row,
    vowel_scores,
)
from smoke import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = load_inputs(args.run_dir)
    protocol = inputs["protocol"]
    if str(torch.__version__) != protocol["models"]["vowel"]["torch"]:
        raise ValueError("vowel inference requires the Phase 6 torch version")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    output = args.run_dir / "vowels"
    output.mkdir(exist_ok=False)
    started = perf_counter()
    encoder = FrozenEncoder.from_bundle(ROOT / protocol["models"]["vowel"]["bundle"])
    before = model_snapshot(encoder)
    audio = AudioCache(inputs["sources"])
    cache = EmbeddingCache(
        audio,
        encoder.identity,
        lambda pcm, sid: encoder.embed(torch.from_numpy(pcm.copy()), sid),
        inputs["config"]["repeats_per_unique_embedding"],
        already_unit=True,
    )
    segments = {}
    for item in inputs["enrollment"] + inputs["queries"]:
        for row in item["segments"]:
            if row["segment_id"] in segments and segments[row["segment_id"]] != row:
                raise ValueError("conflicting original segment identity")
            segments[row["segment_id"]] = row

    class ExactAdapter:
        identity, pipeline, model = encoder.identity, encoder.pipeline, encoder.model

        def embed(self, pcm, segment_id):
            row = segments[segment_id]
            filename = row["source_file"]
            first, last = row["start_frame"], row["end_frame"]
            if not np.array_equal(pcm.numpy(), audio.slice(filename, first, last)):
                raise ValueError("public API PCM differs from declared core")
            first, last = effective_window(
                first, last, inputs["sources"][filename]["frame_count"]
            )
            return cache.embed(filename, first, last, segment_id)

    exact = ExactAdapter()
    native, profiles, references = {}, [], {}
    reference_run = ROOT / protocol["phase6_reference"]["run"]
    profile_checks = []
    for item in inputs["enrollment"]:
        speaker, count = item["user_id"], item["enrollment_count"]
        original = register_user(
            speaker,
            [EnrollmentSegment(**r) for r in item["segments"]],
            exact,
            audio_root=ROOT,
            minimum_segments=count,
        )
        if any(r["status"] != "used" for r in original.data["segments"]):
            raise ValueError("Phase 6 selected enrollment core no longer eligible")
        existing_path = reference_run / f"profiles/validation/{count}/{speaker}.json"
        existing = load_profile(existing_path, encoder=encoder)
        if existing.checksum != original.checksum:
            raise ValueError("exact profile differs from Phase 6")
        profile_checks.append(
            {
                "speaker_id": speaker,
                "enrollment_count": count,
                "native_profile_sha256": original.checksum,
                "phase6_file_sha256": sha256_file(existing_path),
            }
        )
        path = output / f"native-profiles/{count}/{speaker}.json"
        save_profile(original, path)
        if load_profile(path, encoder=encoder).checksum != original.checksum:
            raise ValueError("profile round-trip differs")
        native[speaker, count] = original
        for method, margin in (("vowel_exact", 0), ("vowel_context20", 480)):
            grouped, members = {v: [] for v in VOWELS}, []
            for row in original.data["segments"]:
                filename = row["source_file"]
                first, last = effective_window(
                    row["start_frame"],
                    row["end_frame"],
                    inputs["sources"][filename]["frame_count"],
                    margin=margin,
                )
                grouped[row["vowel"]].append(
                    cache.embed(filename, first, last, row["segment_id"])
                )
                members.append({**row, "input_frames": [first, last]})
            vectors = {v: mean_profile(grouped[v]) for v in VOWELS}
            if method == "vowel_exact" and any(
                not np.array_equal(vectors[v], original.vector(v)) for v in VOWELS
            ):
                raise ValueError("pilot enrollment mean differs from public API")
            profile = {
                "method_id": method,
                "speaker_id": speaker,
                "enrollment_count": count,
                "model_identity": encoder.identity,
                "embedding_dimension": 128,
                "members": members,
                "vectors": {v: vectors[v].tolist() for v in VOWELS},
            }
            profiles.append(profile)
            references[method, speaker, count] = profile
    print(f"vowels: {len(profiles)} profiles; Phase 6 parity exact", flush=True)
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    queries = {q["query_id"]: q for q in inputs["queries"]}
    # Existing validators run on ORIGINAL cores, never on expanded context.
    for item in inputs["queries"]:
        query = VerificationInput(
            item["query_id"], tuple(EnrollmentSegment(**r) for r in item["segments"])
        )
        try:
            prepared = prepare_query(
                query, next(iter(native.values())), exact, audio_root=ROOT
            )
            records, counts = prepared.records, prepared.counts
            prepared.require_unchanged_sources()
        except IncompleteVerification as exc:
            records, counts = exc.records, exc.counts
        if any(r["status"] != "used" for r in records) or counts != item["counts"]:
            raise ValueError("Phase 6 selected query cores no longer eligible")
    scores, api_checks = [], 0
    baseline = {
        r["trial_id"]: r
        for r in read_rows(reference_run / "scores/validation.jsonl")
        if r["query_id"] in queries
    }
    max_phase6_difference = 0.0
    for window in inputs["windows"]:
        item = queries[window["query_id"]]
        by_id = {r["segment_id"]: r for r in item["segments"]}
        retained_query = VerificationInput(
            item["query_id"],
            tuple(
                EnrollmentSegment(**by_id[a["segment_id"]]) for a in window["anchors"]
            ),
        )
        for method, field in (
            ("vowel_exact", "exact_frames"),
            ("vowel_context20", "context_frames"),
        ):
            grouped = {v: [] for v in VOWELS}
            if window["complete_five_vowels"]:
                for anchor in window["anchors"]:
                    first, last = anchor[field]
                    if (
                        not window["recording_frames"][0]
                        <= first
                        < last
                        <= window["recording_frames"][1]
                    ):
                        raise ValueError("vowel input escaped query window")
                    grouped[anchor["vowel"]].append(
                        cache.embed(
                            item["source_file"], first, last, anchor["segment_id"]
                        )
                    )
            for trial in trials[window["query_id"]]:
                speaker, count = trial["claimed_speaker_id"], trial["enrollment_count"]
                profile = references[method, speaker, count]
                values = vowel_scores(
                    {v: np.asarray(profile["vectors"][v]) for v in VOWELS}, grouped
                )
                if method == "vowel_exact":
                    try:
                        result = verify(
                            native[speaker, count],
                            retained_query,
                            exact,
                            audio_root=ROOT,
                        )
                        expected = {
                            "fused": result.score,
                            **{v: result.data["vowels"][v]["score"] for v in VOWELS},
                        }
                    except IncompleteVerification:
                        expected = None
                    if values != expected:
                        raise ValueError("exact scores differ from public API")
                    api_checks += 1
                    if window["condition_id"] == "full":
                        saved = baseline[trial["trial_id"]]
                        if saved["status"] != (
                            "scored" if values is not None else "no_score"
                        ):
                            raise ValueError("Phase 6 no-score parity failed")
                        if values is not None:
                            difference = max(
                                abs(values[k] - saved["scores"][k]) for k in values
                            )
                            max_phase6_difference = max(
                                max_phase6_difference, difference
                            )
                            if (
                                difference
                                > inputs["config"]["phase6_parity_absolute_tolerance"]
                            ):
                                raise ValueError("Phase 6 score parity failed")
                scores.append(score_row(inputs, method, window, trial, profile, values))
    require_unchanged(encoder, before)
    load_inputs(args.run_dir)
    save_worker(
        output,
        profiles,
        scores,
        cache,
        {
            "status": "completed",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
            "model_parameters_and_buffers_unchanged": True,
            "feature_statistics_unchanged": True,
            "original_core_quality_validated": True,
            "expanded_context_quality_filter": False,
            "phase6_profile_checks": profile_checks,
            "public_api_score_checks": api_checks,
            "phase6_full_score_max_absolute_difference": max_phase6_difference,
            "elapsed_seconds": perf_counter() - started,
        },
    )
    print(
        f"vowels: {len(scores)} score slots; {api_checks} public API checks passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
