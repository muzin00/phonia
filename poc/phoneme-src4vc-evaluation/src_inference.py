"""Run the SRC4VC bundle through the existing public enrollment/verification APIs."""

from collections import defaultdict
from time import perf_counter

import numpy as np
import torch
from src_evaluation import ROOT, old, sha256_file, write_json

previous = old.previous
VOWELS = old.VOWELS
expanded_encoder = old.expanded_encoder
EnrollmentSegment, load_profile, writer = (
    old.EnrollmentSegment,
    old.load_profile,
    old.writer,
)


def infer_new(run, split, inputs):
    output = run / split / "jvs70_src4vc70"
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    encoder = expanded_encoder(inputs["config"])
    snapshot = previous.model_snapshot(encoder)
    tensors = [encoder.pipeline.mean.clone(), encoder.pipeline.std.clone()]
    audio = (previous.ValidationAudio if split == "validation" else previous.TestAudio)(
        inputs["sources"], 16
    )
    cache = previous.EmbeddingCache(
        audio,
        encoder.identity,
        lambda pcm, sid: encoder.embed(torch.from_numpy(pcm.copy()), sid),
        3,
        already_unit=True,
    )
    segments = {}
    for item in inputs["enrollment"] + inputs["queries"]:
        for row in item["segments"]:
            if row["segment_id"] in segments and segments[row["segment_id"]] != row:
                raise ValueError("conflicting segment identity")
            segments[row["segment_id"]] = row

    class Adapter:
        identity, pipeline, model = encoder.identity, encoder.pipeline, encoder.model

        def embed(self, pcm, sid):
            row = segments[sid]
            name, first, last = row["source_file"], row["start_frame"], row["end_frame"]
            if not np.array_equal(pcm.numpy(), audio.slice(name, first, last)):
                raise ValueError("public API PCM differs")
            first, last = previous.effective_window(
                first, last, inputs["sources"][name]["frame_count"]
            )
            return cache.embed(name, first, last, sid)

    adapter = Adapter()
    profiles = {}
    for item in inputs["enrollment"]:
        speaker = item["user_id"]
        profile = previous.register_user(
            speaker,
            [EnrollmentSegment(**r) for r in item["segments"]],
            adapter,
            audio_root=ROOT,
        )
        if any(r["status"] != "used" for r in profile.data["segments"]):
            raise ValueError("registration eligibility changed")
        if speaker == inputs["speakers"][0]:
            fresh = previous.register_user(
                speaker,
                [EnrollmentSegment(**r) for r in reversed(item["segments"])],
                encoder,
                audio_root=ROOT,
            )
            if fresh.checksum != profile.checksum:
                raise ValueError("cache-free registration mismatch")
        path = output / "profiles" / f"{speaker}.json"
        previous.save_profile(profile, path)
        profiles[speaker] = load_profile(path, encoder=encoder)
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    tested, slots = set(), 0
    with writer(output / "scores.jsonl.gz") as write:
        for index, item in enumerate(inputs["queries"], 1):
            query = previous.VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**r) for r in item["segments"]),
            )
            try:
                prepared = previous.prepare_query(
                    query, next(iter(profiles.values())), adapter, audio_root=ROOT
                )
                records, counts = prepared.records, prepared.counts
                prepared.require_unchanged_sources()
            except previous.IncompleteVerification as exc:
                records, counts = exc.records, exc.counts
            if any(r["status"] != "used" for r in records) or counts != item["counts"]:
                raise ValueError("query eligibility changed")
            grouped = {v: [] for v in VOWELS}
            if all(counts.values()):
                for row in item["segments"]:
                    grouped[row["vowel"]].append(
                        adapter.embed(
                            torch.from_numpy(
                                audio.slice(
                                    row["source_file"],
                                    row["start_frame"],
                                    row["end_frame"],
                                )
                            ),
                            row["segment_id"],
                        )
                    )
            for trial in trials[item["query_id"]]:
                profile = profiles[trial["claimed_speaker_id"]]
                values = previous.vowel_scores(
                    {v: profile.vector(v) for v in VOWELS}, grouped
                )
                sample = (
                    item["speaker_id"],
                    item["role"],
                    tuple(v for v in VOWELS if counts[v]),
                    trial["is_genuine"],
                )
                if sample not in tested:
                    try:
                        result = previous.verify(
                            profile, query, encoder, audio_root=ROOT
                        )
                        expected = {
                            "fused": result.score,
                            **{v: result.data["vowels"][v]["score"] for v in VOWELS},
                        }
                    except previous.IncompleteVerification:
                        expected = None
                    if values != expected:
                        raise ValueError("cache-free public verification mismatch")
                    tested.add(sample)
                write(
                    {
                        **trial,
                        "model_id": "jvs70_src4vc70",
                        "status": "scored" if values is not None else "no_score",
                        "reason": None if values is not None else "missing_vowels",
                        "scores": values,
                        "profile_sha256": profile.checksum,
                    }
                )
                slots += 1
            if index % 100 == 0:
                print(
                    f"{split}: {index}/1200 queries, {len(cache.vectors)} embeddings",
                    flush=True,
                )
    previous.require_unchanged(encoder, snapshot)
    if not all(
        torch.equal(a, b)
        for a, b in zip(
            tensors, [encoder.pipeline.mean, encoder.pipeline.std], strict=True
        )
    ):
        raise ValueError("feature tensors changed")
    audio.require_unchanged()
    cache.save(output)
    write_json(
        output / "report.json",
        {
            "status": "completed",
            "reused": False,
            "profiles": len(profiles),
            "score_slots": slots,
            "encoder_identity": encoder.identity,
            "cache_free_verification_checks": len(tested),
            "cache_free_registration_checks": 1,
            "unique_embeddings": len(cache.vectors),
            "embedding_repeats": 3,
            "all_repeats_bitwise_equal": True,
            "weights_statistics_and_sources_unchanged": True,
            "elapsed_seconds": perf_counter() - started,
            "outputs_sha256": {
                str(p.relative_to(output)): sha256_file(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
