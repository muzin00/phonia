"""Reuse public registration/verification APIs with a checked, in-memory cache."""

import copy
import hashlib
from collections import defaultdict

import numpy as np
import torch
from phase3_data.manifest import json_sha256
from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    load_profile,
    register_user,
    save_profile,
)
from verification import IncompleteVerification, VerificationInput, verify

from .storage import row_writer


class CachedEncoder:
    def __init__(self, encoder: FrozenEncoder):
        self.base = encoder
        self.identity = encoder.identity
        self.pipeline = encoder.pipeline
        self.model = encoder.model
        self.cache = {}
        self.segments = {}
        self.hits = 0
        self.misses = 0

    def select(self, segments):
        self.cache.clear()
        self.segments = {row["segment_id"]: row for row in segments}
        if len(self.segments) != len(segments):
            raise ValueError("duplicate cache input segment")

    def embed(self, pcm, segment_id):
        if segment_id not in self.segments:
            raise ValueError("cache segment is not declared")
        key = json_sha256(
            {
                "encoder": self.identity,
                "segment": self.segments[segment_id],
                "pcm_sha256": hashlib.sha256(pcm.numpy().tobytes()).hexdigest(),
            }
        )
        if key not in self.cache:
            vector = np.asarray(self.base.embed(pcm, segment_id), dtype=np.float64)
            if (
                vector.shape != (128,)
                or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, rtol=0, atol=1e-12)
            ):
                raise ValueError("invalid cached embedding")
            self.cache[key] = vector.copy()
            self.misses += 1
        else:
            self.hits += 1
        return self.cache[key].copy()


def model_snapshot(encoder):
    return {
        "parameters": {
            name: value.clone() for name, value in encoder.model.state_dict().items()
        },
        "statistics": copy.deepcopy(encoder.pipeline.statistics),
        "identity": copy.deepcopy(encoder.identity),
    }


def require_unchanged(encoder, before):
    if (
        before["identity"] != encoder.identity
        or before["statistics"] != encoder.pipeline.statistics
        or any(
            not torch.equal(before["parameters"][name], value)
            for name, value in encoder.model.state_dict().items()
        )
        or any(
            p.requires_grad or p.grad is not None for p in encoder.model.parameters()
        )
    ):
        raise ValueError("frozen encoder or feature statistics changed")


def score_split(
    prepared: dict, trials: list[dict], encoder, output, *, audio_root, check_cache=True
) -> dict:
    split = prepared["split"]
    cached = CachedEncoder(encoder)
    profiles = {}
    checks = []
    before = model_snapshot(encoder)
    for enrollment in prepared["enrollment"]:
        count, speaker = enrollment["enrollment_count"], enrollment["user_id"]
        segments = [EnrollmentSegment(**row) for row in enrollment["segments"]]
        cached.select(enrollment["segments"])
        profile = register_user(
            speaker, segments, cached, audio_root=audio_root, minimum_segments=count
        )
        path = output / f"profiles/{split}/{count}/{speaker}.json"
        save_profile(profile, path)
        loaded = load_profile(path, encoder=encoder)
        if loaded.checksum != profile.checksum:
            raise ValueError("profile round-trip mismatch")
        if check_cache and speaker == prepared["speakers"][0]:
            reference = register_user(
                speaker,
                list(reversed(segments)),
                encoder,
                audio_root=audio_root,
                minimum_segments=count,
            )
            repeated = register_user(
                speaker, segments, cached, audio_root=audio_root, minimum_segments=count
            )
            if (
                reference.checksum != loaded.checksum
                or repeated.checksum != reference.checksum
            ):
                raise ValueError("cached registration differs from public API")
            checks.append(
                {
                    "kind": "registration",
                    "speaker": speaker,
                    "count": count,
                    "checksum": profile.checksum,
                }
            )
        profiles[(speaker, count)] = loaded
    by_query = defaultdict(list)
    for trial in trials:
        by_query[trial["query_id"]].append(trial)
    tested = set()
    scored_queries = 0
    print(
        f"{split}: profiles ready; scoring {len(prepared['queries'])} utterances",
        flush=True,
    )
    with (
        row_writer(output / f"scores/{split}.jsonl") as write_score,
        row_writer(output / f"phase5-results/{split}.jsonl.gz") as write_result,
        row_writer(output / f"queries/{split}.jsonl") as write_query,
    ):
        for index, item in enumerate(prepared["queries"], 1):
            query = VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**row) for row in item["segments"]),
            )
            cached.select(item["segments"])
            query_trials = by_query[item["query_id"]]
            missing = None
            query_records = None
            for trial in query_trials:
                profile = profiles[
                    (trial["claimed_speaker_id"], trial["enrollment_count"])
                ]
                if missing is None:
                    try:
                        result = verify(profile, query, cached, audio_root=audio_root)
                    except IncompleteVerification as exc:
                        missing = exc
                if missing is not None:
                    reason = (
                        "missing_extraction"
                        if not item["segments"]
                        else "missing_vowels"
                    )
                    write_score(
                        {
                            **trial,
                            "status": "no_score",
                            "reason": reason,
                            "scores": None,
                            "missing_vowels": missing.missing_vowels,
                            "counts": missing.counts,
                        }
                    )
                    query_records = missing.records
                    continue
                sample = (item["speaker_id"], item["role"], trial["is_genuine"])
                if (
                    check_cache
                    and trial["enrollment_count"] == 10
                    and sample not in tested
                ):
                    reference = verify(
                        profile,
                        VerificationInput(
                            query.query_id, tuple(reversed(query.segments))
                        ),
                        encoder,
                        audio_root=audio_root,
                    )
                    if reference.checksum != result.checksum:
                        raise ValueError("cached verification differs from public API")
                    tested.add(sample)
                    checks.append(
                        {
                            "kind": "verification",
                            "trial_id": trial["trial_id"],
                            "checksum": result.checksum,
                        }
                    )
                write_result(
                    {
                        "trial_id": trial["trial_id"],
                        "result": {**result.data, "sha256": result.checksum},
                    }
                )
                write_score(
                    {
                        **trial,
                        "status": "scored",
                        "reason": None,
                        "scores": {
                            "fused": result.score,
                            **{
                                v: row["score"]
                                for v, row in result.data["vowels"].items()
                            },
                        },
                        "profile_sha256": profile.checksum,
                        "result_sha256": result.checksum,
                    }
                )
                query_records = [
                    {**row, "score": None} for row in result.data["segments"]
                ]
            scored_queries += missing is None
            counts = {
                v: sum(
                    row["status"] == "used" and row["vowel"] == v
                    for row in query_records
                )
                for v in item["counts"]
            }
            write_query(
                {
                    **item,
                    "status": "scored" if missing is None else "no_score",
                    "missing_vowels": [] if missing is None else missing.missing_vowels,
                    "actual_counts": counts,
                    "quality_records": query_records,
                }
            )
            if index % 50 == 0:
                print(
                    f"{split}: {index}/{len(prepared['queries'])} utterances; {scored_queries} scored",
                    flush=True,
                )
    require_unchanged(encoder, before)
    if any(
        profile.checksum
        != load_profile(
            output / f"profiles/{split}/{count}/{speaker}.json", encoder=encoder
        ).checksum
        for (speaker, count), profile in profiles.items()
    ):
        raise ValueError("profile changed during evaluation")
    return {
        "schema_version": 1,
        "split": split,
        "cache_hits": cached.hits,
        "cache_misses": cached.misses,
        "cache_equivalence_checks": checks,
        "parameters_unchanged": True,
        "feature_statistics_unchanged": True,
        "profiles_unchanged": True,
        "gradients_disabled": True,
    }
