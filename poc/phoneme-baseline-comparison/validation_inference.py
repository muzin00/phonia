"""Full-validation model workers; identical pilot math with streaming scores."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import defaultdict
from importlib import metadata
from time import perf_counter

import numpy as np
import torch

from comparison_inputs import effective_window
from pilot import (
    VOWELS,
    EmbeddingCache,
    digest,
    load_inputs,
    mean_profile,
    read_rows,
    vowel_scores,
)
from smoke import CONFIG, ROOT, check_model_snapshot, sha256_file, write_json
from validation import BoundedAudio, scored_row, writer


def finish(run, output, profiles, cache, report, started):
    cache.audio.require_unchanged()
    load_inputs(run)
    cache.save(output)
    with writer(output / "profiles.jsonl") as write:
        for profile in profiles:
            write(profile)
    report.update(
        {
            "status": "completed",
            "unique_embeddings": len(cache.vectors),
            "all_repeats_bitwise_equal": True,
            "source_audio_unchanged": True,
            "elapsed_seconds": perf_counter() - started,
            "outputs_sha256": {
                name: sha256_file(output / name)
                for name in (
                    "profiles.jsonl",
                    "scores.jsonl.gz",
                    "embeddings.npy",
                    "embedding-inputs.jsonl",
                )
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
            },
        }
    )
    write_json(output / "report.json", report)


def indexes(inputs):
    trials, windows = defaultdict(list), defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    for window in inputs["windows"]:
        windows[window["query_id"]].append(window)
    return trials, windows


def run_vowels(run):
    for project in (
        "phoneme-speaker-encoder",
        "phoneme-user-registration",
        "phoneme-verification",
        "phoneme-verification-evaluation",
    ):
        sys.path.insert(0, str(ROOT / "poc" / project))
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

    inputs = load_inputs(run)
    if str(torch.__version__) != inputs["protocol"]["models"]["vowel"]["torch"]:
        raise ValueError("requires Phase 6 vowel runtime")
    output = run / "vowels"
    output.mkdir(exist_ok=False)
    started = perf_counter()
    encoder = FrozenEncoder.from_bundle(
        ROOT / inputs["protocol"]["models"]["vowel"]["bundle"]
    )
    before = model_snapshot(encoder)
    feature_tensors = [encoder.pipeline.mean.clone(), encoder.pipeline.std.clone()]
    audio = BoundedAudio(
        inputs["sources"], inputs["config"]["audio_cache_maximum_sources"]
    )
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
            name, first, last = row["source_file"], row["start_frame"], row["end_frame"]
            if not np.array_equal(pcm.numpy(), audio.slice(name, first, last)):
                raise ValueError("native API input differs from declared core")
            first, last = effective_window(
                first, last, inputs["sources"][name]["frame_count"]
            )
            return cache.embed(name, first, last, segment_id)

    exact = ExactAdapter()
    native, profiles, references, vectors, hashes, checks = {}, [], {}, {}, {}, []
    phase6 = ROOT / inputs["protocol"]["phase6_reference"]["run"]
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
            raise ValueError("enrollment core quality changed")
        reference = load_profile(
            phase6 / f"profiles/validation/{count}/{speaker}.json", encoder=encoder
        )
        if reference.checksum != original.checksum:
            raise ValueError("Phase 6 profile parity failed")
        save_profile(original, output / f"native-profiles/{count}/{speaker}.json")
        native[speaker, count] = original
        checks.append(
            {
                "speaker_id": speaker,
                "enrollment_count": count,
                "profile_sha256": original.checksum,
            }
        )
        for method, margin in (("vowel_exact", 0), ("vowel_context20", 480)):
            grouped, members = {v: [] for v in VOWELS}, []
            for row in original.data["segments"]:
                name = row["source_file"]
                first, last = effective_window(
                    row["start_frame"],
                    row["end_frame"],
                    inputs["sources"][name]["frame_count"],
                    margin=margin,
                )
                grouped[row["vowel"]].append(
                    cache.embed(name, first, last, row["segment_id"])
                )
                members.append({**row, "input_frames": [first, last]})
            reference_vectors = {v: mean_profile(grouped[v]) for v in VOWELS}
            if method == "vowel_exact" and any(
                not np.array_equal(reference_vectors[v], original.vector(v))
                for v in VOWELS
            ):
                raise ValueError("native enrollment aggregation parity failed")
            profile = {
                "method_id": method,
                "speaker_id": speaker,
                "enrollment_count": count,
                "model_identity": encoder.identity,
                "embedding_dimension": 128,
                "members": members,
                "vectors": {v: reference_vectors[v].tolist() for v in VOWELS},
            }
            key = method, speaker, count
            profiles.append(profile)
            references[key], vectors[key], hashes[key] = (
                profile,
                reference_vectors,
                digest(profile),
            )
    print(
        f"vowels: {len(profiles)} profiles; all {len(checks)} exact profiles match Phase 6",
        flush=True,
    )
    trials, windows = indexes(inputs)
    baseline = {r["trial_id"]: r for r in read_rows(phase6 / "scores/validation.jsonl")}
    max_difference, api_checks, baseline_checks, score_count = 0.0, 0, 0, 0
    with writer(output / "scores.jsonl.gz") as write:
        for index, item in enumerate(inputs["queries"], 1):
            original_query = VerificationInput(
                item["query_id"],
                tuple(EnrollmentSegment(**r) for r in item["segments"]),
            )
            try:
                prepared = prepare_query(
                    original_query, next(iter(native.values())), exact, audio_root=ROOT
                )
                records, counts = prepared.records, prepared.counts
                prepared.require_unchanged_sources()
            except IncompleteVerification as exc:
                records, counts = exc.records, exc.counts
            if any(r["status"] != "used" for r in records) or counts != item["counts"]:
                raise ValueError("selected query core quality changed")
            by_id = {r["segment_id"]: r for r in item["segments"]}
            for window in windows[item["query_id"]]:
                retained = VerificationInput(
                    item["query_id"],
                    tuple(
                        EnrollmentSegment(**by_id[a["segment_id"]])
                        for a in window["anchors"]
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
                                raise ValueError("input escaped query recording window")
                            grouped[anchor["vowel"]].append(
                                cache.embed(
                                    item["source_file"],
                                    first,
                                    last,
                                    anchor["segment_id"],
                                )
                            )
                    # The same sorted segment order and per-segment dot/clip as the pilot.
                    tested_impostor = False
                    for trial in trials[item["query_id"]]:
                        speaker, count = (
                            trial["claimed_speaker_id"],
                            trial["enrollment_count"],
                        )
                        key = method, speaker, count
                        values = vowel_scores(vectors[key], grouped)
                        if method == "vowel_exact":
                            if count == 10 and (
                                trial["is_genuine"] or not tested_impostor
                            ):
                                try:
                                    result = verify(
                                        native[speaker, count],
                                        retained,
                                        exact,
                                        audio_root=ROOT,
                                    )
                                    expected = {
                                        "fused": result.score,
                                        **{
                                            v: result.data["vowels"][v]["score"]
                                            for v in VOWELS
                                        },
                                    }
                                except IncompleteVerification:
                                    expected = None
                                if values != expected:
                                    raise ValueError(
                                        "public API verification parity failed"
                                    )
                                api_checks += 1
                                tested_impostor |= not trial["is_genuine"]
                            if window["condition_id"] == "full":
                                saved = baseline[trial["trial_id"]]
                                if saved["status"] != (
                                    "scored" if values is not None else "no_score"
                                ):
                                    raise ValueError("Phase 6 no-score parity failed")
                                if values is not None:
                                    delta = max(
                                        abs(values[k] - saved["scores"][k])
                                        for k in values
                                    )
                                    max_difference = max(max_difference, delta)
                                    if (
                                        delta
                                        > inputs["config"][
                                            "phase6_parity_absolute_tolerance"
                                        ]
                                    ):
                                        raise ValueError(
                                            "Phase 6 full score parity failed"
                                        )
                                baseline_checks += 1
                        write(
                            scored_row(
                                inputs, method, window, trial, hashes[key], values
                            )
                        )
                        score_count += 1
            if index % 50 == 0:
                print(
                    f"vowels: {index}/1200 queries; {len(cache.vectors)} embeddings; {perf_counter() - started:.1f}s",
                    flush=True,
                )
    require_unchanged(encoder, before)
    if not torch.equal(feature_tensors[0], encoder.pipeline.mean) or not torch.equal(
        feature_tensors[1], encoder.pipeline.std
    ):
        raise ValueError("feature normalization tensors changed")
    finish(
        run,
        output,
        profiles,
        cache,
        {
            "phase6_profile_checks": checks,
            "public_api_score_checks": api_checks,
            "phase6_full_score_checks": baseline_checks,
            "phase6_full_score_max_absolute_difference": max_difference,
            "score_slots": score_count,
            "original_core_quality_validated": True,
            "expanded_context_quality_filter": False,
            "parameters_buffers_and_feature_statistics_unchanged": True,
        },
        started,
    )


def run_ecapa(run, model_dir):
    import torchaudio
    from speechbrain.inference.speaker import SpeakerRecognition

    inputs = load_inputs(run)
    config = json.loads(CONFIG.read_text())
    expected = inputs["protocol"]["models"]["ecapa"]
    if (
        str(torch.__version__) != expected["torch"]
        or torchaudio.__version__ != expected["torchaudio"]
    ):
        raise ValueError("requires pinned ECAPA runtime")
    snapshot = check_model_snapshot(model_dir, config)
    output = run / "ecapa"
    output.mkdir(exist_ok=False)
    started = perf_counter()
    encoder = SpeakerRecognition.from_hparams(
        source=str(model_dir.resolve()),
        savedir=str(output / "loaded-model"),
        overrides={"pretrained_path": str(model_dir.resolve())},
        run_opts={"device": "cpu"},
    )
    encoder.eval()
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("ECAPA weights not frozen")

    def state_hash():
        value = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    initial_state = state_hash()
    resampler = torchaudio.transforms.Resample(
        orig_freq=24000,
        new_freq=16000,
        resampling_method=config["audio"]["resampling_method"],
        lowpass_filter_width=config["audio"]["lowpass_filter_width"],
        rolloff=config["audio"]["rolloff"],
        dtype=torch.float32,
    )

    def forward(pcm, sid):
        with torch.inference_mode():
            waveform = resampler(torch.from_numpy(pcm.copy())).unsqueeze(0)
            if (
                waveform.shape != (1, (len(pcm) * 16000 + 23999) // 24000)
                or not torch.isfinite(waveform).all()
            ):
                raise ValueError("invalid resampled waveform")
            result = encoder.encode_batch(waveform, normalize=False)
            if (
                result.shape != (1, 1, 192)
                or result.dtype != torch.float32
                or result.device.type != "cpu"
            ):
                raise ValueError("invalid ECAPA embedding shape/dtype/device")
            return result.reshape(-1).numpy().copy()

    identity = {
        "model": config["model"],
        "audio": config["audio"],
        "model_state_sha256": initial_state,
    }
    audio = BoundedAudio(
        inputs["sources"], inputs["config"]["audio_cache_maximum_sources"]
    )
    cache = EmbeddingCache(
        audio, identity, forward, inputs["config"]["repeats_per_unique_embedding"]
    )
    profiles, references, vectors, hashes = [], {}, {}, {}
    for item in inputs["enrollment"]:
        members, embeddings = [], []
        for name in sorted({s["source_file"] for s in item["segments"]}):
            source = inputs["sources"][name]
            if (
                source["evaluation_role"] != "enrollment"
                or source["speaker_id"] != item["user_id"]
            ):
                raise ValueError("ECAPA registration role/speaker mismatch")
            last = source["frame_count"]
            embeddings.append(cache.embed(name, 0, last, name))
            members.append(
                {
                    "source_file": name,
                    "source_sha256": source["source_sha256"],
                    "input_frames": [0, last],
                    "resampled_frames": (last * 16000 + 23999) // 24000,
                }
            )
        vector = mean_profile(embeddings)
        profile = {
            "method_id": "ecapa_whole",
            "speaker_id": item["user_id"],
            "enrollment_count": item["enrollment_count"],
            "model_identity": identity,
            "embedding_dimension": 192,
            "members": members,
            "vector": vector.tolist(),
        }
        key = item["user_id"], item["enrollment_count"]
        profiles.append(profile)
        references[key], vectors[key], hashes[key] = profile, vector, digest(profile)
    print(
        f"ecapa: {len(profiles)} profiles; {len(cache.vectors)} registration WAVs",
        flush=True,
    )
    trials, windows = indexes(inputs)
    score_count = 0
    with writer(output / "scores.jsonl.gz") as write:
        for index, item in enumerate(inputs["queries"], 1):
            for window in windows[item["query_id"]]:
                first, last = window["recording_frames"]
                vector = cache.embed(item["source_file"], first, last, item["query_id"])
                for trial in trials[item["query_id"]]:
                    key = trial["claimed_speaker_id"], trial["enrollment_count"]
                    score = float(np.clip(np.dot(vectors[key], vector), -1, 1))
                    write(
                        scored_row(
                            inputs,
                            "ecapa_whole",
                            window,
                            trial,
                            hashes[key],
                            {"fused": score},
                        )
                    )
                    score_count += 1
            if index % 50 == 0:
                print(
                    f"ecapa: {index}/1200 queries; {len(cache.vectors)} embeddings; {perf_counter() - started:.1f}s",
                    flush=True,
                )
    if (
        initial_state != state_hash()
        or snapshot != check_model_snapshot(model_dir, config)
        or any(p.grad is not None or p.requires_grad for p in encoder.parameters())
    ):
        raise ValueError("frozen ECAPA state changed")
    finish(
        run,
        output,
        profiles,
        cache,
        {
            "model_state_sha256": initial_state,
            "model_snapshot": snapshot,
            "model_parameters_buffers_and_snapshot_unchanged": True,
            "score_slots": score_count,
            "ecapa_environment": {
                p: metadata.version(p) for p in ("torchaudio", "speechbrain")
            },
        },
        started,
    )
