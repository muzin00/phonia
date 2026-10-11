"""Noise is added to real waveform slices; pair labels never select augmentation."""

from __future__ import annotations

import math
import wave
from collections import defaultdict

import shared as s


def noisy_clip(clean, seed, snr_db):
    values = s.np.asarray(clean, dtype=s.np.float64)
    rng = s.np.random.Generator(s.np.random.PCG64(seed))
    noise = rng.standard_normal(len(values))
    noise *= s.np.sqrt(s.np.mean(values**2) / s.np.mean(noise**2)) / 10 ** (snr_db / 20)
    before_clip = values + noise
    result = s.np.clip(before_clip, -1, 1).astype(s.np.float32)
    achieved = float(
        10 * s.np.log10(s.np.mean(values**2) / s.np.mean((result - values) ** 2))
    )
    return result, achieved, int(s.np.count_nonzero(s.np.abs(before_clip) > 1))


def make_schedule(protocol, settings, seed, phone_count):
    rng = s.np.random.Generator(
        s.np.random.PCG64(seed + settings["schedule_seed_offset"])
    )
    shape = (protocol["updates"], protocol["queries_per_update"])
    modes = rng.choice(4, size=shape, p=settings["mode_probabilities"]).astype(
        s.np.uint8
    )
    keep = (
        rng.random((*shape, phone_count)) >= settings["missing_consonant_probability"]
    )
    keep[modes < 2] = True
    keep[..., :5] = True
    return {"modes": modes, "keep": keep}


def augmented_batch(clean, noisy, data, pairs, augmentation, update):
    q = s.torch.from_numpy(pairs["queries"][update].astype(s.np.int64))
    negative = s.torch.from_numpy(pairs["negative_claims"][update].astype(s.np.int64))
    index = {speaker: i for i, speaker in enumerate(data["speakers"])}
    positive = s.torch.tensor([index[data["queries"][int(i)]["speaker_id"]] for i in q])
    use_noise = s.torch.from_numpy((augmentation["modes"][update] % 2) == 1)
    query = s.torch.where(
        use_noise[:, None, None], noisy["queries"][q], clean["queries"][q]
    )
    metadata = s.torch.where(
        use_noise[:, None, None], noisy["qmeta"][q], clean["qmeta"][q]
    )
    keep = s.torch.from_numpy(augmentation["keep"][update])
    shared_mask = (
        clean["qmask"][q] & clean["emask"][positive] & clean["emask"][negative] & keep
    )
    if not shared_mask[:, :5].all():
        raise ValueError("augmentation removed a required vowel")
    claims = s.torch.cat((positive, negative))
    features, components = s.backend.pair_features(
        clean["enrollment"][claims],
        clean["emeta"][claims],
        s.torch.cat((query, query)),
        s.torch.cat((metadata, metadata)),
    )
    return (
        features,
        components,
        s.torch.cat((shared_mask, shared_mask)),
        s.torch.cat((s.torch.ones(len(q)), s.torch.zeros(len(q)))),
    )


def noise_bank(config, protocol, previous, run, phones):
    settings = config["augmentation"]
    source = s.ROOT / protocol["source_run"]
    data = s.read_json(previous / "train-inputs.json")
    needed = {
        i for q in data["queries"] for indices in q["groups"].values() for i in indices
    }
    control_indices = set(sorted(needed)[::11000])
    metadata = s.np.empty((data["feature_count"], 2), s.np.float32)
    selected = defaultdict(list)
    control = []
    for row in s.frozen.rows(source / "train-segments.jsonl"):
        i = row["cache_index"]
        metadata[i] = [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
        if i not in needed:
            continue
        if i in control_indices:
            control.append(
                {
                    k: row[k]
                    for k in (
                        "source_file",
                        "start_frame",
                        "end_frame",
                        "segment_id",
                        "cache_index",
                    )
                }
            )
        digest = s.primitives.rank(settings["training_noise_seed"], row["segment_id"])
        if int(digest[:8], 16) / 2**32 < settings["noisy_query_interval_probability"]:
            selected[row["source_file"]].append(
                {
                    k: row[k]
                    for k in (
                        "source_file",
                        "start_frame",
                        "end_frame",
                        "segment_id",
                        "cache_index",
                    )
                }
            )
    if set(selected) & set(data["enrollment_sources"]):
        raise ValueError("enrollment audio selected for augmentation")
    model, _ = s.frozen.load_model(source / "trials" / protocol["source_trial"])
    raw = s.np.memmap(
        source / "train-features.f32",
        mode="r",
        dtype="<f4",
        shape=(data["feature_count"], 128),
    )
    vectors = s.frozen.model_embeddings(model, raw)
    pipe = s.primitives.pipeline(s.read_json(source / "design-freeze.json")["config"])
    controls, expected = [], []
    for row in control:
        with wave.open(str(s.ROOT / row["source_file"]), "rb") as wav:
            wav.setpos(row["start_frame"])
            pcm = (
                s.np.frombuffer(
                    wav.readframes(row["end_frame"] - row["start_frame"]), dtype="<i2"
                ).astype(s.np.float32)
                / 32768
            )
        controls.append(s.primitives.pool_features(pipe, pcm, row["segment_id"]))
        expected.append(vectors[row["cache_index"]])
    replay = s.frozen.model_embeddings(model, s.np.asarray(controls, s.np.float32))
    replay_error = float(s.np.max(s.np.abs(replay - s.np.asarray(expected))))
    if replay_error > 2e-6:
        raise ValueError("waveform/cache encoder arithmetic differs")
    features, indices, snrs = [], [], []
    selected_indices, clipped = [], 0
    for count, (name, intervals) in enumerate(sorted(selected.items()), 1):
        with wave.open(str(s.ROOT / name), "rb") as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (
                24000,
                1,
                2,
            ):
                raise ValueError("wrong training WAV format")
            pcm = (
                s.np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                    s.np.float32
                )
                / 32768
            )
        for row in intervals:
            seed = int(
                s.primitives.rank(settings["training_noise_seed"], row["segment_id"])[
                    :16
                ],
                16,
            )
            corrupted, snr, clips = noisy_clip(
                pcm[row["start_frame"] : row["end_frame"]],
                seed,
                settings["white_noise_snr_db"],
            )
            features.append(
                s.primitives.pool_features(pipe, corrupted, row["segment_id"])
            )
            indices.append(row["cache_index"])
            selected_indices.append(row["cache_index"])
            snrs.append(snr)
            clipped += clips
            metadata[row["cache_index"], 1] = 20 * math.log10(
                float(s.np.sqrt(s.np.mean(corrupted.astype(s.np.float64) ** 2)))
            )
        if len(features) >= 1024:
            vectors[indices] = s.frozen.model_embeddings(
                model, s.np.asarray(features, s.np.float32)
            )
            features, indices = [], []
        if count % 1000 == 0:
            print(
                f"Training noise: {count}/{len(selected)} WAVs, {len(selected_indices)} intervals",
                flush=True,
            )
    if features:
        vectors[indices] = s.frozen.model_embeddings(
            model, s.np.asarray(features, s.np.float32)
        )
    arrays = s.frozen.aggregates(data, vectors, metadata, phones)
    clean = s.frozen.load_arrays(previous, "train")
    for name in ("enrollment", "emeta", "emask", "qmask"):
        if not s.np.array_equal(arrays[name], clean[name].numpy()):
            raise ValueError(f"augmentation changed enrollment or support: {name}")
    s.np.savez(run / "train-noisy-aggregates.npz", **arrays)
    s.write_json(
        run / "training-noise-provenance.json",
        {
            "status": "passed",
            "eligible_training_query_intervals": len(needed),
            "corrupted_intervals": len(selected_indices),
            "corrupted_cache_indices": selected_indices,
            "affected_query_wavs": len(selected),
            "clipped_samples": clipped,
            "achieved_snr_db_minimum": min(snrs),
            "achieved_snr_db_maximum": max(snrs),
            "achieved_snr_db_mean": float(s.np.mean(snrs)),
            "clean_waveform_replay_probes": len(control),
            "clean_replay_max_embedding_error": replay_error,
            "enrollment_and_query_support_unchanged": True,
            "training_queries_only": True,
        },
    )


def prepare(config, run):
    if run.exists():
        raise ValueError("prepare requires an unused run")
    source_design = s.verify_source(config, full=True)
    protocol = source_design["config"]
    previous = s.source(config)
    if config["augmentation"]["training_noise_seed"] == s.stress.STRESS["seed"]:
        raise ValueError("train/test noise seeds must differ")
    run.mkdir(parents=True)
    phones = source_design["phones"]
    print("Preparing waveform augmentation on original training queries", flush=True)
    noise_bank(config, protocol, previous, run, phones)
    for seed in protocol["seeds"]:
        s.np.savez(
            run / f"augmentation-{seed}.npz",
            **make_schedule(protocol, config["augmentation"], seed, len(phones)),
        )
    source = s.ROOT / protocol["source_run"]
    segments = list(s.frozen.rows(source / "test-segments.jsonl"))
    condition = s.original_case(config, "clean")
    vectors = s.np.load(condition / "test-embeddings.npy", allow_pickle=False)
    original = s.read_json(previous / "test-inputs.json")
    original_metadata = s.np.array(
        [
            [(r["end_frame"] - r["start_frame"]) / 24000, r["rms_dbfs"]]
            for r in segments
        ],
        s.np.float32,
    )
    missing = s.read_json(
        s.original_case(config, "missing-half-consonants") / "test-inputs.json"
    )
    s.np.savez(
        run / "test-missing-aggregates.npz",
        **s.frozen.aggregates(missing, vectors, original_metadata, phones),
    )
    print(
        "Replaying exact previous noise diagnostic to retain its metadata", flush=True
    )
    noisy, noisy_metadata, _ = s.stress.noisy_vectors(source, vectors, segments)
    stored = s.np.load(
        s.original_case(config, "noise-20pct-intervals-snr10") / "test-embeddings.npy",
        allow_pickle=False,
    )
    if not s.np.array_equal(noisy, stored):
        raise ValueError("test noise differs from previous diagnostic")
    s.np.savez(
        run / "test-noisy-aggregates.npz",
        **s.frozen.aggregates(original, noisy, noisy_metadata, phones),
    )
    files = {}
    for path in (
        s.CONFIG,
        *s.BASE.glob("*.py"),
        *s.BASE.glob("tests/*.py"),
        previous / "design-freeze.json",
        previous / "selection-freeze.json",
        previous / "completion-verification.json",
        previous / "stress/completion-verification.json",
    ):
        s.primitives.pin(files, path)
    for path in run.glob("*"):
        s.primitives.pin(files, path)
    s.write_json(
        run / "design-freeze.json",
        {
            "status": "augmentation_inputs_code_and_schedules_frozen_before_training",
            "config": config,
            "training_protocol": protocol,
            "phones": phones,
            "runtime": s.frozen.expanded.runtime(),
            "source_fusion_design_sha256": s.sha256_file(
                previous / "design-freeze.json"
            ),
            "files": files,
            "test_noise_embeddings_exactly_reproduced": True,
            "training_pair_schedules_reused_without_change": True,
            "validation_selection_unchanged": True,
        },
    )
    print(
        "Augmentation prepared, original test diagnostics reproduced exactly",
        flush=True,
    )
