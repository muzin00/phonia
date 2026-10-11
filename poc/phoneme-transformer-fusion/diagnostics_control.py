"""Replay raw clean audio and verify that noise never changes enrollment."""

from __future__ import annotations

import importlib
import wave

import data as shared


def audit_control(run):
    config = shared.read_json(shared.CONFIG)
    source = shared.ROOT / config["source_run"]
    condition = shared.ROOT / config["evaluation_run"] / "enrollment-30"
    segments = list(shared.rows(source / "test-segments.jsonl"))
    original = shared.np.load(condition / "test-embeddings.npy", allow_pickle=False)
    noisy = shared.np.load(
        run / "stress/noise-20pct-intervals-snr10/test-embeddings.npy",
        allow_pickle=False,
    )
    recorded = shared.read_json(run / "stress/results.json")["challenges"][
        "noise-20pct-intervals-snr10"
    ]["conditions"]
    corrupted = recorded["corrupted_cache_indices"]
    if (
        len(set(corrupted)) != len(corrupted)
        or len(corrupted) != recorded["corrupted_intervals"]
    ):
        raise ValueError("duplicate or missing corrupted interval")
    untouched = shared.np.ones(len(original), dtype=bool)
    untouched[corrupted] = False
    enrollment = [r["cache_index"] for r in segments if r["role"] == "enrollment"]
    if not untouched[enrollment].all() or not shared.np.array_equal(
        original[untouched], noisy[untouched]
    ):
        raise ValueError("enrollment or unselected query interval changed")
    if not shared.np.allclose(shared.np.linalg.norm(noisy, axis=1), 1, atol=2e-6):
        raise ValueError("invalid frozen-encoder output norm")
    primitives = importlib.import_module("common")
    pipe = primitives.pipeline(
        shared.read_json(source / "design-freeze.json")["config"]
    )
    model, _ = shared.load_model(source / "trials" / config["source_trial"])
    probes = [r for r in segments if r["role"] != "enrollment"][::1500]
    features = []
    for row in probes:
        with wave.open(str(shared.ROOT / row["source_file"]), "rb") as wav:
            wav.setpos(row["start_frame"])
            samples = (
                shared.np.frombuffer(
                    wav.readframes(row["end_frame"] - row["start_frame"]), dtype="<i2"
                ).astype(shared.np.float32)
                / 32768
            )
        features.append(primitives.pool_features(pipe, samples, row["segment_id"]))
    actual = shared.model_embeddings(
        model, shared.np.asarray(features, shared.np.float32)
    )
    error = float(
        shared.np.max(
            shared.np.abs(actual - original[[r["cache_index"] for r in probes]])
        )
    )
    if error > 2e-6:
        raise ValueError("waveform pipeline differs from the frozen feature cache")
    report = {
        "status": "passed",
        "clean_waveform_replay_probes": len(probes),
        "max_embedding_abs_error": error,
        "all_enrollment_embeddings_unchanged": True,
        "all_unselected_query_embeddings_unchanged": True,
        "original_encoder_sha256": shared.sha256_file(
            source / "trials" / config["source_trial"] / "encoder.pt"
        ),
        "control_script_sha256": shared.sha256_file(__file__),
    }
    shared.write_json(run / "stress/noise-clean-control-audit.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    shared.torch.set_num_threads(1)
    audit_control(
        shared.ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1"
    )
