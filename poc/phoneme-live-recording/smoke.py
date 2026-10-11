"""Replay existing JVS audio through the live backend; not new human evidence."""

import argparse
import time
from pathlib import Path

import runtime as r
from store import Store, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if not root.is_relative_to(r.ROOT / "artifacts/phoneme-live-recording"):
        raise ValueError("Smoke output must be under artifacts/phoneme-live-recording")
    root.mkdir(parents=True, exist_ok=False)
    config = {**read(r.BASE / "protocol.json"), "purpose": "replay_smoke"}
    engine = r.Runtime(config)
    source_run = r.ROOT / config["encoder_run"]
    cache_root = source_run.parents[1]
    old_rows = list(r.s.rows(cache_root / "test-segments.jsonl"))
    utterance = old_rows[0]["utterance_id"]
    expected = next(
        row
        for row in r.s.rows(
            r.ROOT
            / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
        )
        if row["utterance_id"] == utterance
    )
    original = r.ROOT / expected["source_file"]
    r.s.checked(original, expected["source_sha256"])
    pcm, info = r.decode_wav(original.read_bytes(), config["maximum_clip_seconds"])
    started = time.monotonic()
    result = engine.process(
        pcm,
        root / "replay",
        {
            "utterance_id": "smoke",
            "speaker_id": "replay",
            "session_id": "replay",
            "transcript": expected["transcript"],
            "role": "query",
            "origin": "replay",
            "recorded_on": None,
            "device": "corpus-jvs",
        },
    )
    elapsed = time.monotonic() - started
    fresh = list(r.s.rows(root / "replay/segments.jsonl"))

    def key(row):
        return row["phoneme"], row["raw_start_frame"], row["raw_end_frame"]

    reference = {key(row): row for row in old_rows if row["utterance_id"] == utterance}
    if set(reference) != {key(row) for row in fresh}:
        raise AssertionError("Original segment boundaries differ")
    features = r.np.load(root / "replay/features.npy", allow_pickle=False)
    vectors = r.np.load(root / "replay/embeddings.npy", allow_pickle=False)
    original_features = r.np.memmap(
        cache_root / "test-features.f32",
        dtype=r.np.float32,
        mode="r",
        shape=(len(old_rows), 128),
    )
    original_vectors = r.np.load(
        source_run / "test-embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    indices = [reference[key(row)]["cache_index"] for row in fresh]
    feature_error = float(r.np.max(r.np.abs(features - original_features[indices])))
    vector_error = float(r.np.max(r.np.abs(vectors - original_vectors[indices])))
    if feature_error > 1e-6 or vector_error > 1e-6:
        raise AssertionError((feature_error, vector_error))
    # Exercise all 20 enrollment steps with distinct recordings from one existing
    # speaker. Replay transcripts are used only in this isolated smoke store.
    sources = {
        row["utterance_id"]
        for row in old_rows
        if row["speaker_id"] == expected["speaker_id"]
    }
    recordings = [
        row
        for row in r.s.rows(
            r.ROOT
            / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
        )
        if row["utterance_id"] in sources
        and 0 < row["duration_sec"] <= config["maximum_clip_seconds"]
    ][:20]
    if len(recordings) != 20:
        raise AssertionError("Need 20 distinct source recordings")
    replay_config = {
        **config,
        "prompts": [
            {"id": row["utterance_id"], "text": row["transcript"]} for row in recordings
        ],
    }
    database = Store(root / "registration", replay_config, engine, r.decode_wav)
    episode = database.create(
        {
            "participant": "replay",
            "device": "corpus-jvs",
            "session": "replay",
            "recorded_on": "2026-10-11",
            "role": "enrollment",
            "consent": True,
        }
    )
    for row in recordings:
        source = r.ROOT / row["source_file"]
        r.s.checked(source, row["source_sha256"])
        episode = database.submit(
            episode["id"],
            source.read_bytes(),
            {"prompt_id": row["utterance_id"], "origin": "replay"},
        )
    if episode["status"] != "complete":
        raise AssertionError(episode)
    profile = read(database.path(episode["id"]) / "profile.json")
    if profile["threshold"] is not None:
        raise AssertionError("Unexpected threshold")
    norm_error = max(
        abs(float(r.np.linalg.norm(p["vector"])) - 1)
        for p in profile["phones"].values()
    )
    if norm_error > 1e-6:
        raise AssertionError(norm_error)
    report = {
        "purpose": "replay_smoke",
        "new_human_recordings": 0,
        "source": expected["source_file"],
        "source_sha256": expected["source_sha256"],
        "source_audio": info,
        "segments": len(fresh),
        "counts": result["counts"],
        "processing_seconds": elapsed,
        "feature_max_absolute_error": feature_error,
        "embedding_max_absolute_error": vector_error,
        "registration_smoke": {
            "clips": len(episode["clips"]),
            "status": episode["status"],
            "counts": episode["counts"],
            "profile_phones": len(profile["phones"]),
            "profile_max_norm_error": norm_error,
            "sources": [
                {k: row[k] for k in ("utterance_id", "source_file", "source_sha256")}
                for row in recordings
            ],
        },
        "prompt_counts": engine.prompt_counts,
        "files": engine.fingerprints(),
        "conclusion": "Original Julius boundaries and all36 feature/embedding replay verified; live microphone and accuracy remain unmeasured.",
    }
    write(root / "verification.json", report)
    print(
        __import__("json").dumps(
            {k: v for k, v in report.items() if k != "files"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
