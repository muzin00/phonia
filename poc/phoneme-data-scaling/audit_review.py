"""Independently check the served sample against frozen train inputs and PCM."""

import array
import hashlib
import json
import math
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

from review import sample_segments
from selection import BASE, ROOT, balanced_take, fixed, rank, read, sha


def main():
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    source = ROOT / config["candidate_run"]
    frozen = read(run / "freeze.json")
    for name, digest in frozen["files"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(run / "freeze.json") == read(run / "asr/freeze.json")["selection_sha256"]
    expected_sources = []
    for group in config["groups"]:
        grouped = defaultdict(list)
        for row in read(source / (group + ".json")):
            grouped[row["speaker_id"]].append({**row, "review_group": group})
        expected_sources.extend(
            balanced_take(
                grouped, config["utterances_per_group"], config["utterance_seed"]
            )
        )
    assert read(run / "selected-sources.json") == expected_sources
    for name, digest in read(run / "runtime-inputs.json")["files"].items():
        assert sha(ROOT / name) == digest, name
    inputs = read(run / "selected-utterances.json")
    assert len(inputs) == 160
    assert len({v["source_id"] for v in inputs}) == 160
    assert len({v["sha256"] for v in inputs}) == 160
    assert len({Path(v["local_path"]).stem for v in inputs}) == 160
    for row, expected in zip(inputs, expected_sources, strict=True):
        assert all(row[k] == value for k, value in expected.items())
        assert sha(ROOT / row["local_path"]) == row["sha256"]
    transcripts = read(run / "asr.json")
    assert len(transcripts) == len(inputs)
    for transcript, row in zip(transcripts, inputs, strict=True):
        assert all(transcript[k] == value for k, value in row.items())
        if row["corpus"] == "common_voice":
            assert transcript["asr"]["text"] == row["sentence"]
    transcript_by_path = {v["local_path"]: v for v in transcripts}
    allowed = {r["local_path"]: r for r in inputs}
    pool = read(run / "post-qc.json")
    raw = read(run / "pre-qc.json")
    excluded = read(run / "excluded.json")
    assert len(raw) == len(pool) + len(excluded)
    assert {r["segment_id"] for r in raw} == {r["segment_id"] for r in pool + excluded}
    for row in pool:
        assert row["original_mp3"] in allowed
        assert row["original_mp3_sha256"] == allowed[row["original_mp3"]]["sha256"]
        assert row["source_split"] == "train" and not row["exclusion_reasons"]
        origin = allowed[row["original_mp3"]]
        assert row["review_group"] == origin["review_group"]
        assert row["speaker_id"] == origin["speaker_id"]
        assert row["corpus"] == origin["corpus"]
        assert (
            row["transcript"]
            == transcript_by_path[row["original_mp3"]]["asr"]["text"].strip()
        )
    selected = read(run / "selected-segments.json")
    expected_sample = []
    for group in config["groups"]:
        expected_sample.extend(
            sample_segments(
                [v for v in pool if v["review_group"] == group],
                config["segments_per_group"],
                config["segment_seed"],
                config["display_seed"],
            )
        )
    expected_sample.sort(key=lambda v: rank(config["display_seed"], v["segment_id"]))
    assert selected == expected_sample
    assert Counter(v["review_group"] for v in selected) == {
        g: 25 for g in config["groups"]
    }
    dataset = read(run / "review-dataset.json")
    audit = read(run / "review-audit.json")
    assert sha(run / "review-dataset.json") == audit["dataset_sha256"]
    assert len(dataset["items"]) == 100
    for item, row, expected in zip(
        dataset["items"], selected, audit["audit"], strict=True
    ):
        assert item["id"] == row["segment_id"] == expected["item_id"]
        assert item["utterance"]["text"] == row["transcript"]
        assert item["target"]["label"] == row["phoneme"]
        assert item["target"]["index"] == row["interval_index"] - 1
        assert item["target"]["unitCount"] == 1
        bounds = item["candidates"][0]["segment"]
        first, last = round(bounds["startSec"] * 24000), round(bounds["endSec"] * 24000)
        assert (first, last) == (row["start_frame"], row["end_frame"])
        media = run / item["utterance"]["audioUrl"].lstrip("/")
        assert sha(media) == sha(ROOT / row["source_file"]) == expected["media_sha256"]
        assert sha(ROOT / row["original_mp3"]) == row["original_mp3_sha256"]
        with wave.open(str(media)) as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (
                24000,
                2,
                1,
            )
            assert 0 <= first < last <= wav.getnframes()
            wav.setpos(first)
            pcm = wav.readframes(last - first)
        assert hashlib.sha256(pcm).hexdigest() == expected["crop_pcm_sha256"]
        samples = array.array("h", pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        rms = 20 * math.log10(
            math.sqrt(sum((s / 32768) ** 2 for s in samples) / len(samples))
        )
        assert abs(rms - row["rms_dbfs"]) < 1e-8
        assert rms >= -50 and 720 <= len(samples) <= 6000
    fixed(
        run / "independent-audit.json",
        {
            "auditor_sha256": sha(Path(__file__)),
            "dataset_sha256": sha(run / "review-dataset.json"),
            "verified_items": 100,
            "verified_source_utterances": 160,
            "train_only": True,
            "sample_reproduction_pcm_boundaries_hashes_rms": "passed",
            "cross_dataset_waveform_overlap": "not yet audited; required before training",
        },
    )
    print(
        json.dumps(
            {
                "verified_items": 100,
                "train_only": True,
                "pcm_boundaries_hashes_rms": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
