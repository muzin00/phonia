"""Independently check the served sample against frozen train inputs and PCM."""

import array
import hashlib
import json
import math
import sys
import wave

from acquire import BASE, ROOT, read, sha
from review import sample_segments, select_utterances


def main():
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    source = ROOT / config["source_run"]
    frozen = read(run / "freeze.json")
    for name, digest in frozen["files"].items():
        assert sha(ROOT / name) == digest, name
    policy = read(run / "alignment-resource-policy.json")
    assert sha(BASE / "review_extract.py") == policy["source_sha256"]
    assert policy["maximum_utterance_seconds"] == 30
    assert policy["julius_timeout_seconds"] == 10
    original = run / "selection-asr-freeze.json"
    if original.exists():
        assert sha(original) == read(run / "asr/freeze.json")["selection_sha256"]
        old = read(original)
        assert old["files"][str((BASE / "review.py").relative_to(ROOT))] == sha(
            run / "review-preparation-original.py"
        )
        assert old["files"][str((BASE / "review_asr.py").relative_to(ROOT))] == sha(
            run / "review-asr-original.py"
        )
    inputs = read(run / "selected-utterances.json")
    assert inputs == select_utterances(
        read(source / "download.json"), config["utterance_seed"]
    )
    assert len(inputs) == len({r["speaker_id"] for r in inputs}) == 140
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
    selected = read(run / "selected-segments.json")
    assert selected == sample_segments(
        pool, 100, config["segment_seed"], config["display_seed"]
    )
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
