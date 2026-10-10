"""Independently verify selected IDs, source PCM, UI ranges and HTTP audio."""

import argparse
import hashlib
import io
import json
import math
import wave
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(map(json.loads, stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(seed, identifier):
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-http", action="store_true")
    args = parser.parse_args()
    config = load(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    freeze = load(run / "review-freeze.json")
    for name, checksum in freeze["files"].items():
        if sha(ROOT / name) != checksum:
            raise ValueError(f"frozen input/output changed: {name}")
    splits = load(ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json")[
        "speaker_splits"
    ]
    cv = ROOT / config["common_voice_run"]
    labels = {
        "JVS": set(splits["train"]),
        "Common Voice": {r["speaker_id"] for r in load(cv / "selection.json")},
    }
    report = {
        "status": "passed",
        "frozen_checksum_checks": len(freeze["files"]),
        "corpora": {},
        "human_answers_written_by_audit": False,
    }
    for corpus, name, port in (
        ("JVS", "jvs", 5176),
        ("Common Voice", "common-voice", 5177),
    ):
        pool = rows(run / name / "eligible.jsonl")
        rejected = rows(run / name / "excluded.jsonl")
        raw = rows(run / name / "raw.jsonl")
        raw_ids = {r["item_id"] for r in raw}
        eligible_ids = {r["item_id"] for r in pool}
        rejected_ids = {r["item_id"] for r in rejected}
        assert (
            raw_ids == eligible_ids | rejected_ids and not eligible_ids & rejected_ids
        )
        assert len(raw_ids) == len(raw) and len(eligible_ids) == len(pool)
        assert all(
            r["quality_flags"]
            and (r["frame_count"] < 720 or r["rms_dbfs"] is None or r["rms_dbfs"] < -50)
            for r in rejected
        )
        assert all(
            not r["quality_flags"]
            and r["rms_dbfs"] >= -50
            and 720 <= r["frame_count"] <= 6000
            and r["speaker_id"] in labels[corpus]
            and r["split"] == "train"
            and r["evaluation_role"] == "training"
            for r in pool
        )
        coverage = Counter(r["speaker_id"] for r in pool)
        assert set(coverage) == labels[corpus] and min(coverage.values()) >= 2
        assert not labels[corpus] & (set(splits["validation"]) | set(splits["test"]))
        expected = sorted(
            pool, key=lambda r: (priority(config["seed"], r["item_id"]), r["item_id"])
        )[:100]
        expected.sort(key=lambda r: priority(config["seed"] + 1, r["item_id"]))
        output = BASE / "data" / f"{name}-review"
        sample = rows(output / "sample.jsonl")
        dataset = load(output / "review-dataset.json")
        assert [r["item_id"] for r in sample] == [r["item_id"] for r in expected]
        assert [i["id"] for i in dataset["items"]] == [r["item_id"] for r in sample]
        assert len(sample) == 100 and len({r["item_id"] for r in sample}) == 100
        http_audio = 0
        if args.check_http:
            address = f"http://127.0.0.1:{port}"
            with urlopen(address + "/review-dataset.json", timeout=10) as response:
                assert json.load(response) == dataset
            query = urlencode(
                {
                    "datasetId": dataset["datasetId"],
                    "datasetVersion": dataset["datasetVersion"],
                }
            )
            with urlopen(address + "/api/reviews?" + query, timeout=10) as response:
                assert isinstance(json.load(response)["records"], list)
        for row, item in zip(sample, dataset["items"]):
            original = load(ROOT / row["raw_alignment_file"])
            interval = original["intervals"][row["interval_index"]]
            assert interval["phoneme"] == "s" and item["target"]["label"] == "s"
            first, last = (
                round(interval["start_sec"] * 24000),
                round(interval["end_sec"] * 24000),
            )
            size = min(6000, last - first)
            begin = first + (last - first - size) // 2
            assert (begin, begin + size) == (row["start_frame"], row["end_frame"])
            assert not row["quality_flags"] and 720 <= size <= 6000
            candidate = item["candidates"][0]["segment"]
            assert (candidate["startSec"], candidate["endSec"]) == (
                begin / 24000,
                (begin + size) / 24000,
            )
            with wave.open(str(ROOT / row["source_file"]), "rb") as source:
                source.setpos(begin)
                pcm = source.readframes(size)
            with wave.open(str(ROOT / row["segment_file"]), "rb") as clip:
                assert (
                    clip.getframerate(),
                    clip.getnchannels(),
                    clip.getsampwidth(),
                    clip.getnframes(),
                ) == (24000, 1, 2, size)
                assert clip.readframes(size) == pcm
            rms = float(
                np.sqrt(
                    np.mean(
                        (np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768)
                        ** 2
                    )
                )
            )
            assert rms >= 10 ** (-50 / 20)
            assert abs(20 * math.log10(rms) - row["rms_dbfs"]) < 1e-10
            copied = run / name / "media/utterances" / f"{row['utterance_id']}.wav"
            assert sha(copied) == row["source_sha256"] == sha(ROOT / row["source_file"])
            if args.check_http:
                with urlopen(address + row["audio_url"], timeout=10) as response:
                    content = response.read()
                    assert response.headers["Content-Type"] == "audio/wav"
                assert hashlib.sha256(content).hexdigest() == row["source_sha256"]
                with wave.open(io.BytesIO(content), "rb") as audio:
                    assert audio.getframerate() == 24000
                request = Request(
                    address + row["audio_url"], headers={"Range": "bytes=0-63"}
                )
                with urlopen(request, timeout=10) as response:
                    assert response.status == 206 and response.read() == content[:64]
                    assert (
                        response.headers["Content-Range"]
                        == f"bytes 0-63/{len(content)}"
                    )
                http_audio += 1
        report["corpora"][corpus] = {
            "raw": len(raw),
            "eligible": len(pool),
            "excluded": len(rejected),
            "deterministic_selected_ids_verified": len(sample),
            "source_pcm_and_ui_ranges_verified": len(sample),
            "http_audio_sha256_checks": http_audio,
        }
    report["http_checked"] = args.check_http
    (run / "independent-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
