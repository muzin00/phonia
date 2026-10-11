"""Select JTubeSpeech-ASV Japanese speakers before decoding any new audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rank(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def select(config):
    source = ROOT / config["source_directory"]
    output = ROOT / config["run_directory"]
    if output.exists():
        raise ValueError("Use a fresh run")
    archive = read(source / "jtube-archive-index.json")
    assert archive["url"] == config["source_url"]
    entries = {v["path"]: v for v in archive["files"]}
    official_test = {
        v["path"].split("/")[3]
        for v in archive["files"]
        if v["path"].startswith("JTubeSpeech-ASV/test/") and v["path"].endswith(".mp3")
    }
    grouped = defaultdict(list)
    for line in (source / "Japanese.lst").read_text().splitlines():
        speaker, relative = line.split()
        assert relative.split("/")[0] == speaker
        path = "JTubeSpeech-ASV/train/" + relative
        assert path in entries
        grouped[speaker].append(path)
    assert not set(grouped) & official_test
    assert all(len(paths) == len(set(paths)) for paths in grouped.values())
    eligible = sorted(
        (
            s
            for s, paths in grouped.items()
            if len(paths) >= config["utterances_per_speaker"]
        ),
        key=lambda s: rank(config["speaker_seed"], s),
    )
    counts = {
        "train": config["training_speakers"],
        "validation": config["validation_speakers"],
        "test": config["test_speakers"],
    }
    assert len(eligible) >= sum(counts.values())
    partitions, rows, offset = {}, [], 0
    for split, count in counts.items():
        partitions[split] = eligible[offset : offset + count]
        offset += count
        for speaker in partitions[split]:
            paths = sorted(
                grouped[speaker], key=lambda p: rank(config["utterance_seed"], p)
            )[: config["utterances_per_speaker"]]
            for index, path in enumerate(paths):
                rows.append(
                    {
                        **entries[path],
                        "speaker_id": "jtube_" + speaker,
                        "source_speaker_id": speaker,
                        "split": split,
                        "position": index,
                        "pilot": split == "train"
                        and speaker in partitions["train"][: config["pilot_speakers"]]
                        and index < config["pilot_utterances_per_speaker"],
                    }
                )
    files = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            source / "jtube-archive-index.json",
            source / "Japanese.lst",
            source / "README.md",
            BASE / "config.json",
            BASE / "acquire.py",
        ]
    }
    write(
        output / "selection-freeze.json",
        {
            "config": config,
            "files": files,
            "archive_etag": archive["etag"],
            "archive_size": archive["size"],
            "japanese_speaker_labels": len(grouped),
            "eligible_speaker_labels": len(eligible),
            "official_test_label_overlap": 0,
            "partitions": partitions,
            "clips": rows,
            "identity_overlap_with_JVS_CV": "unknown",
        },
    )
    print(
        {
            "eligible": len(eligible),
            "partitions": {k: len(v) for k, v in partitions.items()},
            "clips": len(rows),
            "pilot": sum(v["pilot"] for v in rows),
        }
    )


def fetch_member(config, frozen, row):
    output = (
        ROOT
        / config["run_directory"]
        / "audio"
        / row["split"]
        / row["speaker_id"]
        / Path(row["path"]).name
    )
    sidecar = output.with_suffix(".json")
    if output.exists() and sidecar.exists():
        assert sha(output) == read(sidecar)["sha256"]
        return read(sidecar)
    start = row["offset"]
    end = min(
        frozen["archive_size"] - 1, start + row["compressed_size"] + 65535 * 2 + 30 - 1
    )
    request = urllib.request.Request(
        config["source_url"],
        headers={"Range": f"bytes={start}-{end}", "If-Match": frozen["archive_etag"]},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith(f"bytes {start}-")
        payload = response.read(end - start + 2)
    assert len(payload) == end - start + 1
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack(
        "<4s5H3I2H", payload[:30]
    )
    assert signature == b"PK\x03\x04" and not flags & 1
    assert payload[30 : 30 + name_size].decode("utf-8") == row["path"]
    body = payload[
        30 + name_size + extra_size : 30
        + name_size
        + extra_size
        + row["compressed_size"]
    ]
    assert method in (0, 8)
    data = zlib.decompress(body, -15) if method == 8 else body
    assert len(data) == row["size"] and zlib.crc32(data) == row["crc"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    temporary.write_bytes(data)
    temporary.replace(output)
    result = {**row, "local_path": str(output.relative_to(ROOT)), "sha256": sha(output)}
    write(sidecar, result)
    return result


def download(config, pilot):
    output = ROOT / config["run_directory"]
    frozen = read(output / "selection-freeze.json")
    assert frozen["config"] == config
    for name, digest in frozen["files"].items():
        assert sha(ROOT / name) == digest, name
    selected = [v for v in frozen["clips"] if v["pilot"]] if pilot else frozen["clips"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda row: fetch_member(config, frozen, row), selected)
        )
    write(output / ("pilot-download.json" if pilot else "download.json"), results)
    print(f"Downloaded and CRC/SHA checked {len(results)} clips", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("select", "pilot", "download"))
    args = parser.parse_args()
    config = read(BASE / "config.json")
    if args.phase == "select":
        select(config)
    else:
        download(config, args.phase == "pilot")


if __name__ == "__main__":
    main()
