"""Materialize all frozen additions; preserve review audio and original sources."""

import hashlib
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from expansion_common import (
    BASE,
    CONFIG,
    ROOT,
    check_files,
    fixed,
    pin,
    progress,
    read,
    sha,
)


def main():
    from pyarrow import parquet

    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    candidate = ROOT / config["candidate_run"]
    review = ROOT / config["review_run"]
    original_config = read(BASE / "config.json")
    candidate_freeze = read(candidate / "selection-freeze.json")
    check_files(candidate_freeze["files"])
    check_files(read(review / "freeze.json")["files"])
    selected = []
    for group in config["groups"]:
        for row in read(candidate / (group + ".json")):
            selected.append(
                {
                    **row,
                    "addition_group": group,
                    "input_id": hashlib.sha256(row["source_id"].encode()).hexdigest(),
                    "position": len(selected),
                }
            )
    assert len(selected) == 11200 and len({v["source_id"] for v in selected}) == 11200
    fixed(run / "selected-sources.json", selected)
    fixed(
        run / "acquisition-freeze.json",
        {
            "config": config,
            "files": pin(
                [
                    CONFIG,
                    BASE / "acquire_expansion.py",
                    BASE / "expansion_common.py",
                    candidate / "selection-freeze.json",
                    candidate / "selection-audit.json",
                    review / "selected-utterances.json",
                    ROOT / "poc/phoneme-corpus-scaling/acquire.py",
                    run / "selected-sources.json",
                ]
            ),
        },
    )
    results = {}
    prior = {v["source_id"]: v for v in read(review / "selected-utterances.json")}
    counts, started = Counter(), time.monotonic()
    shards = defaultdict(list)
    jtube = []
    for row in selected:
        if row["source_id"] in prior:
            old = prior[row["source_id"]]
            assert (
                old["speaker_id"] == row["speaker_id"]
                and sha(ROOT / old["local_path"]) == old["sha256"]
            )
            results[row["source_id"]] = {
                **row,
                "path": old["path"],
                "local_path": old["local_path"],
                "sha256": old["sha256"],
            }
            counts["reused_review"] += 1
        elif row["corpus"] == "common_voice":
            shards[row["shard"]].append(row)
        else:
            jtube.append(row)
    source = ROOT / original_config["cv_run"] / "source"
    for shard, rows in sorted(shards.items()):
        table = parquet.read_table(
            source / shard, columns=["client_id", "sentence", "audio"]
        )
        for row in rows:
            data = table.slice(row["shard_row"], 1).to_pylist()[0]
            assert (
                data["client_id"] == row["client_id"]
                and data["sentence"] == row["sentence"]
                and data["audio"]["path"] == row["audio_path"]
            )
            path = run / "audio" / row["speaker_id"] / row["audio_path"].split("/")[-1]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = data["audio"]["bytes"]
            if path.exists():
                assert path.read_bytes() == payload
            else:
                path.write_bytes(payload)
            results[row["source_id"]] = {
                **row,
                "path": row["audio_path"],
                "local_path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
            }
            counts["common_voice"] += 1
        progress(run, "download", len(results), len(selected), counts, started)
    sys.path.insert(0, str(ROOT / "poc/phoneme-corpus-scaling"))
    import acquire

    frozen = read(ROOT / original_config["jtube_run"] / "selection-freeze.json")
    fetch_config = {**frozen["config"], "run_directory": config["run_directory"]}

    def fetch(row):
        for attempt in range(3):
            try:
                return acquire.fetch_member(fetch_config, frozen, row)
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=config["download_workers"]) as executor:
        futures = {executor.submit(fetch, row): row for row in jtube}
        for future in as_completed(futures):
            row = future.result()
            results[row["source_id"]] = row
            counts["jtube"] += 1
            progress(run, "download", len(results), len(selected), counts, started)
    fixed(run / "inputs.json", [results[v["source_id"]] for v in selected])
    fixed(
        run / "acquisition-complete.json",
        {
            "source_count": len(selected),
            "counts": dict(counts),
            "inputs_sha256": sha(run / "inputs.json"),
        },
    )


if __name__ == "__main__":
    main()
