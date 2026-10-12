"""Freeze a small review frame, then materialize only its original audio."""

import argparse
import sys
from collections import defaultdict

from selection import BASE, ROOT, balanced_take, check_files, fixed, pin, read, sha


def select(config, run):
    source = ROOT / config["candidate_run"]
    check_files(read(source / "selection-freeze.json")["files"])
    rows = []
    for group in config["groups"]:
        speakers = defaultdict(list)
        for row in read(source / (group + ".json")):
            assert row["split"] == "train"
            speakers[row["speaker_id"]].append({**row, "review_group": group})
        chosen = balanced_take(
            speakers, config["utterances_per_group"], config["utterance_seed"]
        )
        assert len({v["speaker_id"] for v in chosen}) == config["utterances_per_group"]
        rows.extend(chosen)
    fixed(run / "selected-sources.json", rows)
    paths = [
        BASE / n
        for n in [
            "review-config.json",
            "prepare_review.py",
            "review.py",
            "review_asr.py",
            "selection.py",
        ]
    ]
    paths += [
        source / "selection-freeze.json",
        source / "selection-audit.json",
        run / "selected-sources.json",
        ROOT / config["runtime_freeze"],
        ROOT / "poc/phoneme-corpus-scaling/acquire.py",
    ]
    paths += [source / (g + ".json") for g in config["groups"]]
    fixed(run / "freeze.json", {"config": config, "files": pin(paths)})
    print(f"Fixed {len(rows)} review utterances; no training", flush=True)


def download(config, run):
    from pyarrow import parquet

    sys.path.insert(0, str(ROOT / "poc/phoneme-corpus-scaling"))
    import acquire

    check_files(read(run / "freeze.json")["files"])
    selection_config = read(BASE / "config.json")
    source = ROOT / selection_config["cv_run"] / "source"
    rows = read(run / "selected-sources.json")
    by_shard = defaultdict(list)
    results = {}
    for row in rows:
        if row["corpus"] == "common_voice":
            by_shard[row["shard"]].append(row)
    for shard, chosen in sorted(by_shard.items()):
        table = parquet.read_table(
            source / shard, columns=["client_id", "sentence", "audio"]
        )
        for row in chosen:
            record = table.slice(row["shard_row"], 1).to_pylist()[0]
            assert (
                record["client_id"] == row["client_id"]
                and record["sentence"] == row["sentence"]
            )
            assert record["audio"]["path"] == row["audio_path"]
            audio = (
                run / "audio" / row["speaker_id"] / (row["audio_path"].split("/")[-1])
            )
            audio.parent.mkdir(parents=True, exist_ok=True)
            payload = record["audio"]["bytes"]
            if audio.exists():
                assert audio.read_bytes() == payload
            else:
                audio.write_bytes(payload)
            results[row["source_id"]] = {
                **row,
                "path": row["audio_path"],
                "local_path": str(audio.relative_to(ROOT)),
                "sha256": sha(audio),
            }
        print(f"CV audio {len(results)}/80", flush=True)
    frozen = read(ROOT / selection_config["jtube_run"] / "selection-freeze.json")
    fetch_config = {**frozen["config"], "run_directory": config["run_directory"]}
    for row in rows:
        if row["corpus"] == "jtube":
            results[row["source_id"]] = acquire.fetch_member(fetch_config, frozen, row)
            print(f"All audio {len(results)}/{len(rows)}", flush=True)
    fixed(run / "selected-utterances.json", [results[row["source_id"]] for row in rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "download"])
    args = parser.parse_args()
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    {"select": select, "download": download}[args.phase](config, run)


if __name__ == "__main__":
    main()
