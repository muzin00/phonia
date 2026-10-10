#!/usr/bin/env python3
"""Create an SRC4VC-only repeat review of the same 50 previously sampled intervals."""

from __future__ import annotations

import json
import statistics
from collections import Counter

import sample_review as shared

SOURCE = shared.BASE / "data/review"
OUTPUT = shared.BASE / "data/src4vc-review"


def main() -> None:
    source_dataset = shared.read_json(SOURCE / "review-dataset.json")
    source_summary = shared.read_json(SOURCE / "sampling-summary.json")
    for filename, checksum_key in (
        ("review-dataset.json", "review_dataset_sha256"),
        ("sample.jsonl", "sample_sha256"),
    ):
        if shared.sha256(SOURCE / filename) != source_summary[checksum_key]:
            raise ValueError(f"Original sample changed: {filename}")
    records = [
        json.loads(line)
        for line in (SOURCE / "sample.jsonl").read_text().splitlines()
        if line.strip()
    ]
    selected = [row for row in records if row["corpus"] == "SRC4VC"]
    counts = Counter(row["phoneme"] for row in selected)
    if counts != {"m": 25, "n": 25} or any(row["split"] != "train" for row in selected):
        raise ValueError("Expected the original 50 train-only SRC4VC intervals")
    selected_ids = {row["item_id"] for row in selected}
    dataset = dict(source_dataset)
    dataset.update(
        {
            "datasetId": "src4vc-mn-repeat-training-review",
            "datasetVersion": f"seed{source_summary['seed']}-n50",
            "title": "SRC4VC /m/・/n/ 再確認レビュー（50区間）",
            "items": [
                item for item in source_dataset["items"] if item["id"] in selected_ids
            ],
        }
    )
    if [item["id"] for item in dataset["items"]] != [
        row["item_id"] for row in selected
    ]:
        raise ValueError("Dataset and sample must retain their original order")
    shared.write_json(OUTPUT / "review-dataset.json", dataset)
    (OUTPUT / "sample.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    durations = [row["duration_sec"] for row in selected]
    summary = {
        "schema_version": 1,
        "dataset_id": dataset["datasetId"],
        "dataset_version": dataset["datasetVersion"],
        "selection": "SRC4VC_subset_of_original_review_no_resampling_no_boundary_changes",
        "seed": source_summary["seed"],
        "source_dataset_id": source_dataset["datasetId"],
        "source_dataset_version": source_dataset["datasetVersion"],
        "source_review_dataset_sha256": shared.sha256(SOURCE / "review-dataset.json"),
        "source_sample_sha256": shared.sha256(SOURCE / "sample.jsonl"),
        "selected_count": len(selected),
        "selected_phoneme_counts": dict(counts),
        "selected_speaker_count": len({row["speaker_id"] for row in selected}),
        "selected_source_file_count": len({row["source_file"] for row in selected}),
        "duration_sec": {
            "minimum": min(durations),
            "median": statistics.median(durations),
            "maximum": max(durations),
        },
        "audio_volume_filter": False,
        "media_root": source_summary["media_root"],
        "reviewed": False,
        "review_dataset_sha256": shared.sha256(OUTPUT / "review-dataset.json"),
        "sample_sha256": shared.sha256(OUTPUT / "sample.jsonl"),
        "human_answers_used_for_training": False,
    }
    shared.write_json(OUTPUT / "sampling-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
