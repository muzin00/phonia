#!/usr/bin/env python3
"""Summarize the latest completed Phase 2 vowel review records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = PROJECT_DIR / "data" / "reviews" / "stratified"
DEFAULT_SAMPLE = DEFAULT_REVIEW_DIR / "sample.jsonl"
DEFAULT_RECORDS = DEFAULT_REVIEW_DIR / "review-records.jsonl"
DEFAULT_OUTPUT = DEFAULT_REVIEW_DIR / "review-summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the latest completed stratified review records."
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def latest_completed_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        item_id = str(record["itemId"])
        previous = latest.get(item_id)
        if previous is None or int(record["revision"]) > int(previous["revision"]):
            latest[item_id] = record
    return latest


def perceived_content(record: dict[str, Any]) -> str:
    answers = record["candidateAnswers"]
    if len(answers) != 1:
        raise ValueError(f"{record['itemId']}: expected exactly one candidate answer")
    return str(answers[0]["answers"]["perceived_content"])


def group_summary(
    reviewed: list[tuple[dict[str, Any], dict[str, Any]]],
    value_of: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample, review in reviewed:
        grouped.setdefault(value_of(sample), []).append(review)

    result: dict[str, dict[str, Any]] = {}
    for value, reviews in sorted(grouped.items()):
        statuses = Counter(str(review["reviewStatus"]) for review in reviews)
        contents = Counter(perceived_content(review) for review in reviews)
        accepted = statuses["accepted"]
        result[value] = {
            "reviewed": len(reviews),
            "accepted": accepted,
            "rejected": statuses["rejected"],
            "acceptance_rate": round(accepted / len(reviews), 6),
            "perceived_content": dict(sorted(contents.items())),
        }
    return result


def build_summary(
    samples: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    sample_by_id = {
        str(sample["vowel_interval_id"]): sample for sample in samples
    }
    latest = latest_completed_records(records)
    missing = sorted(set(latest) - set(sample_by_id))
    if missing:
        raise ValueError(f"Review records reference unknown items: {missing[:3]}")

    reviewed = [(sample_by_id[item_id], review) for item_id, review in latest.items()]
    reviewed_ids = set(latest)
    pending = [
        sample for item_id, sample in sample_by_id.items() if item_id not in reviewed_ids
    ]
    statuses = Counter(str(review["reviewStatus"]) for _, review in reviewed)
    contents = Counter(perceived_content(review) for _, review in reviewed)
    reviewed_strata = {sample["sampling"]["stratum"] for sample, _ in reviewed}
    all_strata = {sample["sampling"]["stratum"] for sample in sample_by_id.values()}

    return {
        "schema_version": 1,
        "dataset_id": "jvs-phase-2-stratified-vowel-review",
        "dataset_version": "1",
        "coverage": {
            "sampled": len(sample_by_id),
            "reviewed": len(reviewed),
            "pending": len(pending),
            "reviewed_speakers": len(
                {sample["speaker_id"] for sample, _ in reviewed}
            ),
            "sampled_speakers": len(
                {sample["speaker_id"] for sample in sample_by_id.values()}
            ),
            "reviewed_strata": len(reviewed_strata),
            "sampled_strata": len(all_strata),
            "unreviewed_strata": sorted(all_strata - reviewed_strata),
        },
        "overall": {
            "accepted": statuses["accepted"],
            "rejected": statuses["rejected"],
            "acceptance_rate": round(statuses["accepted"] / len(reviewed), 6),
            "perceived_content": dict(sorted(contents.items())),
        },
        "by_quality_group": group_summary(
            reviewed, lambda sample: str(sample["sampling"]["quality_group"])
        ),
        "by_duration_bin": group_summary(
            reviewed, lambda sample: str(sample["sampling"]["duration_bin"])
        ),
        "by_vowel": group_summary(
            reviewed, lambda sample: str(sample["normalized_phoneme"])
        ),
        "by_split": group_summary(reviewed, lambda sample: str(sample["split"])),
        "by_phone_mapping": group_summary(
            reviewed,
            lambda sample: str(sample["sampling"]["phone_mapping"] or "none"),
        ),
    }


def main() -> None:
    args = parse_args()
    summary = build_summary(read_jsonl(args.sample), read_jsonl(args.records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Summarized {summary['coverage']['reviewed']} completed reviews "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
