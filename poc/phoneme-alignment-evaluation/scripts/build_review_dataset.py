#!/usr/bin/env python3
"""Build anonymized review data from normalized aligner outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MFA_INPUT = PROJECT_DIR / "data" / "alignments" / "mfa" / "normalized.jsonl"
DEFAULT_DATASET_OUTPUT = PROJECT_DIR / "data" / "reviews" / "review-dataset.json"
DEFAULT_MAPPING_OUTPUT = PROJECT_DIR / "data" / "reviews" / "candidate-map.json"
DATASET_ID = "jvs-vowel-alignment-review"
DATASET_VERSION = "5"
PROTOCOL_ID = "japanese-vowel-boundary-review"
PROTOCOL_VERSION = "5"

CANDIDATE_QUESTIONS = [
    {
        "id": "perceived_content",
        "prompt": "対象区間で聞こえる音に最も近いものを選んでください。",
        "choices": [
            {"value": "target_vowel", "label": "表示された母音が聞こえる"},
            {
                "value": "target_vowel_with_non_vowel",
                "label": "表示された母音と子音・破裂音が聞こえる",
            },
            {"value": "other_vowel", "label": "別の母音が聞こえる"},
            {"value": "non_vowel_only", "label": "子音・破裂音だけが聞こえる"},
            {"value": "near_silence", "label": "ほぼ音が聞こえない"},
            {"value": "uncertain", "label": "判断できない"},
        ],
    },
]

REVIEW_STATUS = {
    "prompt": "この区間を、表示された母音の学習データとして利用できますか？",
    "choices": [
        {"value": "accepted", "label": "利用できる"},
        {"value": "rejected", "label": "利用できない"},
        {"value": "uncertain", "label": "判断できない"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build anonymized review data from normalized alignments."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Normalized JSONL for one method. May be passed multiple times.",
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_candidate_specs(specs: list[str]) -> dict[str, Path]:
    if not specs:
        return {"mfa": DEFAULT_MFA_INPUT}

    candidates: dict[str, Path] = {}
    for spec in specs:
        method, separator, path_text = spec.partition("=")
        if not separator or not method.strip() or not path_text.strip():
            raise ValueError(f"Invalid candidate specification: {spec!r}")
        method = method.strip()
        if method in candidates:
            raise ValueError(f"Duplicate candidate method: {method}")
        candidates[method] = Path(path_text).expanduser().resolve()
    return candidates


def index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = str(record["vowel_interval_id"])
        if item_id in indexed:
            raise ValueError(f"Duplicate vowel_interval_id: {item_id}")
        indexed[item_id] = record
    return indexed


def target_tags(record: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if record.get("is_long"):
        tags.append("long")
    if record.get("is_devoiced"):
        tags.append("devoiced")
    elif record.get("contains_devoiced"):
        tags.append("contains_devoiced")
    return tags


def validate_matching_target(
    item_id: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    fields = (
        "utterance_id",
        "vowel_index",
        "normalized_phoneme",
        "expected_units",
    )
    mismatches = [
        field for field in fields if candidate.get(field) != reference.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"{item_id}: candidate target differs in {', '.join(mismatches)}"
        )


def build_review_artifacts(
    candidate_records: dict[str, list[dict[str, Any]]],
    *,
    dataset_id: str = DATASET_ID,
    dataset_version: str = DATASET_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(candidate_records) != 1:
        raise ValueError("Exactly one candidate method is required for protocol v5")

    indexed = {
        method: index_records(records) for method, records in candidate_records.items()
    }
    methods = sorted(indexed)
    item_ids = sorted(
        {item_id for records in indexed.values() for item_id in records},
        key=lambda item_id: _item_sort_key(item_id, indexed),
    )

    items: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for item_id in item_ids:
        available_records = [
            indexed[method][item_id] for method in methods if item_id in indexed[method]
        ]
        reference = available_records[0]
        for candidate in available_records[1:]:
            validate_matching_target(item_id, reference, candidate)

        candidates: list[dict[str, Any]] = []
        mapping_candidates: list[dict[str, Any]] = []
        for method in methods:
            candidate_id = "A"
            record = indexed[method].get(item_id)
            if record is None:
                candidates.append({"id": candidate_id, "status": "missing"})
                mapping_candidates.append(
                    {
                        "candidateId": candidate_id,
                        "method": method,
                        "status": "missing",
                    }
                )
                continue

            candidates.append(
                {
                    "id": candidate_id,
                    "status": "available",
                    "segment": {
                        "startSec": record["start_sec"],
                        "endSec": record["end_sec"],
                    },
                }
            )
            mapping_candidates.append(
                {
                    "candidateId": candidate_id,
                    "method": method,
                    "status": "available",
                    "aligner": record.get("aligner"),
                    "alignerVersion": record.get("aligner_version"),
                    "modelId": record.get("model_id"),
                    "modelVersion": record.get("model_version"),
                }
            )

        items.append(
            {
                "id": item_id,
                "utterance": {
                    "id": reference["utterance_id"],
                    "text": reference["transcript"],
                    "audioUrl": reference.get(
                        "audio_url", f"/media/{Path(reference['source_file']).name}"
                    ),
                },
                "target": {
                    "label": reference["normalized_phoneme"],
                    "index": reference["vowel_index"],
                    "unitCount": reference["expected_units"],
                    "tags": target_tags(reference),
                },
                "candidates": candidates,
            }
        )
        mappings.append({"itemId": item_id, "candidates": mapping_candidates})

    dataset = {
        "schemaVersion": 2,
        "datasetId": dataset_id,
        "datasetVersion": dataset_version,
        "title": "日本語母音区間レビュー",
        "protocol": {"id": PROTOCOL_ID, "version": PROTOCOL_VERSION},
        "playback": {"contextPaddingSec": 0.1},
        "form": {
            "candidateQuestions": CANDIDATE_QUESTIONS,
            "reviewStatus": REVIEW_STATUS,
        },
        "items": items,
    }
    mapping = {
        "schemaVersion": 2,
        "datasetId": dataset_id,
        "datasetVersion": dataset_version,
        "items": mappings,
    }
    return dataset, mapping


def _item_sort_key(
    item_id: str, indexed: dict[str, dict[str, dict[str, Any]]]
) -> tuple[str, int, str]:
    record = next(
        records[item_id] for records in indexed.values() if item_id in records
    )
    return str(record["utterance_id"]), int(record["vowel_index"]), item_id


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    candidate_paths = parse_candidate_specs(args.candidate)
    candidate_records = {
        method: read_jsonl(path) for method, path in candidate_paths.items()
    }
    dataset, mapping = build_review_artifacts(
        candidate_records,
        dataset_id=args.dataset_id,
        dataset_version=args.dataset_version,
    )
    write_json(args.output, dataset)
    write_json(args.mapping_output, mapping)
    print(
        f"Wrote {len(dataset['items'])} review items to {args.output} "
        f"and candidate mapping to {args.mapping_output}"
    )


if __name__ == "__main__":
    main()
