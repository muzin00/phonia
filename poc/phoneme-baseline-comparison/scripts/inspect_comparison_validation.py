"""Inspect frozen validation metadata; never read WAVs, scores, or test inputs."""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison_inputs import describe_audio
from smoke import BASE, ROOT, sha256_file, write_json


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def inspect(protocol: dict) -> dict:
    references = protocol["metadata_readiness_inputs"]
    for reference in references.values():
        if sha256_file(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"reference checksum mismatch: {reference['path']}")
    with (ROOT / references["utterance_manifest"]["path"]).open() as stream:
        originals = [json.loads(line) for line in stream]
    sources = {
        row["source_file"]: row for row in originals if row["split"] == "validation"
    }
    split = json.loads((ROOT / references["speaker_split"]["path"]).read_text())
    speakers = sorted(split["speaker_splits"]["validation"])
    if len(speakers) != 15 or any(
        row["speaker_id"] not in speakers for row in sources.values()
    ):
        raise ValueError("invalid validation speaker assignment")
    records = {}
    for category in ("enrollment", "queries"):
        reference = references[category]
        with (ROOT / reference["path"]).open() as stream:
            records[category] = [json.loads(line) for line in stream]
        for row in records[category]:
            speaker = row["user_id"] if category == "enrollment" else row["speaker_id"]
            if row["split"] != "validation" or speaker not in speakers:
                raise ValueError("readiness accepts validation only")
            for segment in row["segments"]:
                source = sources[segment["source_file"]]
                if source["speaker_id"] != speaker or source["evaluation_role"] != (
                    "enrollment" if category == "enrollment" else row["role"]
                ):
                    raise ValueError("anchor speaker/role mismatch")
    enrollments = records["enrollment"]
    expected = {(speaker, count) for speaker in speakers for count in (1, 5, 10)}
    if {
        (row["user_id"], row["enrollment_count"]) for row in enrollments
    } != expected or len(enrollments) != len(expected):
        raise ValueError("missing or duplicate enrollment condition")
    enrollment_ids = {
        (row["user_id"], row["enrollment_count"]): {
            segment["segment_id"] for segment in row["segments"]
        }
        for row in enrollments
    }
    if any(
        not enrollment_ids[speaker, 1]
        <= enrollment_ids[speaker, 5]
        <= enrollment_ids[speaker, 10]
        for speaker in speakers
    ):
        raise ValueError("enrollment anchors are not nested")
    query_ids = [row["query_id"] for row in records["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate validation query")
    if Counter(
        (row["speaker_id"], row["role"]) for row in records["queries"]
    ) != Counter(
        {
            (speaker, role): count
            for speaker in speakers
            for role, count in (("verification", 50), ("cross_text_verification", 30))
        }
    ):
        raise ValueError("unexpected validation query counts")
    summaries = {}
    for category, rows in records.items():
        groups = {}
        for row in rows:
            if category == "enrollment":
                key = str(row["enrollment_count"])
                counts = Counter(segment["vowel"] for segment in row["segments"])
                if counts != Counter(
                    {vowel: row["enrollment_count"] for vowel in "aiueo"}
                ):
                    raise ValueError("incomplete enrollment vowels")
            else:
                key = row["role"]
                counts = Counter(segment["vowel"] for segment in row["segments"])
                if row["counts"] != {vowel: counts[vowel] for vowel in "aiueo"}:
                    raise ValueError("query vowel counts mismatch")
                source = sources[row["source_file"]]
                if (
                    source["speaker_id"] != row["speaker_id"]
                    or source["evaluation_role"] != row["role"]
                    or source["source_sha256"] != row["source_sha256"]
                    or any(
                        segment["source_file"] != row["source_file"]
                        for segment in row["segments"]
                    )
                ):
                    raise ValueError("query source identity mismatch")
            audio = describe_audio(
                row["segments"],
                sources,
                margin=protocol["context"]["margin_frames_per_side"],
            )
            if category == "queries":
                audio["whole_source_frames"] = sources[row["source_file"]][
                    "frame_count"
                ]
            audio["complete_five_vowels"] = all(counts[vowel] > 0 for vowel in "aiueo")
            groups.setdefault(key, []).append(audio)
        summaries[category] = {}
        for key, audio_rows in groups.items():
            item = {
                "conditions": len(audio_rows),
                "source_count": distribution(
                    [row["source_count"] for row in audio_rows]
                ),
                "whole_source_seconds": distribution(
                    [row["whole_source_frames"] / 24000 for row in audio_rows]
                ),
                "complete_five_vowels": sum(
                    row["complete_five_vowels"] for row in audio_rows
                ),
                "zero_budget_queries": sum(
                    row["unique_frames"]["vowel_exact"] == 0 for row in audio_rows
                ),
                "anchor_count": sum(row["anchor_count"] for row in audio_rows),
                "context_changed_anchors": sum(
                    row["context_changed_anchor_count"] for row in audio_rows
                ),
                "context_cropped_anchors": sum(
                    row["context_cropped_anchor_count"] for row in audio_rows
                ),
                "exact_cropped_anchors": sum(
                    row["exact_cropped_anchor_count"] for row in audio_rows
                ),
                "retained_context_seconds_including_repeated_frames": sum(
                    row["retained_context_frames_including_repeated_frames"]
                    for row in audio_rows
                )
                / 24000,
            }
            for method in ("vowel_exact", "vowel_context20"):
                item[f"{method}_unique_seconds"] = distribution(
                    [row["unique_frames"][method] / 24000 for row in audio_rows]
                )
                item[f"{method}_processed_seconds"] = distribution(
                    [row["processed_frames"][method] / 24000 for row in audio_rows]
                )
                item[f"{method}_per_source_control_seconds"] = distribution(
                    [
                        source["budgets"][method] / 24000
                        for row in audio_rows
                        for source in row["sources"].values()
                    ]
                )
            summaries[category][key] = item
    return {
        "schema_version": 1,
        "purpose": "validation_metadata_only_no_audio_models_or_scores",
        "protocol_version": protocol["protocol_version"],
        "inspected_split": "validation",
        "checked_inputs": references,
        "summary": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = BASE / "config/comparison-protocol.json"
    report = inspect(json.loads(protocol_path.read_text()))
    report["protocol_sha256"] = sha256_file(protocol_path)
    report["implementation_sha256"] = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (
            BASE / "comparison_inputs.py",
            Path(__file__).resolve(),
            BASE / "smoke.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
