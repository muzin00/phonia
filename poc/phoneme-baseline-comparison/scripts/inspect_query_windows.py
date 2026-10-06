"""Inspect v2 query recording caps on frozen validation metadata only."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_comparison_validation import distribution, inspect
from smoke import BASE, ROOT, sha256_file, write_json
from utterance_windows import describe_query_window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = BASE / "config/comparison-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["protocol_version"] != "2.0.0" or protocol["query_windows"][
        "scope"
    ] != ("query_only_enrollment_fixed"):
        raise ValueError("requires v2 query-only recording caps")
    readiness = inspect(protocol)
    references = protocol["metadata_readiness_inputs"]
    with (ROOT / references["utterance_manifest"]["path"]).open() as stream:
        sources = {
            row["source_file"]: row
            for row in map(json.loads, stream)
            if row["split"] == "validation"
        }
    with (ROOT / references["queries"]["path"]).open() as stream:
        queries = list(map(json.loads, stream))
    if any(query["split"] != "validation" for query in queries):
        raise ValueError("only validation metadata is allowed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary, ids, manifests = {}, {}, []
    for condition in protocol["query_windows"]["conditions"]:
        rows = [
            describe_query_window(
                query,
                sources[query["source_file"]],
                condition["maximum_seconds"],
                margin=protocol["context"]["margin_frames_per_side"],
            )
            for query in queries
        ]
        filename = f"query-windows-{condition['id']}.jsonl"
        with (args.output_dir / filename).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifests.append(filename)
        ids[condition["id"]] = {row["query_id"] for row in rows}
        role_summary = {}
        for role in protocol["queries_and_trials"]["roles"]:
            selected = [row for row in rows if row["role"] == role]
            long_cohort = [
                row
                for row in selected
                if sources[row["source_file"]]["frame_count"] >= 120000
            ]
            role_summary[role] = {
                "queries": len(selected),
                "complete_five_vowels": sum(
                    row["complete_five_vowels"] for row in selected
                ),
                "shorter_than_cap": sum(
                    row["source_shorter_than_cap"] for row in selected
                ),
                "actual_recording_seconds": distribution(
                    [row["actual_recording_seconds"] for row in selected]
                ),
                "retained_anchors": sum(len(row["anchors"]) for row in selected),
                "complete_query_count_by_speaker": {
                    speaker: sum(
                        row["complete_five_vowels"]
                        for row in selected
                        if row["speaker_id"] == speaker
                    )
                    for speaker in sorted({row["speaker_id"] for row in selected})
                },
                "source_at_least_5s_cohort": {
                    "queries": len(long_cohort),
                    "complete_five_vowels": sum(
                        row["complete_five_vowels"] for row in long_cohort
                    ),
                },
            }
            for method in ("vowel_exact", "vowel_context20"):
                role_summary[role][f"{method}_unique_seconds"] = distribution(
                    [
                        row["used_audio"][method]["unique_frames"] / 24000
                        for row in selected
                    ]
                )
        summary[condition["id"]] = role_summary
    if any(value != ids["full"] for value in ids.values()):
        raise ValueError("query population changed across recording caps")
    report = {
        "schema_version": 1,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(protocol_path),
        "purpose": "validation_query_window_metadata_no_audio_models_or_scores",
        "checked_inputs": readiness["checked_inputs"],
        "query_population_same_across_conditions": True,
        "enrollment_unchanged": True,
        "summary": summary,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                BASE / "utterance_windows.py",
                BASE / "comparison_inputs.py",
                BASE / "scripts/inspect_comparison_validation.py",
                BASE / "smoke.py",
            )
        },
        "outputs_sha256": {
            name: sha256_file(args.output_dir / name) for name in manifests
        },
    }
    write_json(args.output_dir / "query-window-readiness.json", report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
