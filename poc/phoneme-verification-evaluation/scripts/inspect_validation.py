"""Audit validation metadata only; never load audio, models or test scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
ROLES = ("enrollment", "verification", "cross_text_verification")
VOWELS = ("a", "i", "u", "e", "o")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    references = [*protocol["inputs"].values(), *protocol["reused_files"]]
    for reference in references:
        if sha256_file(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"protocol checksum mismatch: {reference['path']}")
    split = json.loads((ROOT / protocol["inputs"]["speaker_split"]["path"]).read_text())
    speakers = set(split["speaker_splits"]["validation"])
    if len(speakers) != protocol["splits"]["expected_speakers_per_split"]:
        raise ValueError("unexpected validation speaker count")
    original = {}
    sessions = Counter()
    with (ROOT / protocol["inputs"]["utterance_manifest"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "validation":
                continue
            if row["speaker_id"] not in speakers or row["evaluation_role"] not in ROLES:
                raise ValueError("invalid validation utterance assignment")
            if row["source_file"] in original:
                raise ValueError("duplicate validation source WAV")
            original[row["source_file"]] = row
            sessions[str(row.get("session_id") or "unknown")] += 1
    groups = defaultdict(Counter)
    seen_ids = set()
    with (ROOT / protocol["inputs"]["segment_manifest"]["path"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "validation":
                continue
            source = row["source_file"]
            utterance = original.get(source)
            if utterance is None or any(
                row[name] != utterance[name]
                for name in (
                    "speaker_id",
                    "evaluation_role",
                    "utterance_id",
                    "source_sha256",
                )
            ):
                raise ValueError("segment/utterance metadata mismatch")
            if row["normalized_phoneme"] not in VOWELS or "near_silent" in row.get(
                "quality_flags", []
            ):
                raise ValueError("ineligible Phase 3 validation segment")
            if row["vowel_interval_id"] in seen_ids:
                raise ValueError("duplicate validation segment ID")
            seen_ids.add(row["vowel_interval_id"])
            groups[source][row["normalized_phoneme"]] += 1
    roles = {}
    for role in ROLES:
        candidates = [
            row for row in original.values() if row["evaluation_role"] == role
        ]
        complete = {
            str(count): sum(
                all(groups[row["source_file"]][vowel] >= count for vowel in VOWELS)
                for row in candidates
            )
            for count in (1, 5, 10)
        }
        by_speaker = {}
        for speaker in sorted(speakers):
            rows = [row for row in candidates if row["speaker_id"] == speaker]
            by_speaker[speaker] = {
                "original_utterances": len(rows),
                "sources_with_eligible_segments": sum(
                    bool(groups[row["source_file"]]) for row in rows
                ),
                "all_five_vowels": sum(
                    all(groups[row["source_file"]][vowel] >= 1 for vowel in VOWELS)
                    for row in rows
                ),
                "total_segments_by_vowel": {
                    vowel: sum(groups[row["source_file"]][vowel] for row in rows)
                    for vowel in VOWELS
                },
            }
        complete_count = complete["1"]
        roles[role] = {
            "original_utterances": len(candidates),
            "sources_with_eligible_segments": sum(
                bool(groups[row["source_file"]]) for row in candidates
            ),
            "complete_vowel_sources_by_minimum_count": complete,
            "coverage_at_minimum_one": complete_count / len(candidates)
            if candidates
            else None,
            "by_speaker": by_speaker,
        }
        if role != "enrollment":
            roles[role]["planned_trials_per_enrollment_count"] = {
                "scored_genuine": complete_count,
                "scored_impostor": complete_count * (len(speakers) - 1),
                "no_score_genuine": len(candidates) - complete_count,
                "no_score_impostor": (len(candidates) - complete_count)
                * (len(speakers) - 1),
            }
    return {
        "schema_version": 1,
        "purpose": "validation_metadata_only_no_audio_or_scores",
        "inspected_split": "validation",
        "protocol_sha256": sha256_file(protocol_path),
        "checked_reference_files": len(references),
        "inputs": protocol["inputs"],
        "validation_speakers": sorted(speakers),
        "session_ids": dict(sorted(sessions.items())),
        "roles": roles,
        "enrollment_has_ten_segments_each_vowel_for_all_speakers": all(
            count >= 10
            for row in roles["enrollment"]["by_speaker"].values()
            for count in row["total_segments_by_vowel"].values()
        ),
    }


def write_report(path: Path, report: dict) -> None:
    payload = (
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=BASE / "config/evaluation-protocol.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = inspect(args.protocol)
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "report": str(args.output),
                    "inspected_split": report["inspected_split"],
                    "roles": {
                        name: {
                            key: value
                            for key, value in role.items()
                            if key != "by_speaker"
                        }
                        for name, role in report["roles"].items()
                    },
                    "enrollment_ready": report[
                        "enrollment_has_ten_segments_each_vowel_for_all_speakers"
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"validation audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
