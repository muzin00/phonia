"""Validation-only input selection and source integrity checks for Phase 7."""

import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CONFIG = BASE / "config/embedding-smoke.json"
ROLES = ("enrollment", "verification", "cross_text_verification")


def sha256_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(file: Path, payload: dict) -> None:
    with file.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_validation(rows: list[dict], splits: dict, selection: dict) -> list[dict]:
    if selection["split"] != "validation" or selection["roles"] != list(ROLES):
        raise ValueError("smoke must use validation and the three specified roles")
    speakers = sorted(splits["speaker_splits"]["validation"])
    all_speakers = [
        speaker for group in splits["speaker_splits"].values() for speaker in group
    ]
    if len(all_speakers) != len(set(all_speakers)):
        raise ValueError("speaker split overlap")
    count = selection["speaker_count"]
    if not 1 <= count <= len(speakers):
        raise ValueError("invalid speaker count")
    if selection["additional_shortest_per_query_role"] < 0:
        raise ValueError("invalid shortest utterance count")
    seen_ids, seen_sources = set(), set()
    for row in rows:
        if (
            row["split"] != "validation"
            or row["speaker_id"] not in speakers
            or row["evaluation_role"] not in ROLES
        ):
            raise ValueError("invalid validation assignment")
        if row["utterance_id"] in seen_ids or row["source_file"] in seen_sources:
            raise ValueError("duplicate validation utterance")
        seen_ids.add(row["utterance_id"])
        seen_sources.add(row["source_file"])
    selected = {}
    for speaker in speakers[:count]:
        for role in ROLES:
            candidates = sorted(
                (
                    row
                    for row in rows
                    if row["speaker_id"] == speaker and row["evaluation_role"] == role
                ),
                key=lambda row: row["utterance_id"],
            )
            if not candidates:
                raise ValueError(f"missing validation role: {speaker}/{role}")
            selected[candidates[0]["utterance_id"]] = candidates[0]
    for role in ROLES[1:]:
        candidates = sorted(
            (row for row in rows if row["evaluation_role"] == role),
            key=lambda row: (row["frame_count"], row["utterance_id"]),
        )
        for row in candidates[: selection["additional_shortest_per_query_role"]]:
            selected[row["utterance_id"]] = row
    return [selected[key] for key in sorted(selected)]


def prepare_inputs(config: dict) -> tuple[list[dict], dict]:
    paths = {}
    for name, reference in config["inputs"].items():
        file = ROOT / reference["path"]
        if sha256_file(file) != reference["sha256"]:
            raise ValueError(f"input checksum mismatch: {name}")
        paths[name] = file
    splits = json.loads(paths["speaker_split"].read_text())
    rows = []
    with paths["utterance_manifest"].open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "validation":
                rows.append(row)
    selected = select_validation(rows, splits, config["selection"])
    summaries = {}
    for role in ROLES:
        candidates = [row for row in rows if row["evaluation_role"] == role]
        durations = sorted(row["frame_count"] / 24000 for row in candidates)
        summaries[role] = {
            "utterances": len(candidates),
            "minimum_duration_sec": min(durations),
            "maximum_duration_sec": max(durations),
            "speaker_counts": dict(Counter(row["speaker_id"] for row in candidates)),
        }
    return selected, summaries


def read_pcm(row: dict, *, root: Path = ROOT) -> bytes:
    if row["split"] != "validation":
        raise ValueError("only validation WAVs may be opened")
    source = (root / row["source_file"]).resolve()
    if Path(row["source_file"]).is_absolute() or not source.is_relative_to(
        root.resolve()
    ):
        raise ValueError("source outside audio root")
    if sha256_file(source) != row["source_sha256"]:
        raise ValueError("source checksum mismatch")
    with wave.open(str(source), "rb") as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
            wav.getnframes(),
        ) != (24000, 1, 2, "NONE", row["frame_count"]):
            raise ValueError("source PCM metadata mismatch")
        return wav.readframes(wav.getnframes())


def check_model_snapshot(directory: Path, config: dict) -> dict:
    manifest = json.loads((directory / "model-manifest.json").read_text())
    model = config["model"]
    if (manifest["model_id"], manifest["revision"]) != (
        model["id"],
        model["revision"],
    ) or set(manifest["files"]) != set(model["files"]):
        raise ValueError("model snapshot identity mismatch")
    if set(model["file_sha256"]) != set(model["files"]):
        raise ValueError("missing pinned model file checksums")
    for name, entry in manifest["files"].items():
        if (
            entry["sha256"] != model["file_sha256"][name]
            or sha256_file(directory / name) != model["file_sha256"][name]
        ):
            raise ValueError(f"model file checksum mismatch: {name}")
    return manifest
