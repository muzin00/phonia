"""Reuse the frozen encoder aggregates, models and metric definitions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-transformer-fusion"))
import data as original

spec = importlib.util.spec_from_file_location(
    "missing_vowels_source_fusion", BASE.parent / "phoneme-transformer-fusion/study.py"
)
fusion = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fusion
spec.loader.exec_module(fusion)
np = original.np
torch = original.torch
read_json = original.read_json
write_json = original.write_json
rows = original.rows
write_rows = original.write_rows
sha256_file = original.sha256_file
checked = original.checked
VOWELS = original.VOWELS
ROLES = original.ROLES
CONFIG = BASE / "config/protocol.json"


def source(config):
    return ROOT / config["source_run"]


def control(config):
    settings = read_json(source(config) / "design-freeze.json")["config"]
    return ROOT / settings["evaluation_run"] / "enrollment-30"


def original_path(config, model, split, suffix):
    directory = control(config) if model == "equal" else source(config) / model
    return directory / f"{split}-{suffix}"


def accepted(phones, policy):
    supported = set(phones)
    return (
        len(supported & set(VOWELS)) >= policy["minimum_vowels"]
        and len(supported) >= policy["minimum_phones"]
    )


def apply_policy(values, policy):
    result = []
    for row in values:
        item = dict(row)
        if not accepted(row["used_phones"], policy):
            item.update(
                status="no_score", score=None, reason="insufficient_phone_support"
            )
        result.append(item)
    return result


def verify(config, run):
    design = read_json(run / "design-freeze.json")
    if design["config"] != config or design["runtime"] != original.expanded.runtime():
        raise ValueError("protocol or numerical runtime changed after freeze")
    for path, checksum in design["files"].items():
        checked(ROOT / path, checksum)
    return design


def selected(config, run):
    result = read_json(run / "threshold-freeze.json")
    checked(run / "design-freeze.json", result["design_sha256"])
    for path, checksum in result["files"].items():
        checked(ROOT / path, checksum)
    return result


def numeric(value):
    return float(value)


def diagnostics(values, baseline, points, original_points):
    """Whole-input pattern rates and explicit old/new input decision changes."""
    result = {}
    for role in ROLES:
        chosen = [(r, b) for r, b in zip(values, baseline) if r["role"] == role]
        patterns = {}
        for row, before in chosen:
            missing = "".join(p for p in VOWELS if p not in row["used_phones"])
            key = missing or "none"
            patterns.setdefault(key, []).append((row, before))
        role_result = {}
        for mode, thresholds in (
            ("recalibrated", points),
            ("original", original_points),
        ):
            mode_result = {}
            for point, threshold in thresholds.items():
                groups = {}
                for name, pairs in patterns.items():
                    g = [r for r, _ in pairs if r["is_genuine"]]
                    i = [r for r, _ in pairs if not r["is_genuine"]]
                    positive = lambda r, threshold=threshold: (
                        r["status"] == "scored" and r["score"] >= numeric(threshold)
                    )
                    groups[name] = {
                        "queries": len(g),
                        "speakers": len({r["speaker_id"] for r in g}),
                        "scored_queries": sum(r["status"] == "scored" for r in g),
                        "impostor_trials": len(i),
                        "false_accepts": sum(positive(r) for r in i),
                        "all_false_rejects": sum(not positive(r) for r in g),
                    }
                changes = {}
                for status, pairs in (
                    (
                        "previously_scored",
                        [(r, b) for r, b in chosen if b["status"] == "scored"],
                    ),
                    (
                        "previously_unscored",
                        [(r, b) for r, b in chosen if b["status"] != "scored"],
                    ),
                ):
                    changes[status] = {
                        "genuine_accept_gain": 0,
                        "genuine_accept_loss": 0,
                        "impostor_accept_gain": 0,
                        "impostor_accept_loss": 0,
                    }
                    for r, b in pairs:
                        old = b["status"] == "scored" and b["score"] >= numeric(
                            original_points[point]
                        )
                        new = r["status"] == "scored" and r["score"] >= numeric(
                            threshold
                        )
                        label = "genuine" if r["is_genuine"] else "impostor"
                        if new != old:
                            changes[status][
                                f"{label}_accept_{'gain' if new else 'loss'}"
                            ] += 1
                mode_result[point] = {"patterns": groups, "changes": changes}
            role_result[mode] = mode_result
        result[role] = role_result
    return result
