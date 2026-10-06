"""Select score-independent validation inputs for ECAPA duration feasibility."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from comparison_inputs import describe_audio, matched_center_window


def select_cases(records: dict, sources: dict, *, margin: int = 480) -> list[dict]:
    """Use min/max per condition; ties use source path then stable case ID."""
    if not sources or any(row["split"] != "validation" for row in sources.values()):
        raise ValueError("only validation sources are allowed")
    groups = defaultdict(list)
    methods = {
        "vowel_exact": "ecapa_time_exact",
        "vowel_context20": "ecapa_time_context20",
    }
    for category in ("enrollment", "queries"):
        for row in records[category]:
            if row["split"] != "validation":
                raise ValueError("only validation records are allowed")
            speaker = row["user_id"] if category == "enrollment" else row["speaker_id"]
            role = "enrollment" if category == "enrollment" else row["role"]
            condition = row["enrollment_count"] if category == "enrollment" else role
            audio = describe_audio(row["segments"], sources, margin=margin)
            for filename, source_audio in audio["sources"].items():
                source = sources[filename]
                if source["speaker_id"] != speaker or source["evaluation_role"] != role:
                    raise ValueError("source speaker or role mismatch")
                if category == "queries" and filename != row["source_file"]:
                    raise ValueError("query must use one source")
                for vowel_method, ecapa_method in methods.items():
                    window = source_audio["ecapa_duration_windows"][vowel_method]
                    if window is None:
                        continue
                    identity = [
                        category,
                        condition,
                        ecapa_method,
                        filename,
                        source["source_sha256"],
                        window,
                    ]
                    case = {
                        "case_id": hashlib.sha256(
                            json.dumps(identity, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "purpose": "actual_comparison_input",
                        "category": category,
                        "condition": condition,
                        "method": ecapa_method,
                        "source": source,
                        "start_frame": window[0],
                        "end_frame": window[1],
                        "budget_frames": source_audio["budgets"][vowel_method],
                        "anchor_ids": sorted(
                            segment["segment_id"]
                            for segment in row["segments"]
                            if segment["source_file"] == filename
                        ),
                    }
                    groups[category, str(condition), ecapa_method].append(case)
    selected = {}
    for group, candidates in sorted(groups.items()):
        candidates.sort(
            key=lambda case: (
                case["budget_frames"],
                case["source"]["source_file"],
                case["case_id"],
            )
        )
        for label, case in (("shortest", candidates[0]), ("longest", candidates[-1])):
            stored = selected.setdefault(case["case_id"], {**case, "selection": []})
            stored["selection"].append(f"{':'.join(group)}:{label}")
    if not selected:
        raise ValueError("no positive validation audio budgets")
    return sorted(selected.values(), key=lambda case: case["case_id"])


def add_length_diagnostics(cases: list[dict], lengths_ms: list[int]) -> list[dict]:
    """Diagnose fixed lengths on the shortest source; never add comparison rows."""
    shortest = min(
        cases,
        key=lambda case: (
            case["budget_frames"],
            case["source"]["source_file"],
            case["case_id"],
        ),
    )
    source = shortest["source"]
    if source["split"] != "validation":
        raise ValueError("diagnostics require validation")
    if len(lengths_ms) != len(set(lengths_ms)) or any(
        type(length) is not int or length < 1 for length in lengths_ms
    ):
        raise ValueError("diagnostic lengths must be unique positive integer ms")
    diagnostics = []
    for length in sorted(lengths_ms):
        frames = length * 24
        start, end = matched_center_window(source["frame_count"], frames)
        identity = ["length_diagnostic", source["source_sha256"], start, end]
        diagnostics.append(
            {
                "case_id": hashlib.sha256(
                    json.dumps(identity, separators=(",", ":")).encode()
                ).hexdigest(),
                "purpose": "length_diagnostic_excluded_from_comparison",
                "category": "diagnostic",
                "condition": length,
                "method": "ecapa_length_diagnostic",
                "source": source,
                "start_frame": start,
                "end_frame": end,
                "budget_frames": frames,
                "anchor_ids": [],
                "selection": ["predeclared_length_grid_on_shortest_actual_source"],
            }
        )
    return cases + diagnostics
