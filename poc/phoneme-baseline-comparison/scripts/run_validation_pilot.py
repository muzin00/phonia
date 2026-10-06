"""Prepare, run both pinned runtimes, and audit a validation-only v2 pilot."""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import METHODS, digest, load_inputs, prepare, read_rows, write_rows
from smoke import BASE, ROOT, sha256_file, write_json


def audit(run):
    inputs = load_inputs(run)
    scores, workers = [], {}
    for worker in ("vowels", "ecapa"):
        report = json.loads((run / worker / "report.json").read_text())
        for name, expected in report["outputs_sha256"].items():
            if sha256_file(run / worker / name) != expected:
                raise ValueError("worker artifact changed")
        workers[worker] = report
        scores.extend(read_rows(run / worker / "scores.jsonl"))
    expected = {
        digest([inputs["protocol"]["protocol_version"], method, c["id"], t["trial_id"]])
        for method in METHODS
        for c in inputs["protocol"]["query_windows"]["conditions"]
        for t in inputs["trials"]
    }
    if len(scores) != len(expected) or {r["score_id"] for r in scores} != expected:
        raise ValueError("missing or duplicate score slots")
    windows = {(w["query_id"], w["condition_id"]): w for w in inputs["windows"]}
    profile_hashes, query_status = {}, {}
    groups = defaultdict(list)
    for row in scores:
        w = windows[row["query_id"], row["condition_id"]]
        should_score = row["method_id"] == "ecapa_whole" or w["complete_five_vowels"]
        if row["window_id"] != w["window_id"] or row["status"] != (
            "scored" if should_score else "no_score"
        ):
            raise ValueError("score population/window/no-score invariant failed")
        if (row["scores"] is not None) != should_score:
            raise ValueError("no-score carried a numeric score")
        profile_key = (
            row["method_id"],
            row["claimed_speaker_id"],
            row["enrollment_count"],
        )
        if (
            profile_key in profile_hashes
            and profile_hashes[profile_key] != row["profile_sha256"]
        ):
            raise ValueError("enrollment profile changed across query caps")
        profile_hashes[profile_key] = row["profile_sha256"]
        query_key = row["method_id"], row["condition_id"], row["query_id"]
        if query_key in query_status and query_status[query_key] != row["status"]:
            raise ValueError("query coverage changed across claims/enrollment counts")
        query_status[query_key] = row["status"]
        groups[row["method_id"], row["condition_id"], row["role"]].append(row)
    summary = []
    for (method, condition, role), rows in sorted(groups.items()):
        query_ids = {r["query_id"] for r in rows}
        scored = {r["query_id"] for r in rows if r["status"] == "scored"}
        genuine = [r for r in rows if r["is_genuine"]]
        summary.append(
            {
                "method_id": method,
                "condition_id": condition,
                "role": role,
                "queries": len(query_ids),
                "scored_queries": len(scored),
                "no_score_queries": len(query_ids - scored),
                "score_slots": len(rows),
                "scored_genuine_slots": sum(r["status"] == "scored" for r in genuine),
                "scored_impostor_slots": sum(
                    r["status"] == "scored" and not r["is_genuine"] for r in rows
                ),
            }
        )
    write_rows(run / "scores.jsonl", sorted(scores, key=lambda r: r["score_id"]))
    return {
        "schema_version": 1,
        "status": "completed",
        "purpose": inputs["purpose"],
        "protocol_sha256": inputs["protocol_sha256"],
        "config_sha256": inputs["config_sha256"],
        "speakers": inputs["speakers"],
        "queries": len(inputs["queries"]),
        "trial_ids": len(inputs["trials"]),
        "score_slots": len(scores),
        "profiles": len(profile_hashes),
        "summary": summary,
        "workers": workers,
        "checks": {
            "all_score_slots_present_once": True,
            "same_query_population_all_caps": True,
            "same_enrollment_profiles_all_caps": True,
            "vowel_no_score_matches_metadata": True,
            "ecapa_scores_independent_of_vowels": True,
            "thresholds_calibrated": False,
            "test_audio_or_scores_read": False,
        },
        "outputs_sha256": {"scores.jsonl": sha256_file(run / "scores.jsonl")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/phoneme-baseline-comparison/models/ecapa-0f99f2d0",
    )
    args = parser.parse_args()
    # Selection and input checks precede any output creation or model inference.
    inputs = prepare()
    run = args.output_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "inputs.json", inputs)
    implementations = sorted(
        {
            *BASE.glob("*.py"),
            *BASE.glob("scripts/*.py"),
            *BASE.glob("config/*.json"),
            BASE / "pyproject.toml",
            BASE / "uv.lock",
            ROOT / "poc/phoneme-verification-evaluation/uv.lock",
            *(ROOT / "poc/phoneme-user-registration/registration").glob("*.py"),
            *(ROOT / "poc/phoneme-verification/verification").glob("*.py"),
            ROOT / "poc/phoneme-speaker-encoder/phase3_data/input.py",
            ROOT / "poc/phoneme-speaker-encoder/phase3_train/models.py",
            ROOT / "poc/phoneme-verification-evaluation/evaluation/inference.py",
        }
    )
    write_json(
        run / "preflight.json",
        {
            "inputs_sha256": sha256_file(run / "inputs.json"),
            "implementation_sha256": {
                str(p.relative_to(ROOT)): sha256_file(p) for p in implementations
            },
        },
    )
    print(
        f"pilot: {len(inputs['speakers'])} speakers, {len(inputs['queries'])} queries, "
        f"{len(inputs['trials']) * 15} score slots; no test",
        flush=True,
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    try:
        for project, script, extra in (
            ("phoneme-verification-evaluation", "run_pilot_vowels.py", []),
            (
                "phoneme-baseline-comparison",
                "run_pilot_ecapa.py",
                ["--model-dir", str(args.model_dir)],
            ),
        ):
            command = [
                str(ROOT / "poc" / project / ".venv/bin/python"),
                str(BASE / "scripts" / script),
                "--run-dir",
                str(run),
                *extra,
            ]
            subprocess.run(command, check=True, cwd=ROOT, env=env)
        report = audit(run)
        write_json(run / "pilot-report.json", report)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "speakers",
                        "queries",
                        "profiles",
                        "score_slots",
                        "checks",
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        write_json(
            run / "failure.json",
            {"status": "failed", "type": type(exc).__name__, "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
