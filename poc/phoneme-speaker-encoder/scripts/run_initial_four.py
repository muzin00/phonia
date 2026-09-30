"""Finish only the four already-started full-comparison runs, then stop."""

from __future__ import annotations

import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from run_full_matrix import (
    BUDGET_PATH,
    ROOT,
    _code_sha256,
    _execute_one,
)


def main() -> None:
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    root = (
        ROOT / "artifacts/phoneme-speaker-encoder/comparisons" / budget["comparison_id"]
    )
    matrix = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
    if matrix["execution_budget_sha256"] != sha256_file(BUDGET_PATH):
        raise ValueError("frozen execution budget differs")
    if matrix["training_code_sha256"] != _code_sha256():
        raise ValueError("frozen training code differs")
    if matrix["runner_sha256"] != sha256_file(BASE / "scripts/run_full_matrix.py"):
        raise ValueError("frozen full runner differs")
    selected = matrix["runs"][:4]
    if len(selected) != 4:
        raise ValueError("expected four initial runs")
    for row in selected:
        path = root / "runs" / row["run_id"] / "full-result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
    with (root / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_execute_one, row, root, budget, matrix["sha256"]): row
                for row in selected
            }
            for future in as_completed(futures):
                row = futures[future]
                result = future.result()
                print(
                    json.dumps(
                        {
                            "run_id": row["run_id"],
                            "status": result["status"],
                            "reason": result.get("reason"),
                            "macro_eer": result.get("validation_macro_eer"),
                        }
                    ),
                    flush=True,
                )
    results = []
    for row in matrix["runs"]:
        path = root / "runs" / row["run_id"] / "full-result.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {
                "run_id": row["run_id"],
                "config_id": row["config_id"],
                "status": "not_started",
            }
        )
    write_json(
        root / "results.json", {"matrix_sha256": matrix["sha256"], "results": results}
    )
    print(
        json.dumps(
            {
                status: sum(result["status"] == status for result in results)
                for status in (
                    "completed",
                    "running",
                    "excluded",
                    "failed",
                    "not_started",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
