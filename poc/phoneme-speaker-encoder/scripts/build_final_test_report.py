"""Rebuild the frozen Phase 3 test summary from stored per-seed results."""

from __future__ import annotations

from evaluate_final_test import OUTPUT, _read, _report

if __name__ == "__main__":
    _report(_read(OUTPUT / "plan.json"))
