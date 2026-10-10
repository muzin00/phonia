"""Reuse the fixed fourteen-phone model and frozen evaluation utilities."""

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
prior = BASE.parent / "phoneme-voicing-comparison"
sys.path.insert(0, str(prior))
spec = importlib.util.spec_from_file_location("count_fixed14", prior / "study.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
sys.path.insert(0, str(BASE))
read_json, write_json, checked, pin, sha256_file = (
    fixed.read_json,
    fixed.write_json,
    fixed.checked,
    fixed.pin,
    fixed.sha256_file,
)
