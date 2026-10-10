"""Load the immutable fourteen-phone encoder and existing metric utilities."""

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
prior = BASE.parent / "phoneme-voicing-comparison"
sys.path.insert(0, str(prior))
spec = importlib.util.spec_from_file_location("dependence_fixed14", prior / "study.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
sys.path.insert(0, str(BASE))
metrics = importlib.import_module("phase3_train.metrics")
read_json, write_json, checked, pin, sha256_file = (
    fixed.read_json,
    fixed.write_json,
    fixed.checked,
    fixed.pin,
    fixed.sha256_file,
)
