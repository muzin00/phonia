"""Reuse the frozen encoder, input pipeline and held-out metric implementations."""

import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
spec = importlib.util.spec_from_file_location(
    "discriminability_core",
    BASE.parent / "phoneme-consonant-triple-evaluation/study.py",
)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
metrics = importlib.import_module("phase3_train.metrics")
PHONES = core.PHONEMES
ROLES = core.ROLES
read_json = core.read_json
write_json = core.write_json
checked = core.checked
pin = core.pin
sha256_file = core.sha256_file
