"""Load immutable helpers from the original fusion experiment by exact path."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = BASE.parent / "phoneme-transformer-fusion"
CONFIG = BASE / "config/protocol.json"
sys.path.insert(0, str(PREVIOUS))
frozen = importlib.import_module("data")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PREVIOUS / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


fusion = module("augmentation_source_fusion", "study.py")
checker = module("augmentation_source_verification", "verification.py")
stress = module("augmentation_source_stress", "stress.py")
primitives = importlib.import_module("common")
backend = importlib.import_module("backend")
np, torch = frozen.np, frozen.torch
read_json, write_json = frozen.read_json, frozen.write_json
sha256_file, checked = frozen.sha256_file, frozen.checked
ROLES = frozen.ROLES


def source(config):
    return ROOT / config["source_fusion_run"]


def verify_source(config, full=False):
    directory = source(config)
    design = frozen.verify(directory, full=full)
    selection = read_json(directory / "selection-freeze.json")
    checked(directory / "design-freeze.json", selection["design_sha256"])
    for name, files in selection["models"].items():
        for filename, checksum in files.items():
            checked(directory / name / filename, checksum)
    for filename in (
        "completion-verification.json",
        "stress/completion-verification.json",
    ):
        report = read_json(directory / filename)
        if report["status"] != "passed":
            raise ValueError("source comparison or diagnostic incomplete")
        for name, checksum in report["output_sha256"].items():
            checked(ROOT / name, checksum)
    return design


def verify(config, run, full=False):
    design = read_json(run / "design-freeze.json")
    if (
        design["config"] != config
        or config != read_json(CONFIG)
        or design["runtime"] != frozen.expanded.runtime()
    ):
        raise ValueError("augmentation protocol or runtime changed")
    for name, checksum in design["files"].items():
        checked(ROOT / name, checksum)
    verify_source(config, full=full)
    return design


def original_case(config, name, split="test"):
    directory = source(config)
    protocol = read_json(directory / "design-freeze.json")["config"]
    if name == "clean":
        return ROOT / protocol["evaluation_run"] / "enrollment-30"
    if split != "test":
        raise ValueError("diagnostics are not a checkpoint selection set")
    return directory / "stress" / name


def original_model_file(config, case, variant, filename, split="test"):
    if case == "clean":
        return source(config) / variant / f"{split}-{filename}"
    return original_case(config, case) / f"{variant}-{filename}"


def selected_models(config, run):
    selection = read_json(run / "selection-freeze.json")
    checked(run / "design-freeze.json", selection["design_sha256"])
    for name, files in selection["models"].items():
        for filename, checksum in files.items():
            checked(run / name / filename, checksum)
    return selection
