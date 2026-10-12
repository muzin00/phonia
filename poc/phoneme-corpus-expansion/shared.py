"""Adapters for a fourth training corpus; legacy experiments stay immutable."""

import importlib.util
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-data-budget"))
import data_budget as d

s, e, b = d.s, d.e, d.budget
read, sha, pin, fixed, atomic, check_files = (
    d.read,
    d.sha,
    d.pin,
    d.fixed,
    d.atomic,
    d.check_files,
)
CONFIG = BASE / "config.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


primitives = module(
    "src490_primitives", BASE.parent / "phoneme-greedy-selection/prepare.py"
)
mining = module("src490_mining", BASE.parent / "phoneme-hard-negatives/mining.py")
cv_schedule = module(
    "src490_cv_schedule", BASE.parent / "phoneme-cv-hard-negatives/schedule.py"
)


def corpus(speaker):
    for prefix, name in (
        ("ext_src4vc", "src4vc"),
        ("cv17_", "common_voice"),
        ("jtube_", "jtube"),
        ("jvs", "jvs"),
    ):
        if speaker.startswith(prefix):
            return name
    raise ValueError(f"Unknown speaker namespace: {speaker}")


def install_corpus_adapter():
    # These functions share the legacy inputs module globals. Patch only this
    # process, never its frozen source files or the three evaluation corpora.
    e.training_inputs.__globals__["corpus"] = corpus
    mining.corpus = corpus


def old_sources(config):
    sources = read(ROOT / config["baseline_run"] / "results.json")["sources"]
    return {
        seed: sources[f"combined420-seed{seed}-e{config['encoder_updates']}"]
        for seed in config["encoder_seeds"]
    }
