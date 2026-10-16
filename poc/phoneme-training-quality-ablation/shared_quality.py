"""New protocol around the immutable, previously validated uniform learner."""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
LEGACY = BASE.parent / "phoneme-segment-selection-rl"
sys.path.insert(0, str(LEGACY))
import runtime as rt
from final_fusion import fit
from pool import Pool
from student import Student
from verification import Bank, evaluate, load_evaluations

sys.path.insert(0, str(BASE.parent / "phoneme-common-voice-ablation"))
sys.path.insert(0, str(BASE))
ARMS = ("original", "screened", "random_matched")
__all__ = ["Bank", "Pool", "Student", "evaluate", "fit", "load_evaluations", "rt"]


def configuration(smoke=False):
    config = rt.read(LEGACY / "config.json")
    config.update(rt.read(BASE / "config.json"))
    config["prepared_directory"] = config["run_directory"] + "/prepared"
    if smoke:
        config.update(
            run_directory=config["run_directory"] + "-smoke-v1",
            smoke_test_only=True,
            updates=400,
            warmup_updates=10,
            evaluation_seeds=config["evaluation_seeds"][:1],
            final_fusion_updates=40,
            final_fusion_validation_every=20,
            bootstrap_replicates=20,
        )
    return config


def target(run, arm, seed):
    return run / arm / f"seed-{seed}"


def prepared(config):
    return ROOT / config["prepared_directory"]


def evaluation_prepared(config):
    return ROOT / config["reference_run"] / "prepared"


def variant(config, arm, seed=None):
    path = prepared(config) / arm
    return path / f"seed-{seed}" if arm == "random_matched" else path


def pool_paths(config, arm, seed=None):
    if arm == "original":
        return ROOT / config["source_run"], evaluation_prepared(config)
    path = (
        ROOT / config["quality_run"] / "screened"
        if arm == "screened"
        else variant(config, arm, seed)
    )
    return path, path
