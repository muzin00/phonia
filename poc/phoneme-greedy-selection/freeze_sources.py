"""Freeze completed source tools and reuse unchanged pre-training feature caches."""

import os
import shutil

from common import (
    BASE,
    CONFIG,
    ROOT,
    checked,
    expanded,
    pin,
    read_json,
    relative,
    sha256_file,
    torch,
    write_json,
)


def main():
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    prepared = ROOT / config["preparation_run_directory"]
    run = ROOT / config["run_directory"]
    if (
        run.exists()
        or (prepared / "trials").exists()
        or (prepared / "search-state.json").exists()
    ):
        raise ValueError("unused training run and never-trained preparation required")
    original = read_json(prepared / "design-freeze.json")
    former = dict(config)
    del former["preparation_run_directory"]
    former["run_directory"] = relative(prepared)
    if original["config"] != former:
        raise ValueError("preparation numerical protocol changed")
    files = {}
    for name, sha in original["files"].items():
        if name.startswith("poc/phoneme-greedy-selection/"):
            continue
        checked(ROOT / name, sha)
        files[name] = sha
    run.mkdir(parents=True)
    for source in prepared.iterdir():
        if source.is_file() and source.name != "design-freeze.json":
            os.link(source, run / source.name)
            pin(files, run / source.name)
    shutil.copytree(prepared / "listening", run / "listening", symlinks=True)
    for source in (run / "listening").rglob("*"):
        if source.is_file():
            pin(files, source)
    pin(files, prepared / "design-freeze.json")
    for path in [CONFIG, *BASE.glob("*.py"), *BASE.glob("tests/*.py")]:
        pin(files, path)
    payload = {
        **original,
        "config": config,
        "files": files,
        "source_finalization": {
            "stage": "before_all_training_and_scored_encoder_inference",
            "preparation_design_sha256": sha256_file(prepared / "design-freeze.json"),
            "reason": "finish and lint audit/publication tools before training; preparation files reused byte-for-byte; numerical training/selection protocol unchanged",
            "preparation_had_training": False,
        },
        "runtime": expanded.runtime(),
    }
    write_json(run / "design-freeze.json", payload)
    print(f"source finalization complete; {len(files)} immutable files verified")


if __name__ == "__main__":
    main()
