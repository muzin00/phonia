"""Pin extraction dependencies and completed transcripts before alignment."""

import importlib.metadata
import runpy
import sys
from pathlib import Path

from selection import BASE, ROOT, check_files, fixed, pin, read


def main():
    config = read(BASE / "review-config.json")
    run = ROOT / config["run_directory"]
    check_files(read(run / "freeze.json")["files"])
    # The historical runtime imports its own greedy-selection module by this
    # short name. Keep the candidate selector out of that import chain.
    candidate_selection = sys.modules.pop("selection")
    sys.path = [p for p in sys.path if Path(p).resolve() != BASE]
    sys.path.insert(0, str(ROOT / "poc/phoneme-live-recording"))
    import runtime

    engine = runtime.Runtime(read(ROOT / config["runtime_freeze"])["config"])
    files = engine.fingerprints()
    files.update(
        pin(
            [
                Path(__file__),
                BASE / "review.py",
                run / "asr.json",
                run / "asr/freeze.json",
                run / "selected-utterances.json",
            ]
        )
    )
    frozen = {
        "files": files,
        "packages": {
            n: importlib.metadata.version(n) for n in ["numpy", "torch", "pyopenjtalk"]
        },
        "extract_config": engine.extract_config,
        "maximum_utterance_seconds": config["maximum_utterance_seconds"],
        "julius_timeout_seconds": config["julius_timeout_seconds"],
    }
    fixed(run / "runtime-inputs.json", frozen)
    check_files(frozen["files"])
    sys.modules["selection"] = candidate_selection
    sys.path.insert(0, str(BASE))
    sys.argv = [str(BASE / "review.py"), "extract"]
    runpy.run_path(str(BASE / "review.py"), run_name="__main__")


if __name__ == "__main__":
    main()
