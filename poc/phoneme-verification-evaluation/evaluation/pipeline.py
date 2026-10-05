"""Immutable evaluation stages: validation, freeze, test, then reporting."""

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from phase3_data.manifest import json_sha256, sha256_file
from registration import FrozenEncoder

from . import BASE, ROOT
from .inference import score_split
from .metrics import bootstrap_draws, calibrate, evaluate, validate_scores
from .report import build_report
from .storage import load_document, read_rows, save_document, save_rows
from .trials import build_trials, prepare_split


def runtime():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def implementation_files():
    files = [
        *(BASE / "evaluation").glob("*.py"),
        *(BASE / "scripts").glob("*.py"),
        *(BASE / "tests").glob("*.py"),
        BASE / "pyproject.toml",
        BASE / "uv.lock",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}


def require_files(files: dict, root: Path):
    for path, expected in files.items():
        if sha256_file(root / path) != expected:
            raise ValueError(f"frozen file changed: {path}")


def load_protocol(path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol["protocol_version"] != "1.0.0"
        or protocol["scores"]["kinds"] != ["fused", "a", "i", "u", "e", "o"]
        or protocol["enrollment"]["counts_per_vowel"] != [1, 5, 10]
        or protocol["thresholds"]["far_targets"] != [0.01, 0.001]
        or protocol["bootstrap"]["replicates"] != 10000
        or protocol["queries"]["unit"] != "one_source_wav_utterance"
    ):
        raise ValueError("unsupported evaluation protocol")
    references = [*protocol["inputs"].values(), *protocol["reused_files"]]
    references.extend(
        {"path": str(Path(protocol["model"]["bundle"]) / name), "sha256": value}
        for name, value in protocol["model"]["bundle_sha256"].items()
    )
    require_files({row["path"]: row["sha256"] for row in references}, ROOT)
    if (
        sys.version_info[:2] != (3, 12)
        or np.__version__ != protocol["model"]["numpy"]
        or str(torch.__version__) != protocol["model"]["torch"]
    ):
        raise ValueError("runtime differs from protocol")
    return protocol


def run_checks(output):
    commands = [
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(BASE.parent / name / "tests"),
            "-v",
        ]
        for name in (
            "phoneme-speaker-encoder",
            "phoneme-user-registration",
            "phoneme-verification",
            "phoneme-verification-evaluation",
        )
    ]
    commands.extend(
        [
            [sys.executable, "-m", "ruff", "check", str(BASE)],
            [sys.executable, "-m", "ruff", "format", "--check", str(BASE)],
            ["git", "diff", "--check"],
        ]
    )
    results = []
    for index, command in enumerate(commands):
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        path = output / f"checks/{index}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout + result.stderr, encoding="utf-8")
        results.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "log": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
        )
        print(
            f"check {index + 1}/{len(commands)}: exit {result.returncode}", flush=True
        )
        if result.returncode:
            raise ValueError(f"required check failed; see {path}")
    save_document(output / "checks.json", {"schema_version": 1, "results": results})


def store_inputs(prepared, output, protocol):
    split = prepared["split"]
    save_rows(output / f"inputs/{split}/enrollment.jsonl", prepared["enrollment"])
    save_rows(output / f"inputs/{split}/queries.jsonl", prepared["queries"])
    save_rows(
        output / f"trials/{split}.jsonl",
        build_trials(
            prepared["queries"],
            prepared["speakers"],
            protocol["enrollment"]["counts_per_vowel"],
            protocol["protocol_version"],
        ),
    )
    return {
        str(p.relative_to(output)): sha256_file(p)
        for p in [
            output / f"inputs/{split}/enrollment.jsonl",
            output / f"inputs/{split}/queries.jsonl",
            output / f"trials/{split}.jsonl",
        ]
    }


def read_inputs(output, split, speakers):
    return {
        "split": split,
        "speakers": speakers,
        "enrollment": list(read_rows(output / f"inputs/{split}/enrollment.jsonl")),
        "queries": list(read_rows(output / f"inputs/{split}/queries.jsonl")),
    }


def prepare(output: Path, protocol_path: Path):
    if output.exists():
        raise FileExistsError(output)
    protocol = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    run_checks(output)
    prepared = prepare_split(protocol, "validation")
    files = store_inputs(prepared, output, protocol)
    save_document(output / "protocol.json", protocol)
    save_document(
        output / "prepared.json",
        {
            "schema_version": 1,
            "protocol_sha256": sha256_file(protocol_path),
            "protocol_content_sha256": json_sha256(protocol),
            "implementation_files": implementation_files(),
            "runtime": runtime(),
            "input_files": files,
            "speakers": prepared["speakers"],
            "checks_sha256": sha256_file(output / "checks.json"),
        },
    )
    print(f"prepared validation: {len(prepared['queries'])} utterances", flush=True)


def require_prepared(output):
    prepared = load_document(output / "prepared.json")
    protocol = load_document(output / "protocol.json")
    load_protocol(BASE / "config/evaluation-protocol.json")
    if (
        prepared["protocol_sha256"]
        != sha256_file(BASE / "config/evaluation-protocol.json")
        or prepared["protocol_content_sha256"] != json_sha256(protocol)
        or prepared["runtime"] != runtime()
    ):
        raise ValueError("prepared protocol/runtime changed")
    require_files(prepared["implementation_files"], ROOT)
    if prepared["implementation_files"] != implementation_files():
        raise ValueError("implementation files added or removed after preparation")
    require_files(prepared["input_files"], output)
    if prepared["checks_sha256"] != sha256_file(output / "checks.json"):
        raise ValueError("checks changed after preparation")
    return prepared, protocol


def start_stage(output, stage):
    save_document(
        output / f"stages/{stage}-started.json",
        {
            "stage": stage,
            "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "runtime": runtime(),
        },
    )


def finish_stage(output, stage):
    save_document(
        output / f"stages/{stage}-completed.json",
        {
            "stage": stage,
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )


def validation(output):
    prepared, protocol = require_prepared(output)
    start_stage(output, "validation")
    inputs = read_inputs(output, "validation", prepared["speakers"])
    trials = list(read_rows(output / "trials/validation.jsonl"))
    encoder = FrozenEncoder.from_bundle(ROOT / protocol["model"]["bundle"])
    invariants = score_split(inputs, trials, encoder, output, audio_root=ROOT)
    rows = list(read_rows(output / "scores/validation.jsonl"))
    validate_scores(rows, trials)
    thresholds = calibrate(
        [r for r in rows if r["role"] == "verification"],
        protocol["enrollment"]["counts_per_vowel"],
        prepared["speakers"],
    )
    save_document(output / "thresholds/validation.json", thresholds)
    metrics, curves = evaluate(
        rows,
        thresholds,
        protocol["enrollment"]["counts_per_vowel"],
        protocol["splits"]["roles"],
        prepared["speakers"],
    )
    save_document(output / "metrics/validation.json", metrics)
    save_rows(output / "curves/validation.jsonl", curves)
    save_document(output / "invariants/validation.json", invariants)
    require_prepared(output)
    finish_stage(output, "validation")
    print("validation completed; thresholds calibrated without test scores", flush=True)


def freeze(output):
    prepared, protocol = require_prepared(output)
    load_document(output / "stages/validation-completed.json")
    start_stage(output, "freeze")
    test = prepare_split(protocol, "test")
    files = store_inputs(test, output, protocol)
    save_document(
        output / "test-prepared.json",
        {"schema_version": 1, "speakers": test["speakers"], "input_files": files},
    )
    draws = bootstrap_draws(
        test["speakers"],
        replicates=protocol["bootstrap"]["replicates"],
        seed=protocol["bootstrap"]["seed"],
    )
    save_document(output / "bootstrap-draws.json", draws)
    thresholds = load_document(output / "thresholds/validation.json")
    if thresholds["split"] != "validation" or thresholds["role"] != "verification":
        raise ValueError("invalid calibration source")
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    save_document(
        output / "evaluation-plan.json",
        {
            "schema_version": 1,
            "frozen_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "protocol_sha256": prepared["protocol_sha256"],
            "implementation_files": prepared["implementation_files"],
            "runtime": runtime(),
            "files": files,
            "test_speakers": test["speakers"],
            "test_encoder_inference_before_freeze": False,
        },
    )
    finish_stage(output, "freeze")
    print("evaluation plan frozen before any test encoder inference", flush=True)


def require_plan(output):
    prepared, protocol = require_prepared(output)
    plan = load_document(output / "evaluation-plan.json")
    if (
        plan["implementation_files"] != prepared["implementation_files"]
        or plan["runtime"] != runtime()
    ):
        raise ValueError("frozen implementation/runtime mismatch")
    require_files(plan["files"], output)
    return plan, protocol


def test(output):
    plan, protocol = require_plan(output)
    start_stage(output, "test")
    inputs = read_inputs(output, "test", plan["test_speakers"])
    trials = list(read_rows(output / "trials/test.jsonl"))
    encoder = FrozenEncoder.from_bundle(ROOT / protocol["model"]["bundle"])
    invariants = score_split(
        inputs, trials, encoder, output, audio_root=ROOT, check_cache=False
    )
    rows = list(read_rows(output / "scores/test.jsonl"))
    validate_scores(rows, trials)
    thresholds = load_document(output / "thresholds/validation.json")
    draws = load_document(output / "bootstrap-draws.json")
    print(
        "test scores completed; computing shared speaker confidence intervals",
        flush=True,
    )
    metrics, curves = evaluate(
        rows,
        thresholds,
        protocol["enrollment"]["counts_per_vowel"],
        protocol["splits"]["roles"],
        plan["test_speakers"],
        draws=draws,
    )
    save_document(output / "metrics/test.json", metrics)
    save_rows(output / "curves/test.jsonl", curves)
    save_document(output / "invariants/test.json", invariants)
    require_plan(output)
    finish_stage(output, "test")
    print("test completed with frozen validation thresholds", flush=True)


def report(output):
    require_plan(output)
    load_document(output / "stages/test-completed.json")
    start_stage(output, "report")
    build_report(output)
    finish_stage(output, "report")
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    save_document(
        output / "execution.json",
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "plan_sha256": sha256_file(output / "evaluation-plan.json"),
            "runtime": runtime(),
            "files": files,
            "model_statistics_profiles_unchanged": True,
        },
    )
    print(f"report: {output / 'report.md'}", flush=True)
