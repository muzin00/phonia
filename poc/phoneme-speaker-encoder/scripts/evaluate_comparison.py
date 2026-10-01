"""Complete validation-only benchmark, paired bootstrap and selection evidence."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))

from phase3_data.manifest import Segment, iter_segments, seeded_hash, sha256_file
from weighted_bootstrap import speaker_draws, weighted_eers

VOWELS = ("a", "i", "u", "e", "o")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_hash():
    return {
        name: sha256_file(BASE / "scripts" / name)
        for name in ("evaluate_comparison.py", "weighted_bootstrap.py")
    }


def setup(comparison):
    output = comparison / "selection-evaluation"
    output.mkdir(exist_ok=True)
    matrix = read(comparison / "matrix.json")
    summary = read(comparison / "results.json")
    if len(summary["results"]) != 54 or any(
        r["status"] != "completed" or not r["repeat_evaluation_identical"]
        for r in summary["results"]
    ):
        raise ValueError("54 complete, repeated validation results required")
    if summary["matrix_sha256"] != matrix["sha256"] or {
        row["run_id"] for row in matrix["runs"]
    } != {row["run_id"] for row in summary["results"]}:
        raise ValueError("results differ from frozen matrix")
    protocol = {
        "schema_version": 1,
        "matrix_sha256": matrix["sha256"],
        "results_sha256": sha256_file(comparison / "results.json"),
        "implementation_sha256": source_hash(),
        "bootstrap": {
            "replicates": 10000,
            "seed": 20260929,
            "generator": "PCG64",
            "numpy_version": np.__version__,
            "genuine_weight": "n_s",
            "impostor_weight": "n_s*n_t",
            "percentile_method": "linear",
            "seed_resampling": False,
        },
        "benchmark": {
            "query_count": 1000,
            "query_hash_seed": 20260930,
            "batch_size": 1,
            "warmup_passes": 1,
            "measured_passes": 5,
            "device": "cpu",
            "precision": "float32",
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
            "parallel_runs": 1,
            "timing_api": "time.perf_counter_ns; CPU operations synchronous",
            "memory_api": "libproc.proc_pidinfo(PROC_PIDTASKINFO).pti_resident_size",
            "memory_poll_seconds": 0.005,
            "memory_reset": "reset sampled RSS maximum after model load and warmup, before measured passes",
            "includes": [
                "crop",
                "DC removal",
                "optional RMS",
                "features",
                "feature normalization",
                "batch padding",
                "encoder",
            ],
            "excludes": ["WAV reading", "profile creation", "model load", "warmup"],
        },
        "protocol_deviations": [
            "CPU benchmark API and thread count were not registered in the original execution budget. They are fixed here after viewing validation results, before these measurements. Sampled RSS is not an exact allocator peak. The eligible candidate set already contains only one configuration; these measurements do not change it."
        ],
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": importlib.metadata.version("torch"),
        "test_used": False,
    }
    path = output / "evaluation-budget.json"
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != "registered_at"} != protocol:
            raise ValueError("frozen evaluation protocol differs")
    else:
        write(path, {**protocol, "registered_at": timestamp()})
    return output, summary["results"]


def prepare_trials(comparison, output, results):
    metadata_path = output / "primary-trials.json"
    if all(
        path.exists()
        for path in (
            metadata_path,
            output / "benchmark-queries.json",
            output / "bootstrap-draws.json",
            output / "bootstrap-counts.npy",
        )
    ):
        return read(metadata_path)
    trial_path = comparison / "fixed-validation/selections/trials.jsonl"
    expected = read(comparison / "fixed-validation/data.json")["trials_sha256"]
    if sha256_file(trial_path) != expected:
        raise ValueError("fixed trials checksum differs")
    primary = []
    with trial_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "validation":
                raise ValueError("non-validation trial")
            if row["role"] == "verification" and row["enrollment_count"] == 10:
                primary.append(row)
    speakers = sorted(
        {row["speaker_id"] for row in primary}
        | {row["claimed_speaker_id"] for row in primary}
    )
    if len(speakers) != 15:
        raise ValueError("expected 15 validation speakers")
    index = {speaker: i for i, speaker in enumerate(speakers)}
    metadata = {
        "trials_sha256": expected,
        "speakers": speakers,
        "trial_ids": [row["trial_id"] for row in primary],
        "query": [index[row["speaker_id"]] for row in primary],
        "claimed": [index[row["claimed_speaker_id"]] for row in primary],
        "vowels": [VOWELS.index(row["vowel"]) for row in primary],
    }
    query_ids = sorted(
        {row["segment_id"] for row in primary},
        key=lambda segment: seeded_hash("benchmark", 20260930, segment),
    )[:1000]
    manifest = Path(
        read(comparison / "runs" / results[0]["run_id"] / "run.json")["manifest"]
    )
    query_set = set(query_ids)
    found = {
        segment.segment_id: segment
        for segment in iter_segments(manifest)
        if segment.segment_id in query_set
    }
    from dataclasses import asdict

    if len(found) != len(query_ids):
        raise ValueError("benchmark query missing from manifest")
    if any(
        found[key].split != "validation" or found[key].role != "verification"
        for key in query_ids
    ):
        raise ValueError("invalid benchmark query split/role")
    write(
        output / "benchmark-queries.json",
        {
            "segments": [asdict(found[key]) for key in query_ids],
            "manifest_sha256": sha256_file(manifest),
            "trials_sha256": expected,
        },
    )
    draws, counts = speaker_draws(15)
    write(
        output / "bootstrap-draws.json",
        {
            "seed": 20260929,
            "generator": "PCG64",
            "numpy_version": np.__version__,
            "speakers": speakers,
            "drawn_speaker_ids": [
                [speakers[i] for i in draw] for draw in draws.tolist()
            ],
        },
    )
    np.save(output / "bootstrap-counts.npy", counts, allow_pickle=False)
    write(metadata_path, metadata)
    return metadata


def bootstrap_run(comparison, output, result, metadata):
    run_id = result["run_id"]
    destination = output / "bootstrap" / f"{run_id}.npy"
    evidence = destination.with_suffix(".json")
    if destination.exists() and evidence.exists():
        record = read(evidence)
        if record["score_sha256"] != result["validation_score_sha256"] or record[
            "replicate_sha256"
        ] != sha256_file(destination):
            raise ValueError("bootstrap cache checksum differs")
        return
    started = time.monotonic()
    score_path = (
        comparison
        / "runs"
        / run_id
        / f"validation/update-{result['selected_checkpoint_update']:06d}/scores/validation.jsonl"
    )
    scores = np.empty(len(metadata["trial_ids"]), dtype=np.float64)
    digest = hashlib.sha256()
    number = 0
    with score_path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if (
                b'"role":"verification"' not in line
                or b'"enrollment_count":10,' not in line
            ):
                continue
            row = json.loads(line)
            if (
                row["split"] != "validation"
                or row["role"] != "verification"
                or row["enrollment_count"] != 10
            ):
                raise ValueError("unexpected primary trial")
            if (
                number >= len(scores)
                or row["trial_id"] != metadata["trial_ids"][number]
            ):
                raise ValueError("paired primary trial order differs")
            scores[number] = row["cosine_score"]
            number += 1
    if number != len(scores) or digest.hexdigest() != result["validation_score_sha256"]:
        raise ValueError("primary score coverage/checksum differs")
    counts = np.load(output / "bootstrap-counts.npy", allow_pickle=False)
    query, claimed, vowels = (
        np.asarray(metadata[key]) for key in ("query", "claimed", "vowels")
    )
    replicas = np.zeros(len(counts), dtype=np.float64)
    point = 0.0
    for vowel in range(5):
        mask = vowels == vowel
        values = weighted_eers(
            scores[mask],
            query[mask],
            claimed[mask],
            np.vstack([np.ones((1, 15), dtype=np.int64), counts]),
        )
        point += values[0] / 5
        replicas += values[1:] / 5
    if (
        not np.isfinite(replicas).all()
        or abs(point - result["validation_macro_eer"]) > 1e-12
    ):
        raise ValueError(
            f"unweighted EER differs: {point} vs {result['validation_macro_eer']}"
        )
    destination.parent.mkdir(exist_ok=True)
    np.save(destination, replicas, allow_pickle=False)
    write(
        evidence,
        {
            "run_id": run_id,
            "score_sha256": digest.hexdigest(),
            "replicate_sha256": sha256_file(destination),
            "point_eer": point,
            "replicates": len(replicas),
            "wall_seconds": time.monotonic() - started,
            "completed_at": timestamp(),
        },
    )


def resident_bytes():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int
    buffer = (ctypes.c_uint64 * 12)()  # proc_taskinfo: six uint64 + twelve int32
    if function(
        os.getpid(), 4, 0, ctypes.byref(buffer), ctypes.sizeof(buffer)
    ) != ctypes.sizeof(buffer):
        raise OSError(ctypes.get_errno(), "proc_pidinfo failed")
    return int(buffer[1])


def benchmark_one(comparison, output, result):
    import torch
    from phase3_data.input import InputPipeline, collate_segments, read_pcm_slice
    from phase3_train.models import create_encoder

    run_dir = comparison / "runs" / result["run_id"]
    run = read(run_dir / "run.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(run["settings"]["seed"])
    checkpoint = run_dir / "checkpoints/best.pt"
    if sha256_file(checkpoint) != result["selected_checkpoint_sha256"]:
        raise ValueError("benchmark checkpoint checksum differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = create_encoder(run["settings"]["encoder"])
    model.load_state_dict(state["model"])
    model.eval()
    del state
    stats = (
        read(run_dir / "feature-statistics.json")
        if model.input_kind == "log_mel"
        else None
    )
    pipeline = InputPipeline(
        read(BASE / "config/baseline-log-mel.json")["input"],
        rms_enabled=run["settings"]["rms_enabled"],
        statistics=stats,
    )
    segments = [
        Segment(**row) for row in read(output / "benchmark-queries.json")["segments"]
    ]
    pcm = [read_pcm_slice(ROOT, segment) for segment in segments]
    times = []
    baseline = 0
    peak = 0
    samples = 0
    finished = threading.Event()
    errors = []

    def sample():
        nonlocal peak, samples
        try:
            while not finished.wait(0.005):
                peak = max(peak, resident_bytes())
                samples += 1
        except OSError as error:
            errors.append(str(error))

    sampler = None
    started = time.monotonic()
    with torch.inference_mode():
        for iteration in range(6):
            if iteration == 1:
                baseline = peak = resident_bytes()
                sampler = threading.Thread(target=sample, daemon=True)
                sampler.start()
            for segment, waveform in zip(segments, pcm):
                tick = time.perf_counter_ns()
                item = pipeline.prepare(
                    waveform, segment.segment_id, mode="center", kind=model.input_kind
                )
                batch = collate_segments([item])
                embedding = model(batch["input"], batch["mask"])
                elapsed = time.perf_counter_ns() - tick
                if iteration > 0:
                    times.append(elapsed / 1e6)
                if not torch.isfinite(embedding).all():
                    raise ValueError("nonfinite benchmark embedding")
    peak = max(peak, resident_bytes())
    finished.set()
    sampler.join()
    if errors or not samples:
        raise ValueError(f"benchmark RSS sampler failed: {errors}")
    write(
        output / "benchmark" / f"{result['run_id']}.json",
        {
            "run_id": result["run_id"],
            "config_id": result["config_id"],
            "checkpoint_sha256": result["selected_checkpoint_sha256"],
            "query_file_sha256": sha256_file(output / "benchmark-queries.json"),
            "protocol_sha256": sha256_file(output / "evaluation-budget.json"),
            "median_ms_per_segment": statistics.median(times),
            "pass_median_ms": [
                statistics.median(times[i * len(segments) : (i + 1) * len(segments)])
                for i in range(5)
            ],
            "sampled_peak_rss_bytes": peak,
            "baseline_rss_bytes": baseline,
            "rss_samples": samples,
            "timing_samples": len(times),
            "query_count": len(segments),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "wall_seconds": time.monotonic() - started,
            "completed_at": timestamp(),
        },
    )


def finalize(comparison, output):
    from build_comparison_report import build

    report = build(comparison)
    group_replicas = {}
    for group in report["groups"]:
        run_replicas = []
        benches = []
        for run in group["runs"]:
            replica_path = output / "bootstrap" / f"{run['run_id']}.npy"
            evidence = read(replica_path.with_suffix(".json"))
            if evidence["replicate_sha256"] != sha256_file(replica_path):
                raise ValueError("bootstrap replicate checksum differs")
            replicas = np.load(replica_path, allow_pickle=False)
            if replicas.shape != (10000,) or not np.isfinite(replicas).all():
                raise ValueError("incomplete bootstrap replicas")
            run_replicas.append(replicas)
            bench = read(output / "benchmark" / f"{run['run_id']}.json")
            if bench["checkpoint_sha256"] != run["checkpoint_sha256"] or bench[
                "protocol_sha256"
            ] != sha256_file(output / "evaluation-budget.json"):
                raise ValueError("benchmark provenance differs")
            if bench["parameter_count"] != group["parameter_count"]:
                raise ValueError("benchmark encoder size differs")
            run["benchmark"] = bench
            benches.append(bench)
        group_replicas[group["config_id"]] = np.mean(run_replicas, axis=0)
        group["inference_median_ms"] = statistics.median(
            b["median_ms_per_segment"] for b in benches
        )
        group["benchmark_peak_rss_bytes"] = max(
            b["sampled_peak_rss_bytes"] for b in benches
        )
        if group["cross_text_eer"] is None or group["short_eer"] is None:
            raise ValueError("mandatory selection metric missing")
    reference = report["groups"][0]
    for group in report["groups"]:
        difference = (
            group_replicas[group["config_id"]] - group_replicas[reference["config_id"]]
        )
        group["delta_eer_ci95"] = np.percentile(
            difference, [2.5, 97.5], method="linear"
        ).tolist()
    eligible = [g for g in report["groups"] if g["eligible"]]

    def key(group):
        return (
            group["std_eer"],
            group["cross_text_eer"],
            group["short_eer"],
            group["parameter_count"],
            group["inference_median_ms"],
            group["benchmark_peak_rss_bytes"],
            group["config_id"],
        )

    selected = min(eligible, key=key)
    bundle = output / "selected-bundle"
    bundle.mkdir(exist_ok=True)
    files = []
    for run in selected["runs"]:
        run_dir = comparison / "runs" / run["run_id"]
        destination = bundle / str(run["seed"])
        destination.mkdir(exist_ok=True)
        threshold_path = (
            run_dir
            / f"validation/update-{run['selected_update']:06d}/selections/thresholds.json"
        )
        thresholds = read(threshold_path)
        if thresholds["source_role"] != "verification" or set(
            thresholds["per_count"]
        ) != {"1", "5", "10"}:
            raise ValueError("missing validation-only thresholds")
        for relative, source in (
            ("best.pt", run_dir / "checkpoints/best.pt"),
            ("thresholds.json", threshold_path),
            ("run.json", run_dir / "run.json"),
            ("feature-statistics.json", run_dir / "feature-statistics.json"),
        ):
            target = destination / relative
            shutil.copyfile(source, target)
            files.append(
                {
                    "seed": run["seed"],
                    "file": str(target.relative_to(comparison)),
                    "sha256": sha256_file(target),
                }
            )
        if sha256_file(destination / "best.pt") != run["checkpoint_sha256"]:
            raise ValueError("selected bundle checkpoint checksum differs")
    protocol = read(output / "evaluation-budget.json")
    selection = {
        "selected_config_id": selected["config_id"],
        "selected_mean_eer": selected["mean_eer"],
        "eligible_config_ids": [g["config_id"] for g in eligible],
        "tie_break_values": dict(
            zip(
                (
                    "std_eer",
                    "cross_text_eer",
                    "short_eer",
                    "encoder_parameters",
                    "inference_median_ms",
                    "benchmark_peak_rss_bytes",
                    "config_id",
                ),
                key(selected),
            )
        ),
        "rule": "mean_eer <= minimum_mean + 0.001; then lexicographic predefined tie-break",
        "matrix_sha256": report["matrix_sha256"],
        "results_sha256": report["results_sha256"],
        "evaluation_protocol_sha256": sha256_file(output / "evaluation-budget.json"),
        "bootstrap_draws_sha256": sha256_file(output / "bootstrap-draws.json"),
        "benchmark_queries_sha256": sha256_file(output / "benchmark-queries.json"),
        "reference_config_id": reference["config_id"],
        "single_distribution_seed": 20260926,
        "test_used": False,
        "files": files,
        "protocol_deviations": protocol["protocol_deviations"],
        "completed_at": timestamp(),
    }
    write(output / "selection.json", selection)
    report["selection_finalized"] = True
    report["missing_selection_evidence"] = []
    report["selection"] = selection
    write(output / "ranking-with-evidence.json", report)
    print(
        json.dumps(
            {
                "selected": selected["config_id"],
                "mean_eer": selected["mean_eer"],
                "inference_ms": selected["inference_median_ms"],
                "bundle": str(bundle),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "bootstrap", "benchmark", "finalize", "run", "benchmark-one"),
    )
    budget = read(BASE / "config/full-execution-budget.json")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=ROOT
        / "artifacts/phoneme-speaker-encoder/comparisons"
        / budget["comparison_id"],
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    comparison = args.comparison_dir.resolve()
    output, results = setup(comparison)
    if args.command == "benchmark-one":
        benchmark_one(
            comparison, output, next(r for r in results if r["run_id"] == args.run_id)
        )
        return
    with (output / "evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = prepare_trials(comparison, output, results)
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "protocol": str(output / "evaluation-budget.json"),
                        "primary_trials": len(metadata["trial_ids"]),
                    }
                )
            )
            return
        if args.command in ("bootstrap", "run"):
            for index, result in enumerate(results, 1):
                bootstrap_run(comparison, output, result, metadata)
                print(
                    json.dumps(
                        {
                            "stage": "bootstrap",
                            "completed": index,
                            "total": 54,
                            "run_id": result["run_id"],
                        }
                    ),
                    flush=True,
                )
        if args.command in ("benchmark", "run"):
            for index, result in enumerate(results, 1):
                path = output / "benchmark" / f"{result['run_id']}.json"
                if not path.exists():
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "benchmark-one",
                            "--comparison-dir",
                            str(comparison),
                            "--run-id",
                            result["run_id"],
                        ],
                        check=True,
                        env={
                            **os.environ,
                            "OPENBLAS_NUM_THREADS": "1",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                        },
                    )
                else:
                    cached = read(path)
                    if cached["checkpoint_sha256"] != result[
                        "selected_checkpoint_sha256"
                    ] or cached["protocol_sha256"] != sha256_file(
                        output / "evaluation-budget.json"
                    ):
                        raise ValueError("benchmark cache provenance differs")
                print(
                    json.dumps(
                        {
                            "stage": "benchmark",
                            "completed": index,
                            "total": 54,
                            "run_id": result["run_id"],
                        }
                    ),
                    flush=True,
                )
        if args.command in ("finalize", "run"):
            finalize(comparison, output)


if __name__ == "__main__":
    main()
