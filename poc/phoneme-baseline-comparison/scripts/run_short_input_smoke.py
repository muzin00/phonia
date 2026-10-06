"""Check actual duration-matched ECAPA inputs, recording failures without padding."""

import argparse
import hashlib
import json
import platform
import sys
import traceback
from importlib import metadata
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_comparison_validation import inspect
from short_input_smoke import add_length_diagnostics, select_cases
from smoke import BASE, ROOT, check_model_snapshot, read_pcm, sha256_file, write_json


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = BASE / "config/short-input-smoke.json"
    config = json.loads(config_path.read_text())
    protocol_path = BASE / config["protocol"]
    require(
        sha256_file(protocol_path) == config["protocol_sha256"],
        "comparison protocol checksum mismatch",
    )
    protocol = json.loads(protocol_path.read_text())
    embedding_path = BASE / config["embedding_config"]
    require(
        sha256_file(embedding_path) == config["embedding_config_sha256"],
        "embedding config checksum mismatch",
    )
    embedding_config = json.loads(embedding_path.read_text())
    readiness = inspect(protocol)
    references = protocol["metadata_readiness_inputs"]
    with (ROOT / references["utterance_manifest"]["path"]).open() as stream:
        rows = [json.loads(line) for line in stream]
    sources = {row["source_file"]: row for row in rows if row["split"] == "validation"}
    records = {}
    for category in ("enrollment", "queries"):
        with (ROOT / references[category]["path"]).open() as stream:
            records[category] = [json.loads(line) for line in stream]
    cases = add_length_diagnostics(
        select_cases(
            records, sources, margin=protocol["context"]["margin_frames_per_side"]
        ),
        config["diagnostic_lengths_ms"],
    )
    require(config["repeats"] >= 2, "repeat check needs at least two calls")
    model_dir = args.model_dir.resolve()
    snapshot = check_model_snapshot(model_dir, embedding_config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "selected-inputs.json", {"cases": cases})

    import numpy as np
    import torch
    import torchaudio
    from speechbrain.inference.speaker import SpeakerRecognition

    inference = embedding_config["inference"]
    audio = embedding_config["audio"]
    require(
        inference["device"] == "cpu"
        and inference["dtype"] == "float32"
        and inference["batch_size"] == 1,
        "requires CPU float32 batch one",
    )
    torch.set_num_threads(inference["torch_threads"])
    torch.set_num_interop_threads(inference["torch_interop_threads"])
    torch.use_deterministic_algorithms(inference["deterministic_algorithms"])
    torch.manual_seed(0)
    encoder = SpeakerRecognition.from_hparams(
        source=str(model_dir),
        savedir=str(args.output_dir / "loaded-model"),
        overrides={"pretrained_path": str(model_dir)},
        run_opts={"device": "cpu"},
    )
    encoder.eval()
    require(not any(p.requires_grad for p in encoder.parameters()), "unfrozen model")

    def state_hash():
        digest = hashlib.sha256()
        for name, value in sorted(encoder.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    before_state = state_hash()
    resampler = torchaudio.transforms.Resample(
        orig_freq=audio["source_sample_rate"],
        new_freq=audio["target_sample_rate"],
        resampling_method=audio["resampling_method"],
        lowpass_filter_width=audio["lowpass_filter_width"],
        rolloff=audio["rolloff"],
        dtype=torch.float32,
    )

    def waveform_for(case):
        pcm = read_pcm(case["source"])
        start, end = case["start_frame"], case["end_frame"]
        require(0 <= start < end <= case["source"]["frame_count"], "invalid window")
        require(end - start == case["budget_frames"], "budget/window mismatch")
        cropped = pcm[start * 2 : end * 2]
        waveform = torch.from_numpy(np.frombuffer(cropped, dtype="<i2").copy()).float()
        waveform = resampler(waveform / audio["pcm_scaling_divisor"]).unsqueeze(0)
        expected = (case["budget_frames"] * 16000 + 23999) // 24000
        require(waveform.shape == (1, expected), "resampled length mismatch")
        require(torch.isfinite(waveform).all(), "nonfinite input")
        return waveform, hashlib.sha256(cropped).hexdigest()

    vectors, results = [], []
    with torch.inference_mode():
        warmup_case = max(
            (case for case in cases if case["purpose"] == "actual_comparison_input"),
            key=lambda case: case["budget_frames"],
        )
        warmup, _ = waveform_for(warmup_case)
        encoder.encode_batch(warmup, normalize=False)
        for case in cases:
            waveform, pcm_hash = waveform_for(case)
            features = encoder.mods.compute_features(waveform)
            attempts, repeated = [], []
            for repeat in range(config["repeats"]):
                started = perf_counter()
                try:
                    embedding = encoder.encode_batch(waveform, normalize=False)
                except RuntimeError as error:
                    attempts.append(
                        {
                            "status": "inference_error",
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
                else:
                    require(
                        embedding.shape == (1, 1, 192)
                        and embedding.dtype == torch.float32
                        and embedding.device.type == "cpu",
                        "unexpected embedding shape, dtype, or device",
                    )
                    require(torch.isfinite(embedding).all(), "nonfinite embedding")
                    vector = embedding.reshape(-1).cpu().numpy().copy()
                    require(np.linalg.norm(vector.astype(np.float64)) > 0, "zero norm")
                    repeated.append(vector)
                    attempts.append({"status": "success"})
                attempts[-1]["repeat"] = repeat
                attempts[-1]["seconds"] = perf_counter() - started
            successful = len(repeated) == config["repeats"]
            consistent = len({attempt["status"] for attempt in attempts}) == 1
            require(consistent, "inference success/error status varies across repeats")
            result = {
                "case_id": case["case_id"],
                "purpose": case["purpose"],
                "method": case["method"],
                "category": case["category"],
                "condition": case["condition"],
                "source_sha256": case["source"]["source_sha256"],
                "cropped_pcm_sha256": pcm_hash,
                "duration_sec": case["budget_frames"] / 24000,
                "resampled_frames": waveform.shape[1],
                "feature_frames": features.shape[1],
                "status": "success" if successful else "inference_error",
                "attempts": attempts,
                "repeat_status_consistent": consistent,
            }
            if successful:
                require(
                    all(np.array_equal(repeated[0], item) for item in repeated[1:]),
                    "embedding changed across repeats",
                )
                result.update(
                    embedding_row=len(vectors),
                    embedding_l2_norm=float(
                        np.linalg.norm(repeated[0].astype(np.float64))
                    ),
                    bitwise_equal_repeats=True,
                )
                vectors.append(repeated[0])
            else:
                require(
                    len({(item["type"], item["message"]) for item in attempts}) == 1,
                    "error changed across repeats",
                )
            results.append(result)
            print(
                f"{case['category']}/{case['condition']}/{case['method']}: "
                f"{result['duration_sec']:.3f}s, {result['feature_frames']} feature frames, "
                f"{result['status']}",
                flush=True,
            )
    require(before_state == state_hash(), "model state changed")
    require(
        snapshot == check_model_snapshot(model_dir, embedding_config), "weights changed"
    )
    np.save(
        args.output_dir / "embeddings.npy",
        np.stack(vectors) if vectors else np.empty((0, 192), dtype=np.float32),
        allow_pickle=False,
    )
    actual = [row for row in results if row["purpose"] == "actual_comparison_input"]
    errors = [row for row in actual if row["status"] == "inference_error"]
    report = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "status": "completed_with_inference_errors" if errors else "completed",
        "config": config,
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "embedding_config_sha256": sha256_file(embedding_path),
        "model_snapshot": snapshot,
        "model_state_sha256": before_state,
        "checked_metadata": readiness["checked_inputs"],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                BASE / "short_input_smoke.py",
                BASE / "comparison_inputs.py",
                BASE / "scripts/inspect_comparison_validation.py",
                BASE / "smoke.py",
                BASE / "pyproject.toml",
                BASE / "uv.lock",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: metadata.version(package)
                for package in ("numpy", "torch", "torchaudio", "speechbrain")
            },
        },
        "summary": {
            "actual_cases": len(actual),
            "actual_successes": len(actual) - len(errors),
            "actual_inference_errors": len(errors),
            "length_diagnostic_cases": len(results) - len(actual),
            "comparison_ready": not errors,
            "model_parameters_and_buffers_unchanged": True,
            "model_snapshot_unchanged": True,
            "successful_embeddings_repeat_bitwise_equal": True,
            "warmup_case_id": warmup_case["case_id"],
        },
        "results": results,
        "outputs_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("selected-inputs.json", "embeddings.npy")
        },
    }
    write_json(args.output_dir / "short-input-report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
