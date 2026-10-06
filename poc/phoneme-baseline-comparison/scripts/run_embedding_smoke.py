"""Check frozen ECAPA embeddings on a small deterministic validation sample."""

import argparse
import hashlib
import json
import platform
import resource
import sys
from importlib import metadata
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import (
    BASE,
    CONFIG,
    ROOT,
    check_model_snapshot,
    prepare_inputs,
    read_pcm,
    sha256_file,
    write_json,
)


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    require(
        config["inference"]["device"] == "cpu"
        and config["inference"]["dtype"] == "float32"
        and config["inference"]["batch_size"] == 1
        and config["inference"]["repeats"] >= 2,
        "smoke requires repeated CPU/float32/batch=1 inference",
    )
    require(
        config["audio"]["source_sample_rate"] == 24000
        and config["audio"]["channels"] == 1
        and config["audio"]["sample_width_bytes"] == 2
        and config["audio"]["pcm_scaling_divisor"] == 32768.0
        and not any(
            config["audio"][key] for key in ("vad", "crop", "rms_normalization")
        ),
        "smoke requires unchanged mono 24kHz PCM16 source audio",
    )
    model_dir = args.model_dir.resolve()
    snapshot = check_model_snapshot(model_dir, config)
    selected, population = prepare_inputs(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "selected-inputs.json", {"utterances": selected})

    import numpy as np
    import torch
    import torchaudio
    from speechbrain.inference.speaker import SpeakerRecognition

    torch.set_num_threads(config["inference"]["torch_threads"])
    torch.set_num_interop_threads(config["inference"]["torch_interop_threads"])
    torch.use_deterministic_algorithms(config["inference"]["deterministic_algorithms"])
    torch.manual_seed(0)
    started = perf_counter()
    encoder = SpeakerRecognition.from_hparams(
        source=str(model_dir),
        savedir=str(args.output_dir / "loaded-model"),
        overrides={"pretrained_path": str(model_dir)},
        run_opts={"device": config["inference"]["device"]},
    )
    encoder.eval()
    load_seconds = perf_counter() - started
    require(
        not any(parameter.requires_grad for parameter in encoder.parameters()),
        "model parameters are not frozen",
    )

    def state_hash() -> str:
        digest = hashlib.sha256()
        for key, value in sorted(encoder.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    initial_state = state_hash()
    resampler = torchaudio.transforms.Resample(
        orig_freq=config["audio"]["source_sample_rate"],
        new_freq=config["audio"]["target_sample_rate"],
        resampling_method=config["audio"]["resampling_method"],
        lowpass_filter_width=config["audio"]["lowpass_filter_width"],
        rolloff=config["audio"]["rolloff"],
        dtype=torch.float32,
    )
    embeddings, records = [], []
    warmup_seconds = None
    with torch.inference_mode():
        for row in selected:
            started = perf_counter()
            pcm = read_pcm(row)
            waveform = (
                torch.from_numpy(np.frombuffer(pcm, dtype="<i2").copy()).to(
                    dtype=torch.float32
                )
                / config["audio"]["pcm_scaling_divisor"]
            )
            waveform = resampler(waveform).unsqueeze(0)
            preprocessing_seconds = perf_counter() - started
            expected_frames = (
                row["frame_count"] * config["audio"]["target_sample_rate"]
                + config["audio"]["source_sample_rate"]
                - 1
            ) // config["audio"]["source_sample_rate"]
            require(
                waveform.shape == (1, expected_frames), "resampling length mismatch"
            )
            require(torch.isfinite(waveform).all(), "nonfinite resampled waveform")
            if warmup_seconds is None:
                started = perf_counter()
                for _ in range(config["inference"]["warmup_calls"]):
                    encoder.encode_batch(
                        waveform,
                        normalize=config["model"]["normalize_embedding_statistics"],
                    )
                warmup_seconds = perf_counter() - started
            times, repeated = [], []
            for _ in range(config["inference"]["repeats"]):
                started = perf_counter()
                embedding = encoder.encode_batch(
                    waveform,
                    normalize=config["model"]["normalize_embedding_statistics"],
                )
                times.append(perf_counter() - started)
                require(
                    embedding.shape == (1, 1, config["model"]["dimension"]),
                    "embedding shape mismatch",
                )
                require(
                    embedding.dtype == torch.float32 and embedding.device.type == "cpu",
                    "embedding dtype or device mismatch",
                )
                require(torch.isfinite(embedding).all(), "nonfinite embedding")
                vector = embedding.reshape(-1).cpu().numpy().copy()
                require(np.linalg.norm(vector) > 0, "zero embedding")
                repeated.append(vector)
            require(
                all(np.array_equal(repeated[0], item) for item in repeated[1:]),
                "repeated inference changed embedding",
            )
            embeddings.append(repeated[0])
            duration = row["frame_count"] / config["audio"]["source_sample_rate"]
            records.append(
                {
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "role": row["evaluation_role"],
                    "source_sha256": row["source_sha256"],
                    "duration_sec": duration,
                    "resampled_frames": expected_frames,
                    "preprocessing_seconds": preprocessing_seconds,
                    "inference_seconds": times,
                    "mean_inference_seconds": float(np.mean(times)),
                    "real_time_factor": float(np.mean(times)) / duration,
                    "embedding_l2_norm": float(np.linalg.norm(repeated[0])),
                    "repeat_max_absolute_difference": max(
                        float(np.max(np.abs(repeated[0] - item)))
                        for item in repeated[1:]
                    ),
                    "bitwise_equal_repeats": True,
                }
            )
            print(
                f"{row['utterance_id']}: {duration:.3f}s audio, "
                f"{np.mean(times):.3f}s inference, repeat exact",
                flush=True,
            )
    require(initial_state == state_hash(), "model parameters or buffers changed")
    require(
        snapshot == check_model_snapshot(model_dir, config), "model snapshot changed"
    )
    embeddings = np.stack(embeddings)
    np.save(args.output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    normalized = embeddings.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    cosine = normalized @ normalized.T
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss *= 1024
    report = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "status": "completed",
        "config": config,
        "config_sha256": sha256_file(CONFIG),
        "model_snapshot": snapshot,
        "model_state_sha256": initial_state,
        "implementation_sha256": {
            str(file.relative_to(ROOT)): sha256_file(file)
            for file in (
                BASE / "smoke.py",
                Path(__file__),
                BASE / "pyproject.toml",
                BASE / "uv.lock",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                package: metadata.version(package)
                for package in (
                    "numpy",
                    "torch",
                    "torchaudio",
                    "speechbrain",
                    "huggingface-hub",
                    "scipy",
                )
            },
        },
        "validation_population_metadata": population,
        "utterances": records,
        "summary": {
            "utterances": len(records),
            "speakers": len({row["speaker_id"] for row in records}),
            "dimension": embeddings.shape[1],
            "model_load_seconds": load_seconds,
            "warmup_seconds": warmup_seconds,
            "mean_inference_seconds": float(
                np.mean([row["mean_inference_seconds"] for row in records])
            ),
            "mean_real_time_factor": float(
                np.mean([row["real_time_factor"] for row in records])
            ),
            "peak_process_rss_bytes": peak_rss,
            "all_repeats_bitwise_equal": True,
            "model_parameters_and_buffers_unchanged": True,
            "model_snapshot_unchanged": True,
        },
        "diagnostic_cosine": {
            "purpose": "numerical_sanity_only_not_authentication_performance",
            "utterance_order": [row["utterance_id"] for row in records],
            "matrix": cosine.tolist(),
        },
        "outputs_sha256": {
            name: sha256_file(args.output_dir / name)
            for name in ("embeddings.npy", "selected-inputs.json")
        },
    }
    write_json(args.output_dir / "smoke-report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
