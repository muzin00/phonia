"""Run frozen ECAPA enrollment and common-window queries in its own runtime."""

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from pilot import (
    AudioCache,
    EmbeddingCache,
    load_inputs,
    mean_profile,
    save_worker,
    score_row,
)
from smoke import CONFIG, check_model_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = load_inputs(args.run_dir)
    config = json.loads(CONFIG.read_text())
    model_config = inputs["protocol"]["models"]["ecapa"]
    if (
        str(torch.__version__) != model_config["torch"]
        or torchaudio.__version__ != model_config["torchaudio"]
    ):
        raise ValueError("ECAPA requires its pinned torch/torchaudio environment")
    snapshot = check_model_snapshot(args.model_dir, config)
    output = args.run_dir / "ecapa"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    started = perf_counter()
    encoder = SpeakerRecognition.from_hparams(
        source=str(args.model_dir.resolve()),
        savedir=str(output / "loaded-model"),
        overrides={"pretrained_path": str(args.model_dir.resolve())},
        run_opts={"device": "cpu"},
    )
    encoder.eval()
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("ECAPA parameters are not frozen")

    def state_hash():
        value = hashlib.sha256()
        for name, tensor in sorted(encoder.state_dict().items()):
            value.update(name.encode())
            value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return value.hexdigest()

    initial_state = state_hash()
    audio = AudioCache(inputs["sources"])
    resampler = torchaudio.transforms.Resample(
        orig_freq=24000,
        new_freq=16000,
        resampling_method=config["audio"]["resampling_method"],
        lowpass_filter_width=config["audio"]["lowpass_filter_width"],
        rolloff=config["audio"]["rolloff"],
        dtype=torch.float32,
    )

    def forward(pcm, segment_id):
        with torch.inference_mode():
            waveform = resampler(torch.from_numpy(pcm.copy())).unsqueeze(0)
            if waveform.shape != (1, (len(pcm) * 16000 + 23999) // 24000):
                raise ValueError("resampling frame count mismatch")
            vector = encoder.encode_batch(waveform, normalize=False)
            if (
                vector.shape != (1, 1, 192)
                or vector.dtype != torch.float32
                or vector.device.type != "cpu"
            ):
                raise ValueError("unexpected ECAPA embedding shape/device/dtype")
            return vector.reshape(-1).numpy().copy()

    identity = {
        "model": config["model"],
        "audio": config["audio"],
        "model_state_sha256": initial_state,
    }
    cache = EmbeddingCache(
        audio, identity, forward, inputs["config"]["repeats_per_unique_embedding"]
    )
    profiles, references = [], {}
    for item in inputs["enrollment"]:
        members, vectors = [], []
        for filename in sorted({s["source_file"] for s in item["segments"]}):
            source = inputs["sources"][filename]
            if (
                source["evaluation_role"] != "enrollment"
                or source["speaker_id"] != item["user_id"]
            ):
                raise ValueError("ECAPA enrollment speaker/role mismatch")
            last = source["frame_count"]
            vectors.append(cache.embed(filename, 0, last, filename))
            members.append(
                {
                    "source_file": filename,
                    "source_sha256": source["source_sha256"],
                    "input_frames": [0, last],
                    "resampled_frames": (last * 16000 + 23999) // 24000,
                }
            )
        profile = {
            "method_id": "ecapa_whole",
            "speaker_id": item["user_id"],
            "enrollment_count": item["enrollment_count"],
            "model_identity": identity,
            "embedding_dimension": 192,
            "members": members,
            "vector": mean_profile(vectors).tolist(),
        }
        profiles.append(profile)
        references[item["user_id"], item["enrollment_count"]] = profile
    print(
        f"ecapa: {len(profiles)} profiles from {len(cache.vectors)} source WAVs",
        flush=True,
    )
    trials = defaultdict(list)
    for trial in inputs["trials"]:
        trials[trial["query_id"]].append(trial)
    scores = []
    for window in inputs["windows"]:
        first, last = window["recording_frames"]
        vector = cache.embed(window["source_file"], first, last, window["query_id"])
        for trial in trials[window["query_id"]]:
            profile = references[trial["claimed_speaker_id"], trial["enrollment_count"]]
            score = float(np.clip(np.dot(np.asarray(profile["vector"]), vector), -1, 1))
            scores.append(
                score_row(
                    inputs, "ecapa_whole", window, trial, profile, {"fused": score}
                )
            )
    if initial_state != state_hash() or snapshot != check_model_snapshot(
        args.model_dir, config
    ):
        raise ValueError("ECAPA model state or snapshot changed")
    if any(p.grad is not None or p.requires_grad for p in encoder.parameters()):
        raise ValueError("ECAPA gradients changed")
    load_inputs(args.run_dir)
    save_worker(
        output,
        profiles,
        scores,
        cache,
        {
            "status": "completed",
            "environment": {
                "python": platform.python_version(),
                **{
                    p: metadata.version(p)
                    for p in ("numpy", "torch", "torchaudio", "speechbrain")
                },
            },
            "model_parameters_and_buffers_unchanged": True,
            "model_snapshot_unchanged": True,
            "model_state_sha256": initial_state,
            "model_snapshot": snapshot,
            "elapsed_seconds": perf_counter() - started,
        },
    )
    print(f"ecapa: {len(scores)} score slots; all query windows scored", flush=True)


if __name__ == "__main__":
    main()
