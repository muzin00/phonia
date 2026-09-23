#!/usr/bin/env python3
"""Run ONNX Wav2Vec2 phoneme CTC inference and forced alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import warnings
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "samples" / "manifest.jsonl"
DEFAULT_EXPECTED = PROJECT_DIR / "data" / "phonemes" / "expected.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "raw"
DEFAULT_RUN_METADATA = (
    PROJECT_DIR / "data" / "alignments" / "wav2vec2" / "run.json"
)
MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_REVISION = "2c733782da5604684829819a5eb744c193fe9398"
DEFAULT_MODEL_ROOT = (
    Path.home() / ".local" / "share" / "phonia" / "wav2vec2" / MODEL_REVISION
)
DEFAULT_ONNX_MODEL = DEFAULT_MODEL_ROOT / "onnx" / "model.onnx"
DEFAULT_SOURCE_DIR = (
    Path.home() / ".cache" / "phonia" / "wav2vec2" / f"source-{MODEL_REVISION}"
)
TARGET_SAMPLE_RATE_HZ = 16_000
BLANK_ID = 0

# pyopenjtalk/Julius phone labels to the model's eSpeak IPA token vocabulary.
PHONE_TO_MODEL_TOKEN = {
    "a": "a",
    "b": "b",
    "ch": "tɕ",
    "d": "d",
    "e": "e",
    "g": "ɡ",
    "h": "h",
    "i": "i",
    "j": "dʑ",
    "k": "k",
    "m": "m",
    "my": "mʲ",
    "n": "n",
    "o": "o",
    "r": "ɾ",
    "s": "s",
    "sh": "ɕ",
    "t": "t",
    "u": "ɯ",
    "y": "j",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX phoneme CTC inference and forced alignment."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--source-directory", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-metadata", type=Path, default=DEFAULT_RUN_METADATA)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def resolve_source_file(source_file: str) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() else REPOSITORY_DIR / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_phonemes_by_utterance(path: Path) -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        phonemes = record.get("raw_phonemes")
        if not isinstance(phonemes, list) or not phonemes:
            raise ValueError(f"{path}: raw_phonemes is required")
        expected[str(record["utterance_id"])] = [str(phone) for phone in phonemes]
    return expected


def load_audio(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"{path}: compressed WAV is not supported")
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit PCM WAV")
        source_rate = source.getframerate()
        source_frames = source.getnframes()
        pcm = source.readframes(source_frames)

    if source_rate != TARGET_SAMPLE_RATE_HZ:
        pcm, _ = audioop.ratecv(
            pcm, 2, 1, source_rate, TARGET_SAMPLE_RATE_HZ, None
        )
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    variance = float(samples.var())
    samples = (samples - samples.mean()) / math.sqrt(variance + 1e-7)
    return samples, {
        "source_sample_rate_hz": source_rate,
        "source_frame_count": source_frames,
        "derived_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "derived_frame_count": len(samples),
    }


def log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def map_phonemes_to_tokens(
    phonemes: list[str], vocabulary: dict[str, int]
) -> tuple[list[str], list[int], list[int]]:
    tokens: list[str] = []
    token_ids: list[int] = []
    source_indices: list[int] = []
    for source_index, phone in enumerate(phonemes):
        if phone == "pau":
            continue
        if phone not in PHONE_TO_MODEL_TOKEN:
            raise ValueError(f"No Wav2Vec2 token mapping for phone: {phone}")
        token = PHONE_TO_MODEL_TOKEN[phone]
        if token not in vocabulary:
            raise ValueError(f"Wav2Vec2 vocabulary does not contain token: {token}")
        tokens.append(token)
        token_ids.append(int(vocabulary[token]))
        source_indices.append(source_index)
    return tokens, token_ids, source_indices


def ctc_forced_align(
    log_probabilities: np.ndarray,
    target_ids: list[int],
    *,
    blank_id: int = BLANK_ID,
) -> tuple[list[dict[str, Any]], float]:
    """Return target-token emission spans using the best valid CTC path."""
    if log_probabilities.ndim != 2:
        raise ValueError("log_probabilities must have shape [frames, vocabulary]")
    frame_count, vocabulary_size = log_probabilities.shape
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    if any(token < 0 or token >= vocabulary_size for token in target_ids):
        raise ValueError("target_ids contain an out-of-range token")

    states: list[int] = [blank_id]
    for token in target_ids:
        states.extend((token, blank_id))
    state_count = len(states)
    scores = np.full(state_count, -np.inf, dtype=np.float64)
    backpointers = np.full((frame_count, state_count), -1, dtype=np.int32)
    scores[0] = float(log_probabilities[0, blank_id])
    if state_count > 1:
        scores[1] = float(log_probabilities[0, states[1]])

    for frame in range(1, frame_count):
        next_scores = np.full(state_count, -np.inf, dtype=np.float64)
        for state, token in enumerate(states):
            predecessors = [(scores[state], state)]
            if state >= 1:
                predecessors.append((scores[state - 1], state - 1))
            if (
                state >= 2
                and state % 2 == 1
                and states[state] != states[state - 2]
            ):
                predecessors.append((scores[state - 2], state - 2))
            predecessor_score, predecessor = max(predecessors)
            if np.isfinite(predecessor_score):
                next_scores[state] = predecessor_score + float(
                    log_probabilities[frame, token]
                )
                backpointers[frame, state] = predecessor
        scores = next_scores

    final_candidates = [(scores[-1], state_count - 1)]
    if state_count > 1:
        final_candidates.append((scores[-2], state_count - 2))
    final_score, state = max(final_candidates)
    if not np.isfinite(final_score):
        raise RuntimeError("No valid CTC alignment path")

    path = [state]
    for frame in range(frame_count - 1, 0, -1):
        state = int(backpointers[frame, state])
        if state < 0:
            raise RuntimeError("CTC alignment backtracking failed")
        path.append(state)
    path.reverse()

    spans: list[dict[str, Any]] = []
    for target_index, token_id in enumerate(target_ids):
        target_state = target_index * 2 + 1
        frames = [index for index, path_state in enumerate(path) if path_state == target_state]
        if not frames:
            raise RuntimeError(f"CTC target {target_index} has no aligned frame")
        token_log_probabilities = [
            float(log_probabilities[frame, token_id]) for frame in frames
        ]
        spans.append(
            {
                "target_index": target_index,
                "token_id": token_id,
                "begin_frame": frames[0],
                "end_frame": frames[-1],
                "mean_log_probability": sum(token_log_probabilities)
                / len(token_log_probabilities),
            }
        )
    return spans, float(final_score / frame_count)


def spans_to_phone_intervals(
    phonemes: list[str],
    source_indices: list[int],
    spans: list[dict[str, Any]],
    *,
    frame_shift_sec: float,
    audio_duration_sec: float,
) -> list[dict[str, Any]]:
    """Expand token peaks to phone boundaries and preserve explicit pauses.

    Blank frames between adjacent phones are split at their midpoint. When the
    expected sequence contains ``pau``, the blank gap is assigned to that pause
    instead of either neighbouring phone.
    """
    if len(source_indices) != len(spans):
        raise ValueError("source_indices and spans must have the same length")
    intervals: list[dict[str, Any] | None] = [None] * len(phonemes)
    for source_index, span in zip(source_indices, spans, strict=True):
        start_sec = span["begin_frame"] * frame_shift_sec
        end_sec = (span["end_frame"] + 1) * frame_shift_sec
        intervals[source_index] = {
            "phoneme": phonemes[source_index],
            "start_sec": start_sec,
            "end_sec": end_sec,
            **span,
        }

    for pair_index in range(len(source_indices) - 1):
        left_index = source_indices[pair_index]
        right_index = source_indices[pair_index + 1]
        left = intervals[left_index]
        right = intervals[right_index]
        assert left is not None and right is not None
        gap_start = float(left["end_sec"])
        gap_end = float(right["start_sec"])
        pause_indices = [
            index
            for index in range(left_index + 1, right_index)
            if phonemes[index] == "pau"
        ]
        if not pause_indices:
            boundary = (gap_start + gap_end) / 2
            left["end_sec"] = boundary
            right["start_sec"] = boundary
            continue

        pause_width = max(0.0, gap_end - gap_start) / len(pause_indices)
        for offset, pause_index in enumerate(pause_indices):
            pause_start = gap_start + offset * pause_width
            pause_end = gap_start + (offset + 1) * pause_width
            intervals[pause_index] = {
                "phoneme": "pau",
                "start_sec": pause_start,
                "end_sec": pause_end,
                "begin_frame": round(pause_start / frame_shift_sec),
                "end_frame": max(
                    round(pause_start / frame_shift_sec),
                    round(pause_end / frame_shift_sec) - 1,
                ),
                "token_id": BLANK_ID,
                "mean_log_probability": None,
            }

    resolved = []
    for index, interval in enumerate(intervals):
        if interval is None:
            raise RuntimeError(f"Could not derive interval for phone {index}")
        interval["start_sec"] = round(max(0.0, float(interval["start_sec"])), 7)
        interval["end_sec"] = round(
            min(audio_duration_sec, float(interval["end_sec"])), 7
        )
        if interval["end_sec"] <= interval["start_sec"]:
            raise RuntimeError(f"Invalid derived phone interval: {interval}")
        resolved.append(interval)
    return resolved


def collapse_greedy_ids(ids: np.ndarray, blank_id: int = BLANK_ID) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for value in ids.tolist():
        token = int(value)
        if token != previous and token != blank_id:
            collapsed.append(token)
        previous = token
    return collapsed


def edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_value in left:
        current = [previous[0] + 1]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def validate_intervals(
    intervals: list[dict[str, Any]], phonemes: list[str], audio_duration_sec: float
) -> None:
    if [entry["phoneme"] for entry in intervals] != phonemes:
        raise RuntimeError("Aligned phone sequence differs from expected sequence")
    previous_end = 0.0
    for interval in intervals:
        start = float(interval["start_sec"])
        end = float(interval["end_sec"])
        if start < previous_end - 1e-9 or end <= start:
            raise RuntimeError(f"Invalid Wav2Vec2 interval: {interval}")
        previous_end = end
    if previous_end > audio_duration_sec + 1e-6:
        raise RuntimeError("Wav2Vec2 alignment exceeds the audio duration")


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    expected = expected_phonemes_by_utterance(args.expected)
    onnx_model = args.onnx_model.expanduser().resolve()
    source_directory = args.source_directory.expanduser().resolve()
    vocabulary_path = source_directory / "vocab.json"
    config_path = source_directory / "config.json"
    if not onnx_model.is_file():
        raise FileNotFoundError(onnx_model)
    if not vocabulary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Pinned model vocabulary/configuration is missing")

    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_to_logits_ratio = int(
        config.get("inputs_to_logits_ratio", math.prod(config["conv_stride"]))
    )
    frame_shift_sec = input_to_logits_ratio / TARGET_SAMPLE_RATE_HZ
    id_to_token = {int(value): key for key, value in vocabulary.items()}
    session = ort.InferenceSession(
        str(onnx_model), providers=["CPUExecutionProvider"]
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    total_duration = 0.0
    for record in manifest:
        utterance_id = str(record["utterance_id"])
        if utterance_id not in expected:
            raise ValueError(f"No expected phonemes for {utterance_id}")
        source_path = resolve_source_file(str(record["source_file"]))
        samples, audio_metadata = load_audio(source_path)
        audio_duration_sec = len(samples) / TARGET_SAMPLE_RATE_HZ
        phonemes = expected[utterance_id]
        tokens, target_ids, source_indices = map_phonemes_to_tokens(
            phonemes, vocabulary
        )

        started_at = time.perf_counter()
        logits = session.run(None, {"input_values": samples[None, :]})[0][0]
        log_probabilities = log_softmax(logits)
        spans, path_score = ctc_forced_align(log_probabilities, target_ids)
        intervals = spans_to_phone_intervals(
            phonemes,
            source_indices,
            spans,
            frame_shift_sec=frame_shift_sec,
            audio_duration_sec=audio_duration_sec,
        )
        elapsed = time.perf_counter() - started_at
        total_duration += elapsed
        validate_intervals(intervals, phonemes, audio_duration_sec)

        greedy_ids = collapse_greedy_ids(logits.argmax(axis=-1))
        distance = edit_distance(target_ids, greedy_ids)
        raw = {
            "utterance_id": utterance_id,
            "source_file": record["source_file"],
            "aligner": "wav2vec2-ctc-onnx",
            "aligner_version": ort.__version__,
            "model_id": MODEL_ID,
            "model_version": MODEL_REVISION,
            "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "frame_shift_sec": frame_shift_sec,
            "logit_frame_count": int(logits.shape[0]),
            "ctc_path_mean_log_probability": path_score,
            "target_tokens": tokens,
            "greedy_tokens": [id_to_token[token] for token in greedy_ids],
            "greedy_token_edit_distance": distance,
            "greedy_token_error_rate": distance / len(target_ids),
            "intervals": intervals,
        }
        output_path = args.output_directory / f"{source_path.stem}.json"
        output_path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        run_records.append(
            {
                "utterance_id": utterance_id,
                "duration_sec": round(elapsed, 3),
                "phone_interval_count": len(intervals),
                "target_token_count": len(target_ids),
                "logit_frame_count": int(logits.shape[0]),
                "ctc_path_mean_log_probability": path_score,
                "greedy_token_edit_distance": distance,
                "greedy_token_error_rate": distance / len(target_ids),
                **audio_metadata,
            }
        )
        print(
            f"{utterance_id}: {len(intervals)} phones aligned in {elapsed:.3f}s; "
            f"greedy token error rate={distance / len(target_ids):.3f}"
        )

    metadata = {
        "aligner": "wav2vec2-ctc-onnx",
        "aligner_version": ort.__version__,
        "architecture": platform.machine(),
        "execution_provider": "CPUExecutionProvider",
        "model_id": MODEL_ID,
        "model_version": MODEL_REVISION,
        "model_sha256": sha256(onnx_model),
        "model_format": "ONNX FP32",
        "manifest": str(args.manifest.relative_to(PROJECT_DIR)),
        "manifest_sha256": sha256(args.manifest),
        "expected_phonemes": str(args.expected.relative_to(PROJECT_DIR)),
        "expected_phonemes_sha256": sha256(args.expected),
        "utterance_count": len(run_records),
        "duration_sec": round(total_duration, 3),
        "duration_scope": (
            "onnx_inference_and_ctc_alignment_excluding_model_load_"
            "and_audio_preprocessing"
        ),
        "input_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "frame_shift_sec": frame_shift_sec,
        "output_format": "json",
        "utterances": run_records,
    }
    args.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.run_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aligned {len(run_records)} utterances in {total_duration:.3f}s")


if __name__ == "__main__":
    main()
