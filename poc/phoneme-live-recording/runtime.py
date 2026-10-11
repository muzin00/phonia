"""Immutable all36 encoder and original Julius/feature pipeline for local audio."""

from __future__ import annotations

import importlib.metadata
import io
import math
import sys
import wave
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-recording-quality"))
import report as qr

t, s = qr.t, qr.t.s
np = s.np
VOWELS = tuple(s.VOWELS)


def decode_wav(content, maximum_seconds):
    from scipy.signal import resample_poly

    try:
        opened = wave.open(io.BytesIO(content), "rb")  # noqa: SIM115 - managed by with below after format validation.
    except (wave.Error, EOFError) as error:
        raise ValueError("有効なPCM WAVファイルを使用してください。") from error
    with opened as wav:
        rate, channels, width, frames = (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getnframes(),
        )
        if (
            channels != 1
            or width != 2
            or wav.getcomptype() != "NONE"
            or rate not in (16000, 22050, 24000, 32000, 44100, 48000, 96000)
        ):
            raise ValueError("モノラル16bit PCMのWAVを使用してください。")
        if not 0 < frames <= round(maximum_seconds * rate):
            raise ValueError("録音時間が上限を超えているか、音声が空です。")
        raw = wav.readframes(frames)
        if len(raw) != frames * 2:
            raise ValueError("WAVデータが途中で切れています。")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    if rate != 24000:
        gcd = math.gcd(rate, 24000)
        pcm = resample_poly(pcm, 24000 // gcd, rate // gcd)
    pcm = np.clip(np.rint(pcm * 32768), -32768, 32767).astype("<i2")
    return pcm, {
        "original_sample_rate": rate,
        "original_frames": frames,
        "seconds": frames / rate,
        "resampled": rate != 24000,
    }


class Runtime:
    def __init__(self, config):
        import pyopenjtalk

        self.config = config
        s.checked(ROOT / config["encoder_run"] / "encoder.pt", config["encoder_sha256"])
        s.checked(
            ROOT / config["feature_statistics"], config["feature_statistics_sha256"]
        )
        self.model, bundle = s.original.load_model(ROOT / config["encoder_run"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        s.torch.set_num_threads(1)
        self.phones = bundle["phonemes"]
        old = s.read_json(
            ROOT / "artifacts/phoneme-all-training/all36-20261011-v1/design-freeze.json"
        )["config"]
        self.extract_config = {
            **old,
            "minimum_frames": config["minimum_frames"],
            "maximum_frames": config["maximum_frames"],
            "minimum_rms_dbfs": config["minimum_rms_dbfs"],
        }
        self.pipe = t.h.primitives.pipeline(self.extract_config)
        self.preparation = t.h.preparation
        self.alignment = self.preparation.alignment
        self.model_path = (
            self.alignment.DEFAULT_SEGMENTATION_KIT
            / "models/hmmdefs_monof_mix16_gid.binhmm"
        )
        ac = s.read_json(
            ROOT / "poc/phoneme-speaker-dataset/config/vowel-dataset.json"
        )["alignment"]
        s.checked(self.model_path, ac["model_sha256"])
        if self.alignment.executable_version(self.alignment.DEFAULT_JULIUS) != "4.6":
            raise ValueError("Unexpected Julius version")
        dictionary = pyopenjtalk.OPEN_JTALK_DICT_DIR
        self.dictionary = (
            dictionary.decode() if isinstance(dictionary, bytes) else dictionary
        )
        self.g2p_version = importlib.metadata.version("pyopenjtalk")
        self.prompt_counts = {
            p["id"]: dict(
                Counter(
                    x.lower()
                    for x in pyopenjtalk.g2p(p["text"]).split()
                    if x.lower() in VOWELS
                )
            )
            for p in config["prompts"]
        }
        if not all(
            all(counts.get(v, 0) >= 2 for v in VOWELS)
            for counts in self.prompt_counts.values()
        ):
            raise ValueError(
                f"Prompts need >=2 of every canonical vowel: {self.prompt_counts}"
            )

    def fingerprints(self):
        paths = [
            ROOT / self.config["encoder_run"] / "encoder.pt",
            ROOT / self.config["encoder_run"] / "training-summary.json",
            ROOT / self.config["feature_statistics"],
            ROOT
            / "artifacts/phoneme-all-training/all36-20261011-v1/design-freeze.json",
            ROOT / "poc/phoneme-speaker-dataset/config/vowel-dataset.json",
            self.model_path,
            self.alignment.DEFAULT_JULIUS,
        ]
        paths += [p for p in Path(self.dictionary).rglob("*") if p.is_file()]
        paths += [
            Path(m.__file__)
            for m in tuple(sys.modules.values())
            if getattr(m, "__file__", None)
            and str(m.__file__).endswith(".py")
            and Path(m.__file__).resolve().is_relative_to(ROOT / "poc")
        ]
        return {t.rel(p): s.sha256_file(p) for p in paths}

    def process(self, pcm, destination, metadata):
        destination.mkdir(parents=True, exist_ok=True)
        source = destination / "audio.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm.tobytes())
        record = {
            **metadata,
            "source_file": t.rel(source),
            "source_sha256": s.sha256_file(source),
            "duration_sec": len(pcm) / 24000,
            "source_subset": "human_pilot"
            if metadata["origin"] == "microphone"
            else metadata["origin"],
            "split": "pilot",
            "evaluation_role": metadata["role"],
            "learning_curve_cohorts": [],
        }
        expected = self.preparation.expected.convert_record(
            record, self.g2p_version, self.dictionary
        )
        s.write_json(destination / "expected.json", expected)
        results = self.alignment.process_records(
            [expected],
            destination / "alignment",
            lambda item, raw, fail: self.alignment.align_record(
                item, raw, fail, self.alignment.DEFAULT_JULIUS, self.model_path
            ),
        )
        raw_path = (
            destination
            / "alignment/raw"
            / metadata["speaker_id"]
            / f"{metadata['utterance_id']}.json"
        )
        segments, features, qc, files = [], [], Counter(), {}
        if raw_path.exists():
            for segment, clip in t.h.primitives.extract(
                raw_path, expected, self.extract_config, files, qc
            ):
                if not segment["quality_flags"] and segment["phoneme"] in self.phones:
                    segment["cache_index"] = len(segments)
                    segments.append(segment)
                    features.append(
                        t.h.primitives.pool_features(
                            self.pipe, clip, segment["segment_id"]
                        )
                    )
        features = np.asarray(features, dtype=np.float32).reshape(-1, 128)
        vectors = (
            s.original.model_embeddings(self.model, features)
            if len(features)
            else np.empty((0, 128), np.float32)
        )
        np.save(destination / "features.npy", features, allow_pickle=False)
        np.save(destination / "embeddings.npy", vectors, allow_pickle=False)
        s.write_rows(destination / "segments.jsonl", segments)
        result = {
            "utterance": record,
            "counts": dict(Counter(r["phoneme"] for r in segments)),
            "alignment": results,
            "qc": dict(qc),
            "status": "processed" if len(segments) else "unusable",
            "files": {
                t.rel(p): s.sha256_file(p)
                for p in destination.rglob("*")
                if p.is_file()
            },
        }
        s.write_json(destination / "result.json", result)
        return result

    def profile(self, directory, episode):
        rows, vectors = [], []
        for clip in episode["clips"]:
            if clip["status"] != "processed":
                continue
            selected = list(s.rows(directory / clip["id"] / "segments.jsonl"))
            v = np.load(directory / clip["id"] / "embeddings.npy", allow_pickle=False)
            for row, vector in zip(selected, v, strict=True):
                rows.append({**row, "cache_index": len(rows)})
                vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float32)
        profile = {}
        for phone in self.phones:
            chosen = t.h.primitives.diverse(
                [r for r in rows if r["phoneme"] == phone],
                self.config["enrollment_limit_per_phone"],
                self.config["selection_seed"],
            )
            if not chosen:
                continue
            mean = matrix[[r["cache_index"] for r in chosen]].mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm < 1e-12:
                raise ValueError("Degenerate profile")
            profile[phone] = {
                "vector": (mean / norm).tolist(),
                "segments": [r["segment_id"] for r in chosen],
            }
        if not all(
            len(profile.get(v, {}).get("segments", []))
            >= self.config["enrollment_minimum_per_vowel"]
            for v in VOWELS
        ):
            raise ValueError("Incomplete profile")
        s.write_json(
            directory / "profile.json",
            {
                "participant": episode["participant"],
                "encoder_sha256": self.config["encoder_sha256"],
                "phones": profile,
                "threshold": None,
                "status": "pilot_profile_not_calibrated",
            },
        )
