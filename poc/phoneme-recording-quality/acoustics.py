"""Explicit acoustic proxies and deterministic whole-waveform interventions."""

import hashlib
import wave

import numpy as np
from scipy import signal

RATE = 24000


def read_wave(path):
    with wave.open(str(path)) as wav:
        if (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getcomptype(),
        ) != (RATE, 1, 2, "NONE"):
            raise ValueError("expected 24kHz mono PCM16")
        return (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                np.float32
            )
            / 32768
        )


def write_wave(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(
            np.rint(np.clip(values, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
        )


def frame_power(values):
    size = RATE // 50
    x = np.asarray(values, np.float64)
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    return np.mean(x[: len(x) // size * size].reshape(-1, size) ** 2, axis=1)


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-12))


def quality(values):
    x = np.asarray(values, np.float64)
    power = frame_power(x)
    levels = db(power)
    p10, p90 = np.quantile(levels, [0.1, 0.9])
    active = levels > max(p90 - 25, -50)
    quiet = np.sort(power)[: max(1, int(np.ceil(len(power) * 0.1)))]
    f, psd = signal.welch(x, RATE, nperseg=min(1024, len(x)))
    total = psd.sum()
    rolloff = f[np.searchsorted(np.cumsum(psd), total * 0.95)] if total > 0 else 0
    return {
        "duration_seconds": len(x) / RATE,
        "rms_dbfs": float(db(np.mean(x**2))),
        "active_rms_dbfs": float(db(power[active].mean())) if active.any() else -120.0,
        "quiet_rms_dbfs": float(db(quiet.mean())),
        "energy_contrast_db": float(p90 - p10),
        "active_seconds": float(active.sum() / 50),
        "active_fraction": float(active.mean()),
        "near_clip_fraction": float(np.mean(np.abs(x) >= 0.999)),
        "high_frequency_fraction": float(psd[f > 4000].sum() / total)
        if total > 0
        else 0.0,
        "rolloff95_hz": float(rolloff),
    }


def seed_for(seed, identifier):
    return int(hashlib.sha256(f"{seed}/{identifier}".encode()).hexdigest()[:16], 16)


def transform(values, condition, seed):
    x = np.asarray(values, np.float64)
    diagnostics = {}
    if condition == "clean":
        y = x.copy()
    elif condition == "gain-minus12":
        y = x * 10 ** (-12 / 20)
    elif condition.startswith("white-snr"):
        snr = float(condition.removeprefix("white-snr"))
        rms = 10 ** (quality(x)["active_rms_dbfs"] / 20)
        noise = np.random.default_rng(seed).standard_normal(len(x))
        noise *= rms / np.sqrt(np.mean(noise**2)) / 10 ** (snr / 20)
        y = x + noise
        diagnostics["added_noise_snr_db"] = float(
            20 * np.log10(rms / np.sqrt(np.mean(noise**2)))
        )
    elif condition == "band-300-3400":
        y = signal.sosfiltfilt(
            signal.butter(6, [300, 3400], btype="bandpass", fs=RATE, output="sos"), x
        )
        y *= np.sqrt(np.sum(x * x) / max(np.sum(y * y), 1e-20))
    elif condition == "reverb-rt60-0.3":
        rng = np.random.default_rng(20261025)
        t = np.arange(int(0.3 * RATE)) / RATE
        tail = rng.standard_normal(len(t)) * np.exp(-np.log(1000) * t / 0.3)
        tail[: int(0.01 * RATE)] = 0
        tail /= np.linalg.norm(tail)
        tail[0] = 1
        y = signal.fftconvolve(x, tail)[: len(x)]
        y *= np.sqrt(np.sum(x * x) / max(np.sum(y * y), 1e-20))
    else:
        raise ValueError(condition)
    diagnostics["pre_quantization_clip_fraction"] = float(
        np.mean((y < -1) | (y > 32767 / 32768))
    )
    y = (np.rint(np.clip(y, -1, 32767 / 32768) * 32768) / 32768).astype(np.float32)
    if condition.startswith("white-snr"):
        rms = 10 ** (quality(x)["active_rms_dbfs"] / 20)
        diagnostics["effective_added_noise_snr_db"] = float(
            20 * np.log10(rms / np.sqrt(np.mean((y - x) ** 2)))
        )
    return y, diagnostics
