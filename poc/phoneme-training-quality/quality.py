"""Versioned training-only quality decisions; heuristic holds are not error labels."""

from itertools import groupby

import numpy as np

REASONS = {
    "low_rms_after_dc": 1,
    "near_clipping": 2,
    "majority_exact_zero": 4,
    "long_raw_interval": 8,
    "asr_low_logprob": 16,
    "asr_high_compression": 32,
    "asr_character_repetition": 64,
    "asr_diagnostics_missing": 128,
}
STATUS = {0: "accepted", 1: "held", 2: "rejected"}


def reason_names(bits):
    return [name for name, bit in REASONS.items() if int(bits) & bit]


def measure_pcm(pcm):
    """Measure the already-cropped training interval, including DC removal first."""
    pcm = np.asarray(pcm)
    if pcm.dtype != np.int16 or pcm.ndim != 1 or not len(pcm):
        raise ValueError("Expected a nonempty cropped mono PCM16 interval")
    x = pcm.astype(np.float64) / 32768
    centered = x - x.mean()
    return {
        "measured_rms_dbfs": float(10 * np.log10(max(np.mean(x * x), 1e-20))),
        "dc_removed_rms_dbfs": float(
            10 * np.log10(max(np.mean(centered * centered), 1e-20))
        ),
        "dc_offset": float(x.mean()),
        "near_clip_fraction": float(np.mean(np.abs(x) >= 0.999)),
        "rail_fraction": float(np.mean((pcm == 32767) | (pcm == -32768))),
        "zero_fraction": float(np.mean(pcm == 0)),
    }


def source_quality(source, asr, policy):
    """Missing mandatory diagnostics are held, never implicitly treated as clean."""
    result = {
        "source_file": source["source_file"],
        "source_sha256": source["source_sha256"],
        "speaker_id": source["speaker_id"],
        "corpus": source["corpus"],
        "asr_status": "not_required",
        "asr": None,
        "reason_bits": 0,
    }
    if asr is None:
        if source["corpus"] in policy["asr_required_corpora"]:
            result.update(
                asr_status="missing", reason_bits=REASONS["asr_diagnostics_missing"]
            )
        return result
    if any(asr[k] != source[k] for k in ("source_file", "speaker_id")):
        raise ValueError("ASR/source identity mismatch")
    names = (
        "minimum_avg_logprob",
        "maximum_compression_ratio",
        "maximum_no_speech_prob",
    )
    metrics = {k: asr.get(k) for k in names}
    if any(v is None or not np.isfinite(v) for v in metrics.values()):
        result.update(
            asr_status="missing", reason_bits=REASONS["asr_diagnostics_missing"]
        )
        return result
    text = asr.get("text", "")
    longest = max((sum(1 for _ in g) for _, g in groupby(text)), default=0)
    bits = 0
    if not text.strip():
        bits |= REASONS["asr_diagnostics_missing"]
    if metrics["minimum_avg_logprob"] < policy["hold_asr_avg_logprob_lt"]:
        bits |= REASONS["asr_low_logprob"]
    if metrics["maximum_compression_ratio"] > policy["hold_asr_compression_ratio_gt"]:
        bits |= REASONS["asr_high_compression"]
    if longest >= policy["hold_asr_same_character_run_ge"]:
        bits |= REASONS["asr_character_repetition"]
    result.update(
        asr_status="measured",
        asr={
            **metrics,
            "text": text,
            "maximum_same_character_run": longest,
            "asr_file": asr.get("asr_file"),
        },
        reason_bits=bits,
    )
    return result


def decide(measures, source_bits, policy, sample_rate):
    """Vectorized, identical policy for cached or newly measured training data."""
    required = (
        "dc_removed_rms_dbfs",
        "near_clip_fraction",
        "zero_fraction",
        "alignment_score",
        "raw_start",
        "raw_end",
    )
    arrays = {k: np.asarray(measures[k]) for k in required}
    size = len(arrays["raw_start"])
    if any(v.shape != (size,) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Invalid or missing segment quality measurements")
    if np.any(arrays["raw_end"] <= arrays["raw_start"]):
        raise ValueError("Nonpositive raw interval")
    for name in ("near_clip_fraction", "zero_fraction"):
        if np.any((arrays[name] < 0) | (arrays[name] > 1)):
            raise ValueError("Invalid sample fraction")
    bits = np.asarray(source_bits, dtype=np.uint16).copy()
    if bits.shape != (size,):
        raise ValueError("Source decisions do not align with segments")
    tests = {
        "low_rms_after_dc": arrays["dc_removed_rms_dbfs"]
        < policy["minimum_dc_removed_rms_dbfs"],
        "near_clipping": arrays["near_clip_fraction"]
        >= policy["hold_near_clip_fraction_ge"],
        "majority_exact_zero": arrays["zero_fraction"]
        > policy["hold_zero_fraction_gt"],
        "long_raw_interval": (arrays["raw_end"] - arrays["raw_start"]) / sample_rate
        > policy["hold_raw_duration_seconds_gt"],
    }
    for name, mask in tests.items():
        bits[mask] |= REASONS[name]
    status = np.where(bits == 0, 0, 1).astype(np.uint8)
    status[(bits & REASONS["low_rms_after_dc"]) != 0] = 2
    return bits, status
