"""Read-only census of the accepted training pool, audio and forced alignment."""

import hashlib
import io
import json
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SOURCE = ROOT / "artifacts/phoneme-corpus-expansion/src490-20261013-v1"
OUTPUT = ROOT / "artifacts/phoneme-training-quality-audit/src490-20261016-v1"
RATE = 24000


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def corpus(speaker):
    for prefix, name in (
        ("cv17_", "common_voice"),
        ("jvs", "jvs"),
        ("jtube_", "jtube"),
        ("ext_src4vc", "src4vc"),
    ):
        if speaker.startswith(prefix):
            return name
    raise ValueError(speaker)


def summary(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    q = np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1])
    return {
        "count": len(values),
        **dict(zip(("min", "p01", "p10", "median", "p90", "p99", "max"), q.tolist())),
    }


def interval_measures(pcm, start, end):
    """Measure the exact accepted samples, before and after the existing DC removal."""
    x = np.asarray(pcm, np.float64) / 32768
    length = end - start

    def aggregate(values):
        c = np.r_[0, np.cumsum(values, dtype=np.float64)]
        return (c[end] - c[start]) / length

    mean = aggregate(x)
    power = aggregate(x * x)
    return {
        "measured_rms_dbfs": 10 * np.log10(np.maximum(power, 1e-20)),
        "dc_removed_rms_dbfs": 10 * np.log10(np.maximum(power - mean * mean, 1e-20)),
        "dc_offset": mean,
        "near_clip_fraction": aggregate(np.abs(x) >= 0.999),
        "rail_fraction": aggregate((pcm == 32767) | (pcm == -32768)),
        "below_minus50_fraction": aggregate(np.abs(x) < 10 ** (-50 / 20)),
        "zero_fraction": aggregate(pcm == 0),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "complete.json").exists():
        raise RuntimeError("Audit already complete; inspect saved results")
    start_time = time.time()
    report = read(SOURCE / "preparation-report.json")
    size = report["train_feature_count"]
    manifest = SOURCE / "train-segments.jsonl"
    inputs = {
        str(manifest.relative_to(ROOT)): sha(manifest),
        str((SOURCE / "train-features.f32").relative_to(ROOT)): sha(
            SOURCE / "train-features.f32"
        ),
    }
    for path in BASE.glob("*.py"):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    phones = list(read(SOURCE / "design-freeze.json")["phones"])
    phone_map = {p: i for i, p in enumerate(phones)}
    arrays = {
        k: np.empty(size, np.int32)
        for k in ("source", "start", "end", "raw_start", "raw_end", "interval", "phone")
    }
    arrays["stored_rms_dbfs"] = np.empty(size, np.float64)
    sources, source_map, issues = [], {}, Counter()
    hashes, groups = defaultdict(set), defaultdict(list)
    with manifest.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            name = row["source_file"]
            if name not in source_map:
                source_map[name] = len(sources)
                sources.append(
                    {
                        k: row[k]
                        for k in (
                            "source_file",
                            "source_sha256",
                            "alignment_file",
                            "speaker_id",
                            "utterance_id",
                        )
                    }
                )
                sources[-1].update(corpus=corpus(row["speaker_id"]), indices=[])
            sid = source_map[name]
            source = sources[sid]
            source["indices"].append(i)
            issues["cache_index_mismatch"] += row["cache_index"] != i
            issues["nontraining_rows"] += (
                row["split"] != "train" or row["role"] != "training"
            )
            issues["flagged_accepted_rows"] += bool(row["quality_flags"])
            issues["source_identity_mismatch"] += any(
                source[k] != row[k]
                for k in (
                    "source_sha256",
                    "alignment_file",
                    "speaker_id",
                    "utterance_id",
                )
            )
            arrays["source"][i] = sid
            for key, field in (
                ("start", "start_frame"),
                ("end", "end_frame"),
                ("raw_start", "raw_start_frame"),
                ("raw_end", "raw_end_frame"),
                ("interval", "interval_index"),
            ):
                arrays[key][i] = row[field]
            arrays["phone"][i] = phone_map[row["phoneme"]]
            arrays["stored_rms_dbfs"][i] = row["rms_dbfs"]
            hashes[row["source_sha256"]].add(row["speaker_id"])
            groups[row["speaker_id"], row["phoneme"]].append(i)
    if i + 1 != size:
        raise ValueError("Population changed")
    duration = arrays["end"] - arrays["start"]
    raw_duration = arrays["raw_end"] - arrays["raw_start"]
    issues["duration_gate_violation"] = int(
        ((duration < 720) | (duration > 6000)).sum()
    )
    issues["rms_gate_violation"] = int((arrays["stored_rms_dbfs"] < -50).sum())
    expected_start = (
        arrays["raw_start"] + (raw_duration - np.minimum(raw_duration, 6000)) // 2
    )
    issues["center_crop_mismatch"] = int(
        (
            (arrays["start"] != expected_start)
            | (duration != np.minimum(raw_duration, 6000))
        ).sum()
    )
    key = np.rec.fromarrays([arrays[k] for k in ("source", "interval", "phone")])
    issues["duplicate_accepted_interval_rows"] = size - len(np.unique(key))
    issues["source_hash_shared_across_speaker_labels"] = sum(
        len(v) > 1 for v in hashes.values()
    )
    features = np.memmap(
        SOURCE / "train-features.f32", mode="r", dtype="<f4", shape=(size, 128)
    )
    issues["nonfinite_feature_rows"] = sum(
        int((~np.isfinite(features[i : i + 65536]).all(1)).sum())
        for i in range(0, size, 65536)
    )
    issues["feature_byte_count_mismatch"] = (
        SOURCE / "train-features.f32"
    ).stat().st_size != size * 128 * 4
    metrics = (
        "measured_rms_dbfs",
        "dc_removed_rms_dbfs",
        "dc_offset",
        "near_clip_fraction",
        "rail_fraction",
        "below_minus50_fraction",
        "zero_fraction",
        "alignment_score",
    )
    for k in metrics:
        arrays[k] = np.full(size, np.nan, np.float64)
    pcm_seen, duplicate_pcm, audio_rows = {}, [], []
    source_fingerprints = hashlib.sha256()
    alignment_fingerprints = hashlib.sha256()
    for sid, source in enumerate(sources):
        ids = np.asarray(source["indices"], np.int64)
        item = {k: v for k, v in source.items() if k != "indices"}
        try:
            contents = (ROOT / source["source_file"]).read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            issues["source_hash_mismatch"] += digest != source["source_sha256"]
            source_fingerprints.update(
                (source["source_file"] + ":" + digest + "\n").encode()
            )
            with wave.open(io.BytesIO(contents)) as wav:
                spec = (
                    wav.getframerate(),
                    wav.getnchannels(),
                    wav.getsampwidth(),
                    wav.getcomptype(),
                )
                if spec != (RATE, 1, 2, "NONE"):
                    raise ValueError(f"Unexpected WAV format: {spec}")
                frame_count = wav.getnframes()
                pcm_bytes = wav.readframes(frame_count)
            pcm = np.frombuffer(pcm_bytes, dtype="<i2")
            if len(pcm) != frame_count:
                raise ValueError("Truncated WAV")
            item.update(
                duration_seconds=len(pcm) / RATE,
                segments=len(ids),
                pcm_sha256=hashlib.sha256(pcm_bytes).hexdigest(),
            )
            previous = pcm_seen.setdefault(item["pcm_sha256"], sid)
            if previous != sid:
                duplicate_pcm.append(
                    {
                        "source": source["source_file"],
                        "other": sources[previous]["source_file"],
                        "same_speaker_label": source["speaker_id"]
                        == sources[previous]["speaker_id"],
                    }
                )
            s, e = arrays["start"][ids], arrays["end"][ids]
            if not ((0 <= s) & (s < e) & (e <= len(pcm))).all():
                raise ValueError("Accepted boundary outside WAV")
            values = interval_measures(pcm, s, e)
            for k, v in values.items():
                arrays[k][ids] = v
            item["rms_max_absolute_error"] = float(
                np.max(
                    np.abs(values["measured_rms_dbfs"] - arrays["stored_rms_dbfs"][ids])
                )
            )
            issues["rms_recalculation_mismatch"] += int(
                (
                    np.abs(values["measured_rms_dbfs"] - arrays["stored_rms_dbfs"][ids])
                    > 1e-5
                ).sum()
            )
            x = pcm.astype(np.float64) / 32768
            frame_len = 480
            frame_power = np.mean(
                x[: len(x) // frame_len * frame_len].reshape(-1, frame_len) ** 2, axis=1
            )
            levels = 10 * np.log10(np.maximum(frame_power, 1e-12))
            p10, p90 = np.quantile(levels, [0.1, 0.9])
            item.update(
                near_clip_fraction=float(np.mean(np.abs(x) >= 0.999)),
                rail_fraction=float(np.mean((pcm == 32767) | (pcm == -32768))),
                energy_contrast_db=float(p90 - p10),
                active_seconds=float((levels > max(p90 - 25, -50)).sum() * 0.02),
            )
            apath = ROOT / source["alignment_file"]
            acontent = apath.read_bytes()
            alignment_fingerprints.update(
                (
                    source["alignment_file"]
                    + ":"
                    + hashlib.sha256(acontent).hexdigest()
                    + "\n"
                ).encode()
            )
            alignment = json.loads(acontent)
            intervals = alignment["intervals"]
            indices = arrays["interval"][ids]
            matched = [intervals[j] for j in indices]
            arrays["alignment_score"][ids] = [v["score"] for v in matched]
            astart = np.array([round(v["start_sec"] * RATE) for v in matched])
            aend = np.array([round(v["end_sec"] * RATE) for v in matched])
            issues["alignment_boundary_mismatch"] += int(
                (
                    (astart != arrays["raw_start"][ids])
                    | (aend != arrays["raw_end"][ids])
                ).sum()
            )
            issues["alignment_source_identity_mismatch"] += any(
                alignment[k] != source[k]
                for k in ("source_file", "source_sha256", "speaker_id", "utterance_id")
            )
            raw = alignment["raw_phonemes"]
            normalized = [
                raw[j - 1].lower() if raw[j - 1] in "AIUEO" else raw[j - 1]
                for j in indices
            ]
            issues["raw_phoneme_label_mismatch"] += sum(
                phones[p] != actual
                for p, actual in zip(arrays["phone"][ids], normalized)
            )
        except (OSError, ValueError, KeyError, IndexError, wave.Error) as error:
            item["audit_error"] = repr(error)
            issues["source_audit_errors"] += 1
        audio_rows.append(item)
        if (sid + 1) % 1000 == 0 or sid + 1 == len(sources):
            state = {
                "phase": "scanning_audio_and_alignment",
                "sources_done": sid + 1,
                "sources_total": len(sources),
                "elapsed_seconds": time.time() - start_time,
            }
            write(OUTPUT / "state.json", state)
            print(state, flush=True)
    np.savez_compressed(OUTPUT / "segment-measures.npz", **arrays)
    with (OUTPUT / "source-measures.jsonl").open("w") as f:
        for item in audio_rows:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    masks = {
        "near_clip_ge_1pct": arrays["near_clip_fraction"] >= 0.01,
        "near_clip_any": arrays["near_clip_fraction"] > 0,
        "rail_ge_1pct": arrays["rail_fraction"] >= 0.01,
        "rms_below_gate_after_dc": arrays["dc_removed_rms_dbfs"] < -50,
        "absolute_dc_ge_0_01": np.abs(arrays["dc_offset"]) >= 0.01,
        "more_than_half_below_minus50": arrays["below_minus50_fraction"] > 0.5,
        "more_than_half_exact_zero": arrays["zero_fraction"] > 0.5,
        "raw_interval_gt_1_second": raw_duration > RATE,
        "raw_interval_gt_250_ms": raw_duration > 6000,
        "accepted_30_ms": duration == 720,
    }
    by_corpus, candidates = {}, defaultdict(set)
    segment_corpus = np.array([s["corpus"] for s in sources])[arrays["source"]]
    for c in sorted(set(segment_corpus)):
        keep = segment_corpus == c
        src = [x for x in audio_rows if x["corpus"] == c]
        result = {
            "segments": int(keep.sum()),
            "sources": len(src),
            "speaker_labels": len({x["speaker_id"] for x in src}),
            "duration_ms": summary(duration[keep] / 24),
            "raw_duration_ms": summary(raw_duration[keep] / 24),
            "metrics": {k: summary(arrays[k][keep]) for k in metrics},
            "screens": {k: int((v & keep).sum()) for k, v in masks.items()},
            "sources_with_any_near_clip": sum(
                v.get("near_clip_fraction", 0) > 0 for v in src
            ),
            "sources_with_ge_1pct_near_clip": sum(
                v.get("near_clip_fraction", 0) >= 0.01 for v in src
            ),
        }
        result["source_metrics"] = {
            k: summary([x[k] for x in src if k in x])
            for k in ("duration_seconds", "active_seconds", "energy_contrast_db")
        }
        result["phones"] = {
            p: {
                "segments": int((keep & (arrays["phone"] == j)).sum()),
                "screens": {
                    k: int((v & keep & (arrays["phone"] == j)).sum())
                    for k, v in masks.items()
                },
            }
            for j, p in enumerate(phones)
        }
        by_corpus[c] = result
        for name, values in masks.items():
            if name in (
                "near_clip_any",
                "raw_interval_gt_250_ms",
                "accepted_30_ms",
                "absolute_dc_ge_0_01",
            ):
                continue
            ids = np.flatnonzero(values & keep)
            for index in ids[:5]:
                candidates[int(index)].add(name)
        ids = np.flatnonzero(keep & np.isfinite(arrays["alignment_score"]))
        for index in ids[np.argsort(arrays["alignment_score"][ids])[:5]]:
            candidates[int(index)].add(
                "lowest_alignment_score_for_manual_review_not_calibrated"
            )
    examples = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            if i in candidates:
                examples.append(
                    {
                        "reasons": sorted(candidates[i]),
                        "segment": json.loads(line),
                        "measures": {k: float(arrays[k][i]) for k in metrics},
                    }
                )
    write(OUTPUT / "review-candidates.json", examples)
    result = {
        "status": "complete",
        "scope": "All accepted training segments and their source WAVs; no evaluation metrics or training performed",
        "inputs": inputs,
        "segments": size,
        "sources": len(sources),
        "structural_issues": dict(issues),
        "by_corpus": by_corpus,
        "duplicate_pcm": duplicate_pcm,
        "source_fingerprint_digest": source_fingerprints.hexdigest(),
        "alignment_fingerprint_digest": alignment_fingerprints.hexdigest(),
        "screening_not_quality_labels": True,
        "limitations": "Screen thresholds identify inspection candidates, not established bad-data prevalence. Low amplitude/long phones/low forced-alignment scores can be valid speech. No human listening, speaker identity validation, ASR correctness verification, calibrated SNR or near-duplicate search was performed.",
        "elapsed_seconds": time.time() - start_time,
    }
    write(OUTPUT / "results.json", result)
    for name, digest in inputs.items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Input changed during audit: {name}")
    write(
        OUTPUT / "complete.json",
        {
            "status": "complete",
            "inputs": inputs,
            "files": {
                p.name: sha(p)
                for p in OUTPUT.iterdir()
                if p.name
                in (
                    "results.json",
                    "segment-measures.npz",
                    "source-measures.jsonl",
                    "review-candidates.json",
                )
            },
        },
    )
    write(
        OUTPUT / "state.json",
        {
            "phase": "complete",
            "sources": len(sources),
            "segments": size,
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(
        json.dumps(
            {
                "structural_issues": dict(issues),
                "by_corpus": {
                    c: {k: v[k] for k in ("segments", "sources", "screens")}
                    for c, v in by_corpus.items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
