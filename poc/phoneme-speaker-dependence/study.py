"""Freeze input, cluster validation variance scores, then evaluate other speakers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import dependence_math as dm
import numpy as np
import torch
from bridge import (
    BASE,
    ROOT,
    checked,
    fixed,
    pin,
    read_json,
    sha256_file,
    write_json,
)
from phase3_data.input import InputPipeline, SegmentDataset, collate_segments
from phase3_data.manifest import Segment

CONFIG = BASE / "config/protocol.json"
ROLES = ("verification", "cross_text_verification")
STOPS = {"t", "d", "k", "g"}


def seeded_key(row, seed):
    return hashlib.sha256(f"{seed}/{row['segment_id']}".encode()).hexdigest()


def choose(rows, count, seed, distinct_sources=False):
    ordered = sorted(rows, key=lambda r: seeded_key(r, seed))
    selected, sources = [], set()
    for row in ordered:
        if row["source_file"] in sources:
            continue
        selected.append(row)
        sources.add(row["source_file"])
        if len(selected) == count:
            return selected
    if not distinct_sources:
        ids = {r["segment_id"] for r in selected}
        selected += [r for r in ordered if r["segment_id"] not in ids][
            : count - len(selected)
        ]
    if len(selected) != count:
        raise ValueError(
            f"insufficient candidates: need {count}, found {len(selected)}"
        )
    return selected


def candidates(source, metadata, phones, files, qc):
    meta = metadata[source]
    path = (
        ROOT
        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
        / meta["speaker_id"]
        / f"{meta['utterance_id']}.json"
    )
    raw = read_json(path)
    for key in ("source_file", "source_sha256", "speaker_id", "raw_phonemes"):
        if raw[key] != meta[key]:
            raise ValueError("raw alignment metadata mismatch")
    pin(files, path)
    checked(ROOT / source, meta["source_sha256"])
    files[source] = meta["source_sha256"]
    with wave.open(str(ROOT / source), "rb") as audio:
        if (
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getcomptype(),
        ) != (24000, 1, 2, "NONE"):
            raise ValueError("unexpected PCM format")
        pcm = (
            np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
                float
            )
            / 32768
        )
    result = {"variance": [], "verification": []}
    intervals = raw["intervals"]
    for i, interval in enumerate(intervals):
        phone = interval["phoneme"]
        if phone not in phones:
            continue
        lo, hi = (
            round(interval["start_sec"] * 24000),
            round(interval["end_sec"] * 24000),
        )
        if not 0 <= lo < hi <= len(pcm):
            raise ValueError("invalid raw boundary")
        for purpose, selected in result.items():
            qc[f"{purpose}/{phone}/raw"] += 1
            size = min(hi - lo, 6000) if purpose == "variance" else 1200
            if hi - lo < size or size < 720:
                qc[f"{purpose}/{phone}/duration_excluded"] += 1
                continue
            first = (
                hi - size
                if purpose == "verification" and phone in STOPS
                else lo + (hi - lo - size) // 2
            )
            energy = float(np.mean(pcm[first : first + size] ** 2))
            rms = 10 * np.log10(energy) if energy else -np.inf
            if rms < -50:
                qc[f"{purpose}/{phone}/rms_excluded"] += 1
                continue
            qc[f"{purpose}/{phone}/eligible"] += 1
            selected.append(
                {
                    "segment_id": f"{meta['utterance_id']}--{i:03d}-{phone}--{purpose}-{first}-{size}",
                    "source_file": source,
                    "source_sha256": meta["source_sha256"],
                    "speaker_id": meta["speaker_id"],
                    "split": meta["split"],
                    "role": meta["evaluation_role"],
                    "vowel": phone,
                    "start_frame": first,
                    "end_frame": first + size,
                    "raw_index": i,
                    "raw_start_frame": lo,
                    "raw_end_frame": hi,
                    "previous_phone": intervals[i - 1]["phoneme"] if i else "sil",
                    "following_phone": intervals[i + 1]["phoneme"]
                    if i + 1 < len(intervals)
                    else "sil",
                    "rms_dbfs": float(rms),
                }
            )
    return result


def all_rows(data):
    return [*data["variance"], *data["enrollment"], *data["queries"]]


def validate_inputs(data, config, training_hashes):
    speakers, phones = data["speakers"], config["phones"]
    groups = defaultdict(list)
    enroll_hashes = {r["source_sha256"] for r in data["enrollment"]}
    for category in ("variance", "enrollment", "queries"):
        rows = data[category]
        if len({r["segment_id"] for r in rows}) != len(rows):
            raise ValueError("repeated selected token")
        for r in rows:
            if (
                r["speaker_id"] not in speakers
                or r["vowel"] not in phones
                or r["split"] != data["split"]
            ):
                raise ValueError("wrong speaker/phone/split")
            if r["source_sha256"] in training_hashes:
                raise ValueError("training audio overlap")
            if category != "enrollment" and r["source_sha256"] in enroll_hashes:
                raise ValueError("enrollment/query source overlap")
            size = r["end_frame"] - r["start_frame"]
            if (
                not r["raw_start_frame"]
                <= r["start_frame"]
                < r["end_frame"]
                <= r["raw_end_frame"]
            ):
                raise ValueError("outside source phone boundary")
            if not 720 <= size <= 6000 or (category != "variance" and size != 1200):
                raise ValueError("unexpected selected duration")
            groups[category, r["role"], r["speaker_id"], r["vowel"]].append(r)
    expected = [
        ("variance", "verification", 12),
        ("enrollment", "enrollment", 5),
        ("queries", "verification", 10),
        ("queries", "cross_text_verification", 3),
    ]
    for category, role, count in expected:
        for s in speakers:
            for p in phones:
                rows = groups[category, role, s, p]
                if len(rows) != count or (
                    category != "enrollment"
                    and len({r["source_file"] for r in rows}) != count
                ):
                    raise ValueError("unbalanced sample or repeated query source")


def prepare(config, run):
    if run.exists():
        raise ValueError("new run directory required")
    if config["phones"] != list(fixed.PHONEMES):
        raise ValueError("requires the complete fixed14 inventory")
    files, qc = {}, Counter()
    metadata_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = {
        r["source_file"]: r for r in fixed.old.previous.read_rows(metadata_path)
    }
    pin(files, metadata_path)
    split_path = ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    labels = read_json(split_path)["speaker_splits"]
    pin(files, split_path)
    training = ROOT / config["training_run"]
    summary = read_json(training / "training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["selected_update"] != 30000
        or summary["test_used"]
    ):
        raise ValueError("unexpected training snapshot")
    checked(training / "training-freeze.json", summary["freeze_sha256"])
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        checked(training / name, summary["outputs_sha256"][name])
        pin(files, training / name)
    manifest = training / "training-segments.jsonl"
    training_hashes = {
        r["source_sha256"] for r in fixed.old.previous.read_rows(manifest)
    }
    pin(files, manifest)
    pin(files, training / "training/summary.json")
    cache = {}
    inputs = {}
    for split in ("validation", "test"):
        ref_path = ROOT / config["reference_inputs"] / f"{split}-inputs.json"
        original = read_json(ref_path)
        pin(files, ref_path)
        if set(original["speakers"]) != set(labels[split]) or set(labels[split]) & set(
            labels["train"]
        ):
            raise ValueError("speaker split overlap")
        result = {
            "split": split,
            "speakers": original["speakers"],
            "variance": [],
            "enrollment": [],
            "queries": [],
        }

        def get(source):
            if source not in cache:
                cache[source] = candidates(
                    source, metadata, config["phones"], files, qc
                )
            return cache[source]

        for item in original["enrollment"]:
            sources = sorted({r["source_file"] for r in item["segments"]})
            pool = [r for s in sources for r in get(s)["verification"]]
            for p in config["phones"]:
                result["enrollment"] += choose(
                    [r for r in pool if r["vowel"] == p], 5, config["selection_seed"]
                )
        for s in result["speakers"]:
            for role in ROLES:
                queries = [
                    q
                    for q in original["queries"]
                    if q["speaker_id"] == s and q["role"] == role
                ]
                pools = {
                    purpose: [
                        r for q in queries for r in get(q["source_file"])[purpose]
                    ]
                    for purpose in ("variance", "verification")
                }
                for p in config["phones"]:
                    if role == "verification":
                        result["variance"] += choose(
                            [r for r in pools["variance"] if r["vowel"] == p],
                            12,
                            config["selection_seed"],
                            True,
                        )
                    result["queries"] += choose(
                        [r for r in pools["verification"] if r["vowel"] == p],
                        10 if role == "verification" else 3,
                        config["selection_seed"],
                        True,
                    )
        validate_inputs(result, config, training_hashes)
        inputs[split] = result
    for path in BASE.rglob("*"):
        if (
            path.is_file()
            and path.suffix in (".py", ".json")
            and "__pycache__" not in path.parts
        ):
            pin(files, path)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.is_file()
                and path.is_relative_to(ROOT)
                and path.suffix == ".py"
                and ".venv" not in path.parts
            ):
                pin(files, path)
    pin(files, ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json")
    run.mkdir(parents=True)
    for split, data in inputs.items():
        write_json(run / f"{split}-inputs.json", data)
        pin(files, run / f"{split}-inputs.json")
    write_json(
        run / "design-freeze.json",
        {
            "status": "frozen_before_inference",
            "config": config,
            "files": files,
            "qc": dict(qc),
            "runtime": fixed.old.previous.runtime(),
        },
    )
    print(
        json.dumps(
            {
                s: {k: len(v) for k, v in d.items() if isinstance(v, list)}
                for s, d in inputs.items()
            }
        ),
        flush=True,
    )


def verify(run):
    freeze = read_json(run / "design-freeze.json")
    if (
        freeze["config"] != read_json(CONFIG)
        or freeze["runtime"] != fixed.old.previous.runtime()
    ):
        raise ValueError("protocol or runtime changed")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)


def infer(config, run, split):
    verify(run)
    data = read_json(run / f"{split}-inputs.json")
    output = run / split
    if output.exists():
        raise ValueError("inference already performed")
    training = ROOT / config["training_run"]
    bundle = torch.load(
        training / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if list(bundle["phonemes"]) != config["phones"] or bundle["checkpoint_update"] != 30000:
        raise ValueError("wrong model inventory/update")
    torch.set_num_threads(1)
    model = fixed.create_encoder("statistics_mlp").eval().requires_grad_(False)
    model.load_state_dict(bundle["model"], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    settings = read_json(
        ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json"
    )
    pipeline = InputPipeline(
        settings["input"],
        rms_enabled=False,
        statistics=read_json(training / "bundle/feature-statistics.json"),
    )
    rows = sorted(
        {r["segment_id"]: r for r in all_rows(data)}.values(),
        key=lambda r: r["segment_id"],
    )
    segments = [
        Segment(
            segment_id=r["segment_id"],
            speaker_id=r["speaker_id"],
            vowel=r["vowel"],
            split=split,
            role="verification",
            cohorts=(),
            source_file=r["source_file"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            source_sha256=r["source_sha256"],
        )
        for r in rows
    ]
    dataset = SegmentDataset(ROOT, segments, pipeline, mode="center")
    vectors = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(segments), 64):
            batch = collate_segments(
                [dataset[i] for i in range(offset, min(offset + 64, len(segments)))]
            )
            embeddings = model(batch["input"], batch["mask"])
            if not torch.isfinite(embeddings).all() or not torch.equal(
                embeddings, model(batch["input"], batch["mask"])
            ):
                raise ValueError("nonfinite or nonrepeatable inference")
            values = embeddings.numpy().astype(np.float64)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            vectors.update(zip(batch["segment_ids"], values, strict=True))
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items()):
        raise ValueError("encoder mutated")
    profiles = {}
    for s in data["speakers"]:
        for p in config["phones"]:
            rows = [
                r for r in data["enrollment"] if (r["speaker_id"], r["vowel"]) == (s, p)
            ]
            mean = np.mean([vectors[r["segment_id"]] for r in rows], axis=0)
            profiles[s, p] = mean / np.linalg.norm(mean)
    scores = []
    for q in data["queries"]:
        for s in data["speakers"]:
            scores.append(
                {
                    "query_id": q["segment_id"],
                    "speaker_id": q["speaker_id"],
                    "claimed_speaker_id": s,
                    "condition": q["vowel"],
                    "role": q["role"],
                    "split": split,
                    "status": "scored",
                    "is_genuine": s == q["speaker_id"],
                    "score": float(vectors[q["segment_id"]] @ profiles[s, q["vowel"]]),
                }
            )
    output.mkdir()
    np.savez_compressed(output / "embedding-vectors.npz", **vectors)
    with (output / "scores.jsonl").open("w") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        output / "inference.json",
        {
            "unique_tokens": len(vectors),
            "score_rows": len(scores),
            "elapsed_sec": time.perf_counter() - start,
            "bitwise_repeat_equal": True,
            "encoder_unchanged": True,
            "encoder_sha256": sha256_file(training / "bundle/encoder.pt"),
        },
    )
    return data, vectors, scores


def variance(config, data, vectors, bootstrap=False):
    components, blocks = {}, {}
    for p in config["phones"]:
        rows = [r for r in data["variance"] if r["vowel"] == p]
        y = np.stack([vectors[r["segment_id"]] for r in rows])
        blocks[p], dimensions = dm.sufficient_statistics(
            dm.design(rows), y, [r["speaker_id"] for r in rows], data["speakers"]
        )
        components[p] = dm.fit_components(blocks[p], dimensions)
    groups = dm.cluster({p: c["R"] for p, c in components.items()})
    result = {
        "components": components,
        "groups": groups,
        "nuisance": config["variance_covariates"],
        "interpretation": "stable speaker dependence in this fixed encoder; anatomy and learned articulation not separated",
        "session_ids_available": False,
        "recording_conditions_not_identifiable_separately_from_speaker": True,
    }
    if not bootstrap:
        return result, {}
    draws, _ = fixed.old.speaker_draws(
        len(data["speakers"]),
        config["grouping_bootstrap_replicates"],
        config["grouping_bootstrap_seed"],
    )
    rs = np.zeros((len(draws), len(config["phones"])))
    high = np.zeros_like(rs, dtype=bool)
    for index, draw in enumerate(draws):
        for j, p in enumerate(config["phones"]):
            rs[index, j] = dm.fit_components(blocks[p], dimensions, draw)["R"]
        clustered = dm.cluster(dict(zip(config["phones"], rs[index], strict=True)))
        high[index] = [p in clustered["high"] for p in config["phones"]]
        if (index + 1) % 100 == 0:
            print(f"variance bootstrap {index + 1}/{len(draws)}", flush=True)
    for j, p in enumerate(config["phones"]):
        components[p]["ci95"] = fixed.old.interval(rs[:, j])
        components[p]["bootstrap_high_membership_fraction"] = float(high[:, j].mean())
    return result, {"speaker_draws": draws, "R": rs, "high": high}


def score_cells(config, data, scores, thresholds):
    cells, selected = {}, {}
    for p in config["phones"]:
        for role in ROLES:
            key = f"{p}/{role}"
            rows = [r for r in scores if (r["condition"], r["role"]) == (p, role)]
            cells[key], _ = fixed.old.evaluate_cell(
                rows, thresholds[p], data["speakers"]
            )
            selected[key] = rows
    return cells, selected


def validation(config, run):
    if (run / "group-freeze.json").exists():
        raise ValueError("validation grouping already frozen")
    data, vectors, scores = infer(config, run, "validation")
    result, arrays = variance(config, data, vectors, True)
    thresholds = {
        p: fixed.old.calibrate_cell(
            [r for r in scores if r["condition"] == p and r["role"] == "verification"],
            data["speakers"],
        )
        for p in config["phones"]
    }
    cells, _ = score_cells(config, data, scores, thresholds)
    write_json(run / "validation-components.json", result)
    write_json(run / "validation-thresholds.json", thresholds)
    write_json(run / "validation-metrics.json", cells)
    np.savez_compressed(run / "grouping-bootstrap.npz", **arrays)
    verify(run)
    files = {}
    for p in [
        run / "design-freeze.json",
        run / "validation-components.json",
        run / "validation-thresholds.json",
        run / "grouping-bootstrap.npz",
        run / "validation/embedding-vectors.npz",
        run / "validation/scores.jsonl",
        run / "validation-metrics.json",
    ]:
        pin(files, p)
    write_json(
        run / "group-freeze.json",
        {
            "status": "groups_and_thresholds_frozen_before_test_inference",
            "groups": result["groups"],
            "files": files,
        },
    )
    print("frozen groups: " + json.dumps(result["groups"]), flush=True)


def group_summary(cells, groups, role, field="pooled_eer"):
    def value(p):
        c = cells[f"{p}/{role}"]
        if field == "pooled_eer":
            return c[field]
        point, metric = field.split("/")
        return c["operating_points"][point][metric]

    means = {g: float(np.mean([value(p) for p in groups[g]])) for g in ("low", "high")}
    return {**means, "high_minus_low": means["high"] - means["low"]}


def test(config, run):
    freeze = read_json(run / "group-freeze.json")
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    data, vectors, scores = infer(config, run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    cells, rows = score_cells(config, data, scores, thresholds)
    components, _ = variance(config, data, vectors)
    groups = freeze["groups"]
    draws, counts = fixed.old.speaker_draws(
        len(data["speakers"]),
        config["verification_bootstrap_replicates"],
        config["verification_bootstrap_seed"],
    )
    replicas, bootstrap_cis = {}, {}
    for key, cell in cells.items():
        p = key.split("/")[0]
        bootstrap_cis[key], replicas[key] = fixed.old.bootstrap_cell(
            rows[key], cell, thresholds[p], data["speakers"], counts
        )
        print("test bootstrap " + key, flush=True)
    summary, arrays = {}, {"speaker_draws": draws, "speaker_counts": counts}
    fields = [
        "pooled_eer",
        "far_1pct/all_input_far",
        "far_1pct/all_input_frr",
        "far_0_1pct/all_input_far",
        "far_0_1pct/all_input_frr",
    ]
    for role in ROLES:
        for field in fields:
            key = f"{role}/{field}"
            item = group_summary(cells, groups, role, field)
            means = {
                g: np.mean([replicas[f"{p}/{role}"][field] for p in groups[g]], axis=0)
                for g in ("low", "high")
            }
            delta = means["high"] - means["low"]
            item["ci95_high_minus_low"] = fixed.old.interval(delta)
            summary[key] = item
            arrays[key] = delta
    for key, value in replicas.items():
        for field, a in value.items():
            arrays[f"phone/{key}/{field}"] = a
    val = read_json(run / "validation-components.json")["components"]
    rs = [val[p]["R"] for p in config["phones"]]
    eer = [cells[f"{p}/verification"]["pooled_eer"] for p in config["phones"]]
    primary = summary["verification/pooled_eer"]
    write_json(
        run / "test-results.json",
        {
            "cells": cells,
            "cell_bootstrap_ci": bootstrap_cis,
            "test_components": components,
            "group_summary": summary,
            "groups": groups,
            "validation_R_vs_test_EER_spearman": dm.rank_correlation(rs, eer),
            "validation_R_vs_test_R_spearman": dm.rank_correlation(
                rs, [components["components"][p]["R"] for p in config["phones"]]
            ),
            "supports_lower_normal_macro_EER_in_high_group": primary[
                "ci95_high_minus_low"
            ]["upper"]
            < 0,
            "bootstrap_ci_scope": "test speakers conditional on validation-frozen group and thresholds; exploratory previously observed JVS",
            "not_group_fusion": "macro average of per-phone EER; unequal group sizes; different utterances matched count and time",
        },
    )
    np.savez_compressed(run / "test-bootstrap.npz", **arrays)
    verify(run)
    for name, checksum in freeze["files"].items():
        checked(ROOT / name, checksum)
    print(
        json.dumps(
            {
                "primary": primary,
                "supports": primary["ci95_high_minus_low"]["upper"] < 0,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "validation", "test"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    {"prepare": prepare, "validation": validation, "test": test}[args.stage](
        config, run
    )


if __name__ == "__main__":
    main()
