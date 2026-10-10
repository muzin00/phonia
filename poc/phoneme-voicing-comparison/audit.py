"""Independently check PCM plans, score fusion, ROC and bootstrap arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def close(left, right):
    if not math.isclose(float(left), float(right), abs_tol=1e-12, rel_tol=1e-12):
        raise AssertionError((left, right))


def input_audit(config):
    training = list(rows(ROOT / config["training_run"] / "training-segments.jsonl"))
    train_speakers = {r["speaker_id"] for r in training}
    train_hashes = {r["source_sha256"] for r in training}
    assert len(train_speakers) == 140
    assert all(
        r["split"] == "train"
        and r["evaluation_role"] == "training"
        and not r["quality_flags"]
        for r in training
    )
    assert {r["normalized_phoneme"] for r in training} == set(config["phonemes"])
    del training
    totals, wav_frames, raw_cache, verified_sources = Counter(), {}, {}, set()
    for diagnostic, run in (
        (False, ROOT / config["run_directory"]),
        (True, ROOT / config["diagnostics"]["run_directory"]),
    ):
        freeze = read(run / "input-freeze.json")
        assert freeze["config"] == config
        for name, checksum in freeze["files"].items():
            assert digest(ROOT / name) == checksum, name
        for split in ("validation", "test"):
            data = read(run / f"{split}-inputs.json")
            assert not set(data["speakers"]) & train_speakers
            reference = read(ROOT / config["reference_inputs"] / f"{split}-inputs.json")
            originals = {
                r["segment_id"]: r
                for item in [*reference["enrollment"], *reference["queries"]]
                for r in item["segments"]
            }
            items = data["studies"] if diagnostic else {"main": data}
            for item in items.values():
                enrollment_hashes = {
                    r["source_sha256"]
                    for p in item["profiles"].values()
                    for selected in p["conditions"].values()
                    for r in selected
                }
                for role in ("verification", "cross_text_verification"):
                    assert {
                        q["speaker_id"] for q in item["queries"] if q["role"] == role
                    } == set(data["speakers"])
                for sample in [*item["profiles"].values(), *item["queries"]]:
                    for condition, selected in sample["conditions"].items():
                        expected_phones = (
                            {condition}
                            if diagnostic
                            else set(config["condition_phones"][condition])
                        )
                        assert {r["vowel"] for r in selected} == expected_phones
                        assert (
                            sum(r["end_frame"] - r["start_frame"] for r in selected)
                            == sample["budget_frames"]
                        )
                        assert len({r["original_segment_id"] for r in selected}) == len(
                            selected
                        )
                        ranges = defaultdict(list)
                        for row in selected:
                            assert row["source_sha256"] not in train_hashes
                            if "query_id" in sample:
                                assert row["source_sha256"] not in enrollment_hashes
                            first, last = row["start_frame"], row["end_frame"]
                            assert 720 <= last - first <= 6000
                            origin = originals.get(row["original_segment_id"])
                            if origin is None:
                                utterance, suffix = row["original_segment_id"].split(
                                    "--phone-"
                                )
                                raw_path = (
                                    ROOT
                                    / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
                                    / utterance.split("_")[0]
                                    / f"{utterance}.json"
                                )
                                if raw_path not in raw_cache:
                                    raw_cache[raw_path] = read(raw_path)
                                raw = raw_cache[raw_path]
                                interval = raw["intervals"][int(suffix.split("-")[0])]
                                start, end = (
                                    round(interval["start_sec"] * 24000),
                                    round(interval["end_sec"] * 24000),
                                )
                                if interval["phoneme"] not in ("m", "n"):
                                    length = min(6000, end - start)
                                    start += (end - start - length) // 2
                                    end = start + length
                                origin = {
                                    "vowel": interval["phoneme"],
                                    "source_file": raw["source_file"],
                                    "source_sha256": raw["source_sha256"],
                                    "start_frame": start,
                                    "end_frame": end,
                                }
                            assert all(
                                row[k] == origin[k]
                                for k in ("vowel", "source_file", "source_sha256")
                            )
                            expected = (
                                origin["end_frame"] - (last - first)
                                if row["vowel"] in ("t", "d", "k", "g")
                                else origin["start_frame"]
                                + (
                                    origin["end_frame"]
                                    - origin["start_frame"]
                                    - (last - first)
                                )
                                // 2
                            )
                            assert first == expected and last <= origin["end_frame"]
                            source = ROOT / row["source_file"]
                            if source not in verified_sources:
                                assert digest(source) == row["source_sha256"]
                                verified_sources.add(source)
                            with wave.open(str(source), "rb") as audio:
                                if source not in wav_frames:
                                    assert (
                                        audio.getframerate(),
                                        audio.getnchannels(),
                                        audio.getsampwidth(),
                                        audio.getcomptype(),
                                    ) == (24000, 1, 2, "NONE")
                                    wav_frames[source] = audio.getnframes()
                                assert 0 <= first < last <= wav_frames[source]
                                audio.setpos(first)
                                pcm = (
                                    np.frombuffer(
                                        audio.readframes(last - first), dtype="<i2"
                                    ).astype(np.float64)
                                    / 32768
                                )
                            assert len(pcm) == last - first and math.sqrt(
                                float(np.mean(pcm * pcm))
                            ) >= 10 ** (-50 / 20)
                            ranges[row["source_file"]].append((first, last))
                            totals["source_slices"] += 1
                        for selected_ranges in ranges.values():
                            ordered = sorted(selected_ranges)
                            assert all(
                                a[1] <= b[0] for a, b in pairwise(ordered)
                            )
                        totals["matched_plans"] += 1
                    if diagnostic:
                        lengths = [
                            [r["end_frame"] - r["start_frame"] for r in selected]
                            for selected in sample["conditions"].values()
                        ]
                        assert lengths[0] == lengths[1]
                totals["queries"] += len(item["queries"])
        write(
            run / "input-independent-audit.json",
            {
                "status": "passed",
                "checks": dict(totals),
                "training_speakers": 140,
                "training_audio_overlap": False,
                "post_plan_rms_passed": True,
            },
        )
    return {**dict(totals), "unique_source_wavs": len(verified_sources)}


def eer(values, genuine, weights=None):
    order = np.argsort(-values, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(values) - 1]
    weights = np.ones(len(values), dtype=np.int64) if weights is None else weights
    positive = weights * genuine
    negative = weights * (~genuine)
    far = np.r_[0.0, np.cumsum(negative[order])[ends] / negative.sum()]
    frr = np.r_[
        1.0, (positive.sum() - np.cumsum(positive[order])[ends]) / positive.sum()
    ]
    gap = far - frr
    right = int(np.flatnonzero(gap >= 0)[0])
    if gap[right] == 0:
        return float(far[right])
    fraction = -gap[right - 1] / (gap[right] - gap[right - 1])
    return float(far[right - 1] + fraction * (far[right] - far[right - 1]))


def score_audit(data, records, vectors, diagnostic):
    items = data["studies"] if diagnostic else {"main": data}
    expected = {}
    for key, item in items.items():
        profiles = {}
        for speaker, profile in item["profiles"].items():
            for condition, selected in profile["conditions"].items():
                for phone in {r["vowel"] for r in selected}:
                    mean = np.mean(
                        [
                            vectors[r["segment_id"]]
                            for r in selected
                            if r["vowel"] == phone
                        ],
                        axis=0,
                    )
                    profiles[speaker, condition, phone] = mean / np.linalg.norm(mean)
        for query in item["queries"]:
            for condition, selected in query["conditions"].items():
                phone_vectors = {
                    phone: np.mean(
                        [
                            vectors[r["segment_id"]]
                            for r in selected
                            if r["vowel"] == phone
                        ],
                        axis=0,
                    )
                    for phone in {r["vowel"] for r in selected}
                }
                for claimed in data["speakers"]:
                    components = {
                        phone: float(vector @ profiles[claimed, condition, phone])
                        for phone, vector in phone_vectors.items()
                    }
                    expected[key, query["query_id"], condition, claimed] = (
                        query,
                        components,
                    )
    seen = set()
    for row in records:
        key = (
            row["study"] if diagnostic else "main",
            row["query_id"],
            row["condition"],
            row["claimed_speaker_id"],
        )
        assert key in expected and key not in seen
        seen.add(key)
        query, components = expected[key]
        assert all(row[name] == query[name] for name in ("split", "role", "speaker_id"))
        assert (
            row["is_genuine"] == (row["speaker_id"] == row["claimed_speaker_id"])
            and row["status"] == "scored"
        )
        assert row["used_frames"] == query["budget_frames"]
        close(row["score"], np.mean(list(components.values())))
        if not diagnostic:
            assert set(row["phone_scores"]) == set(components)
            for phone, value in components.items():
                close(row["phone_scores"][phone], value)
    assert seen == set(expected)
    return len(records)


def numerical_audit(config):
    totals = Counter()
    for diagnostic, run in (
        (False, ROOT / config["run_directory"]),
        (True, ROOT / config["diagnostics"]["run_directory"]),
    ):
        thresholds = read(run / "validation-thresholds.json")
        counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
        metrics = {
            split: read(run / f"{split}-metrics.json")["conditions"]
            for split in ("validation", "test")
        }
        bootstrap = np.load(run / "bootstrap-metrics.npz", allow_pickle=False)
        for split in ("validation", "test"):
            data = read(run / f"{split}-inputs.json")
            records = list(rows(run / split / "scores.jsonl"))
            inference = read(run / split / "inference.json")
            assert digest(run / split / "scores.jsonl") == inference["scores_sha256"]
            assert (
                digest(run / split / "embedding-vectors.npz")
                == inference["embedding_vectors_sha256"]
            )
            with np.load(
                run / split / "embedding-vectors.npz", allow_pickle=False
            ) as saved:
                vectors = dict(
                    zip(saved["segment_ids"].tolist(), saved["vectors"], strict=True)
                )
            assert all(
                np.isfinite(vector).all()
                and math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-12)
                for vector in vectors.values()
            )
            totals["independent_score_fusion_checks"] += score_audit(
                data, records, vectors, diagnostic
            )
            for key, cell in metrics[split].items():
                prefix, role = key.rsplit("/", 1)
                if diagnostic:
                    pair, condition = prefix.split("/")
                    selected = [
                        r
                        for r in records
                        if r["study"] == pair
                        and r["condition"] == condition
                        and r["role"] == role
                    ]
                else:
                    selected = [
                        r
                        for r in records
                        if r["condition"] == prefix and r["role"] == role
                    ]
                values = np.array([r["score"] for r in selected])
                genuine = np.array([r["is_genuine"] for r in selected], dtype=bool)
                close(eer(values, genuine), cell["pooled_eer"])
                assert (
                    cell["queries"] == int(genuine.sum())
                    and cell["query_coverage"] == 1
                )
                totals["eer_checks"] += 1
                for point, rate in cell["operating_points"].items():
                    threshold = float(
                        thresholds[prefix]["operating_points"][point]["threshold"]
                    )
                    accepted = values >= threshold
                    fa = int(np.count_nonzero(accepted & ~genuine))
                    fr = int(np.count_nonzero(~accepted & genuine))
                    assert (
                        rate["false_accepts"],
                        rate["false_rejects"],
                        rate["genuine"],
                        rate["impostor"],
                    ) == (fa, fr, int(genuine.sum()), int((~genuine).sum()))
                    for metric, value in (
                        ("far", fa / (~genuine).sum()),
                        ("all_input_far", fa / (~genuine).sum()),
                        ("frr", fr / genuine.sum()),
                        ("all_input_frr", fr / genuine.sum()),
                    ):
                        close(rate[metric], value)
                        totals["rate_checks"] += 1
                    if (
                        split == "validation"
                        and role == "verification"
                        and point in ("far_1pct", "far_0_1pct")
                    ):
                        target = 0.01 if point == "far_1pct" else 0.001
                        impostors = np.sort(values[~genuine])
                        expected = np.nextafter(
                            impostors[
                                len(impostors) - math.floor(target * len(impostors)) - 1
                            ],
                            np.inf,
                        )
                        assert threshold == expected
                        totals["validation_far_threshold_checks"] += 1
                if split == "test":
                    lookup = {
                        speaker: index for index, speaker in enumerate(data["speakers"])
                    }
                    qi = np.array([lookup[r["speaker_id"]] for r in selected])
                    ci = np.array([lookup[r["claimed_speaker_id"]] for r in selected])
                    for index, draw in enumerate(counts[:64]):
                        weights = np.where(genuine, draw[qi], draw[qi] * draw[ci])
                        close(
                            eer(values, genuine, weights),
                            bootstrap[f"{key}/pooled_eer"][index],
                        )
                        totals["independent_bootstrap_eer_checks"] += 1
                        for point, rate in cell["operating_points"].items():
                            accepted = values >= float(rate["threshold"])
                            for metric, wrong, denominator in (
                                ("all_input_far", accepted & ~genuine, ~genuine),
                                ("all_input_frr", ~accepted & genuine, genuine),
                            ):
                                close(
                                    weights[wrong].sum() / weights[denominator].sum(),
                                    bootstrap[f"{key}/{point}/{metric}"][index],
                                )
                                totals["independent_bootstrap_rate_checks"] += 1
        test = read(run / "test-metrics.json")
        with np.load(
            run / "bootstrap-differences.npz", allow_pickle=False
        ) as differences:
            assert set(differences.files) == set(test["paired_differences"])
            for key, entry in test["paired_differences"].items():
                if diagnostic:
                    pair, contrast, role, metric = key.split("/", 3)
                    right, left = contrast.split("_minus_")
                    first, second = f"{pair}/{left}/{role}", f"{pair}/{right}/{role}"
                else:
                    _, role, metric = key.split("/", 2)
                    first, second = (
                        f"{entry['subtrahend']}/{role}",
                        f"{entry['minuend']}/{role}",
                    )
                expected = 100 * (
                    bootstrap[f"{second}/{metric}"] - bootstrap[f"{first}/{metric}"]
                )
                assert np.array_equal(differences[key], expected)
                interval = np.percentile(expected, [2.5, 97.5], method="linear")
                close(entry["ci95_percentage_points"]["lower"], interval[0])
                close(entry["ci95_percentage_points"]["upper"], interval[1])
                totals["paired_difference_checks"] += 1
            for key, entry in test.get("primary_contrasts", {}).items():
                interval = np.percentile(
                    differences[key], [1.25, 98.75], method="linear"
                )
                close(entry["ci_percentage_points"]["lower"], interval[0])
                close(entry["ci_percentage_points"]["upper"], interval[1])
                assert entry["voiced_lower_eer_supported"] == bool(interval[1] < 0)
                totals["primary_contrast_checks"] += 1
        bootstrap.close()
        write(
            run / "independent-numerical-audit.json",
            {
                "status": "passed",
                "checks": dict(totals),
                "independent_bootstrap_draws_per_cell": 64,
            },
        )
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "metrics"))
    args = parser.parse_args()
    config = read(BASE / "config/evaluation.json")
    result = input_audit(config) if args.stage == "inputs" else numerical_audit(config)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
