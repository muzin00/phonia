"""Recompute raw PCM, trial scores, error rates, EER and diagnostics independently."""

import csv
import hashlib
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eer(rows):
    levels = defaultdict(lambda: [0, 0])
    for row in rows:
        levels[row["score"]][0 if row["is_genuine"] else 1] += 1
    ng = sum(r["is_genuine"] for r in rows)
    ni = len(rows) - ng
    accepted_g = accepted_i = 0
    previous_far, previous_frr = 0.0, 1.0
    for g, i in (counts for _, counts in sorted(levels.items(), reverse=True)):
        accepted_g += g
        accepted_i += i
        far, frr = accepted_i / ni, (ng - accepted_g) / ng
        if far >= frr:
            old_delta = previous_far - previous_frr
            fraction = -old_delta / ((far - frr) - old_delta)
            return previous_far + fraction * (far - previous_far)
        previous_far, previous_frr = far, frr
    raise AssertionError("no EER crossing")


def check_distribution(values, measured):
    values = np.asarray(values)
    assert measured["count"] == len(values)
    assert abs(values.mean() - measured["mean"]) < 1e-12
    assert abs(values.std() - measured["std"]) < 1e-12
    assert np.allclose(
        np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99]),
        [
            measured["quantiles"][name]
            for name in ("p01", "p25", "median", "p75", "p99")
        ],
        rtol=0,
        atol=1e-12,
    )
    if "histogram_counts" in measured:
        counts, edges = np.histogram(values, bins=np.linspace(-1, 1, 81))
        assert counts.tolist() == measured["histogram_counts"]
        assert edges.tolist() == measured["histogram_edges"]


def check_separation(rows, measured):
    g = np.array([r["score"] for r in rows if r["is_genuine"]])
    i = np.array([r["score"] for r in rows if not r["is_genuine"]])
    check_distribution(g, measured["genuine"])
    check_distribution(i, measured["impostor"])
    gap = g.mean() - i.mean()
    assert abs(gap - measured["mean_score_gap"]) < 1e-12
    assert abs(gap / np.sqrt((g.var() + i.var()) / 2) - measured["dprime"]) < 1e-12


def main():
    config = read(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    report = read(BASE / "evaluation-results.json")
    assert report["protocol"] == config
    thresholds = read(run / "validation-thresholds.json")
    raw_trials = rate_checks = eer_checks = speaker_checks = slice_checks = 0
    raw_cache = {}
    training_hashes, training_speakers = set(), set()
    training = ROOT / config["training_run"]
    with (training / "training-segments.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            training_hashes.add(row["source_sha256"])
            training_speakers.add(row["speaker_id"])
    for split in ("validation", "test"):
        inputs = read(run / f"{split}-inputs.json")
        assert not set(inputs["speakers"]) & training_speakers
        queries = {q["query_id"]: q for q in inputs["queries"]}
        support = read(ROOT / config["source_support"] / f"{split}-inputs.json")
        assert set(queries) <= {q["query_id"] for q in support["queries"]}
        enrollment_hashes = {
            r["source_sha256"]
            for item in inputs["profiles"].values()
            for rows in item["conditions"].values()
            for r in rows
        }
        for item in [*inputs["profiles"].values(), *inputs["queries"]]:
            count = 1 if "query_id" in item else 10
            assert set(item["conditions"]) == set(config["phonemes"])
            assert item["budget_frames"] == count * 1200
            for phone, rows in item["conditions"].items():
                assert len(rows) == count
                assert len({r["original_segment_id"] for r in rows}) == count
                for row in rows:
                    assert (
                        row["vowel"] == phone
                        and row["source_sha256"] not in training_hashes
                    )
                    if "query_id" in item:
                        assert row["source_sha256"] not in enrollment_hashes
                    utterance, suffix = row["original_segment_id"].split("--phone-")
                    path = (
                        ROOT
                        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
                        / utterance.split("_")[0]
                        / f"{utterance}.json"
                    )
                    if path not in raw_cache:
                        raw_cache[path] = read(path)
                    raw = raw_cache[path]
                    interval = raw["intervals"][int(suffix.split("-")[0])]
                    first, last = (
                        round(interval["start_sec"] * 24000),
                        round(interval["end_sec"] * 24000),
                    )
                    assert interval["phoneme"] == phone
                    assert (
                        row["source_file"] == raw["source_file"]
                        and row["source_sha256"] == raw["source_sha256"]
                    )
                    assert (
                        row["original_start_frame"] == first
                        and row["original_end_frame"] == last
                    )
                    assert row["start_frame"] == first + (last - first - 1200) // 2
                    assert (
                        row["end_frame"] - row["start_frame"] == 1200
                        and first <= row["start_frame"] < row["end_frame"] <= last
                    )
                    with wave.open(str(ROOT / row["source_file"]), "rb") as audio:
                        assert (
                            audio.getframerate(),
                            audio.getnchannels(),
                            audio.getsampwidth(),
                            audio.getcomptype(),
                        ) == (24000, 1, 2, "NONE")
                        assert row["end_frame"] <= audio.getnframes()
                        audio.setpos(row["start_frame"])
                        x = (
                            np.frombuffer(audio.readframes(1200), dtype="<i2").astype(
                                float
                            )
                            / 32768
                        )
                    dbfs = 20 * np.log10(np.sqrt(np.mean(x * x)))
                    assert dbfs >= -50 and abs(dbfs - row["selected_rms_dbfs"]) < 1e-12
                    slice_checks += 1
        with (run / split / "scores.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        raw_trials += len(rows)
        measured = read(run / f"{split}-metrics.json")
        assert report[split] == measured
        matrix, groups = set(), defaultdict(list)
        with (
            np.load(run / split / "query-vectors.npz", allow_pickle=False) as vectors,
            np.load(
                run / split / "profile-vectors.npz", allow_pickle=False
            ) as profiles,
        ):
            for row in rows:
                q = queries[row["query_id"]]
                key = row["query_id"], row["condition"], row["claimed_speaker_id"]
                assert key not in matrix
                matrix.add(key)
                assert (
                    row["condition"] in config["phonemes"]
                    and row["claimed_speaker_id"] in inputs["speakers"]
                )
                assert row["is_genuine"] == (
                    q["speaker_id"] == row["claimed_speaker_id"]
                )
                assert all(
                    row[k] == q[k] for k in ("speaker_id", "split", "role", "common8")
                )
                assert row["status"] == "scored" and row["used_frames"] == 1200
                assert row["phone_scores"] == {row["condition"]: row["score"]}
                expected = float(
                    vectors[f"{row['condition']}/{row['query_id']}"]
                    @ profiles[f"{row['condition']}/{row['claimed_speaker_id']}"]
                )
                assert expected == row["score"]
                groups[row["condition"], row["role"]].append(row)
            assert len(matrix) == len(queries) * len(inputs["speakers"]) * 8
            for (phone, role), subset in groups.items():
                cell = measured["conditions"][f"{phone}/{role}"]
                genuine = [r for r in subset if r["is_genuine"]]
                impostor = [r for r in subset if not r["is_genuine"]]
                for point, threshold in thresholds[phone]["operating_points"].items():
                    t = float(threshold["threshold"])
                    rates = cell["operating_points"][point]
                    fa = sum(r["score"] >= t for r in impostor)
                    fr = sum(r["score"] < t for r in genuine)
                    assert (rates["false_accepts"], rates["false_rejects"]) == (fa, fr)
                    assert rates["all_input_far"] == fa / len(impostor) and rates[
                        "all_input_frr"
                    ] == fr / len(genuine)
                    rate_checks += 1
                assert abs(eer(subset) - cell["pooled_eer"]) < 1e-12
                eer_checks += 1
                d = cell["diagnostics"]
                check_separation(subset, d["score_separation"])
                qrows = [q for q in inputs["queries"] if q["role"] == role]
                samples = [
                    np.stack(
                        [
                            vectors[f"{phone}/{q['query_id']}"]
                            for q in qrows
                            if q["speaker_id"] == speaker
                        ]
                    )
                    for speaker in inputs["speakers"]
                ]
                centers = np.stack([x.mean(0) for x in samples])
                center = centers.mean(0)
                between = np.mean(np.sum((centers - center) ** 2, axis=1))
                within = np.mean(
                    [
                        np.mean(np.sum((x - c) ** 2, axis=1))
                        for x, c in zip(samples, centers)
                    ]
                )
                scatter = d["embedding_scatter"]
                assert (
                    abs(between - scatter["speaker_balanced_between_scatter"]) < 1e-12
                )
                assert abs(within - scatter["speaker_balanced_within_scatter"]) < 1e-12
                assert abs(between / within - scatter["between_within_ratio"]) < 1e-12
                for speaker, values in d["claimed_speaker_diagnostics"].items():
                    claimed = [r for r in subset if r["claimed_speaker_id"] == speaker]
                    assert abs(eer(claimed) - values["eer"]) < 1e-12
                    check_separation(claimed, values["separation"])
                    speaker_checks += 1
        if split == "validation":
            for phone in config["phonemes"]:
                scores = sorted(
                    r["score"]
                    for r in groups[phone, "verification"]
                    if not r["is_genuine"]
                )
                for point, target in (("far_1pct", 0.01), ("far_0_1pct", 0.001)):
                    expected = float(
                        np.nextafter(
                            scores[len(scores) - int(target * len(scores)) - 1], np.inf
                        )
                    )
                    assert (
                        thresholds[phone]["operating_points"][point]["threshold"]
                        == expected
                    )
        for name, expected in read(run / split / "inference.json")[
            "outputs_sha256"
        ].items():
            assert sha(run / split / name) == expected
    test = read(run / "test-metrics.json")
    with (
        np.load(run / "bootstrap-metrics.npz", allow_pickle=False) as replicas,
        np.load(run / "bootstrap-differences.npz", allow_pickle=False) as arrays,
    ):
        assert len(arrays.files) == 168 and set(arrays.files) == set(
            test["paired_differences"]
        )
        for key, d in test["paired_differences"].items():
            _, role, metric = key.split("/", 2)
            delta = 100 * (
                replicas[f"{d['minuend']}/{role}/{metric}"]
                - replicas[f"{d['subtrahend']}/{role}/{metric}"]
            )
            assert np.array_equal(delta, arrays[key], equal_nan=True)
            bounds = np.quantile(delta[np.isfinite(delta)], [0.025, 0.975])
            ci = d["ci95_percentage_points"]
            assert np.allclose(bounds, [ci["lower"], ci["upper"]], rtol=0, atol=1e-12)
        primary = test["primary_hypothesis"]
        for phone, ref in config["primary_contrasts"]:
            d = primary["contrasts"][f"{phone}_minus_{ref}"]
            delta = 100 * (
                replicas[f"{phone}/verification/pooled_eer"]
                - replicas[f"{ref}/verification/pooled_eer"]
            )
            low, high = np.quantile(delta[np.isfinite(delta)], [0.0125, 0.9875])
            assert d["ci975_bonferroni_percentage_points"] == {
                "lower": low,
                "upper": high,
            }
            assert d["supports_lower_eer_than_s"] == (high < 0)
            assert d["supports_higher_eer_than_s"] == (low > 0)
        assert primary["both_m_and_n_better_than_s"] == all(
            d["supports_lower_eer_than_s"] for d in primary["contrasts"].values()
        )
    with (BASE / "evaluation-cells.csv").open() as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == 96
    for row in cells:
        cell = report[row["split"]]["conditions"][f"{row['phone']}/{row['role']}"]
        assert float(row["eer"]) == cell["pooled_eer"]
        for metric in (
            "all_input_far",
            "all_input_frr",
            "false_accepts",
            "false_rejects",
        ):
            assert (
                float(row[metric])
                == cell["operating_points"][row["operating_point"]][metric]
            )
    for name, expected in read(run / "design-freeze.json")["files"].items():
        assert sha(ROOT / name) == expected
    for name, expected in read(run / "evaluation-freeze.json")["files"].items():
        assert sha(run / name) == expected
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    assert counts.shape == (10000, 15) and np.all(counts.sum(1) == 15)
    summary = read(training / "training/summary.json")
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        assert sha(training / name) == summary["outputs_sha256"][name]
    result = {
        "status": "passed",
        "raw_trials": raw_trials,
        "selected_pcm_slices": slice_checks,
        "rate_cells_recomputed": rate_checks,
        "eer_cells_recomputed": eer_checks,
        "speaker_diagnostic_cells": speaker_checks,
        "validation_far_thresholds_recomputed": 16,
        "paired_arrays_checked": 168,
        "primary_adjusted_intervals_checked": 2,
        "csv_cells_checked": 96,
        "test_threshold_recalibration": False,
    }
    with (run / "independent-audit.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
