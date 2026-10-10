"""Independently recompute raw trial rates, EER, paired intervals and frozen inputs."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
base = Path(__file__).resolve().parent
config = json.loads((base / "config/protocol.json").read_text())
run = root / config["run_directory"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


thresholds = read(run / "validation-thresholds.json")
report = read(base / "evaluation-results.json")
checks = eer_checks = trial_rows = plan_checks = slice_checks = baseline_checks = 0
previous = root / config["prior_pair_run"]
for split in ("validation", "test"):
    inputs = read(run / f"{split}-inputs.json")
    queries = {q["query_id"]: q for q in inputs["queries"]}
    prior_inputs = read(previous / f"{split}-inputs.json")
    assert inputs["speakers"] == prior_inputs["speakers"]
    assert [q["query_id"] for q in inputs["queries"]] == [
        q["query_id"] for q in prior_inputs["queries"]
    ]
    old_items = list(prior_inputs["profiles"].values()) + prior_inputs["queries"]
    new_items = list(inputs["profiles"].values()) + inputs["queries"]
    enrollment_hashes = {
        r["source_sha256"]
        for item in inputs["profiles"].values()
        for r in item["conditions"]["vowels_mns"]
    }
    for old_item, item in zip(old_items, new_items):
        assert item["budget_frames"] == old_item["budget_frames"]
        assert set(item["conditions"]) == set(config["conditions"])
        for condition, rows in item["conditions"].items():
            if condition in old_item["conditions"]:
                assert rows == old_item["conditions"][condition]
            assert {r["vowel"] for r in rows} == set(
                config["condition_phones"][condition]
            )
            assert (
                sum(r["end_frame"] - r["start_frame"] for r in rows)
                == item["budget_frames"]
            )
            assert len({r["original_segment_id"] for r in rows}) == len(rows)
            for row in rows:
                assert 720 <= row["end_frame"] - row["start_frame"] <= 6000
                if "common8" in item:
                    assert row["source_sha256"] not in enrollment_hashes
                slice_checks += 1
            plan_checks += 1
    groups = defaultdict(list)
    with (run / split / "scores.jsonl").open() as stream:
        scores = [json.loads(line) for line in stream]
    trial_rows += len(scores)
    assert (
        sha(run / split / "scores.jsonl")
        == read(run / split / "inference.json")["scores_sha256"]
    )
    matrix = set()
    for row in scores:
        q = queries[row["query_id"]]
        key = row["query_id"], row["condition"], row["claimed_speaker_id"]
        assert key not in matrix
        matrix.add(key)
        assert row["is_genuine"] == (q["speaker_id"] == row["claimed_speaker_id"])
        assert row["claimed_speaker_id"] in inputs["speakers"]
        assert row["role"] == q["role"] and row["split"] == split
        assert row["common8"] and row["status"] == "scored"
        assert row["used_frames"] == q["budget_frames"]
        assert set(row["phone_scores"]) == set(
            config["condition_phones"][row["condition"]]
        )
        assert (
            abs(
                row["score"]
                - sum(row["phone_scores"].values()) / len(row["phone_scores"])
            )
            < 1e-15
        )
        groups[row["condition"], row["role"]].append(row)
    assert len(matrix) == len(queries) * len(inputs["speakers"]) * len(
        config["conditions"]
    )
    measured = read(run / f"{split}-metrics.json")
    for (condition, role), rows in groups.items():
        cell = measured["conditions"][f"{condition}/{role}"]
        assert cell == report[split]["conditions"][f"{condition}/{role}"]
        if condition != "vowels_mns":
            old_cell = read(previous / f"{split}-metrics.json")["conditions"][
                f"{condition}/{role}"
            ]
            for point, values in cell["operating_points"].items():
                previous_values = old_cell["operating_points"][point]
                assert abs(values["threshold"] - previous_values["threshold"]) <= 1e-6
                assert {
                    name: value for name, value in values.items() if name != "threshold"
                } == {
                    name: value
                    for name, value in previous_values.items()
                    if name != "threshold"
                }
                baseline_checks += 1
            assert abs(cell["pooled_eer"] - old_cell["pooled_eer"]) < 1e-12
        genuine = [r for r in rows if r["is_genuine"]]
        impostor = [r for r in rows if not r["is_genuine"]]
        assert len(genuine) == cell["queries"] and len(impostor) == cell["queries"] * 14
        for point, t in thresholds[condition]["operating_points"].items():
            threshold = float(t["threshold"])
            false_accepts = sum(r["score"] >= threshold for r in impostor)
            false_rejects = sum(r["score"] < threshold for r in genuine)
            rates = cell["operating_points"][point]
            assert (
                false_accepts == rates["false_accepts"]
                and false_rejects == rates["false_rejects"]
            )
            assert rates["all_input_far"] == false_accepts / len(impostor)
            assert rates["all_input_frr"] == false_rejects / len(genuine)
            assert rates["all_genuine"] == len(genuine) and rates[
                "all_impostor"
            ] == len(impostor)
            checks += 1
        # Independent grouped threshold scan with interpolation at FAR=FRR.
        levels = defaultdict(lambda: [0, 0])
        for r in rows:
            levels[r["score"]][0 if r["is_genuine"] else 1] += 1
        accepted_g = accepted_i = 0
        previous_far, previous_frr = 0.0, 1.0
        for score, (g, i) in sorted(levels.items(), reverse=True):
            accepted_g += g
            accepted_i += i
            far, frr = (
                accepted_i / len(impostor),
                (len(genuine) - accepted_g) / len(genuine),
            )
            if far >= frr:
                old_delta = previous_far - previous_frr
                fraction = -old_delta / ((far - frr) - old_delta)
                eer = previous_far + fraction * (far - previous_far)
                break
            previous_far, previous_frr = far, frr
        assert abs(eer - cell["pooled_eer"]) < 1e-12
        eer_checks += 1
    if split == "validation":
        for condition, t in thresholds.items():
            for point, target in (("far_1pct", 0.01), ("far_0_1pct", 0.001)):
                imp = sorted(
                    r["score"]
                    for r in groups[condition, "verification"]
                    if not r["is_genuine"]
                )
                allowed = int(target * len(imp))
                expected = float(np.nextafter(imp[len(imp) - allowed - 1], np.inf))
                assert t["operating_points"][point]["threshold"] == expected
arrays = np.load(run / "bootstrap-differences.npz", allow_pickle=False)
metrics = np.load(run / "bootstrap-metrics.npz", allow_pickle=False)
test = read(run / "test-metrics.json")
assert set(arrays.files) == set(test["paired_differences"])
for key, d in test["paired_differences"].items():
    prefix, role, name = key.split("/", 2)
    actual = 100 * (
        metrics[f"{d['minuend']}/{role}/{name}"]
        - metrics[f"{d['subtrahend']}/{role}/{name}"]
    )
    assert np.array_equal(arrays[key], actual, equal_nan=True)
    bounds = np.quantile(actual[np.isfinite(actual)], [0.025, 0.975])
    assert np.allclose(
        bounds,
        [d["ci95_percentage_points"]["lower"], d["ci95_percentage_points"]["upper"]],
        rtol=0,
        atol=1e-12,
    )

    def value(condition, role=role, name=name):
        cell = test["conditions"][f"{condition}/{role}"]
        if "/" in name:
            point, metric = name.split("/")
            return cell["operating_points"][point][metric]
        return cell[name]

    assert d["difference_percentage_points"] == 100 * (
        value(d["minuend"]) - value(d["subtrahend"])
    )
with (base / "evaluation-cells.csv").open() as stream:
    cells = list(csv.DictReader(stream))
assert len(cells) == 60
for row in cells:
    cell = report[row["split"]]["conditions"][f"{row['condition']}/{row['role']}"]
    assert int(row["queries"]) == cell["queries"]
    assert float(row["eer"]) == cell["pooled_eer"]
    for name in ("all_input_far", "all_input_frr", "far", "frr"):
        assert (
            float(row[name]) == cell["operating_points"][row["operating_point"]][name]
        )
counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
assert np.array_equal(
    counts, np.load(previous / "bootstrap-counts.npy", allow_pickle=False)
)
assert counts.shape == (10000, 15) and np.all(counts.sum(axis=1) == 15)
for freeze_name in ("input-freeze.json", "design-freeze.json"):
    for name, expected in read(run / freeze_name)["files"].items():
        assert sha(root / name) == expected, name
for name, expected in read(run / "evaluation-freeze.json")["files"].items():
    assert sha(run / name) == expected, name
training = root / config["training_run"]
summary = read(training / "training/summary.json")
for name, expected in summary["outputs_sha256"].items():
    assert sha(training / name) == expected, name
result = {
    "status": "passed",
    "raw_trial_rows": trial_rows,
    "rate_cells_recomputed": checks,
    "eer_cells_recomputed": eer_checks,
    "validation_far_thresholds_recomputed": 10,
    "bootstrap_pair_arrays_checked": len(arrays.files),
    "csv_cells_checked": len(cells),
    "threshold_recalibration_on_test": False,
    "baseline_inputs": read(run / "baseline-input-audit.json"),
    "baseline_metrics": baseline_checks,
    "input_plan_checks": plan_checks,
    "source_slice_checks": slice_checks,
}
with (run / "independent-numerical-audit.json").open("x") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
