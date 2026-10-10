"""Independent raw-PCM, fusion, EER, thresholds and paired-bootstrap audit."""

import importlib.util
import json
import math
import wave
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from bridge import BASE, ROOT, checked, fixed, pin, read_json, write_json

prior = BASE.parent / "phoneme-voicing-comparison/audit.py"
spec = importlib.util.spec_from_file_location("count_prior_audit", prior)
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
close, eer = independent.close, independent.eer


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    design = read_json(run / "design-freeze.json")
    for freeze in (design, read_json(run / "evaluation-freeze.json")):
        for name, checksum in freeze["files"].items():
            checked(ROOT / name, checksum)
    schedule = design["schedule"]
    for count, expected in ((7, 2), (10, 5)):
        totals = Counter(
            p
            for k in schedule["by_count"][str(count)]
            for p in schedule["conditions"][k]
            if p not in config["phones"][:5]
        )
        assert totals == Counter({p: expected for p in config["phones"][5:]})
    training = list(
        fixed.old.previous.read_rows(
            ROOT / config["training_run"] / "training-segments.jsonl"
        )
    )
    train_speakers = {r["speaker_id"] for r in training}
    train_hashes = {r["source_sha256"] for r in training}
    assert len(train_speakers) == 140 and all(
        r["split"] == "train" and not r["quality_flags"] for r in training
    )
    del training
    thresholds = read_json(run / "validation-thresholds.json")
    test = read_json(run / "test-results.json")
    with np.load(run / "bootstrap-speakers.npz", allow_pickle=False) as saved:
        counts = saved["counts"]
    bootstrap = np.load(run / "bootstrap-metrics.npz", allow_pickle=False)
    totals, raw_cache, wave_cache, verified = Counter(), {}, {}, set()
    for split in ("validation", "test"):
        data = read_json(run / f"{split}-inputs.json")
        assert not set(data["speakers"]) & train_speakers
        enroll_hashes = {
            r["source_sha256"]
            for p in data["profiles"].values()
            for rows in p["conditions"].values()
            for r in rows
        }
        for item in [*data["profiles"].values(), *data["queries"]]:
            assert set(item["conditions"]) == set(schedule["conditions"])
            for condition, rows in item["conditions"].items():
                assert {r["vowel"] for r in rows} == set(
                    schedule["conditions"][condition]
                )
                assert (
                    sum(r["end_frame"] - r["start_frame"] for r in rows)
                    == item["budget_frames"]
                )
                assert len({r["original_segment_id"] for r in rows}) == len(rows)
                ranges = defaultdict(list)
                for r in rows:
                    source = r["source_file"]
                    if source not in raw_cache:
                        utterance = r["original_segment_id"].split("--raw-")[0]
                        raw_cache[source] = read_json(
                            ROOT
                            / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
                            / r["speaker_id"]
                            / f"{utterance}.json"
                        )
                    raw = raw_cache[source]
                    interval = raw["intervals"][r["raw_index"]]
                    assert interval["phoneme"] == r["vowel"]
                    lo, hi = (
                        round(interval["start_sec"] * 24000),
                        round(interval["end_sec"] * 24000),
                    )
                    assert (lo, hi) == (r["raw_start_frame"], r["raw_end_frame"])
                    candidate_size = min(6000, hi - lo)
                    first = lo + (hi - lo - candidate_size) // 2
                    size = r["end_frame"] - r["start_frame"]
                    expected = (
                        first + candidate_size - size
                        if r["vowel"] in ("t", "d", "k", "g")
                        else first + (candidate_size - size) // 2
                    )
                    assert (
                        r["start_frame"] == expected
                        and lo <= expected < r["end_frame"] <= hi
                        and 720 <= size <= 6000
                    )
                    assert (
                        r["source_sha256"] == raw["source_sha256"]
                        and r["source_sha256"] not in train_hashes
                    )
                    assert (
                        raw["source_file"] == source
                        and raw["speaker_id"] == r["speaker_id"]
                    )
                    if "query_id" in item:
                        assert (
                            source == item["source_file"]
                            and r["source_sha256"] not in enroll_hashes
                        )
                    else:
                        assert source in item["candidate_source_files"]
                    assert r["speaker_id"] == item["speaker_id"]
                    if source not in verified:
                        checked(ROOT / source, r["source_sha256"])
                        verified.add(source)
                    if source not in wave_cache:
                        with wave.open(str(ROOT / source), "rb") as audio:
                            assert (
                                audio.getframerate(),
                                audio.getnchannels(),
                                audio.getsampwidth(),
                                audio.getcomptype(),
                            ) == (24000, 1, 2, "NONE")
                            wave_cache[source] = (
                                np.frombuffer(
                                    audio.readframes(audio.getnframes()), dtype="<i2"
                                ).astype(float)
                                / 32768
                            )
                    pcm = wave_cache[source][r["start_frame"] : r["end_frame"]]
                    assert len(pcm) == size and math.sqrt(
                        float(np.mean(pcm**2))
                    ) >= 10 ** (-50 / 20)
                    ranges[source].append((r["start_frame"], r["end_frame"]))
                    totals["raw_source_slices"] += 1
                assert all(
                    a[1] <= b[0]
                    for rs in ranges.values()
                    for a, b in pairwise(sorted(rs))
                )
                totals["equal_time_plans"] += 1
        inference = read_json(run / split / "inference.json")
        assert inference["bitwise_repeat_equal"] and inference["encoder_unchanged"]
        checked(run / split / "scores.jsonl", inference["scores_sha256"])
        checked(
            run / split / "embedding-vectors.npz", inference["embedding_vectors_sha256"]
        )
        with np.load(
            run / split / "embedding-vectors.npz", allow_pickle=False
        ) as saved:
            vectors = dict(
                zip(saved["segment_ids"].tolist(), saved["vectors"], strict=True)
            )
        assert all(
            np.isfinite(v).all()
            and math.isclose(float(np.linalg.norm(v)), 1, abs_tol=1e-12)
            for v in vectors.values()
        )
        records = list(independent.rows(run / split / "scores.jsonl"))
        totals["independent_fusion_scores"] += independent.score_audit(
            data, records, vectors, False
        )
        metrics = (
            read_json(run / "validation-metrics.json")
            if split == "validation"
            else test["cells"]
        )
        lookup = {s: i for i, s in enumerate(data["speakers"])}
        for key, cell in metrics.items():
            condition, role = key.split("/")
            selected = [
                r for r in records if r["condition"] == condition and r["role"] == role
            ]
            if not selected:
                assert cell["pooled_eer"] is None
                continue
            values = np.array([r["score"] for r in selected])
            genuine = np.array([r["is_genuine"] for r in selected], dtype=bool)
            close(eer(values, genuine), cell["pooled_eer"])
            assert cell["queries"] == genuine.sum()
            assert cell["query_speaker_count"] == len(
                {r["speaker_id"] for r in selected}
            )
            totals["independent_EER_cells"] += 1
            for point, rate in cell["operating_points"].items():
                threshold = float(
                    thresholds[condition]["operating_points"][point]["threshold"]
                )
                accepted = values >= threshold
                fa, fr = (
                    int(np.count_nonzero(accepted & ~genuine)),
                    int(np.count_nonzero(~accepted & genuine)),
                )
                assert (rate["false_accepts"], rate["false_rejects"]) == (fa, fr)
                for field, value in (
                    ("far", fa / (~genuine).sum()),
                    ("all_input_far", fa / (~genuine).sum()),
                    ("frr", fr / genuine.sum()),
                    ("all_input_frr", fr / genuine.sum()),
                ):
                    close(rate[field], value)
                    totals["independent_FAR_FRR"] += 1
                if (
                    split == "validation"
                    and role == "verification"
                    and point in ("far_1pct", "far_0_1pct")
                ):
                    target = 0.01 if point == "far_1pct" else 0.001
                    negative = np.sort(values[~genuine])
                    expected = np.nextafter(
                        negative[
                            len(negative) - math.floor(target * len(negative)) - 1
                        ],
                        np.inf,
                    )
                    close(threshold, expected)
                    totals["validation_thresholds"] += 1
            if split == "test" and role == "verification":
                qi = np.array([lookup[r["speaker_id"]] for r in selected])
                ci = np.array([lookup[r["claimed_speaker_id"]] for r in selected])
                for i, draw in enumerate(counts[:64]):
                    weights = np.where(genuine, draw[qi], draw[qi] * draw[ci])
                    close(
                        eer(values, genuine, weights),
                        bootstrap[f"cell/{condition}/pooled_eer"][i],
                    )
                    totals["independent_weighted_EER"] += 1
                    for point in thresholds[condition]["operating_points"]:
                        threshold = float(
                            thresholds[condition]["operating_points"][point][
                                "threshold"
                            ]
                        )
                        accepted = values >= threshold
                        for field, mask, denominator in (
                            ("all_input_far", ~genuine & accepted, ~genuine),
                            ("all_input_frr", genuine & ~accepted, genuine),
                        ):
                            expected = weights[mask].sum() / weights[denominator].sum()
                            close(
                                expected,
                                bootstrap[f"cell/{condition}/{point}/{field}"][i],
                            )
                            totals["independent_weighted_FAR_FRR"] += 1
        for role, fields in test["count_summary"].items():
            if split != "test":
                continue
            for field, by_count in fields.items():
                for count, result in by_count.items():
                    if result is None:
                        continue

                    def value(k, metrics=metrics, role=role, field=field):
                        cell = metrics[f"{k}/{role}"]
                        if field == "pooled_eer":
                            return cell[field]
                        point, name = field.split("/")
                        return cell["operating_points"][point][name]

                    vs = [value(k) for k in schedule["by_count"][count]]
                    close(result["mean"], sum(vs) / len(vs))
                    close(result["minimum"], min(vs))
                    close(result["maximum"], max(vs))
                    totals["count_summary_checks"] += 1
                    if role == "verification":
                        expected = np.mean(
                            [
                                bootstrap[f"cell/{k}/{field}"]
                                for k in schedule["by_count"][count]
                            ],
                            axis=0,
                        )
                        np.testing.assert_allclose(
                            expected,
                            bootstrap[f"count/{count}/{field}"],
                            atol=1e-14,
                            rtol=0,
                        )
                        limits = np.percentile(expected, [2.5, 97.5])
                        close(result["ci95"]["lower"], limits[0])
                        close(result["ci95"]["upper"], limits[1])
    for key, item in test["differences"].items():
        pair, field = key.split("/", 1)
        hi, lo = pair.split("_minus_")
        expected = bootstrap[f"count/{hi}/{field}"] - bootstrap[f"count/{lo}/{field}"]
        np.testing.assert_allclose(
            expected, bootstrap[f"difference/{key}"], atol=1e-14, rtol=0
        )
        limits = np.percentile(expected, [2.5, 97.5])
        close(item["ci95"]["lower"], limits[0])
        close(item["ci95"]["upper"], limits[1])
        point = test["count_summary"]["verification"][field]
        close(item["difference"], point[hi]["mean"] - point[lo]["mean"])
        totals["paired_difference_arrays_and_CIs"] += 1
    assert test["primary"] == test["differences"]["14_minus_5/pooled_eer"]
    assert not test["threshold_recalibrated_on_test"]
    report = {
        "status": "passed",
        "checks": dict(totals),
        "unique_wavs": len(verified),
        "training_speakers": 140,
        "new_training": False,
    }
    write_json(run / "independent-audit.json", report)
    files = {}
    for path in [
        run / "test-results.json",
        run / "bootstrap-metrics.npz",
        run / "independent-audit.json",
        *[
            run / split / name
            for split in ("validation", "test")
            for name in ("scores.jsonl", "embedding-vectors.npz", "inference.json")
        ],
    ]:
        pin(files, path)
    write_json(run / "audit-freeze.json", {"status": "audited", "files": files})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
