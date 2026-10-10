"""Independently recompute source slices, dense REML, cosine and weighted ROC."""

import json
import wave
from collections import defaultdict

import numpy as np
from bridge import BASE, ROOT, checked, read_json, write_json


def roc_eer(rows, weights=None):
    levels = defaultdict(lambda: np.zeros(2))
    if weights is None:
        weights = np.ones(len(rows))
    for r, w in zip(rows, weights, strict=True):
        levels[r["score"]][0 if r["is_genuine"] else 1] += w
    total = sum(levels.values())
    accepted = np.zeros(2)
    previous = np.array([0.0, 1.0])
    for _, amount in sorted(levels.items(), reverse=True):
        accepted += amount
        current = np.array([accepted[1] / total[1], 1 - accepted[0] / total[0]])
        if current[0] >= current[1]:
            delta = previous[0] - previous[1]
            fraction = -delta / (current[0] - current[1] - delta)
            return float(previous[0] + fraction * (current[0] - previous[0]))
        previous = current
    raise AssertionError("ROC never crossed")


def dense_components(rows, vectors, measured):
    # Independent design construction; intercept plus grouped adjacent labels.
    contexts = [
        "a",
        "i",
        "u",
        "e",
        "o",
        "unvoiced_vowel",
        "nasal",
        "voiced_obstruent",
        "unvoiced_obstruent",
        "approximant",
    ]

    def label(p):
        if p in contexts[:5]:
            return p
        for name, phones in [
            ("unvoiced_vowel", ["A", "I", "U", "E", "O"]),
            ("nasal", ["m", "n", "N", "my", "ny"]),
            ("voiced_obstruent", ["b", "d", "g", "z", "j", "v", "by", "dy", "gy"]),
            (
                "unvoiced_obstruent",
                [
                    "p",
                    "t",
                    "k",
                    "s",
                    "sh",
                    "ch",
                    "ts",
                    "h",
                    "f",
                    "ky",
                    "py",
                    "ty",
                    "hy",
                    "cl",
                ],
            ),
            ("approximant", ["r", "ry", "w", "y"]),
        ]:
            if p in phones:
                return name
        return "other"

    x = np.array(
        [
            [1.0, np.log((r["end_frame"] - r["start_frame"]) / 2400)]
            + [int(label(r["previous_phone"]) == c) for c in contexts]
            + [int(label(r["following_phone"]) == c) for c in contexts]
            for r in rows
        ]
    )
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    rank = int(np.sum(singular**2 > max(singular[0] ** 2 * 1e-10, 1e-12)))
    x = u[:, :rank]
    speakers = sorted({r["speaker_id"] for r in rows})
    z = np.array([[r["speaker_id"] == s for s in speakers] for r in rows], dtype=float)
    y = np.stack([vectors[r["segment_id"]] for r in rows])
    n, d = y.shape

    def profile(lam):
        v = np.eye(n) + lam * (z @ z.T)
        inverse = np.linalg.inv(v)
        information = x.T @ inverse @ x
        beta = np.linalg.solve(information, x.T @ inverse @ y)
        residual = y - x @ beta
        w = float(np.sum(residual * (inverse @ residual)) / (d * (n - rank)))
        objective = d * (
            (n - rank) * np.log(w)
            + np.linalg.slogdet(v)[1]
            + np.linalg.slogdet(information)[1]
        )
        return objective, w

    lam = measured["lambda"]
    objective, w = profile(lam)
    assert abs(d * w - measured["W_trace"]) < 1e-8
    assert abs(d * lam * w - measured["B_trace"]) < 1e-8
    for candidate in [max(0, lam * 0.99), lam * 1.01] if lam else [0.0001]:
        assert profile(candidate)[0] >= objective - 1e-7
    return 1


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    freeze = read_json(run / "design-freeze.json")
    groups = read_json(run / "group-freeze.json")
    for ledger in (freeze, groups):
        for name, sha in ledger["files"].items():
            checked(ROOT / name, sha)
    thresholds = read_json(run / "validation-thresholds.json")
    validation = read_json(run / "validation-components.json")
    test = read_json(run / "test-results.json")
    arrays = np.load(run / "test-bootstrap.npz")
    counters = defaultdict(int)
    for split in ("validation", "test"):
        inputs = read_json(run / f"{split}-inputs.json")
        vectors = np.load(run / split / "embedding-vectors.npz")
        rows = list(
            {
                r["segment_id"]: r
                for k in ("variance", "enrollment", "queries")
                for r in inputs[k]
            }.values()
        )
        audio = {}
        for r in rows:
            source = r["source_file"]
            if source not in audio:
                with wave.open(str(ROOT / source), "rb") as w:
                    assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
                        24000,
                        1,
                        2,
                    )
                    audio[source] = (
                        np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(
                            float
                        )
                        / 32768
                    )
            clip = audio[source][r["start_frame"] : r["end_frame"]]
            assert len(clip) == r["end_frame"] - r["start_frame"]
            assert (
                r["raw_start_frame"]
                <= r["start_frame"]
                < r["end_frame"]
                <= r["raw_end_frame"]
            )
            rms = 10 * np.log10(np.mean(clip**2))
            assert rms >= -50 and abs(rms - r["rms_dbfs"]) < 1e-10
            assert abs(np.linalg.norm(vectors[r["segment_id"]]) - 1) < 1e-12
            counters["source_slices"] += 1
        profiles = {}
        for s in inputs["speakers"]:
            for p in config["phones"]:
                enrolled = [
                    r
                    for r in inputs["enrollment"]
                    if r["speaker_id"] == s and r["vowel"] == p
                ]
                assert len(enrolled) == 5 and all(
                    r["end_frame"] - r["start_frame"] == 1200 for r in enrolled
                )
                mean = np.mean([vectors[r["segment_id"]] for r in enrolled], axis=0)
                profiles[s, p] = mean / np.linalg.norm(mean)
        scored = [
            json.loads(line)
            for line in (run / split / "scores.jsonl").read_text().splitlines()
        ]
        by_cell = defaultdict(list)
        for r in scored:
            expected = (
                vectors[r["query_id"]]
                @ profiles[r["claimed_speaker_id"], r["condition"]]
            )
            assert abs(expected - r["score"]) < 1e-12
            assert r["is_genuine"] == (r["speaker_id"] == r["claimed_speaker_id"])
            by_cell[f"{r['condition']}/{r['role']}"].append(r)
            counters["cosine_scores"] += 1
        cells = (
            read_json(run / "validation-metrics.json")
            if split == "validation"
            else test["cells"]
        )
        for p in config["phones"]:
            selected = [r for r in inputs["variance"] if r["vowel"] == p]
            comp = (
                validation["components"][p]
                if split == "validation"
                else test["test_components"]["components"][p]
            )
            counters["dense_REML_components"] += dense_components(
                selected, vectors, comp
            )
            for role in ("verification", "cross_text_verification"):
                key = f"{p}/{role}"
                rr = by_cell[key]
                n = 10 if role == "verification" else 3
                assert len(rr) == 15 * 15 * n
                assert abs(roc_eer(rr) - cells[key]["pooled_eer"]) < 1e-12
                counters["EER_cells"] += 1
                for point, settings in thresholds[p]["operating_points"].items():
                    threshold = float(settings["threshold"])
                    genuine = [r for r in rr if r["is_genuine"]]
                    impostor = [r for r in rr if not r["is_genuine"]]
                    far = sum(r["score"] >= threshold for r in impostor) / len(impostor)
                    frr = sum(r["score"] < threshold for r in genuine) / len(genuine)
                    reported = cells[key]["operating_points"][point]
                    assert abs(far - reported["all_input_far"]) < 1e-12
                    assert abs(frr - reported["all_input_frr"]) < 1e-12
                    counters["FAR_FRR_values"] += 2
                if split == "test":
                    lookup = {s: i for i, s in enumerate(inputs["speakers"])}
                    for index, count in enumerate(arrays["speaker_counts"][:64]):
                        weights = [
                            count[lookup[r["speaker_id"]]]
                            if r["is_genuine"]
                            else count[lookup[r["speaker_id"]]]
                            * count[lookup[r["claimed_speaker_id"]]]
                            for r in rr
                        ]
                        expected = roc_eer(rr, weights)
                        assert (
                            abs(expected - arrays[f"phone/{key}/pooled_eer"][index])
                            < 1e-12
                        )
                        counters["weighted_bootstrap_EERs"] += 1
        if split == "validation":
            # Independent exhaustive scan of all allowed sorted boundaries.
            order = sorted(
                config["phones"], key=lambda p: (validation["components"][p]["R"], p)
            )
            values = np.array([validation["components"][p]["R"] for p in order])
            losses = [
                (np.var(values[:k]) * k + np.var(values[k:]) * (len(values) - k), k)
                for k in range(2, len(values) - 1)
            ]
            _, boundary = min(losses)
            assert groups["groups"]["low"] == order[:boundary]
            assert groups["groups"]["high"] == order[boundary:]
    for key, item in test["group_summary"].items():
        role, field = key.split("/", 1)
        high = np.mean(
            [arrays[f"phone/{p}/{role}/{field}"] for p in groups["groups"]["high"]],
            axis=0,
        )
        low = np.mean(
            [arrays[f"phone/{p}/{role}/{field}"] for p in groups["groups"]["low"]],
            axis=0,
        )
        assert np.allclose(high - low, arrays[key], atol=1e-14, rtol=0)
        lo, hi = np.quantile(high - low, [0.025, 0.975])
        assert abs(lo - item["ci95_high_minus_low"]["lower"]) < 1e-12
        assert abs(hi - item["ci95_high_minus_low"]["upper"]) < 1e-12
        counters["group_difference_arrays_and_CIs"] += 1
    write_json(
        run / "independent-audit.json", {"status": "passed", "checks": dict(counters)}
    )
    print(json.dumps(dict(counters)), flush=True)


if __name__ == "__main__":
    main()
