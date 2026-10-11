"""Recompute EER, forward-search decisions and fusion independently of the runner."""

from __future__ import annotations

import json
import sys
import wave

from common import (
    BASE,
    CONFIG,
    PHASE3,
    ROOT,
    checked,
    np,
    pipeline,
    read_json,
    rows,
    sha256_file,
    torch,
    write_json,
)
from learner import load_model
from selection import baseline_state, next_candidate, record_attempt

sys.path.insert(0, str(PHASE3 / "scripts"))
from weighted_bootstrap import speaker_draws, weighted_eers


def independent_eer(scores, genuine, weights=None):
    scores, genuine = np.asarray(scores), np.asarray(genuine, dtype=bool)
    weights = (
        np.ones(len(scores), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    levels, inverse = np.unique(scores, return_inverse=True)
    positives = np.bincount(inverse, weights=weights * genuine, minlength=len(levels))
    negatives = np.bincount(inverse, weights=weights * ~genuine, minlength=len(levels))
    p, n = positives.sum(), negatives.sum()
    if p <= 0 or n <= 0:
        raise ValueError("empty independent ROC")
    frr = np.r_[0, np.cumsum(positives) / p]
    far = np.r_[1, 1 - np.cumsum(negatives) / n]
    delta = far - frr
    exact = np.flatnonzero(np.abs(delta) < 1e-14)
    if len(exact):
        return float((far[exact[0]] + frr[exact[0]]) / 2)
    right = int(np.flatnonzero(delta < 0)[0])
    left = right - 1
    ratio = delta[left] / (delta[left] - delta[right])
    return float(far[left] + ratio * (far[right] - far[left]))


def audit_trial(run, trial, split, report):
    data = read_json(run / f"{split}-inputs.json")
    vectors = np.load(trial / f"{split}-embeddings.npy", allow_pickle=False).astype(
        np.float64
    )
    checked(trial / f"{split}-embeddings.npy", report["embeddings_sha256"])
    checked(trial / f"{split}-scores.jsonl", report["scores_sha256"])
    profiles = {}
    used = set(report["phonemes"]) & set(data["universally_registered"])
    for p in used:
        means = np.stack(
            [
                np.add.reduce(vectors[data["profiles"][speaker][p]], axis=0)
                / len(data["profiles"][speaker][p])
                for speaker in data["speakers"]
            ]
        )
        profiles[p] = means / np.sqrt(np.einsum("ij,ij->i", means, means))[:, None]
    expected = {}
    for query in data["queries"]:
        if query["scorable"]:
            components = {}
            for p in sorted(used):
                indices = query["groups"][p]
                if indices:
                    query_mean = np.add.reduce(vectors[indices], axis=0) / len(indices)
                    components[p] = profiles[p] @ query_mean
            for i, speaker in enumerate(data["speakers"]):
                expected[query["query_id"], speaker] = {
                    p: float(v[i]) for p, v in components.items()
                }
        else:
            for speaker in data["speakers"]:
                expected[query["query_id"], speaker] = None
    values = list(rows(trial / f"{split}-scores.jsonl"))
    seen = set()
    max_error = 0.0
    for row in values:
        key = row["query_id"], row["claimed_speaker_id"]
        if key in seen or key not in expected:
            raise ValueError("duplicate/unexpected score identity")
        seen.add(key)
        if row["is_genuine"] != (row["speaker_id"] == row["claimed_speaker_id"]):
            raise ValueError("wrong trial label")
        components = expected[key]
        if components is None:
            if row["status"] != "no_score" or row["score"] is not None:
                raise ValueError("missing vowel must remain no_score")
        else:
            if row["status"] != "scored" or set(row["phone_scores"]) != set(components):
                raise ValueError("query phone support differs")
            errors = [
                abs(row["phone_scores"][p] - score) for p, score in components.items()
            ]
            errors.append(
                abs(row["score"] - sum(components.values()) / len(components))
            )
            max_error = max(max_error, *errors)
    if seen != set(expected) or max_error > 2e-6:
        raise ValueError(f"independent score reconstruction failed: {max_error}")
    rate_checks = 0
    for role, metric in report["metrics"].items():
        chosen = [r for r in values if r["role"] == role]
        scored = [r for r in chosen if r["status"] == "scored"]
        eer = independent_eer(
            [r["score"] for r in scored], [r["is_genuine"] for r in scored]
        )
        if abs(eer - metric["eer"]) > 1e-12:
            raise ValueError("independent EER differs")
        for point in metric["operating_points"].values():
            threshold = (
                np.inf
                if point["threshold"] == "+inf"
                else (-np.inf if point["threshold"] == "-inf" else point["threshold"])
            )
            fa = sum(r["score"] >= threshold and not r["is_genuine"] for r in scored)
            fr = sum(r["score"] < threshold and r["is_genuine"] for r in scored)
            g, i = (
                sum(r["is_genuine"] for r in scored),
                sum(not r["is_genuine"] for r in scored),
            )
            all_g, all_i = (
                sum(r["is_genuine"] for r in chosen),
                sum(not r["is_genuine"] for r in chosen),
            )
            recomputed = {
                "far": fa / i,
                "frr": fr / g,
                "all_input_far": fa / all_i,
                "all_input_frr": (fr + all_g - g) / all_g,
            }
            if (
                any(abs(point[k] - v) > 1e-12 for k, v in recomputed.items())
                or point["false_accepts"] != fa
                or point["false_rejects"] != fr
            ):
                raise ValueError("independent FAR/FRR differs")
            rate_checks += 4
    return {
        "scores": len(values),
        "max_score_error": max_error,
        "eer_checks": 2,
        "rate_checks": rate_checks,
    }


def audit_bootstrap(config, run, frozen, final):
    speakers = read_json(run / "test-inputs.json")["speakers"]
    indices, counts = speaker_draws(
        len(speakers), config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    lookup = {s: i for i, s in enumerate(speakers)}
    estimates, checks = {}, 0
    for label, name in (
        ("baseline", frozen["baseline_trial"]),
        ("final", frozen["final_trial"]),
    ):
        values = list(rows(run / "trials" / name / "test-scores.jsonl"))
        for role in ("verification", "cross_text_verification"):
            selected = [
                r for r in values if r["role"] == role and r["status"] == "scored"
            ]
            scores = np.array([r["score"] for r in selected])
            q = np.array([lookup[r["speaker_id"]] for r in selected])
            c = np.array([lookup[r["claimed_speaker_id"]] for r in selected])
            values_eer = weighted_eers(scores, q, c, counts)
            for sample in range(0, len(counts), max(1, len(counts) // 40)):
                weights = np.where(
                    q == c, counts[sample, q], counts[sample, q] * counts[sample, c]
                )
                if (
                    abs(independent_eer(scores, q == c, weights) - values_eer[sample])
                    > 1e-12
                ):
                    raise ValueError("independent speaker bootstrap differs")
                checks += 1
            estimates[f"{label}/{role}"] = values_eer
    intervals = {}
    for role in ("verification", "cross_text_verification"):
        delta = estimates[f"final/{role}"] - estimates[f"baseline/{role}"]
        estimates[f"difference/{role}"] = delta
        intervals[role] = {
            "eer_difference": final["final"]["metrics"][role]["eer"]
            - final["baseline"]["metrics"][role]["eer"],
            "eer_difference95": np.percentile(delta, [2.5, 97.5]).tolist(),
            "scope": "paired_test_speaker_bootstrap_conditional_on_selected_model; excludes_search_and_training_uncertainty",
        }
    np.savez_compressed(
        run / "bootstrap.npz",
        speaker_indices=indices,
        speaker_counts=counts,
        **estimates,
    )
    write_json(run / "bootstrap-results.json", intervals)
    return checks


def audit_slices(run, config, frozen):
    source_sets = {}
    probes = {}
    inspected = 0
    for split in ("train", "validation", "test"):
        hashes = set()
        for i, row in enumerate(rows(run / f"{split}-segments.jsonl")):
            if row["split"] != split or row["cache_index"] != i or row["quality_flags"]:
                raise ValueError("cache identity/split/quality mismatch")
            length = row["end_frame"] - row["start_frame"]
            if not (
                720 <= length <= 6000
                and row["raw_start_frame"]
                <= row["start_frame"]
                < row["end_frame"]
                <= row["raw_end_frame"]
                and row["rms_dbfs"] >= -50
            ):
                raise ValueError("slice outside original bounds or QC")
            hashes.add(row["source_sha256"])
            probes.setdefault((split, row["phoneme"]), row)
            inspected += 1
        source_sets[split] = hashes
    if any(
        source_sets[a] & source_sets[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("train/validation/test audio hash overlap")
    pipe = pipeline(config)
    cache_errors = []
    for name in dict.fromkeys((frozen["baseline_trial"], frozen["final_trial"])):
        model, _ = load_model(run / "trials" / name)
        for (split, phone), row in probes.items():
            with wave.open(str(ROOT / row["source_file"]), "rb") as wav:
                wav.setpos(row["start_frame"])
                signal = torch.from_numpy(
                    np.frombuffer(
                        wav.readframes(row["end_frame"] - row["start_frame"]),
                        dtype="<i2",
                    ).astype(np.float32)
                    / 32768
                )
            raw = pipe.prepare(signal, row["segment_id"], mode="center")["input"]
            cache = np.memmap(
                run / f"{split}-features.f32", mode="r", dtype="<f4"
            ).reshape(-1, 128)
            pooled = torch.from_numpy(np.array(cache[row["cache_index"]]))[None]
            with torch.inference_mode():
                full = model(
                    raw[None], torch.ones((1, raw.shape[-1]), dtype=torch.bool)
                )
                fast = torch.nn.functional.normalize(
                    model.projection(pooled), dim=1, eps=model.l2_epsilon
                )
            error = float(torch.max(torch.abs(full - fast)))
            if error > 2e-6:
                raise ValueError(
                    "exported raw-waveform encoder differs from cached inference"
                )
            cache_errors.append(error)
    return {
        "slice_checks": inspected,
        "trained_raw_waveform_cache_checks": len(cache_errors),
        "trained_cache_max_abs_error": max(cache_errors),
    }


def audit():
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    torch.set_num_threads(1)
    frozen = read_json(run / "selection-freeze.json")
    final = read_json(run / "final-test.json")
    state = read_json(run / "search-state.json")
    baseline_eer = read_json(run / "trials/trial-000-baseline/validation-metrics.json")[
        "metrics"
    ]["verification"]["eer"]
    replay = baseline_state(state["order"], baseline_eer)
    for entry in state["attempts"]:
        phone = next_candidate(replay)
        if phone != entry["phoneme"]:
            raise ValueError("search did not draw the next random phone")
        computed = record_attempt(
            replay, phone, entry["trial"], entry["validation_eer"], entry["reason"]
        )
        if computed != entry:
            raise ValueError("candidate acceptance or retry differs")
    if next_candidate(replay) is not None or replay != state:
        raise ValueError("search did not stop at a no-improvement pass")
    initial, head_initial, trials, score_count, eer_checks, rate_checks = (
        None,
        None,
        0,
        0,
        0,
        0,
    )
    for trial in sorted((run / "trials").iterdir()):
        if not (trial / "training-summary.json").exists():
            continue
        summary = read_json(trial / "training-summary.json")
        if (
            summary["completed_updates"] != config["maximum_updates"]
            or summary["encoder_parameters"] != 65920
        ):
            raise ValueError("wrong training budget/architecture")
        initial = initial or summary["initial_encoder_sha256"]
        head_initial = head_initial or summary["initial_head_sha256"]
        if (
            initial != summary["initial_encoder_sha256"]
            or head_initial != summary["initial_head_sha256"]
        ):
            raise ValueError("candidate initial weights differ")
        checked(trial / "history.jsonl", summary["history_sha256"])
        _model, _ = load_model(trial)
        for split in ("validation", "test"):
            path = trial / f"{split}-metrics.json"
            if not path.exists():
                continue
            if split == "test" and trial.name not in frozen["models"]:
                raise ValueError("unselected trial saw test")
            inspected = audit_trial(run, trial, split, read_json(path))
            score_count += inspected["scores"]
            eer_checks += inspected["eer_checks"]
            rate_checks += inspected["rate_checks"]
        trials += 1
    bootstrap_checks = audit_bootstrap(config, run, frozen, final)
    slice_checks = audit_slices(run, config, frozen)
    report = {
        "status": "passed",
        "selection_steps": len(state["attempts"]),
        "training_trials": trials,
        "score_checks": score_count,
        "eer_checks": eer_checks,
        "rate_checks": rate_checks,
        "independent_bootstrap_eer_checks": bootstrap_checks,
        "selected_phonemes": state["accepted"],
        "test_used_for_selection": False,
        "audit_source_sha256": sha256_file(BASE / "audit.py"),
        **slice_checks,
    }
    write_json(run / "independent-audit.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    audit()
