"""Pre-frozen, duration- and token-count-matched single-phone diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import study
import torch

BASE = Path(__file__).resolve().parent


def matched_tokens(left, right, config):
    size = min(study.capacity(left), study.capacity(right))
    rows = [study.plan([row], (row["vowel"],), size)[0] for row in (left, right)]
    if any(
        study.pcm_rms_dbfs(study.ROOT, row) < config["minimum_consonant_rms_dbfs"]
        for row in rows
    ):
        return None
    return rows


def ranked_pairs(pools, phones, config, maximum):
    count = min(len(pools[phone]) for phone in phones)
    ordered = [
        study.prior.diversify(pools[phone], len(pools[phone]), config["selection_seed"])
        for phone in phones
    ]
    accepted, rejected = [], 0
    for left, right in zip(ordered[0][:count], ordered[1][:count], strict=True):
        rows = matched_tokens(left, right, config)
        if rows is None:
            rejected += 1
            continue
        accepted.append(rows)
        if len(accepted) == maximum:
            break
    return accepted, rejected


def validate_inputs(inputs, config):
    checks = 0
    for key, item in inputs["studies"].items():
        phones = config["diagnostics"]["pairs"][key]
        enrollment_hashes = {
            r["source_sha256"]
            for p in item["profiles"].values()
            for rows in p["conditions"].values()
            for r in rows
        }
        if set(item["profiles"]) != set(inputs["speakers"]):
            raise ValueError("incomplete diagnostic enrollment")
        for sample in [*item["profiles"].values(), *item["queries"]]:
            if set(sample["conditions"]) != set(phones):
                raise ValueError("wrong diagnostic phones")
            sizes = []
            for phone, rows in sample["conditions"].items():
                if not rows or {r["vowel"] for r in rows} != {phone}:
                    raise ValueError("invalid single-phone plan")
                if len({r["original_segment_id"] for r in rows}) != len(rows):
                    raise ValueError("repeated diagnostic token")
                if any(
                    not 720 <= r["end_frame"] - r["start_frame"] <= 6000 for r in rows
                ):
                    raise ValueError("invalid diagnostic duration")
                if (
                    sum(r["end_frame"] - r["start_frame"] for r in rows)
                    != sample["budget_frames"]
                ):
                    raise ValueError("unequal diagnostic time")
                sizes.append([r["end_frame"] - r["start_frame"] for r in rows])
                if "query_id" in sample and any(
                    r["source_sha256"] in enrollment_hashes for r in rows
                ):
                    raise ValueError("diagnostic enrollment/query overlap")
                checks += len(rows)
            if sizes[0] != sizes[1]:
                raise ValueError("token counts or rank durations differ")
        for role in study.ROLES:
            present = {q["speaker_id"] for q in item["queries"] if q["role"] == role}
            if present != set(inputs["speakers"]):
                raise ValueError(f"incomplete diagnostic speaker support: {key}/{role}")
    return checks


def prepare_split(config, split, files, metadata):
    reference = study.ROOT / config["reference_inputs"] / f"{split}-inputs.json"
    original = study.read_json(reference)
    study.pin(files, reference)
    labels = study.read_json(
        study.ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    )["speaker_splits"]
    if set(original["speakers"]) != set(labels[split]) or set(
        original["speakers"]
    ) & set(labels["train"]):
        raise ValueError("diagnostic training/evaluation speaker overlap")
    cache, qc = {}, Counter()

    def candidates(source, speaker, role):
        if source not in cache:
            cache[source] = study.extract_consonants(
                source, metadata, split, speaker, role, config, files, qc
            )
        return cache[source]

    studies = {
        key: {
            "phones": phones,
            "profiles": {},
            "queries": [],
            "excluded_queries": Counter(),
            "enrollment_rejected_rank_pairs": {},
        }
        for key, phones in config["diagnostics"]["pairs"].items()
    }
    source_hashes = set()
    for enrollment in original["enrollment"]:
        speaker = enrollment["user_id"]
        sources = sorted({r["source_file"] for r in enrollment["segments"]})
        source_hashes.update(metadata[source]["source_sha256"] for source in sources)
        pool = [
            r for source in sources for r in candidates(source, speaker, "enrollment")
        ]
        grouped = {
            phone: [r for r in pool if r["vowel"] == phone]
            for phone in study.CONSONANTS
        }
        for key, item in studies.items():
            tokens, rejected = ranked_pairs(
                grouped,
                item["phones"],
                config,
                config["diagnostics"]["enrollment_candidates_per_phone_maximum"],
            )
            if (
                len(tokens)
                < config["diagnostics"]["minimum_enrollment_candidates_per_phone"]
            ):
                raise ValueError(f"insufficient diagnostic enrollment: {speaker}/{key}")
            plans = {
                phone: [pair[index] for pair in tokens]
                for index, phone in enumerate(item["phones"])
            }
            item["profiles"][speaker] = {
                "budget_frames": sum(
                    r["end_frame"] - r["start_frame"]
                    for r in next(iter(plans.values()))
                ),
                "conditions": plans,
                "candidate_source_files": sources,
            }
            item["enrollment_rejected_rank_pairs"][speaker] = rejected
    for query in original["queries"]:
        if query["source_sha256"] in source_hashes:
            raise ValueError("diagnostic enrollment/query content overlap")
        pool = candidates(query["source_file"], query["speaker_id"], query["role"])
        grouped = {
            phone: [r for r in pool if r["vowel"] == phone]
            for phone in study.CONSONANTS
        }
        for key, item in studies.items():
            tokens, rejected = ranked_pairs(grouped, item["phones"], config, 1)
            if not tokens:
                item["excluded_queries"][
                    f"{query['role']}/missing_phone_or_post_crop_qc"
                ] += 1
                continue
            matched = tokens[0]
            item["queries"].append(
                {
                    **{
                        k: query[k]
                        for k in (
                            "query_id",
                            "utterance_id",
                            "speaker_id",
                            "source_file",
                            "source_sha256",
                            "role",
                            "split",
                        )
                    },
                    "budget_frames": matched[0]["end_frame"]
                    - matched[0]["start_frame"],
                    "conditions": {
                        phone: [row]
                        for phone, row in zip(item["phones"], matched, strict=True)
                    },
                    "rejected_rank_pairs_before_selection": rejected,
                }
            )
    for item in studies.values():
        item["excluded_queries"] = dict(item["excluded_queries"])
        item["preparation"] = {
            "queries_by_role": dict(Counter(q["role"] for q in item["queries"])),
            "original_queries_by_role": dict(
                Counter(q["role"] for q in original["queries"])
            ),
            "selection_uses_embeddings": False,
            "same_number_and_duration_of_enrollment_tokens_within_pair": True,
        }
    result = {
        "split": split,
        "speakers": original["speakers"],
        "studies": studies,
        "consonant_qc": dict(qc),
    }
    validate_inputs(result, config)
    for item in studies.values():
        study.audit_source_slices(
            {"split": split, "profiles": item["profiles"], "queries": item["queries"]},
            config,
        )
    return result


def prepare_inputs(config, run):
    if run.exists():
        raise ValueError("unused diagnostic run directory required")
    meta = (
        study.ROOT
        / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = {r["source_file"]: r for r in study.old.previous.read_rows(meta)}
    files = {}
    study.pin(files, meta)
    study.pin(
        files, study.ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
    )
    run.mkdir(parents=True)
    for split in ("validation", "test"):
        data = prepare_split(config, split, files, metadata)
        study.write_json(run / f"{split}-inputs.json", data)
        study.pin(files, run / f"{split}-inputs.json")
        print(
            f"{split} diagnostic support: "
            + json.dumps(
                {
                    key: item["preparation"]["queries_by_role"]
                    for key, item in data["studies"].items()
                }
            ),
            flush=True,
        )
    for path in (
        study.CONFIG,
        BASE / "diagnostics.py",
        BASE / "study.py",
        BASE / "voicing_audio.py",
        BASE / "tests/test_diagnostics.py",
    ):
        study.pin(files, path)
    for module in list(study.sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.is_relative_to(study.ROOT)
                and ".venv" not in path.parts
            ):
                study.pin(files, path)
    study.write_json(
        run / "input-freeze.json",
        {
            "status": "inputs_frozen_before_inference",
            "config": config,
            "files": files,
            "runtime": study.old.previous.runtime(),
        },
    )


def infer(config, run, split):
    data = study.frozen(config, run, split)
    validate_inputs(data, config)
    training = study.ROOT / config["training_run"]
    export = torch.load(
        training / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if (
        tuple(export["phonemes"]) != study.PHONEMES
        or export["checkpoint_update"] != 30000
    ):
        raise ValueError("unexpected diagnostic encoder")
    model = study.create_encoder("statistics_mlp").eval().requires_grad_(False)
    model.load_state_dict(export["model"], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    settings = study.read_json(
        study.ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json"
    )
    pipeline = study.InputPipeline(
        settings["input"],
        rms_enabled=False,
        statistics=study.read_json(training / "bundle/feature-statistics.json"),
    )
    selected = {}
    for item in data["studies"].values():
        for sample in [*item["profiles"].values(), *item["queries"]]:
            for rows in sample["conditions"].values():
                for row in rows:
                    if (
                        row["segment_id"] in selected
                        and selected[row["segment_id"]] != row
                    ):
                        raise ValueError("conflicting diagnostic slice")
                    selected[row["segment_id"]] = row
    segments = [
        study.Segment(
            r["segment_id"],
            "unused",
            r["vowel"],
            split,
            "verification",
            (),
            r["source_file"],
            r["start_frame"],
            r["end_frame"],
            r["source_sha256"],
        )
        for r in sorted(selected.values(), key=lambda r: r["segment_id"])
    ]
    dataset = study.SegmentDataset(study.ROOT, segments, pipeline, mode="center")
    vectors = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(segments), 64):
            batch = study.collate_segments(
                [dataset[i] for i in range(offset, min(offset + 64, len(segments)))]
            )
            embedding = model(batch["input"], batch["mask"])
            if not torch.isfinite(embedding).all() or not torch.equal(
                embedding, model(batch["input"], batch["mask"])
            ):
                raise ValueError("nonfinite or nondeterministic diagnostic embedding")
            values = embedding.numpy().astype(np.float64)
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            vectors.update(zip(batch["segment_ids"], values, strict=True))
    output = run / split
    output.mkdir(exist_ok=True)
    rows = []
    for key, item in data["studies"].items():
        profiles = {}
        for speaker, profile in item["profiles"].items():
            for phone, selected_rows in profile["conditions"].items():
                mean = np.mean(
                    [vectors[r["segment_id"]] for r in selected_rows], axis=0
                )
                profiles[speaker, phone] = mean / np.linalg.norm(mean)
        for query in item["queries"]:
            for phone, selected_rows in query["conditions"].items():
                vector = vectors[selected_rows[0]["segment_id"]]
                for claimed in data["speakers"]:
                    rows.append(
                        {
                            "study": key,
                            "condition": phone,
                            "query_id": query["query_id"],
                            "speaker_id": query["speaker_id"],
                            "claimed_speaker_id": claimed,
                            "role": query["role"],
                            "split": split,
                            "is_genuine": claimed == query["speaker_id"],
                            "status": "scored",
                            "score": float(vector @ profiles[claimed, phone]),
                            "used_frames": query["budget_frames"],
                        }
                    )
    if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError("diagnostic inference modified encoder")
    with (output / "scores.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(
        output / "embedding-vectors.npz",
        segment_ids=np.asarray(list(vectors)),
        vectors=np.asarray(list(vectors.values())),
    )
    study.write_json(
        output / "inference.json",
        {
            "status": "completed",
            "unique_embeddings": len(vectors),
            "all_repeats_bitwise_equal": True,
            "encoder_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "scores_sha256": study.sha256_file(output / "scores.jsonl"),
            "embedding_vectors_sha256": study.sha256_file(
                output / "embedding-vectors.npz"
            ),
        },
    )
    study.frozen(config, run, split)
    print(f"{split}: {len(rows)} diagnostic scores", flush=True)


def groups(config, run, split):
    data = study.frozen(config, run, split)
    path = run / split / "scores.jsonl"
    study.checked(
        path, study.read_json(run / split / "inference.json")["scores_sha256"]
    )
    rows = study.read_scores(path)
    expected = {
        (key, q["query_id"], phone, claimed): (
            q["speaker_id"],
            q["role"],
            q["budget_frames"],
        )
        for key, item in data["studies"].items()
        for q in item["queries"]
        for phone in item["phones"]
        for claimed in data["speakers"]
    }
    seen = set()
    for row in rows:
        key = (
            row["study"],
            row["query_id"],
            row["condition"],
            row["claimed_speaker_id"],
        )
        if (
            key not in expected
            or key in seen
            or (row["speaker_id"], row["role"], row["used_frames"]) != expected[key]
        ):
            raise ValueError("invalid diagnostic trial identity")
        if (
            row["split"] != split
            or row["is_genuine"] != (row["speaker_id"] == row["claimed_speaker_id"])
            or row["status"] != "scored"
            or not np.isfinite(row["score"])
            or not -1 <= row["score"] <= 1
        ):
            raise ValueError("invalid diagnostic trial label/score")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("incomplete diagnostic trial matrix")
    return data, {
        (key, phone, role): [
            r
            for r in rows
            if r["study"] == key and r["condition"] == phone and r["role"] == role
        ]
        for key, item in data["studies"].items()
        for phone in item["phones"]
        for role in study.ROLES
    }


def calibrate(config, run):
    data, measured = groups(config, run, "validation")
    thresholds, cells = {}, {}
    for key, item in data["studies"].items():
        for phone in item["phones"]:
            name = f"{key}/{phone}"
            thresholds[name] = study.old.calibrate_cell(
                measured[key, phone, "verification"], data["speakers"]
            )
            if thresholds[name]["status"] != "calibrated":
                raise ValueError("insufficient diagnostic calibration support")
            for role in study.ROLES:
                cells[f"{name}/{role}"], _ = study.old.evaluate_cell(
                    measured[key, phone, role], thresholds[name], data["speakers"]
                )
    study.write_json(run / "validation-thresholds.json", thresholds)
    study.write_json(run / "validation-metrics.json", {"conditions": cells})
    _, counts = study.old.speaker_draws(
        15, config["bootstrap_replicates"], config["bootstrap_seed"]
    )
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    names = (
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-counts.npy",
        "validation/scores.jsonl",
        "validation/inference.json",
        "validation/embedding-vectors.npz",
    )
    study.write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_test_inference",
            "files": {name: study.sha256_file(run / name) for name in names},
        },
    )


def evaluate(config, run):
    data, measured = groups(config, run, "test")
    thresholds = study.read_json(run / "validation-thresholds.json")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, replicas, differences, arrays, primary = {}, {}, {}, {}, {}
    for (key, phone, role), rows in measured.items():
        name = f"{key}/{phone}/{role}"
        threshold = thresholds[f"{key}/{phone}"]
        cells[name], _ = study.old.evaluate_cell(rows, threshold, data["speakers"])
        cells[name]["ci95"], replicas[name] = study.old.bootstrap_cell(
            rows, cells[name], threshold, data["speakers"], counts
        )
        print(f"diagnostic metrics: {name}", flush=True)
    for key, item in data["studies"].items():
        left, right = item["phones"]
        for role in study.ROLES:
            first, second = f"{key}/{left}/{role}", f"{key}/{right}/{role}"
            for metric in config["primary_metrics"]:
                name = f"{key}/{right}_minus_{left}/{role}/{metric}"
                delta = 100 * (replicas[second][metric] - replicas[first][metric])
                arrays[name] = delta
                differences[name] = {
                    "difference_percentage_points": 100
                    * (
                        study.old.metric_value(cells[second], metric)
                        - study.old.metric_value(cells[first], metric)
                    ),
                    "ci95_percentage_points": study.old.interval(delta),
                }
                if (
                    key in config["diagnostics"]["primary_pairs"]
                    and role == config["diagnostics"]["primary_role"]
                    and metric == config["diagnostics"]["primary_metric"]
                ):
                    confidence = config["diagnostics"]["primary_ci_confidence"]
                    ci = np.percentile(
                        delta,
                        [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)],
                        method="linear",
                    )
                    primary[name] = {
                        **differences[name],
                        "confidence": confidence,
                        "ci_percentage_points": {
                            "lower": float(ci[0]),
                            "upper": float(ci[1]),
                        },
                        "voiced_lower_eer_supported": bool(ci[1] < 0),
                    }
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{
            f"{k}/{metric}": values
            for k, items in replicas.items()
            for metric, values in items.items()
        },
    )
    np.savez_compressed(run / "bootstrap-differences.npz", **arrays)
    study.write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "paired_differences": differences,
            "primary_contrasts": primary,
            "threshold_recalibrated_on_test": False,
            "bootstrap_replicates": config["bootstrap_replicates"],
        },
    )
    study.frozen(config, run, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "prepare", "validation", "test"))
    args = parser.parse_args()
    config = study.read_json(study.CONFIG)
    run = study.ROOT / config["diagnostics"]["run_directory"]
    torch.set_num_threads(1)
    if args.stage == "inputs":
        prepare_inputs(config, run)
    elif args.stage == "prepare":
        study.prepare(config, run)
    elif args.stage == "validation":
        infer(config, run, "validation")
        calibrate(config, run)
    else:
        infer(config, run, "test")
        evaluate(config, run)


if __name__ == "__main__":
    main()
