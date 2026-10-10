"""Compare eight individual phones with fixed token counts and actual PCM time."""

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from diagnostics import describe, primary_results
from inputs import prepare_inputs, validate_plans
from support import (
    BASE,
    PHONES,
    ROLES,
    ROOT,
    checked,
    core,
    pin,
    read_json,
    sha256_file,
    write_json,
)

CONFIG = BASE / "config/protocol.json"
PAIRS = tuple(combinations(PHONES, 2))


def prepare(config, run):
    files = prepare_inputs(config, run)
    training = ROOT / config["training_run"]
    summary = read_json(training / "training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["test_used"]
        or summary["selected_update"] != 30000
        or summary["export_reload_bitwise_equal_phonemes"] != list(PHONES)
    ):
        raise ValueError("requires the fixed completed eight-phone encoder")
    checked(training / "training-freeze.json", summary["freeze_sha256"])
    for name in ("bundle/encoder.pt", "bundle/feature-statistics.json"):
        checked(training / name, summary["outputs_sha256"][name])
        pin(files, training / name)
    pin(files, training / "training/summary.json")
    pin(files, training / "training-freeze.json")
    for path in [CONFIG, *BASE.glob("*.py"), *BASE.glob("tests/*.py")]:
        pin(files, path)
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if (
                path.is_file()
                and path.suffix == ".py"
                and path.is_relative_to(ROOT)
                and ".venv" not in path.parts
            ):
                pin(files, path)
    write_json(
        run / "design-freeze.json",
        {
            "status": "frozen_before_inference",
            "config": config,
            "runtime": core.old.previous.runtime(),
            "files": files,
        },
    )


def frozen(config, run, split):
    fixed = read_json(run / "design-freeze.json")
    if fixed["config"] != config or fixed["runtime"] != core.old.previous.runtime():
        raise ValueError("design/runtime changed")
    for path, sha in fixed["files"].items():
        checked(ROOT / path, sha)
    if split == "test":
        thresholds = read_json(run / "evaluation-freeze.json")
        if thresholds["status"] != "thresholds_frozen_before_test_inference":
            raise ValueError("test requires frozen validation thresholds")
        for path, sha in thresholds["files"].items():
            checked(run / path, sha)
    inputs = read_json(run / f"{split}-inputs.json")
    validate_plans(inputs, config)
    return inputs


def infer(config, run, split):
    inputs = frozen(config, run, split)
    training = ROOT / config["training_run"]
    export = torch.load(
        training / "bundle/encoder.pt", map_location="cpu", weights_only=True
    )
    if tuple(export["phonemes"]) != PHONES or export["checkpoint_update"] != 30000:
        raise ValueError("unexpected fixed encoder")
    model = core.create_encoder("statistics_mlp").eval().requires_grad_(False)
    model.load_state_dict(export["model"], strict=True)
    before = {k: value.clone() for k, value in model.state_dict().items()}
    settings = read_json(
        ROOT / "poc/phoneme-speaker-encoder/config/baseline-log-mel.json"
    )
    pipeline = core.InputPipeline(
        settings["input"],
        rms_enabled=False,
        statistics=read_json(training / "bundle/feature-statistics.json"),
    )
    rows = {}
    for item in [*inputs["profiles"].values(), *inputs["queries"]]:
        for selected in item["conditions"].values():
            for row in selected:
                if row["segment_id"] in rows and rows[row["segment_id"]] != row:
                    raise ValueError("conflicting source slice")
                rows[row["segment_id"]] = row
    segments = [
        core.Segment(
            segment_id=r["segment_id"],
            speaker_id="unused",
            vowel=r["vowel"],
            split=split,
            role="verification",
            cohorts=(),
            source_file=r["source_file"],
            start_frame=r["start_frame"],
            end_frame=r["end_frame"],
            source_sha256=r["source_sha256"],
        )
        for r in sorted(rows.values(), key=lambda r: r["segment_id"])
    ]
    dataset = core.SegmentDataset(ROOT, segments, pipeline, mode="center")
    vectors, repeats = {}, 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(segments), 64):
            batch = core.collate_segments(
                [dataset[i] for i in range(offset, min(offset + 64, len(segments)))]
            )
            embeddings = model(batch["input"], batch["mask"])
            if not torch.isfinite(embeddings).all():
                raise ValueError("nonfinite embedding")
            for _ in range(2):
                if not torch.equal(embeddings, model(batch["input"], batch["mask"])):
                    raise ValueError("inference repeat mismatch")
                repeats += len(batch["segment_ids"])
            value = embeddings.numpy().astype(np.float64)
            norms = np.linalg.norm(value, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError("zero embedding")
            vectors.update(zip(batch["segment_ids"], value / norms))
    profiles = {}
    for speaker, item in inputs["profiles"].items():
        for phone, selected in item["conditions"].items():
            mean = np.mean([vectors[r["segment_id"]] for r in selected], axis=0)
            profiles[f"{phone}/{speaker}"] = mean / np.linalg.norm(mean)
    query_vectors, scores = {}, []
    for query in inputs["queries"]:
        for phone, selected in query["conditions"].items():
            if len(selected) != 1:
                raise ValueError("single fixed query token required")
            vector = vectors[selected[0]["segment_id"]]
            query_vectors[f"{phone}/{query['query_id']}"] = vector
            for claimed in inputs["speakers"]:
                score = float(vector @ profiles[f"{phone}/{claimed}"])
                scores.append(
                    {
                        **{
                            k: query[k]
                            for k in (
                                "query_id",
                                "speaker_id",
                                "split",
                                "role",
                                "common8",
                            )
                        },
                        "condition": phone,
                        "claimed_speaker_id": claimed,
                        "is_genuine": claimed == query["speaker_id"],
                        "status": "scored",
                        "score": score,
                        "phone_scores": {phone: score},
                        "used_frames": query["budget_frames"],
                    }
                )
    if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise ValueError("encoder modified by inference")
    validate_scores(inputs, scores, config)
    output = run / split
    output.mkdir()
    with (output / "scores.jsonl").open("x") as stream:
        for row in scores:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    np.savez_compressed(output / "query-vectors.npz", **query_vectors)
    np.savez_compressed(output / "profile-vectors.npz", **profiles)
    write_json(
        output / "inference.json",
        {
            "status": "completed",
            "unique_embeddings": len(vectors),
            "repeated_embeddings": repeats,
            "all_repeats_bitwise_equal": True,
            "encoder_unchanged": True,
            "elapsed_seconds": time.perf_counter() - started,
            "outputs_sha256": {
                name: sha256_file(output / name)
                for name in ("scores.jsonl", "query-vectors.npz", "profile-vectors.npz")
            },
        },
    )
    frozen(config, run, split)
    print(
        f"{split}: {len(scores)} trials; {len(vectors)} fixed 50ms embeddings",
        flush=True,
    )


def validate_scores(inputs, rows, config):
    queries = {q["query_id"]: q for q in inputs["queries"]}
    seen = set()
    for row in rows:
        key = row["query_id"], row["condition"], row["claimed_speaker_id"]
        query = queries.get(row["query_id"])
        if (
            query is None
            or key in seen
            or row["condition"] not in PHONES
            or row["claimed_speaker_id"] not in inputs["speakers"]
        ):
            raise ValueError("invalid/duplicate trial identity")
        seen.add(key)
        if (
            any(row[k] != query[k] for k in ("speaker_id", "split", "role", "common8"))
            or row["is_genuine"] != (row["speaker_id"] == row["claimed_speaker_id"])
            or row["status"] != "scored"
            or row["phone_scores"] != {row["condition"]: row["score"]}
            or not np.isfinite(row["score"])
            or not -1 <= row["score"] <= 1
            or row["used_frames"] != config["segment_frames"]
        ):
            raise ValueError("wrong labels, fusion, duration or finite score")
    if len(seen) != len(queries) * len(PHONES) * len(inputs["speakers"]):
        raise ValueError("incomplete eight-phone trial matrix")


def load_groups(config, run, split):
    inputs = frozen(config, run, split)
    for name, sha in read_json(run / split / "inference.json")[
        "outputs_sha256"
    ].items():
        checked(run / split / name, sha)
    with (run / split / "scores.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    validate_scores(inputs, rows, config)
    return inputs, {
        (phone, role): [
            r for r in rows if r["condition"] == phone and r["role"] == role
        ]
        for phone in PHONES
        for role in ROLES
    }


def calibrate(config, run):
    inputs, groups = load_groups(config, run, "validation")
    thresholds, cells = {}, {}
    with np.load(run / "validation/query-vectors.npz", allow_pickle=False) as vectors:
        for phone in PHONES:
            threshold = core.old.calibrate_cell(
                groups[phone, "verification"], inputs["speakers"]
            )
            if threshold["status"] != "calibrated":
                raise ValueError("missing validation support")
            thresholds[phone] = threshold
            for role in ROLES:
                rows = groups[phone, role]
                cell, _ = core.old.evaluate_cell(rows, threshold, inputs["speakers"])
                cell["diagnostics"] = describe(
                    rows, inputs, phone, role, vectors, threshold
                )
                cells[f"{phone}/{role}"] = cell
    write_json(run / "validation-thresholds.json", thresholds)
    write_json(run / "validation-metrics.json", {"conditions": cells})
    _, counts = core.old.speaker_draws(
        len(inputs["speakers"]),
        config["bootstrap_replicates"],
        config["bootstrap_seed"],
    )
    np.save(run / "bootstrap-counts.npy", counts, allow_pickle=False)
    names = (
        "validation-thresholds.json",
        "validation-metrics.json",
        "bootstrap-counts.npy",
        "validation/inference.json",
        "validation/scores.jsonl",
        "validation/query-vectors.npz",
        "validation/profile-vectors.npz",
    )
    write_json(
        run / "evaluation-freeze.json",
        {
            "status": "thresholds_frozen_before_test_inference",
            "files": {name: sha256_file(run / name) for name in names},
        },
    )
    print("eight validation thresholds frozen before test", flush=True)


def evaluate(config, run):
    inputs, groups = load_groups(config, run, "test")
    thresholds = read_json(run / "validation-thresholds.json")
    counts = np.load(run / "bootstrap-counts.npy", allow_pickle=False)
    cells, replicas = {}, {}
    with np.load(run / "test/query-vectors.npz", allow_pickle=False) as vectors:
        for (phone, role), rows in groups.items():
            cell, _ = core.old.evaluate_cell(
                rows, thresholds[phone], inputs["speakers"]
            )
            cell["ci95"], replicas[f"{phone}/{role}"] = core.old.bootstrap_cell(
                rows, cell, thresholds[phone], inputs["speakers"], counts
            )
            cell["diagnostics"] = describe(
                rows, inputs, phone, role, vectors, thresholds[phone]
            )
            cells[f"{phone}/{role}"] = cell
            print(f"test metrics: {phone}/{role}", flush=True)
    differences, arrays = {}, {}
    for a, b in PAIRS:
        for role in ROLES:
            ka, kb = f"{a}/{role}", f"{b}/{role}"
            for metric in (
                "pooled_eer",
                "far_1pct/all_input_far",
                "far_1pct/all_input_frr",
            ):
                key = f"{b}_minus_{a}/{role}/{metric}"
                delta = 100 * (replicas[kb][metric] - replicas[ka][metric])
                arrays[key] = delta
                differences[key] = {
                    "minuend": b,
                    "subtrahend": a,
                    "difference_percentage_points": 100
                    * (
                        core.old.metric_value(cells[kb], metric)
                        - core.old.metric_value(cells[ka], metric)
                    ),
                    "ci95_percentage_points": core.old.interval(delta),
                }
    np.savez_compressed(
        run / "bootstrap-metrics.npz",
        **{
            f"{key}/{name}": value
            for key, values in replicas.items()
            for name, value in values.items()
        },
    )
    np.savez_compressed(run / "bootstrap-differences.npz", **arrays)
    write_json(
        run / "test-metrics.json",
        {
            "conditions": cells,
            "paired_differences": differences,
            "primary_hypothesis": primary_results(
                config, cells, replicas, core.old.interval
            ),
            "threshold_recalibrated_on_test": False,
        },
    )
    frozen(config, run, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validation", "test"))
    args = parser.parse_args()
    config = read_json(CONFIG)
    if (
        config["phonemes"] != list(PHONES)
        or config["segment_frames"] != 1200
        or config["sample_rate"] != 24000
        or config["primary_family_size"] != 2
        or config["enrollment_segments_per_phone"] != 10
        or config["query_segments_per_phone"] != 1
        or config["primary_contrasts"] != [["m", "s"], ["n", "s"]]
        or config["primary_role"] != "verification"
        or config["primary_ci_confidence"] != 0.975
        or config["additional_training"]
        or config["independent_holdout"]
    ):
        raise ValueError("unsupported fixed discriminability protocol")
    torch.set_num_threads(1)
    run = ROOT / config["run_directory"]
    if args.stage == "prepare":
        prepare(config, run)
    elif args.stage == "validation":
        infer(config, run, "validation")
        calibrate(config, run)
    else:
        infer(config, run, "test")
        evaluate(config, run)


if __name__ == "__main__":
    main()
