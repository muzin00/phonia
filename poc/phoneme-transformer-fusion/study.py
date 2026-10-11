"""Train fusion only; select on validation and freeze all seeds before test."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

from backend import Fusion, pair_features
from data import (
    CONFIG,
    ROOT,
    checked,
    expanded,
    load_arrays,
    metrics,
    np,
    prepare,
    read_json,
    rows,
    sha256_file,
    thresholds,
    torch,
    verify,
    write_json,
    write_rows,
)


def create_model(config, kind):
    return Fusion(
        kind,
        dimension=config["dimension"],
        layers=config["transformer_layers"],
        heads=config["attention_heads"],
    )


def batch(arrays, query_ids, claims, mask=None):
    features, components = pair_features(
        arrays["enrollment"][claims],
        arrays["emeta"][claims],
        arrays["queries"][query_ids],
        arrays["qmeta"][query_ids],
    )
    available = arrays["emask"][claims] & arrays["qmask"][query_ids]
    return features, components, available if mask is None else available & mask


def training_batch(arrays, data, schedule, update):
    q = torch.from_numpy(schedule["queries"][update].astype(np.int64))
    negative = torch.from_numpy(schedule["negative_claims"][update].astype(np.int64))
    index = {s: i for i, s in enumerate(data["speakers"])}
    genuine = torch.tensor([index[data["queries"][int(i)]["speaker_id"]] for i in q])
    shared = arrays["qmask"][q] & arrays["emask"][genuine] & arrays["emask"][negative]
    if not shared[:, :5].all():
        raise ValueError("training pair lacks a fixed vowel")
    features, components, mask = batch(
        arrays,
        torch.cat((q, q)),
        torch.cat((genuine, negative)),
        torch.cat((shared, shared)),
    )
    labels = torch.cat((torch.ones(len(q)), torch.zeros(len(q))))
    return features, components, mask, labels


def score(model, arrays, baseline, phones):
    """Original components/support, with learned nonnegative normalized weights."""
    model.eval()
    result = copy.deepcopy(baseline)
    phone_index = {p: i for i, p in enumerate(phones)}
    active = [(i, r) for i, r in enumerate(result) if r["status"] == "scored"]
    # Baseline rows have one complete, ordered block of claims per query.
    speakers = arrays["enrollment"].shape[0]
    with torch.inference_mode():
        for start in range(0, len(active), 128):
            chosen = active[start : start + 128]
            ids = torch.tensor([i // speakers for i, _ in chosen])
            claims = torch.tensor([i % speakers for i, _ in chosen])
            mask = torch.zeros((len(chosen), len(phones)), dtype=torch.bool)
            for j, (_, row) in enumerate(chosen):
                mask[j, [phone_index[p] for p in row["used_phones"]]] = True
            features, components, mask = batch(arrays, ids, claims, mask)
            _, weights = model(features, components, mask)
            for j, (_, row) in enumerate(chosen):
                w = {p: float(weights[j, phone_index[p]]) for p in row["used_phones"]}
                # Normalize in float64 so published weights sum exactly to one.
                total = sum(w.values())
                row["phone_weights"] = {p: value / total for p, value in w.items()}
                row["score"] = sum(
                    row["phone_weights"][p] * row["phone_scores"][p] for p in w
                )
    return result


def variant(kind, seed):
    return f"{kind}-{seed}"


def train(config, run):
    frozen = verify(run)
    data = read_json(run / "train-inputs.json")
    arrays = load_arrays(run, "train")
    val_arrays = load_arrays(run, "validation")
    baseline = list(
        rows(ROOT / config["evaluation_run"] / "enrollment-30/validation-scores.jsonl")
    )
    for seed in config["seeds"]:
        with np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as archive:
            schedule = {k: archive[k] for k in archive.files}
        for kind in config["architectures"]:
            directory = run / variant(kind, seed)
            if (directory / "training-summary.json").exists():
                continue
            if directory.exists():
                raise ValueError(
                    "partial training needs a fresh run, not silent overwrite"
                )
            directory.mkdir()
            expanded.seed_everything(seed)
            model = create_model(config, kind)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            history, best = [], math.inf
            start = time.perf_counter()
            for update in range(config["updates"] + 1):
                if update > 0:
                    model.train()
                    learning_rate = config["learning_rate"] * (
                        0.1
                        + 0.9
                        * (1 + math.cos(math.pi * (update - 1) / config["updates"]))
                        / 2
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    features, components, mask, labels = training_batch(
                        arrays, data, schedule, update - 1
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(
                        features,
                        components,
                        mask,
                        labels,
                        config["uniform_kl_coefficient"],
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                if update % config["validation_every"]:
                    continue
                values = score(model, val_arrays, baseline, frozen["phones"])
                points = thresholds(values)
                report = metrics(values, points)
                eer = report["verification"]["eer"]
                history.append(
                    {
                        "update": update,
                        "normal_eer": eer,
                        "cross_text_eer": report["cross_text_verification"]["eer"],
                        "training_loss": float(loss.detach()) if update else None,
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
                if eer < best - 1e-12:
                    best = eer
                    selected_update = update
                    torch.save(
                        {
                            "kind": kind,
                            "seed": seed,
                            "update": update,
                            "state_dict": model.state_dict(),
                        },
                        directory / "fusion.pt",
                    )
                    selected_values, selected_points, selected_report = (
                        values,
                        points,
                        report,
                    )
                print(
                    json.dumps(
                        {
                            "variant": directory.name,
                            **history[-1],
                            "selected_update": selected_update,
                        }
                    ),
                    flush=True,
                )
            write_rows(directory / "validation-scores.jsonl", selected_values)
            write_json(
                directory / "validation-thresholds.json",
                {
                    "split": "validation",
                    "role": "verification",
                    "thresholds": selected_points,
                },
            )
            write_json(
                directory / "validation-metrics.json", {"metrics": selected_report}
            )
            write_json(
                directory / "training-summary.json",
                {
                    "kind": kind,
                    "seed": seed,
                    "completed_updates": config["updates"],
                    "selected_update": selected_update,
                    "history": history,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "selected_model_sha256": sha256_file(directory / "fusion.pt"),
                    "schedule_sha256": sha256_file(run / f"schedule-{seed}.npz"),
                    "encoder_sha256": config["encoder_sha256"],
                    "encoder_updated": False,
                    "pairs_presented": config["updates"]
                    * config["queries_per_update"]
                    * 2,
                },
            )
    verify(run)
    models = {}
    for seed in config["seeds"]:
        for kind in config["architectures"]:
            directory = run / variant(kind, seed)
            models[directory.name] = {
                name: sha256_file(directory / name)
                for name in (
                    "fusion.pt",
                    "training-summary.json",
                    "validation-scores.jsonl",
                    "validation-thresholds.json",
                    "validation-metrics.json",
                )
            }
    write_json(
        run / "selection-freeze.json",
        {
            "status": "all_six_models_and_validation_thresholds_frozen_before_new_test_scores",
            "models": models,
            "design_sha256": sha256_file(run / "design-freeze.json"),
            "test_previously_observed": True,
        },
    )


def evaluate(config, run):
    frozen = verify(run)
    selection = read_json(run / "selection-freeze.json")
    checked(run / "design-freeze.json", selection["design_sha256"])
    for name, files in selection["models"].items():
        for file, checksum in files.items():
            checked(run / name / file, checksum)
    arrays = load_arrays(run, "test")
    baseline = list(
        rows(ROOT / config["evaluation_run"] / "enrollment-30/test-scores.jsonl")
    )
    for name in selection["models"]:
        directory = run / name
        if (directory / "test-metrics.json").exists():
            continue
        bundle = torch.load(
            directory / "fusion.pt", map_location="cpu", weights_only=True
        )
        model = create_model(config, bundle["kind"])
        model.load_state_dict(bundle["state_dict"])
        values = score(model, arrays, baseline, frozen["phones"])
        points = read_json(directory / "validation-thresholds.json")["thresholds"]
        report = metrics(values, points)
        write_rows(directory / "test-scores.jsonl", values)
        write_json(directory / "test-metrics.json", {"metrics": report})
        print(
            json.dumps({"stage": "test", "variant": name, "metrics": report}),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = read_json(CONFIG)
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](
        config, args.run
    )


if __name__ == "__main__":
    main()
