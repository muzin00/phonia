"""The original training protocol with only train-input augmentation changed."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import shared as s
from augmentation import augmented_batch, prepare


def load_npz(path):
    with s.np.load(path, allow_pickle=False) as archive:
        return {name: s.torch.from_numpy(archive[name]) for name in archive.files}


def case_arrays(config, run, case):
    if case == "clean":
        return s.frozen.load_arrays(s.source(config), "test")
    name = "missing" if case == "missing-half-consonants" else "noisy"
    return load_npz(run / f"test-{name}-aggregates.npz")


def train(config, run):
    design = s.verify(config, run)
    protocol = design["training_protocol"]
    previous = s.source(config)
    inputs = s.read_json(previous / "train-inputs.json")
    clean = s.frozen.load_arrays(previous, "train")
    noisy = load_npz(run / "train-noisy-aggregates.npz")
    validation = s.frozen.load_arrays(previous, "validation")
    baseline = list(
        s.frozen.rows(s.original_case(config, "clean") / "validation-scores.jsonl")
    )
    for seed in protocol["seeds"]:
        with s.np.load(
            previous / f"schedule-{seed}.npz", allow_pickle=False
        ) as archive:
            pairs = {k: archive[k] for k in archive.files}
        with s.np.load(run / f"augmentation-{seed}.npz", allow_pickle=False) as archive:
            augmentation = {k: archive[k] for k in archive.files}
        for kind in protocol["architectures"]:
            directory = run / s.fusion.variant(kind, seed)
            if (directory / "training-summary.json").exists():
                continue
            if directory.exists():
                raise ValueError("partial run cannot be overwritten")
            directory.mkdir()
            s.frozen.expanded.seed_everything(seed)
            model = s.fusion.create_model(protocol, kind)
            initial_sha = s.frozen.expanded.model_sha256(model)
            optimizer = s.torch.optim.AdamW(
                model.parameters(),
                lr=protocol["learning_rate"],
                weight_decay=protocol["weight_decay"],
            )
            history, best, selected_update = [], math.inf, 0
            started = time.perf_counter()
            for update in range(protocol["updates"] + 1):
                if update:
                    model.train()
                    rate = protocol["learning_rate"] * (
                        0.1
                        + 0.9
                        * (1 + math.cos(math.pi * (update - 1) / protocol["updates"]))
                        / 2
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    features, components, mask, labels = augmented_batch(
                        clean, noisy, inputs, pairs, augmentation, update - 1
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(
                        features,
                        components,
                        mask,
                        labels,
                        protocol["uniform_kl_coefficient"],
                    )
                    if not s.torch.isfinite(loss):
                        raise ValueError("nonfinite augmented training loss")
                    loss.backward()
                    s.torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                if update % protocol["validation_every"]:
                    continue
                values = s.fusion.score(model, validation, baseline, design["phones"])
                points = s.frozen.thresholds(values)
                report = s.frozen.metrics(values, points)
                eer = report["verification"]["eer"]
                history.append(
                    {
                        "update": update,
                        "normal_eer": eer,
                        "cross_text_eer": report["cross_text_verification"]["eer"],
                        "training_loss": float(loss.detach()) if update else None,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if eer < best - 1e-12:
                    best, selected_update = eer, update
                    s.torch.save(
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
            s.frozen.write_rows(directory / "validation-scores.jsonl", selected_values)
            s.write_json(
                directory / "validation-thresholds.json",
                {
                    "split": "validation",
                    "role": "verification",
                    "thresholds": selected_points,
                },
            )
            s.write_json(
                directory / "validation-metrics.json", {"metrics": selected_report}
            )
            s.write_json(
                directory / "training-summary.json",
                {
                    "kind": kind,
                    "seed": seed,
                    "completed_updates": protocol["updates"],
                    "selected_update": selected_update,
                    "history": history,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "initial_model_state_sha256": initial_sha,
                    "selected_model_sha256": s.sha256_file(directory / "fusion.pt"),
                    "pairs_schedule_sha256": s.sha256_file(
                        previous / f"schedule-{seed}.npz"
                    ),
                    "augmentation_schedule_sha256": s.sha256_file(
                        run / f"augmentation-{seed}.npz"
                    ),
                    "presented_queries_by_mode": s.np.bincount(
                        augmentation["modes"].ravel(), minlength=4
                    ).tolist(),
                    "pairs_presented": protocol["updates"]
                    * protocol["queries_per_update"]
                    * 2,
                    "encoder_sha256": protocol["encoder_sha256"],
                    "encoder_updated": False,
                },
            )
    s.verify(config, run)
    models = {}
    for seed in protocol["seeds"]:
        for kind in protocol["architectures"]:
            directory = run / s.fusion.variant(kind, seed)
            models[directory.name] = {
                name: s.sha256_file(directory / name)
                for name in (
                    "fusion.pt",
                    "training-summary.json",
                    "validation-scores.jsonl",
                    "validation-thresholds.json",
                    "validation-metrics.json",
                )
            }
    s.write_json(
        run / "selection-freeze.json",
        {
            "status": "all_six_augmented_models_and_clean_validation_thresholds_frozen_before_new_test",
            "design_sha256": s.sha256_file(run / "design-freeze.json"),
            "models": models,
            "test_scenarios": config["test_scenarios"],
            "test_previously_observed": True,
        },
    )


def evaluate(config, run):
    design = s.verify(config, run)
    selection = s.selected_models(config, run)
    if selection["test_scenarios"] != config["test_scenarios"]:
        raise ValueError("test scenarios changed after checkpoint selection")
    for case in config["test_scenarios"]:
        baseline = list(
            s.frozen.rows(s.original_case(config, case) / "test-scores.jsonl")
        )
        arrays = case_arrays(config, run, case)
        for name in selection["models"]:
            directory = run / name
            output = directory / case
            if (output / "metrics.json").exists():
                continue
            output.mkdir()
            bundle = s.torch.load(
                directory / "fusion.pt", map_location="cpu", weights_only=True
            )
            model = s.fusion.create_model(design["training_protocol"], bundle["kind"])
            model.load_state_dict(bundle["state_dict"])
            values = s.fusion.score(model, arrays, baseline, design["phones"])
            points = s.read_json(directory / "validation-thresholds.json")["thresholds"]
            report = {"metrics": s.frozen.metrics(values, points)}
            s.frozen.write_rows(output / "scores.jsonl", values)
            s.write_json(output / "metrics.json", report)
            print(
                json.dumps(
                    {
                        "case": case,
                        "variant": name,
                        "eer": {r: m["eer"] for r, m in report["metrics"].items()},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    settings = s.read_json(s.CONFIG)
    destination = args.run or s.ROOT / settings["run_directory"]
    {"prepare": prepare, "train": train, "evaluate": evaluate}[args.command](
        settings, destination
    )
