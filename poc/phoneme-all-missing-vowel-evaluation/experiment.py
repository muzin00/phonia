"""Freeze the support policies, calibrate all thresholds, then evaluate test."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import shared as s


def prepare(config, run):
    if run.exists():
        raise ValueError("prepare requires an unused run directory")
    previous = s.source(config)
    frozen = s.original.verify(previous, full=True)
    if config["retrain"] or config["enrollment_limit"] != 30:
        raise ValueError("only fixed enrollment-30 models are supported")
    selection = s.read_json(previous / "selection-freeze.json")
    completion = s.read_json(previous / "completion-verification.json")
    for model in config["models"]:
        if model == "equal":
            continue
        for name, checksum in selection["models"][model].items():
            s.checked(previous / model / name, checksum)
    files = dict(frozen["files"])
    paths = [
        s.CONFIG,
        s.BASE / "README.md",
        *s.BASE.glob("*.py"),
        *s.BASE.glob("tests/*.py"),
    ]
    # Bind every reused Python module under this repository, including metrics.
    import sys

    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if (
            path
            and Path(path).resolve().is_relative_to(s.ROOT / "poc")
            and path.endswith(".py")
        ):
            paths.append(Path(path))
    paths.extend(
        [
            previous / "design-freeze.json",
            previous / "selection-freeze.json",
            previous / "completion-verification.json",
        ]
    )
    for split in ("validation", "test"):
        paths.extend(
            [
                previous / f"{split}-inputs.json",
                previous / f"{split}-aggregates.npz",
                s.control(config) / f"{split}-embeddings.npy",
            ]
        )
        for model in config["models"]:
            for suffix in ("scores.jsonl", "metrics.json"):
                path = s.original_path(config, model, split, suffix)
                if model != "equal":
                    s.checked(
                        path, completion["output_sha256"][str(path.relative_to(s.ROOT))]
                    )
                paths.append(path)
            paths.append(
                s.original_path(config, model, "validation", "thresholds.json")
            )
            if model != "equal":
                paths.append(previous / model / "fusion.pt")
    for path in paths:
        s.original.pin(files, path)
    run.mkdir(parents=True)
    s.write_json(
        run / "design-freeze.json",
        {
            "status": "support_policies_models_and_inputs_frozen_before_new_validation_or_test_scoring",
            "config": config,
            "runtime": s.original.expanded.runtime(),
            "phones": frozen["phones"],
            "training_protocol": frozen["config"],
            "files": files,
        },
    )
    print(f"Frozen {len(files)} input/code files; no retraining", flush=True)


def equal_candidates(config, split, phones):
    """Compute additional components, preserving every existing score exactly."""
    data = s.read_json(s.source(config) / f"{split}-inputs.json")
    vectors = s.np.load(
        s.control(config) / f"{split}-embeddings.npy", allow_pickle=False
    )
    original = list(s.rows(s.original_path(config, "equal", split, "scores.jsonl")))
    result = copy.deepcopy(original)
    use = set(phones) & set(data["universally_registered"])
    profiles = {}
    for speaker in data["speakers"]:
        for phone in use:
            mean = vectors[data["profiles"][speaker][phone]].mean(axis=0)
            profiles[speaker, phone] = mean / s.np.linalg.norm(mean)
    for q, query in enumerate(data["queries"]):
        available = sorted(p for p in use if query["groups"][p])
        means = {p: vectors[query["groups"][p]].mean(axis=0) for p in available}
        for c, speaker in enumerate(data["speakers"]):
            row = result[q * len(data["speakers"]) + c]
            if (row["query_id"], row["claimed_speaker_id"]) != (
                query["query_id"],
                speaker,
            ):
                raise ValueError("source trial order changed")
            if row["status"] == "scored":
                continue
            components = {p: float(means[p] @ profiles[speaker, p]) for p in available}
            row.update(
                phone_scores=components,
                used_phones=available,
                status="scored" if available else "no_score",
                score=float(s.np.mean(list(components.values())))
                if available
                else None,
                reason=None if available else "no_shared_phone",
            )
    return result


def learned_candidates(config, split, model_name, equal, design):
    previous = s.source(config)
    original = list(s.rows(s.original_path(config, model_name, split, "scores.jsonl")))
    result = copy.deepcopy(equal)
    pending = []
    for i, row in enumerate(original):
        if row["status"] == "scored":
            result[i] = row
        elif result[i]["status"] == "scored":
            pending.append(i)
    arrays = s.original.load_arrays(previous, split)
    bundle = s.torch.load(
        previous / model_name / "fusion.pt", map_location="cpu", weights_only=True
    )
    model = s.fusion.create_model(design["training_protocol"], bundle["kind"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    phone_index = {p: i for i, p in enumerate(design["phones"])}
    speakers = arrays["enrollment"].shape[0]
    with s.torch.inference_mode():
        for start in range(0, len(pending), 128):
            indices = pending[start : start + 128]
            qids = s.torch.tensor([i // speakers for i in indices])
            claims = s.torch.tensor([i % speakers for i in indices])
            mask = s.torch.zeros((len(indices), len(phone_index)), dtype=s.torch.bool)
            for j, i in enumerate(indices):
                mask[j, [phone_index[p] for p in result[i]["used_phones"]]] = True
            features, components, mask = s.fusion.batch(arrays, qids, claims, mask)
            _, weights = model(features, components, mask)
            for j, i in enumerate(indices):
                row = result[i]
                raw = {p: float(weights[j, phone_index[p]]) for p in row["used_phones"]}
                total = sum(raw.values())
                row["phone_weights"] = {p: w / total for p, w in raw.items()}
                row["score"] = sum(
                    row["phone_weights"][p] * row["phone_scores"][p] for p in raw
                )
    return result


def process(config, run, split):
    design = s.verify(config, run)
    if split == "test":
        frozen = s.selected(config, run)
    destination = run / split
    if destination.exists():
        raise ValueError("output exists; use a fresh run instead of overwriting")
    destination.mkdir()
    equal = equal_candidates(config, split, design["phones"])
    results, freeze_files, all_points = {}, {}, {}
    for model in config["models"]:
        candidates = (
            equal
            if model == "equal"
            else learned_candidates(config, split, model, equal, design)
        )
        path = destination / f"{model}-candidates.jsonl"
        s.write_rows(path, candidates)
        s.original.pin(freeze_files, path)
        baseline = list(s.rows(s.original_path(config, model, split, "scores.jsonl")))
        original_points = s.read_json(
            s.original_path(config, model, "validation", "thresholds.json")
        )["thresholds"]
        results[model], all_points[model] = {}, {}
        for policy in config["policies"]:
            name = policy["name"]
            values = s.apply_policy(candidates, policy)
            points = (
                s.original.thresholds(values)
                if split == "validation"
                else frozen["thresholds"][model][name]
            )
            all_points[model][name] = points
            reports = {
                "recalibrated": s.original.metrics(values, points),
                "original": s.original.metrics(values, original_points),
            }
            if name == "vowels5":
                expected = s.read_json(
                    s.original_path(config, model, split, "metrics.json")
                )["metrics"]
                if reports["recalibrated"] != expected or points != original_points:
                    raise ValueError("five-vowel control metrics or thresholds changed")
            for r, b in zip(values, baseline):
                if b["status"] == "scored" and (
                    r["score"] != b["score"] or r["phone_scores"] != b["phone_scores"]
                ):
                    raise ValueError("previously scored trial changed")
            results[model][name] = {
                "thresholds": points,
                "original_thresholds": original_points,
                "metrics": reports,
                "diagnostics": s.diagnostics(values, baseline, points, original_points),
            }
            normal = reports["recalibrated"][s.ROLES[0]]
            cross = reports["recalibrated"][s.ROLES[1]]
            print(
                f"{split} {model}/{name}: coverage {normal['scored_queries']}/{normal['all_queries']}, {cross['scored_queries']}/{cross['all_queries']}",
                flush=True,
            )
    s.write_json(destination / "results.json", results)
    s.original.pin(freeze_files, destination / "results.json")
    s.verify(config, run)
    if split == "validation":
        s.write_json(
            run / "threshold-freeze.json",
            {
                "status": "all_twelve_model_policy_thresholds_fixed_before_new_test_scoring",
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "thresholds": all_points,
                "files": freeze_files,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "calibrate", "evaluate"))
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    settings = s.read_json(s.CONFIG)
    run = args.run or s.ROOT / settings["run_directory"]
    if args.command == "prepare":
        prepare(settings, run)
    else:
        process(settings, run, "validation" if args.command == "calibrate" else "test")
