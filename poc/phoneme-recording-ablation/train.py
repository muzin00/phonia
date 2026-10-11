"""Single-condition and exposure-preserving leave-one-out fine-tuning."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-recording-quality"))
import report as qr

t = qr.t
learner = __import__("learner")
s = t.s
np = s.np
torch = s.torch
a = t.a
CONFIG = BASE / "config.json"


def verify(config, run):
    design = s.read_json(run / "design-freeze.json")
    assert design["config"] == config
    for name, digest in design["files"].items():
        s.checked(ROOT / name, digest)
    return design


def freeze(config, run):
    if run.exists():
        raise ValueError("fresh run required")
    source = ROOT / config["source_run"]
    previous = s.read_json(source / "design-freeze.json")
    for name, digest in previous["files"].items():
        s.checked(ROOT / name, digest)
    feature = s.read_json(source / "feature-freeze.json")
    for name, digest in feature["files"].items():
        s.checked(ROOT / name, digest)
    files = dict(previous["files"])
    files.update(feature["files"])
    for path in [
        CONFIG,
        BASE / "README.md",
        BASE / "train.py",
        BASE / "evaluate.py",
        BASE / "summarize.py",
        BASE / "verify.py",
        source / "design-freeze.json",
        source / "feature-freeze.json",
    ]:
        t.pin(files, path)
    for seed in config["seeds"]:
        t.pin(files, source / f"schedule-{seed}.npz")
        for arm in ("clean", "augmented"):
            complete = s.read_json(source / f"{arm}-{seed}" / "complete.json")
            for name, digest in complete["files"].items():
                s.checked(ROOT / name, digest)
                files[name] = digest
    for key in (
        "seeds",
        "updates",
        "learning_rate",
        "weight_decay",
        "warmup_updates",
        "conditions",
        "augmentation_seed",
    ):
        assert config[key] == previous["config"][key]
    run.mkdir(parents=True)
    for name in ("train-selection.jsonl", "train-features.npy"):
        (run / name).symlink_to((source / name).resolve())
    s.write_json(run / "feature-freeze.json", feature)
    s.write_json(
        run / "design-freeze.json",
        {
            **previous,
            "config": config,
            "files": files,
            "status": "all_arms_and_gate_frozen_before_training_or_scoring",
        },
    )
    print(
        "Frozen all 12 arms; reusing verified immutable training features", flush=True
    )


def arm_views(views, arm):
    result = views.copy()
    if arm == "clean":
        result[:] = 0
    elif arm.startswith("single"):
        result[:, 1::2] = int(arm[6:])
    elif arm.startswith("drop"):
        result[result == int(arm[4:])] = 0
    elif arm != "augmented":
        raise ValueError(arm)
    return result


def schedule(config, design, rows, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["speaker_id"], row["phoneme"]].append(i)
    speakers = design["training_speakers"]
    indices, classes, valid = learner.schedule(
        groups,
        design["phones"],
        config["updates"],
        seed,
        {sp: i for i, sp in enumerate(speakers)},
    )
    rng = np.random.default_rng(seed + 101)
    views = np.zeros_like(indices)
    views[:, 1::2] = rng.integers(
        1, len(config["conditions"]), size=(config["updates"], 50)
    )
    return indices, classes, valid, views


def train(config, run):
    design = verify(config, run)
    old = design["source_config"]
    for name, digest in s.read_json(run / "feature-freeze.json")["files"].items():
        s.checked(ROOT / name, digest)
    rows = list(s.rows(run / "train-selection.jsonl"))
    features = torch.from_numpy(np.load(run / "train-features.npy", allow_pickle=False))
    initial = ROOT / old["encoder_run"] / "trials" / old["encoder_trial"] / "last.pt"
    state = torch.load(initial, map_location="cpu", weights_only=False)
    source_speakers = s.read_json(
        ROOT / old["encoder_run"] / "preparation-report.json"
    )["training_speakers"]
    assert source_speakers == design["training_speakers"]
    reference, _ = s.original.load_model(initial.parent)
    for k, v in reference.state_dict().items():
        assert torch.equal(v, state["model"][k])
    for seed in config["seeds"]:
        indices, classes, valid, views = schedule(config, design, rows, seed)
        np.savez(
            run / f"schedule-{seed}.npz",
            indices=indices,
            classes=classes,
            valid=valid,
            views=views,
        )
        indices = torch.from_numpy(np.maximum(indices, 0))
        classes = torch.from_numpy(classes)
        valid = torch.from_numpy(valid)
        for arm in config["arms"]:
            actual_views = arm_views(views, arm)
            view_tensor = torch.from_numpy(actual_views)
            dest = run / f"{arm}-{seed}"
            dest.mkdir(exist_ok=True)
            if (dest / "complete.json").exists():
                for name, digest in s.read_json(dest / "complete.json")[
                    "files"
                ].items():
                    s.checked(ROOT / name, digest)
                continue
            if arm in ("clean", "augmented"):
                source = ROOT / config["source_run"] / f"{arm}-{seed}"
                for filename in ("encoder.pt", "history.json", "complete.json"):
                    (dest / filename).symlink_to((source / filename).resolve())
                print(f"Reused verified {arm}-{seed}", flush=True)
                continue
            np.save(dest / "views.npy", actual_views, allow_pickle=False)
            torch.manual_seed(seed)
            model = learner.create_encoder("statistics_mlp")
            model.load_state_dict(state["model"])
            model.train()
            head = learner.AAMSoftmax(len(source_speakers))
            head.load_state_dict(state["head"])
            head.train()
            params = [*model.parameters(), *head.parameters()]
            optimizer = torch.optim.AdamW(
                params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
            )
            started = time.perf_counter()
            history = []
            for u in range(config["updates"]):
                inputs = features[view_tensor[u], indices[u]]
                embeddings = torch.nn.functional.normalize(
                    model.projection(inputs), dim=1, eps=model.l2_epsilon
                )
                aam = learner.grouped_aam(head, embeddings, classes[u], valid[u])
                supcon = learner.grouped_supcon(embeddings, valid[u])
                loss = aam + 0.5 * supcon
                assert torch.isfinite(loss)
                warmup = config["warmup_updates"]
                progress = max(0.0, (u - warmup) / (config["updates"] - warmup))
                scale = (
                    (u + 1) / warmup
                    if u < warmup
                    else 0.5 * (1 + math.cos(math.pi * progress))
                )
                optimizer.param_groups[0]["lr"] = config["learning_rate"] * scale
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
                optimizer.step()
                if (u + 1) % 1000 == 0:
                    entry = {
                        "update": u + 1,
                        "loss": float(loss.detach()),
                        "aam": float(aam.detach()),
                        "supcon": float(supcon.detach()),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    history.append(entry)
                    print(f"{arm}-{seed} {entry}", flush=True)
            model.eval()
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "seed": seed,
                    "arm": arm,
                    "updates": config["updates"],
                },
                dest / "encoder.pt",
            )
            s.write_json(dest / "history.json", history)
            s.write_json(
                dest / "complete.json",
                {
                    "files": {
                        t.rel(p): s.sha256_file(p)
                        for p in [
                            dest / "encoder.pt",
                            dest / "history.json",
                            dest / "views.npy",
                            run / f"schedule-{seed}.npz",
                        ]
                    },
                    "status": "fixed_final_update",
                    "initial_checkpoint_sha256": s.sha256_file(initial),
                    "training_examples": int(valid.sum()),
                    "test_used_for_checkpoint_selection": False,
                    "valid_view_counts": np.bincount(
                        actual_views[valid.numpy()], minlength=6
                    ).tolist(),
                },
            )
    print("Completed matched-budget fine-tuning", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "train"))
    args = parser.parse_args()
    config = s.read_json(CONFIG)
    run = ROOT / config["run_directory"]
    torch.set_num_threads(1)
    {"freeze": freeze, "train": train}[args.stage](config, run)


if __name__ == "__main__":
    main()
