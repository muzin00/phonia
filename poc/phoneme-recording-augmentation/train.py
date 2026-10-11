"""Matched-budget clean versus corruption-augmented encoder fine-tuning."""

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
    source = ROOT / config["quality_run"]
    previous = t.verify(source)
    old = previous["config"]
    selection = s.read_json(source / "selection.json")
    permitted = {
        r["source_file"] for r in selection["quality"] if r["role"] == "training"
    }
    train_speakers = {
        r["speaker_id"] for r in selection["quality"] if r["role"] == "training"
    }
    eval_speakers = {
        r["speaker_id"] for r in selection["quality"] if r["role"] != "training"
    }
    assert not train_speakers & eval_speakers
    model, _ = s.original.load_model(
        ROOT / old["encoder_run"] / "trials" / old["encoder_trial"]
    )
    del model
    phones = s.read_json(
        ROOT
        / old["encoder_run"]
        / "trials"
        / old["encoder_trial"]
        / "training-summary.json"
    )["phonemes"]
    grouped = defaultdict(list)
    for row in s.rows(ROOT / old["encoder_run"] / "train-segments.jsonl"):
        if row["source_file"] in permitted and row["phoneme"] in phones:
            assert row["split"] == "train" and not row["quality_flags"]
            grouped[row["speaker_id"], row["phoneme"]].append(row)
    chosen = []
    for _, rows in sorted(grouped.items()):
        selected = t.h.primitives.diverse(
            rows,
            config["maximum_intervals_per_speaker_phone"],
            config["selection_seed"],
        )
        if len(selected) >= 2:
            chosen.extend(selected)
    supported = [p for p in phones if any(r["phoneme"] == p for r in chosen)]
    assert len(train_speakers) == 140 and len(chosen) > 10000
    eval_hashes = {
        r["source_sha256"] for r in selection["quality"] if r["role"] != "training"
    }
    assert not eval_hashes & {r["source_sha256"] for r in chosen}
    files = dict(previous["files"])
    for path in [
        CONFIG,
        BASE / "README.md",
        BASE / "train.py",
        BASE / "evaluate.py",
        Path(learner.__file__),
        Path(qr.__file__),
        ROOT / old["encoder_run"] / "train-segments.jsonl",
        ROOT / old["encoder_run"] / "train-features.f32",
        ROOT / old["encoder_run"] / "preparation-report.json",
        ROOT / old["encoder_run"] / "trials" / old["encoder_trial"] / "last.pt",
    ]:
        t.pin(files, path)
    for split in ("validation", "test"):
        for path in (source / split).rglob("*"):
            if path.is_file():
                t.pin(files, path)
        for path in [
            ROOT / old["encoder_run"] / f"{split}-features.f32",
            ROOT / old["cv_run"] / split / "features.npy",
        ]:
            t.pin(files, path)
    run.mkdir(parents=True)
    s.write_rows(run / "train-selection.jsonl", chosen)
    t.pin(files, run / "train-selection.jsonl")
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "source_config": old,
            "extract_config": previous["extract_config"],
            "phones": supported,
            "untrained_in_this_probe": sorted(set(phones) - set(supported)),
            "training_speakers": sorted(train_speakers),
            "files": files,
            "intervals": len(chosen),
            "status": "frozen_before_augmented_feature_extraction_and_fine_tuning",
        },
    )
    print(
        f"Frozen {len(chosen)} intervals, {len(supported)} phones, 140 training speakers",
        flush=True,
    )


def prepare(config, run):
    design = verify(config, run)
    old = design["source_config"]
    selected = list(s.rows(run / "train-selection.jsonl"))
    by_source = defaultdict(list)
    for index, row in enumerate(selected):
        by_source[row["source_file"]].append(index)
    pipe = t.h.primitives.pipeline(design["extract_config"])
    shape = (len(config["conditions"]), len(selected), 128)
    bank = np.lib.format.open_memmap(
        run / "train-features.npy", mode="w+", dtype=np.float32, shape=shape
    )
    original = np.memmap(
        ROOT / old["encoder_run"] / "train-features.f32", mode="r", dtype="<f4"
    ).reshape(-1, 128)
    maximum = 0.0
    clipping = 0.0
    for number, (source, indices) in enumerate(sorted(by_source.items()), 1):
        raw = a.read_wave(ROOT / source)
        for mode, condition in enumerate(config["conditions"]):
            y, diag = a.transform(
                raw, condition, a.seed_for(config["augmentation_seed"], source)
            )
            clipping = max(clipping, diag["pre_quantization_clip_fraction"])
            for index in indices:
                row = selected[index]
                feature = t.h.primitives.pool_features(
                    pipe, y[row["start_frame"] : row["end_frame"]], row["segment_id"]
                )
                bank[mode, index] = feature
                if mode == 0:
                    maximum = max(
                        maximum,
                        float(np.max(abs(feature - original[row["cache_index"]]))),
                    )
        if number % 100 == 0:
            print(
                f"Augmented train features {number}/{len(by_source)} WAVs", flush=True
            )
    bank.flush()
    assert maximum < 2e-6
    s.write_json(
        run / "feature-freeze.json",
        {
            "files": {
                t.rel(run / "train-features.npy"): s.sha256_file(
                    run / "train-features.npy"
                )
            },
            "clean_feature_replay_maximum_error": maximum,
            "maximum_clipped_sample_fraction": clipping,
            "sources": len(by_source),
            "intervals": len(selected),
            "status": "train_only",
        },
    )
    print("Prepared train-only six-view feature bank", flush=True)


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
        views = torch.from_numpy(views)
        for arm in config["arms"]:
            dest = run / f"{arm}-{seed}"
            dest.mkdir(exist_ok=True)
            if (dest / "complete.json").exists():
                for name, digest in s.read_json(dest / "complete.json")[
                    "files"
                ].items():
                    s.checked(ROOT / name, digest)
                continue
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
                if arm == "augmented":
                    inputs = features[views[u], indices[u]]
                else:
                    inputs = features[0, indices[u]]
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
                            run / f"schedule-{seed}.npz",
                        ]
                    },
                    "status": "fixed_final_update",
                    "initial_checkpoint_sha256": s.sha256_file(initial),
                    "training_examples": int(valid.sum()),
                    "test_used_for_checkpoint_selection": False,
                },
            )
    print("Completed matched-budget fine-tuning", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "prepare", "train"))
    args = parser.parse_args()
    config = s.read_json(CONFIG)
    run = ROOT / config["run_directory"]
    torch.set_num_threads(1)
    {"freeze": freeze, "prepare": prepare, "train": train}[args.stage](config, run)


if __name__ == "__main__":
    main()
