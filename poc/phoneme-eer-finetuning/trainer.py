"""Paired fine-tuning with identical starts, pair batches and atomic restart."""

from __future__ import annotations

import copy
import time

from common import (
    ALL_PHONES,
    ARMS,
    assert_nested_equal,
    atomic_torch,
    batch,
    check,
    create_model,
    load_validation,
    model_hash,
    pin,
    read,
    seed_all,
    sha,
    starting_path,
    state,
    torch,
    training_data,
    validation_eers,
    write,
)
from losses import objective


def train_loop(
    config,
    model,
    arrays,
    owners,
    schedule,
    directory,
    design_sha,
    arm,
    evaluate,
    stop_after=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    initial_hash = model_hash(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["design_sha256"] == design_sha and saved["arm"] == arm
        assert saved["initial_model_sha256"] == initial_hash
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        history, best, start = (
            saved["history"],
            saved["best_bundle"],
            saved["update"] + 1,
        )
    else:
        history, best, start = [], None, 0
    began = time.monotonic()
    accumulator = []
    for update in range(start, config["updates"] + 1):
        if update:
            model.train()
            f, c, mask, labels = batch(arrays, owners, schedule, update - 1)
            optimizer.zero_grad(set_to_none=True)
            loss, details = objective(model, f, c, mask, labels, config, arm)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            accumulator.append(details)
        if update % config["validation_every"]:
            continue
        values = evaluate(model)
        macro = sum(values.values()) / len(values)
        if best is None or macro < best["macro_eer"] - 1e-12:
            best = {
                "update": update,
                "macro_eer": macro,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
        means = (
            {
                key: sum(v[key] for v in accumulator) / len(accumulator)
                for key in accumulator[0]
            }
            if accumulator
            else None
        )
        history.append(
            {
                "update": update,
                "validation_eer": values,
                "macro_eer": macro,
                "selected_update": best["update"],
                "training_mean": means,
            }
        )
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "history": history,
            "best_bundle": best,
            "update": update,
            "initial_model_sha256": initial_hash,
            "design_sha256": design_sha,
            "arm": arm,
        }
        atomic_torch(checkpoint, saved)
        write(directory / "history.json", history)
        state(
            directory,
            "training",
            arm=arm,
            update=update,
            selected_update=best["update"],
            validation_eer=values,
            elapsed_this_process=time.monotonic() - began,
        )
        accumulator = []
        if stop_after is not None and update >= stop_after:
            return saved
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for mode, values, update in (
        (
            "selected",
            saved["best_bundle"]["state_dict"],
            saved["best_bundle"]["update"],
        ),
        ("final", saved["model"], config["updates"]),
    ):
        path = directory / f"{mode}.pt"
        bundle = {"state_dict": values, "update": update, "arm": arm}
        if path.exists():
            assert_nested_equal(torch.load(path, weights_only=True), bundle)
        else:
            atomic_torch(path, bundle)
    return saved


def run_training(config, run, seed):
    torch.set_num_threads(1)
    arrays, owners, schedule = training_data(config, seed)
    validation = load_validation(config, seed)
    initial = torch.load(
        starting_path(config, seed), weights_only=True, map_location="cpu"
    )
    for arm in ARMS:
        directory = run / f"acoustic-{seed}" / arm
        if (directory / "complete.json").exists():
            check(read(directory / "complete.json")["files"])
            continue
        seed_all(config["finetuning_seed"])
        model = create_model(config, seed)
        model.load_state_dict(initial["state_dict"])
        # Loading data and model creation occur before the shared dropout seed.
        seed_all(config["finetuning_seed"])
        result = train_loop(
            config,
            model,
            arrays,
            owners,
            schedule,
            directory,
            sha(run / "design-freeze.json"),
            arm,
            lambda m: validation_eers(m, validation, ALL_PHONES),
        )
        write(
            directory / "training-summary.json",
            {
                "seed": seed,
                "arm": arm,
                "updates": result["update"],
                "selected_update": result["best_bundle"]["update"],
                "selected_validation_macro_eer": result["best_bundle"]["macro_eer"],
                "initial_model_sha256": model_hash(initial["state_dict"]),
                "source_model_sha256": sha(starting_path(config, seed)),
                "source_selected_update": initial["update"],
                "encoder_updated": False,
            },
            immutable=True,
        )
        write(
            directory / "complete.json",
            {
                "files": pin(
                    [
                        directory / name
                        for name in (
                            "last.pt",
                            "selected.pt",
                            "final.pt",
                            "history.json",
                            "training-summary.json",
                        )
                    ]
                )
            },
            immutable=True,
        )
        state(directory, "complete", arm=arm, updates=result["update"])
