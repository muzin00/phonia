"""Matched 4k cosine + 60k constant-LR fusion training with atomic restart."""

from __future__ import annotations

import copy
import hashlib
import math
import time

from common import (
    assert_nested_equal,
    atomic_torch,
    create_model,
    family,
    load_training,
    load_validation,
    model_hash,
    read,
    seed_all,
    sha,
    state,
    torch,
    training_batch,
    validation_eers,
    write,
)


def learning_rate(config, update):
    if update <= config["prefix_updates"]:
        return config["learning_rate"] * (
            0.1
            + 0.9
            * (1 + math.cos(math.pi * (update - 1) / config["prefix_updates"]))
            / 2
        )
    return config["continuation_learning_rate"]


def train_loop(
    config,
    model,
    arrays,
    owners,
    schedule,
    subset,
    directory,
    design_sha,
    evaluate,
    total,
    stop_after=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    checkpoint = directory / "last.pt"
    initial_hash = model_hash(model.state_dict())
    allowed = torch.zeros(36, dtype=torch.bool)
    allowed[list(subset)] = True
    resumed = checkpoint.exists()
    if resumed:
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["design_sha256"] == design_sha and saved["subset"] == list(subset)
        assert saved["initial_model_sha256"] == initial_hash
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        history, best, start = (
            saved["history"],
            saved["best_bundle"],
            saved["update"] + 1,
        )
        running_loss = saved["running_loss"]
    else:
        history, best, start, running_loss = [], None, 0, 0.0
    began = time.monotonic()
    for update in range(start, total + 1):
        if update:
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, update)
            f, c, mask, labels = training_batch(
                arrays, owners, schedule, update - 1, allowed
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(f, c, mask, labels, config["uniform_kl_coefficient"])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            running_loss += float(loss.detach())
        if update % config["validation_every"]:
            continue
        values = evaluate(model, subset)
        macro = sum(values.values()) / len(values)
        if best is None or macro < best["macro_eer"] - 1e-12:
            best = {
                "update": update,
                "macro_eer": macro,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
        history.append(
            {
                "update": update,
                "validation_eer": values,
                "macro_eer": macro,
                "selected_update": best["update"],
                "loss": float(loss.detach()) if update else None,
            }
        )
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
            "history": history,
            "best_bundle": best,
            "update": update,
            "subset": list(subset),
            "initial_model_sha256": initial_hash,
            "design_sha256": design_sha,
            "running_loss": 0.0,
        }
        atomic_torch(checkpoint, saved)
        write(directory / "history.json", history)
        state(
            directory,
            "training",
            update=update,
            selected_update=best["update"],
            validation_eer=values,
            macro_eer=macro,
            elapsed_this_process=time.monotonic() - began,
        )
        running_loss = 0.0
        if stop_after is not None and update >= stop_after:
            return saved
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for mode, values, update in (
        (
            "selected",
            saved["best_bundle"]["state_dict"],
            saved["best_bundle"]["update"],
        ),
        ("final", saved["model"], total),
    ):
        path = directory / f"{mode}.pt"
        if path.exists():
            old = torch.load(path, weights_only=True, map_location="cpu")
            assert old["update"] == update
            assert_nested_equal(old["state_dict"], values)
        else:
            atomic_torch(
                path,
                {
                    "state_dict": values,
                    "update": update,
                    "kind": "transformer",
                    "seed": config["fusion_seed"],
                },
            )
    return saved


def run_training(config, run, seed, reference=False):
    torch.set_num_threads(1)
    frozen = read(run / "design-freeze.json")
    directory = run / ("reference-prefix" if reference else f"acoustic-{seed}")
    if (directory / "complete.json").exists():
        from common import check

        check(read(directory / "complete.json")["files"])
        return
    arrays, owners, schedule = load_training(config, seed)
    validation = load_validation(config, seed)
    seed_all(config["fusion_seed"])
    model = create_model(config, seed)
    subset = tuple(range(36)) if reference else tuple(frozen["subset"])
    initial_rng = torch.get_rng_state().clone()
    initial_hash = model_hash(model.state_dict())
    result = train_loop(
        config,
        model,
        arrays,
        owners,
        schedule,
        subset,
        directory,
        sha(run / "design-freeze.json"),
        lambda m, p: validation_eers(m, validation, p),
        config["prefix_updates"] if reference else config["updates"],
    )
    report = {
        "seed": seed,
        "fusion_seed": config["fusion_seed"],
        "updates": result["update"],
        "initial_model_sha256": initial_hash,
        "initial_rng_sha256": hashlib.sha256(initial_rng.numpy().tobytes()).hexdigest(),
        "final_model_sha256": model_hash(result["model"]),
        "selected_update": result["best_bundle"]["update"],
        "selected_validation_macro_eer": result["best_bundle"]["macro_eer"],
        "schedule_sha256": sha(
            family(config, seed) / "continued" / f"schedule-{config['fusion_seed']}.npz"
        ),
        "encoder_sha256": sha(family(config, seed) / "trial/encoder.pt"),
        "subset": list(subset),
        "encoder_updated": False,
    }
    if reference:
        source = (
            family(config, seed) / "fusion" / f"transformer-{config['fusion_seed']}"
        )
        original = torch.load(
            source / "last.pt", weights_only=False, map_location="cpu"
        )
        for key in ("model", "optimizer", "rng", "update"):
            assert_nested_equal(result[key], original[key])
        for a, b in zip(result["history"], original["history"], strict=True):
            for key in (
                "update",
                "validation_eer",
                "macro_eer",
                "selected_update",
                "loss",
            ):
                assert a[key] == b[key], (key, a[key], b[key])
        previous_best = torch.load(
            source / "fusion.pt", weights_only=True, map_location="cpu"
        )
        assert_nested_equal(
            result["best_bundle"]["state_dict"], previous_best["state_dict"]
        )
        report["original_prefix_model_optimizer_rng_history_exact_match"] = True
    else:
        reference_report = read(run / "reference-prefix/training-summary.json")
        for key in ("initial_model_sha256", "initial_rng_sha256"):
            assert report[key] == reference_report[key]
        assert result["update"] == config["updates"]
        assert {int(v["step"]) for v in result["optimizer"]["state"].values()} == {
            config["updates"]
        }
    write(directory / "training-summary.json", report, immutable=True)
    from common import pin

    write(
        directory / "complete.json",
        {
            "files": pin(
                [
                    directory / n
                    for n in (
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
    state(directory, "complete", seed=seed, updates=result["update"])
