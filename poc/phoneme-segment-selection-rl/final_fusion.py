"""Fresh 64k fusion fit after a full acoustic run, on the same frozen schedule."""

from __future__ import annotations

import time

from runtime import (
    ROOT,
    atomic_torch,
    backend,
    clone,
    np,
    seed_for,
    torch,
    tree_hash,
    write,
)
from verification import evaluate, macro


def full_arrays(bank, student):
    return bank.materialize(
        student.encoder,
        np.arange(len(bank.info["queries"])),
        np.arange(len(bank.speakers)),
    )


def batch(bank, arrays, queries, negative):
    genuine = bank.owners[queries]
    if np.any(genuine == negative):
        raise ValueError("Impostor label equals genuine speaker")
    q = torch.from_numpy(np.repeat(queries, 2).astype(np.int64))
    c = torch.from_numpy(np.stack((genuine, negative), 1).reshape(-1).astype(np.int64))
    features, components = backend.pair_features(
        arrays["enrollment"][c],
        arrays["emeta"][c],
        arrays["queries"][q],
        arrays["qmeta"][q],
    )
    mask = arrays["emask"][c] & arrays["qmask"][q]
    shared = mask.reshape(-1, 2, len(bank.phones)).all(1).repeat_interleave(2, 0)
    if not shared[:, :5].all():
        raise ValueError("Final fusion five-vowel gate changed")
    return features, components, shared, torch.tensor([1.0, 0.0]).repeat(len(queries))


def fit(
    config,
    directory,
    student,
    bank,
    selection,
    *,
    schedule_override=None,
    stop_after=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "last.pt"
    start, history, best, elapsed = 0, [], float("inf"), 0.0
    student.reset_fusion()
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if saved["config"] != config or saved["encoder_sha256"] != tree_hash(
            student.encoder.state_dict()
        ):
            raise ValueError("Final fusion checkpoint changed")
        student.fusion.load_state_dict(saved["fusion"])
        student.fusion_optimizer.load_state_dict(saved["optimizer"])
        start, history, best, elapsed = (
            saved["update"],
            saved["history"],
            saved["best"],
            saved["cpu_seconds"],
        )
        if (
            start == config["final_fusion_updates"]
            and (directory / "complete.json").exists()
        ):
            return
    if schedule_override is None:
        source = (
            ROOT
            / config["source_run"]
            / f"acoustic-{config['source_seed']}/continued/schedule-20261015.npz"
        )
        with np.load(source, allow_pickle=False) as archive:
            schedule = {k: archive[k].copy() for k in archive.files}
    else:
        schedule = schedule_override
    if len(schedule["queries"]) < config["final_fusion_updates"]:
        raise ValueError("Insufficient frozen fusion schedule")
    arrays = full_arrays(bank, student)
    cpu_start = time.process_time()

    def validate(update):
        nonlocal best
        eers = evaluate(student, selection)
        objective = macro(eers)
        history.append(
            {"update": update, "selection_eer": eers, "macro_eer": objective}
        )
        if objective < best:
            best = objective
            atomic_torch(
                directory / "selected.pt",
                {
                    "fusion": clone(student.fusion.state_dict()),
                    "update": update,
                    "selection_eer": eers,
                },
            )

    if not history:
        validate(0)
    for update in range(start, config["final_fusion_updates"]):
        for group in student.fusion_optimizer.param_groups:
            group["lr"] = (
                config["fusion_learning_rate"]
                if update < config["final_fusion_initial_updates"]
                else config["final_fusion_continuation_learning_rate"]
            )
        torch.manual_seed(seed_for(student.seed, "final-fusion-dropout", update))
        student.fusion.train()
        values = batch(
            bank,
            arrays,
            schedule["queries"][update],
            schedule["negative_claims"][update],
        )
        student.fusion_optimizer.zero_grad(set_to_none=True)
        loss = student.fusion.loss(*values, regularization=config["fusion_uniform_kl"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            student.fusion.parameters(),
            config["gradient_clip"],
            error_if_nonfinite=True,
        )
        student.fusion_optimizer.step()
        end = update + 1
        if (
            end % config["final_fusion_validation_every"] == 0
            or end == config["final_fusion_updates"]
        ):
            validate(end)
            atomic_torch(
                checkpoint,
                {
                    "config": config,
                    "encoder_sha256": tree_hash(student.encoder.state_dict()),
                    "fusion": student.fusion.state_dict(),
                    "optimizer": student.fusion_optimizer.state_dict(),
                    "update": end,
                    "best": best,
                    "history": history,
                    "cpu_seconds": elapsed + time.process_time() - cpu_start,
                },
            )
            write(
                directory / "state.json",
                {
                    "update": end,
                    "budget": config["final_fusion_updates"],
                    "best_selection_eer": best,
                    "cpu_seconds": elapsed + time.process_time() - cpu_start,
                },
            )
            if stop_after is not None and end >= stop_after:
                return
    atomic_torch(
        directory / "final.pt",
        {
            "fusion": student.fusion.state_dict(),
            "update": config["final_fusion_updates"],
        },
    )
    write(directory / "history.json", history)
    write(
        directory / "complete.json",
        {
            "status": "complete",
            "updates": config["final_fusion_updates"],
            "selected_update": torch.load(directory / "selected.pt", weights_only=True)[
                "update"
            ],
            "selected_eer": best,
            "cpu_seconds": elapsed + time.process_time() - cpu_start,
        },
    )
