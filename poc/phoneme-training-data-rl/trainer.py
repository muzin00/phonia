"""Counterfactual training-data selection with independent optimizer forks."""

from __future__ import annotations

import copy
import hashlib
import json
import time

from common import (
    ARMS,
    assert_nested_equal,
    atomic_torch,
    check,
    evaluate_eers,
    evaluation_data,
    initial_model,
    model_hash,
    np,
    pin,
    read,
    sha,
    state,
    torch,
    training_data,
    write,
)
from selection import Teacher, choose, selected_batch, teacher_loss


def tree_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return digest.hexdigest()


def optimizer_for(config, model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def fork(config, model, optimizer):
    child = copy.deepcopy(model)
    child_optimizer = optimizer_for(config, child)
    child_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert model_hash(child.state_dict()) == model_hash(model.state_dict())
    assert tree_hash(child_optimizer.state_dict()) == tree_hash(optimizer.state_dict())
    return child, child_optimizer


def train_block(config, model, optimizer, arrays, owners, pools, keys, start, phone):
    model.train()
    indices, losses = [], []
    for update in range(start, start + config["block_updates"]):
        # Coupled dropout: deterministic for this update, independent of branch order.
        torch.manual_seed(config["dropout_seed"] + update)
        batch, ids = selected_batch(config, arrays, owners, pools, keys, update, phone)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(*batch, regularization=config["uniform_kl_coefficient"])
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        indices.append(ids)
        losses.append(float(loss.detach()))
    chosen = np.array(indices)
    available = pools["common"][start : start + len(chosen)]
    selected_masks = np.take_along_axis(available, chosen[..., None], axis=1)
    details = {
        "loss": float(np.mean(losses)),
        "phone_pair_counts": selected_masks.sum(axis=(0, 1)).tolist(),
        "indices_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(),
    }
    return chosen, details


def macro(values):
    return sum(values.values()) / len(values)


def reinforce_block(
    config,
    model,
    optimizer,
    teacher,
    teacher_optimizer,
    actions,
    draws,
    block,
    arrays,
    owners,
    pools,
    reward_evaluate,
    current_reward,
):
    start = block * config["block_updates"]
    context = [1.0, start / config["updates"], macro(current_reward)]
    probabilities = teacher.probabilities(context)
    choices = [choose(probabilities, draw) for draw in draws["actions"][block]]
    phones = [actions[index] for index in choices]
    parent_model = model_hash(model.state_dict())
    parent_optimizer = tree_hash(optimizer.state_dict())
    branches = {}
    for phone in dict.fromkeys([-1, *phones]):
        child, opt = fork(config, model, optimizer)
        indices, details = train_block(
            config, child, opt, arrays, owners, pools, draws["keys"], start, phone
        )
        eers = reward_evaluate(child)
        branches[phone] = (child, opt, indices, eers, details)
    assert model_hash(model.state_dict()) == parent_model
    assert tree_hash(optimizer.state_dict()) == parent_optimizer
    control = macro(branches[-1][3])
    rewards = [control - macro(branches[phone][3]) for phone in phones]
    before = teacher.theta.detach().tolist()
    teacher_optimizer.zero_grad(set_to_none=True)
    loss = teacher_loss(probabilities, choices, rewards, config["teacher_uniform_kl"])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        teacher.parameters(), config["gradient_clip"], error_if_nonfinite=True
    )
    teacher_optimizer.step()
    # The first on-policy sample is committed, including a negative reward.
    committed = branches[phones[0]]
    log = {
        "start": start,
        "context": context,
        "probabilities": probabilities.detach().tolist(),
        "choices": choices,
        "phones": phones,
        "draws": draws["actions"][block].tolist(),
        "rewards": rewards,
        "committed_rollout": 0,
        "committed_phone": phones[0],
        "parent_model_sha256": parent_model,
        "parent_optimizer_sha256": parent_optimizer,
        "teacher_before": before,
        "teacher_after": teacher.theta.detach().tolist(),
        "teacher_loss": float(loss.detach()),
        "teacher_gradient_norm": float(norm),
        "branches": {
            str(p): {
                "reward_eer": v[3],
                "model_sha256": model_hash(v[0].state_dict()),
                "optimizer_sha256": tree_hash(v[1].state_dict()),
                **v[4],
            }
            for p, v in branches.items()
        },
        "actual_student_updates": len(branches) * config["block_updates"],
    }
    return *committed[:4], log


def train_loop(
    config,
    model,
    arrays,
    owners,
    pools,
    draws,
    actions,
    directory,
    design_sha,
    arm,
    selection_evaluate,
    reward_evaluate,
    stop_after=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    initial_hash = model_hash(model.state_dict())
    optimizer = optimizer_for(config, model)
    teacher = Teacher(len(actions), config["teacher_exploration"])
    teacher_optimizer = torch.optim.Adam(
        teacher.parameters(), lr=config["teacher_learning_rate"]
    )
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["design_sha256"] == design_sha and saved["arm"] == arm
        assert (
            saved["initial_model_sha256"] == initial_hash
            and saved["actions"] == actions
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        teacher.load_state_dict(saved["teacher"])
        teacher_optimizer.load_state_dict(saved["teacher_optimizer"])
        history, best, logs = saved["history"], saved["best_bundle"], saved["blocks"]
        start, current_reward = saved["update"], saved["current_reward"]
        selected = saved["selected_indices"].numpy().copy()
    else:
        history, best, logs, start = [], None, [], 0
        current_reward = reward_evaluate(model) if arm == "rl" else None
        selected = np.empty((0, config["queries_per_update"]), dtype=np.int64)
    began = time.monotonic()
    for update in range(start, config["updates"] + 1, config["block_updates"]):
        if update > start:
            block = update // config["block_updates"] - 1
            if arm == "rl":
                model, optimizer, ids, current_reward, log = reinforce_block(
                    config,
                    model,
                    optimizer,
                    teacher,
                    teacher_optimizer,
                    actions,
                    draws,
                    block,
                    arrays,
                    owners,
                    pools,
                    reward_evaluate,
                    current_reward,
                )
            else:
                probabilities = torch.ones(len(actions), dtype=torch.float64) / len(
                    actions
                )
                phone = (
                    -1
                    if arm == "uniform"
                    else actions[choose(probabilities, draws["actions"][block, 0])]
                )
                parent_model, parent_optimizer = (
                    model_hash(model.state_dict()),
                    tree_hash(optimizer.state_dict()),
                )
                ids, details = train_block(
                    config,
                    model,
                    optimizer,
                    arrays,
                    owners,
                    pools,
                    draws["keys"],
                    update - config["block_updates"],
                    phone,
                )
                log = {
                    "start": update - config["block_updates"],
                    "committed_phone": phone,
                    "parent_model_sha256": parent_model,
                    "parent_optimizer_sha256": parent_optimizer,
                    "actual_student_updates": config["block_updates"],
                    "branches": {
                        str(phone): {
                            "model_sha256": model_hash(model.state_dict()),
                            "optimizer_sha256": tree_hash(optimizer.state_dict()),
                            **details,
                        }
                    },
                }
            logs.append(log)
            selected = np.concatenate((selected, ids))
        if update % config["validation_every"] == 0 and not any(
            h["update"] == update for h in history
        ):
            values = selection_evaluate(model)
            if best is None or macro(values) < best["macro_eer"] - 1e-12:
                best = {
                    "update": update,
                    "macro_eer": macro(values),
                    "state_dict": copy.deepcopy(model.state_dict()),
                }
            history.append(
                {
                    "update": update,
                    "selection_eer": values,
                    "macro_eer": macro(values),
                    "selected_update": best["update"],
                }
            )
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "teacher": teacher.state_dict(),
            "teacher_optimizer": teacher_optimizer.state_dict(),
            "history": history,
            "best_bundle": best,
            "blocks": logs,
            "update": update,
            "selected_indices": torch.from_numpy(selected),
            "current_reward": current_reward,
            "initial_model_sha256": initial_hash,
            "design_sha256": design_sha,
            "arm": arm,
            "actions": actions,
        }
        atomic_torch(checkpoint, saved)
        write(directory / "history.json", history)
        state(
            directory,
            "training",
            arm=arm,
            update=update,
            selected_update=best["update"],
            selection_eer=history[-1]["selection_eer"],
            elapsed_this_process=time.monotonic() - began,
        )
        if stop_after is not None and update >= stop_after:
            return saved
    for mode, values, update in (
        ("selected", best["state_dict"], best["update"]),
        ("final", model.state_dict(), config["updates"]),
    ):
        path = directory / f"{mode}.pt"
        bundle = {"state_dict": values, "update": update, "arm": arm}
        if path.exists():
            assert_nested_equal(torch.load(path, weights_only=True), bundle)
        else:
            atomic_torch(path, bundle)
    write(directory / "blocks.json", logs)
    return saved


def run_training(config, run, seed):
    torch.set_num_threads(1)
    arrays, owners, pools = training_data(config, seed)
    reward = evaluation_data(config, run, seed, "reward")
    selection = evaluation_data(config, run, seed, "selection")
    with np.load(run / f"draws-{seed}.npz", allow_pickle=False) as saved:
        draws = {k: saved[k].copy() for k in saved.files}
    actions = read(run / "design-freeze.json")["actions"][str(seed)]
    for arm in ARMS:
        directory = run / f"acoustic-{seed}" / arm
        if (directory / "complete.json").exists():
            check(read(directory / "complete.json")["files"])
            continue
        model = initial_model(config, seed)
        initial = torch.load(run / f"acoustic-{seed}/initial.pt", weights_only=True)
        assert_nested_equal(model.state_dict(), initial["state_dict"])
        result = train_loop(
            config,
            model,
            arrays,
            owners,
            pools,
            draws,
            actions,
            directory,
            sha(run / "design-freeze.json"),
            arm,
            lambda m: evaluate_eers(m, selection),
            lambda m: evaluate_eers(m, reward),
        )
        write(
            directory / "training-summary.json",
            {
                "seed": seed,
                "arm": arm,
                "updates": result["update"],
                "actual_student_updates": sum(
                    b["actual_student_updates"] for b in result["blocks"]
                ),
                "selected_update": result["best_bundle"]["update"],
                "selected_selection_macro_eer": result["best_bundle"]["macro_eer"],
                "initial_model_sha256": result["initial_model_sha256"],
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
                            "blocks.json",
                            "training-summary.json",
                        )
                    ]
                )
            },
            immutable=True,
        )
        state(directory, "complete", arm=arm, updates=result["update"])
