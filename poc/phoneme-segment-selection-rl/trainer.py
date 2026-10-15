"""Paired real-EER / virtual verification-loss teachers with delayed probes."""

from __future__ import annotations

import gzip
import time

from policy import (
    Teacher,
    add_gradients,
    gradients,
    sample,
    statistics,
    teacher_step,
    uniform_probabilities,
)
from pool import Candidates, Coverage, policy_features, selected_indices, singleton_for
from runtime import atomic_torch, clone, np, seed_for, torch, tree_hash, write
from student import Student
from verification import evaluate, fit_fusion, macro, proxy_loss


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        torch.save(value, stream)
    temporary.replace(path)


def load_checkpoint(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def teacher_for(config, phone_count):
    torch.manual_seed(config["teacher_seed"])
    teacher = Teacher(
        phone_count, config["teacher_hidden"], config["teacher_exploration"]
    )
    optimizer = torch.optim.Adam(
        teacher.parameters(), lr=config["teacher_learning_rate"]
    )
    return teacher, optimizer


def choices_for(
    config,
    pool,
    student,
    candidates,
    coverage,
    teacher,
    update,
    seed,
    learn=False,
    stream="committed",
):
    draws = np.random.default_rng(seed_for(seed, "choice", stream, update)).random(50)
    if teacher is None:
        probability = uniform_probabilities(candidates["mask"])
    else:
        features, phones = policy_features(
            pool, student, candidates, coverage, update / config["updates"]
        )
        with torch.set_grad_enabled(learn):
            probability = teacher.probabilities(
                features, phones, torch.from_numpy(candidates["mask"])
            )
    choices = sample(probability, draws)
    if not learn:
        return choices, None
    logp, kl, decisions = statistics(probability, choices, candidates["mask"])
    a, b = gradients(teacher, logp, kl)
    return choices, (a, b, decisions)


def uniform_steps(config, pool, student, batches, start, seed, stream="control"):
    losses = []
    for i, candidates in enumerate(batches):
        update = start + i
        # The REINFORCE baseline MUST be independent of sampled policy actions.
        # Shared inverse-CDF draws would correlate baseline and action (and leave
        # an initially uniform policy stuck with exactly zero paired reward).
        choices, _ = choices_for(
            config, pool, student, candidates, None, None, update, seed, stream=stream
        )
        ids, mask = selected_indices(candidates, choices)
        singleton = singleton_for(pool, seed, update, config["singleton_interval"])
        losses.append(student.step(pool, ids, mask, update, singleton))
    return losses


def train_episode(
    config,
    directory,
    pool,
    bank,
    reward_data,
    arm,
    episode,
    teacher,
    teacher_optimizer,
    *,
    training_teacher=True,
    stop_after_blocks=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    seed = config["training_seeds"][episode] if training_teacher else episode
    student = Student(config, len(pool.speakers), seed)
    candidates = Candidates(pool, seed, config["candidate_pairs"])
    coverage = Coverage(len(pool.features))
    history, actual, teacher_updates = [], 0, 0
    cpu_before, wall_before = 0.0, 0.0
    checkpoint = directory / "last.pt.gz"
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint)
        if (
            saved["config"] != config
            or saved["arm"] != arm
            or saved["episode"] != episode
            or saved["training_teacher"] != training_teacher
        ):
            raise ValueError("Checkpoint protocol changed")
        student.load_state_dict(saved["student"])
        candidates.load_state_dict(saved["candidates"])
        coverage.load_state_dict(saved["coverage"])
        if training_teacher:
            teacher.load_state_dict(saved["teacher"])
            teacher_optimizer.load_state_dict(saved["teacher_optimizer"])
        elif (
            teacher is not None
            and tree_hash(teacher.state_dict()) != saved["teacher_hash"]
        ):
            raise ValueError("Deployment teacher changed")
        history, actual, teacher_updates = (
            saved["history"],
            saved["actual_updates"],
            saved["teacher_updates"],
        )
        cpu_before, wall_before = saved["cpu_seconds"], saved["wall_seconds"]
        if (
            student.update == config["updates"]
            and (directory / "complete.json").exists()
        ):
            return student, coverage, True
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    mode = (
        (
            "proxy"
            if arm == "proxy"
            or (arm == "hybrid" and episode < config["hybrid_proxy_episodes"])
            else "eer"
        )
        if training_teacher
        else "frozen"
    )
    # Recover a crash after the final block checkpoint but before exports/seal.
    state = {
        "arm": arm,
        "episode": episode,
        "mode": mode,
        "update": student.update,
        "budget": config["updates"],
        "actual_updates": actual,
        "teacher_updates": teacher_updates,
        "cpu_seconds": cpu_before,
        "wall_seconds": wall_before,
        "last_reward": history[-1].get("reward") if history else None,
        "last_eer": history[-1].get("sampled_eer") if history else None,
    }
    while student.update < config["updates"]:
        start = student.update
        count = min(config["block_updates"], config["updates"] - start)
        block = start // config["block_updates"]
        batches = [candidates.draw(u) for u in range(start, start + count)]
        parent_hash = tree_hash(student.state_dict()) if mode == "eer" else None
        control = student.fork() if mode == "eer" else None
        log_gradient, kl_gradient, decisions = None, None, 0
        loss_values, proxy_rewards, choice_digest = [], [], []
        for offset, batch in enumerate(batches):
            update = start + offset
            use_proxy = mode == "proxy" and update % config["proxy_interval"] == 0
            choice, policy = choices_for(
                config,
                pool,
                student,
                batch,
                coverage,
                teacher,
                update,
                seed,
                learn=mode == "eer" or use_proxy,
            )
            virtual_control = student.fork() if use_proxy else None
            indices, valid = selected_indices(batch, choice)
            singleton = singleton_for(pool, seed, update, config["singleton_interval"])
            loss_values.append(student.step(pool, indices, valid, update, singleton))
            actual += 1
            coverage.record(batch, indices, singleton)
            choice_digest.append(choice.astype(np.int8))
            if mode == "eer":
                a, b, n = policy
                log_gradient, kl_gradient = (
                    add_gradients(log_gradient, a),
                    add_gradients(kl_gradient, b),
                )
                decisions += n
            if use_proxy:
                uniform_steps(config, pool, virtual_control, [batch], update, seed)
                actual += 1
                probe_seed = seed_for(seed, "proxy", update)
                advantage = proxy_loss(
                    virtual_control, reward_data, config, probe_seed
                ) - proxy_loss(student, reward_data, config, probe_seed)
                teacher_step(
                    config,
                    teacher,
                    teacher_optimizer,
                    *policy,
                    advantage,
                    config["proxy_reward_scale"],
                )
                teacher_updates += 1
                proxy_rewards.append(advantage)
        record = {
            "start": start,
            "end": student.update,
            "mode": mode,
            "loss": float(np.mean(loss_values)),
            "choices_sha256": tree_hash(np.stack(choice_digest)),
            "candidate_sha256": tree_hash([b["pairs"] for b in batches]),
            "actual_updates_in_block": count,
            "teacher_updates_in_block": len(proxy_rewards),
        }
        if training_teacher:
            fusion_seed = seed_for(seed, "block-fusion", block)
            record["fusion_loss"] = fit_fusion(
                student, bank, config, fusion_seed, config["fusion_steps_per_block"]
            )
        if mode == "eer":
            if tree_hash(control.state_dict()) != parent_hash:
                raise AssertionError("Control fork mutated with sampled student")
            uniform_steps(config, pool, control, batches, start, seed)
            actual += count
            fit_fusion(
                control, bank, config, fusion_seed, config["fusion_steps_per_block"]
            )
            control_eer, sampled_eer = (
                evaluate(control, reward_data),
                evaluate(student, reward_data),
            )
            short_reward = macro(control_eer) - macro(sampled_eer)
            reward, lookahead = short_reward, 0
            if (block + 1) % config["long_probe_every_blocks"] == 0:
                lookahead = min(
                    config["long_probe_updates"], config["updates"] - student.update
                )
            if lookahead:
                future = Candidates(pool, seed, config["candidate_pairs"])
                future.load_state_dict(candidates.state_dict())
                future_batches = [
                    future.draw(u)
                    for u in range(student.update, student.update + lookahead)
                ]
                later_sampled, later_control = student.fork(), control.fork()
                uniform_steps(
                    config,
                    pool,
                    later_sampled,
                    future_batches,
                    student.update,
                    seed,
                    stream="tail",
                )
                uniform_steps(
                    config,
                    pool,
                    later_control,
                    future_batches,
                    student.update,
                    seed,
                    stream="tail",
                )
                extra_fusion = (
                    max(1, lookahead // config["block_updates"])
                    * config["fusion_steps_per_block"]
                )
                later_seed = seed_for(seed, "delayed-fusion", block)
                fit_fusion(later_sampled, bank, config, later_seed, extra_fusion)
                fit_fusion(later_control, bank, config, later_seed, extra_fusion)
                delayed_control, delayed_sampled = (
                    evaluate(later_control, reward_data),
                    evaluate(later_sampled, reward_data),
                )
                delayed_reward = macro(delayed_control) - macro(delayed_sampled)
                fraction = config["long_reward_fraction"]
                reward = (1 - fraction) * short_reward + fraction * delayed_reward
                record.update(
                    delayed_control_eer=delayed_control,
                    delayed_sampled_eer=delayed_sampled,
                    delayed_reward=delayed_reward,
                    lookahead_updates=lookahead,
                )
                actual += 2 * lookahead
            teacher_update = teacher_step(
                config,
                teacher,
                teacher_optimizer,
                log_gradient,
                kl_gradient,
                decisions,
                reward,
                config["teacher_reward_scale"],
            )
            teacher_updates += 1
            record.update(
                control_eer=control_eer,
                sampled_eer=sampled_eer,
                short_reward=short_reward,
                reward=reward,
                teacher_update=teacher_update,
                committed="sampled_even_if_negative",
                parent_sha256=parent_hash,
                actual_updates_in_block=2 * count + 2 * lookahead,
                teacher_updates_in_block=1,
            )
        elif proxy_rewards:
            record.update(
                proxy_rewards=proxy_rewards,
                actual_updates_in_block=count + len(proxy_rewards),
            )
        record["student_sha256"] = tree_hash(student.state_dict())
        record["teacher_sha256"] = (
            tree_hash(teacher.state_dict()) if teacher is not None else None
        )
        history.append(record)
        saved = {
            "config": config,
            "arm": arm,
            "episode": episode,
            "training_teacher": training_teacher,
            "student": student.state_dict(),
            "candidates": candidates.state_dict(),
            "coverage": coverage.state_dict(),
            "teacher": teacher.state_dict() if teacher is not None else None,
            "teacher_hash": record["teacher_sha256"],
            "teacher_optimizer": teacher_optimizer.state_dict()
            if teacher_optimizer is not None
            else None,
            "history": history,
            "actual_updates": actual,
            "teacher_updates": teacher_updates,
            "cpu_seconds": cpu_before + time.process_time() - cpu_start,
            "wall_seconds": wall_before + time.perf_counter() - wall_start,
        }
        save_checkpoint(checkpoint, saved)
        state = {
            "arm": arm,
            "episode": episode,
            "mode": mode,
            "update": student.update,
            "budget": config["updates"],
            "actual_updates": actual,
            "teacher_updates": teacher_updates,
            "cpu_seconds": saved["cpu_seconds"],
            "wall_seconds": saved["wall_seconds"],
            "last_reward": record.get("reward"),
            "last_eer": record.get("sampled_eer"),
        }
        write(directory / "state.json", state)
        print(state, flush=True)
        if stop_after_blocks is not None and len(history) >= stop_after_blocks:
            return student, coverage, False
    atomic_torch(directory / "encoder.pt", clone(student.state_dict()))
    if training_teacher:
        atomic_torch(
            directory / "teacher.pt",
            {
                "teacher": teacher.state_dict(),
                "optimizer": teacher_optimizer.state_dict(),
                "episodes": episode + 1,
            },
        )
    write(directory / "history.json", history)
    write(directory / "coverage.json", coverage.summary(pool))
    write(
        directory / "complete.json",
        {
            **state,
            "status": "complete",
            "blocks": len(history),
            "student_sha256": tree_hash(student.state_dict()),
            "teacher_sha256": tree_hash(teacher.state_dict())
            if teacher is not None
            else None,
        },
    )
    return student, coverage, True
