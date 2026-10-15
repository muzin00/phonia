"""Uniform training over available positive-pair phones, with singleton loss."""

import time

from pool import Candidates, Coverage, selected_indices, singleton_for
from shared_ablation import Student, rt
from trainer import choices_for, load_checkpoint, save_checkpoint


class AvailableCandidates(Candidates):
    def __init__(self, pool, seed, count=4):
        self.pool, self.seed, self.count = pool, seed, count
        self.cursor = rt.np.zeros(len(pool.groups), dtype=rt.np.int64)
        self.permutations, self.phone_groups = {}, []
        for p, phone in enumerate(pool.phones):
            groups = [g for g in pool.eligible if pool.group_phone[g] == p]
            groups.sort(
                key=lambda g: (
                    rt.rank(
                        seed, f"speaker/{phone}/{pool.speakers[pool.group_speaker[g]]}"
                    ),
                    int(g),
                )
            )
            self.phone_groups.append(rt.np.array(groups, dtype=rt.np.int64))
        self.active_phones = [
            p for p, groups in enumerate(self.phone_groups) if len(groups)
        ]
        if not self.active_phones:
            raise ValueError("No training phone has a positive pair")

    def group_slots(self, update):
        slots = rt.np.full(50, -1, dtype=rt.np.int64)
        for i in range(5):
            slot = update * 5 + i
            groups = self.phone_groups[
                self.active_phones[slot % len(self.active_phones)]
            ]
            width = min(10, len(groups))
            offset = (slot // len(self.active_phones)) * width
            slots[i * 10 : i * 10 + width] = groups[
                (offset + rt.np.arange(width)) % len(groups)
            ]
        return slots


def train_uniform(config, directory, pool, seed, *, stop_after_blocks=None):
    directory.mkdir(parents=True, exist_ok=True)
    student = Student(config, len(pool.speakers), seed)
    candidates = AvailableCandidates(pool, seed, config["candidate_pairs"])
    coverage = Coverage(len(pool.features))
    history, cpu_before, wall_before = [], 0.0, 0.0
    checkpoint = directory / "last.pt.gz"
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint)
        if saved["config"] != config or saved["seed"] != seed:
            raise ValueError("Uniform checkpoint identity changed")
        student.load_state_dict(saved["student"])
        candidates.load_state_dict(saved["candidates"])
        coverage.load_state_dict(saved["coverage"])
        history = saved["history"]
        cpu_before, wall_before = saved["cpu_seconds"], saved["wall_seconds"]
        if (
            student.update == config["updates"]
            and (directory / "complete.json").exists()
        ):
            return student, coverage, True
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    while student.update < config["updates"]:
        start = student.update
        losses, choices = [], []
        batches = [
            candidates.draw(u)
            for u in range(
                start, min(start + config["block_updates"], config["updates"])
            )
        ]
        for offset, batch in enumerate(batches):
            update = start + offset
            choice, _ = choices_for(
                config, pool, student, batch, coverage, None, update, seed
            )
            indices, valid = selected_indices(batch, choice)
            singleton = singleton_for(pool, seed, update, config["singleton_interval"])
            losses.append(student.step(pool, indices, valid, update, singleton))
            coverage.record(batch, indices, singleton)
            choices.append(choice.astype(rt.np.int8))
        history.append(
            {
                "start": start,
                "end": student.update,
                "loss": float(rt.np.mean(losses)),
                "choices_sha256": rt.tree_hash(rt.np.stack(choices)),
                "candidate_sha256": rt.tree_hash([b["pairs"] for b in batches]),
                "student_sha256": rt.tree_hash(student.state_dict()),
            }
        )
        save_checkpoint(
            checkpoint,
            {
                "config": config,
                "seed": seed,
                "student": student.state_dict(),
                "candidates": candidates.state_dict(),
                "coverage": coverage.state_dict(),
                "history": history,
                "cpu_seconds": cpu_before + time.process_time() - cpu_start,
                "wall_seconds": wall_before + time.perf_counter() - wall_start,
            },
        )
        state = {
            "mode": "uniform",
            "seed": seed,
            "update": student.update,
            "budget": config["updates"],
            "actual_updates": student.update,
            "teacher_updates": 0,
            "cpu_seconds": cpu_before + time.process_time() - cpu_start,
            "wall_seconds": wall_before + time.perf_counter() - wall_start,
            "paired_phone_count": len(candidates.active_phones),
        }
        rt.write(directory / "state.json", state)
        print(state, flush=True)
        if stop_after_blocks is not None and len(history) >= stop_after_blocks:
            return student, coverage, False
    # This also recovers a crash after the final checkpoint but before exports.
    state = {
        "mode": "uniform",
        "seed": seed,
        "update": student.update,
        "budget": config["updates"],
        "actual_updates": student.update,
        "teacher_updates": 0,
        "cpu_seconds": cpu_before + time.process_time() - cpu_start,
        "wall_seconds": wall_before + time.perf_counter() - wall_start,
        "paired_phone_count": len(candidates.active_phones),
    }
    rt.atomic_torch(directory / "encoder.pt", rt.clone(student.state_dict()))
    rt.write(directory / "history.json", history)
    rt.write(directory / "coverage.json", coverage.summary(pool))
    rt.write(directory / "state.json", state)
    rt.write(
        directory / "complete.json",
        {
            **state,
            "status": "complete",
            "blocks": len(history),
            "student_sha256": rt.tree_hash(student.state_dict()),
        },
    )
    return student, coverage, True
