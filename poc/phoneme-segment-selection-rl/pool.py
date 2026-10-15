"""Full-corpus candidates with matched phone/speaker slots and resumable coverage."""

from __future__ import annotations

import itertools

from runtime import np, rank, read, seed_for, torch


class Pool:
    def __init__(self, source, prepared):
        info = read(prepared / "pool.json")
        self.phones, self.speakers = info["phones"], info["speakers"]
        with np.load(prepared / "pool.npz", allow_pickle=False) as data:
            for key in data.files:
                setattr(self, key, data[key].copy())
        self.features = np.memmap(
            source / "train-features.f32",
            mode="r",
            dtype="<f4",
            shape=(len(self.metadata), 128),
        )
        self.groups = [
            self.indices[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])
        ]
        self.eligible = np.array([i for i, v in enumerate(self.groups) if len(v) >= 2])
        self.singletons = np.array(
            [v[0] for v in self.groups if len(v) == 1], dtype=np.int64
        )


class Candidates:
    def __init__(self, pool, seed, count=4):
        self.pool, self.seed, self.count = pool, seed, count
        self.cursor = np.zeros(len(pool.groups), dtype=np.int64)
        self.permutations = {}
        self.phone_groups = []
        for p, phone in enumerate(pool.phones):
            groups = [g for g in pool.eligible if pool.group_phone[g] == p]
            groups.sort(
                key=lambda g: (
                    rank(
                        seed, f"speaker/{phone}/{pool.speakers[pool.group_speaker[g]]}"
                    ),
                    int(g),
                )
            )
            if not groups:
                raise ValueError(f"No complete positive pair for {phone}")
            self.phone_groups.append(np.array(groups, dtype=np.int64))

    def state_dict(self):
        return {"cursor": self.cursor.copy()}

    def load_state_dict(self, state):
        if state["cursor"].shape != self.cursor.shape:
            raise ValueError("Candidate pool changed")
        self.cursor[:] = state["cursor"]
        self.permutations.clear()

    def group_slots(self, update):
        slots = np.full(50, -1, dtype=np.int64)
        for i in range(5):
            slot = update * 5 + i
            phone = slot % len(self.pool.phones)
            groups = self.phone_groups[phone]
            width = min(10, len(groups))
            offset = (slot // len(self.pool.phones)) * width
            slots[i * 10 : i * 10 + width] = groups[
                (offset + np.arange(width)) % len(groups)
            ]
        return slots

    def _segments(self, group):
        ids = self.pool.groups[group]
        wanted = min(len(ids), 2 * self.count)
        chosen, seen = [], set()
        while len(chosen) < wanted:
            cycle, position = divmod(int(self.cursor[group]), len(ids))
            cached = self.permutations.get(group)
            if cached is None or cached[0] != cycle:
                rng = np.random.default_rng(
                    seed_for(self.seed, "queue", int(group), cycle)
                )
                cached = self.permutations[group] = (cycle, rng.permutation(ids))
            value = int(cached[1][position])
            self.cursor[group] += 1
            if value not in seen:
                chosen.append(value)
                seen.add(value)
        return chosen

    def draw(self, update):
        groups = self.group_slots(update)
        pairs = np.full((50, self.count, 2), -1, dtype=np.int64)
        mask = np.zeros((50, self.count), dtype=bool)
        for row, group in enumerate(groups):
            if group < 0:
                continue
            segments = self._segments(int(group))
            if len(segments) >= 2 * self.count:
                options = np.array(segments).reshape(self.count, 2)
            else:
                options = np.array(list(itertools.combinations(segments, 2)))
                rng = np.random.default_rng(
                    seed_for(self.seed, "pairs", update, int(group))
                )
                options = options[rng.permutation(len(options))[: self.count]]
            pairs[row, : len(options)] = options
            mask[row, : len(options)] = True
        return {"pairs": pairs, "mask": mask, "groups": groups}


class Coverage:
    def __init__(self, size):
        self.candidate = np.zeros(size, dtype=np.int32)
        self.selected = np.zeros(size, dtype=np.int32)
        self.singleton = np.zeros(size, dtype=np.int32)

    def record(self, candidates, selected, singleton=None):
        values = candidates["pairs"][candidates["mask"]].reshape(-1)
        np.add.at(self.candidate, np.unique(values), 1)
        np.add.at(self.selected, selected[selected >= 0], 1)
        if singleton is not None:
            self.singleton[singleton] += 1

    def state_dict(self):
        return {
            k: getattr(self, k).copy() for k in ("candidate", "selected", "singleton")
        }

    def load_state_dict(self, state):
        for key, values in state.items():
            getattr(self, key)[:] = values

    def summary(self, pool):
        selected = self.selected + self.singleton
        return {
            "total_segments": len(selected),
            "unique_candidates": int(np.count_nonzero(self.candidate)),
            "unique_selected": int(np.count_nonzero(selected)),
            "training_segment_presentations": int(selected.sum()),
            "pair_eligible_segments": sum(len(pool.groups[g]) for g in pool.eligible),
            "classification_only_singletons": len(pool.singletons),
            "per_phone": {
                phone: {
                    "available": int((pool.segment_phone == p).sum()),
                    "candidate_unique": int(
                        np.count_nonzero(self.candidate[pool.segment_phone == p])
                    ),
                    "selected_unique": int(
                        np.count_nonzero(selected[pool.segment_phone == p])
                    ),
                    "presentations": int(selected[pool.segment_phone == p].sum()),
                }
                for p, phone in enumerate(pool.phones)
            },
        }


def selected_indices(candidates, choices):
    ids = candidates["pairs"][np.arange(50), choices].reshape(100)
    return ids, ids >= 0


def singleton_for(pool, seed, update, interval):
    if not len(pool.singletons) or update % interval:
        return None
    return int(pool.singletons[(update // interval + seed) % len(pool.singletons)])


@torch.no_grad()
def policy_features(pool, student, candidates, coverage, progress):
    """Symmetric pair information; no label identity or validation label is exposed."""
    indices = np.maximum(candidates["pairs"], 0)
    raw = torch.from_numpy(np.array(pool.features[indices]))
    student.encoder.eval()
    embedded = torch.nn.functional.normalize(
        student.encoder.projection(raw.reshape(-1, 128)), dim=1
    )
    embedded = embedded.reshape(*indices.shape, 128)
    true_labels = torch.from_numpy(pool.segment_speaker[indices].astype(np.int64))
    cosine = student.head.cosine(embedded.reshape(-1, 128))
    target = cosine.gather(1, true_labels.reshape(-1, 1)).reshape(indices.shape)
    other = (
        cosine.scatter(1, true_labels.reshape(-1, 1), -2)
        .amax(dim=1)
        .reshape(indices.shape)
    )
    metadata = torch.from_numpy(pool.metadata[indices])
    counts = torch.from_numpy(np.log1p(coverage.selected[indices]).astype(np.float32))
    similarity = (embedded[..., 0, :] * embedded[..., 1, :]).sum(-1)
    same_recording = torch.from_numpy(
        (pool.recording[indices[..., 0]] == pool.recording[indices[..., 1]]).astype(
            np.float32
        )
    )
    summary = torch.stack(
        (
            target.mean(-1),
            target.amin(-1),
            (target - other).mean(-1),
            (target - other).amin(-1),
            similarity,
            counts.mean(-1),
            counts.amax(-1),
            same_recording,
            torch.full_like(similarity, progress),
        ),
        -1,
    )
    result = torch.cat(
        (
            raw.mean(-2),
            (raw[..., 0, :] - raw[..., 1, :]).abs(),
            metadata.mean(-2),
            (metadata[..., 0, :] - metadata[..., 1, :]).abs(),
            summary,
        ),
        -1,
    )
    phone = torch.from_numpy(pool.segment_phone[indices[..., 0]].astype(np.int64))
    return result, phone
