"""Recompute pooling with the CURRENT encoder; no stale learned feature cache."""

from __future__ import annotations

from runtime import backend, metrics, np, read, seed_for, torch


class Bank:
    def __init__(self, info, features, metadata, phones):
        self.info, self.features, self.phones = info, features, phones
        self.speakers = info["speakers"]
        speaker_index = {s: i for i, s in enumerate(self.speakers)}
        self.owners = np.array(
            [speaker_index[q["speaker_id"]] for q in info["queries"]]
        )
        self.groups = []
        for speaker in self.speakers:
            self.groups.extend(
                np.asarray(info["profiles"][speaker].get(p, []), dtype=np.int64)
                for p in phones
            )
        for query in info["queries"]:
            self.groups.extend(
                np.asarray(query["groups"].get(p, []), dtype=np.int64) for p in phones
            )
        self.counts = np.array([len(g) for g in self.groups], dtype=np.int64)
        self.static = np.zeros((len(self.groups), 4), np.float32)
        for i, ids in enumerate(self.groups):
            if len(ids):
                self.static[i, [0, 1, 3]] = [
                    np.log1p(len(ids)),
                    np.log(metadata[ids, 0].mean()),
                    metadata[ids, 1].mean() / 50,
                ]
        self.allowed = np.array(
            [p in info.get("universally_registered", phones) for p in phones]
        )

    @torch.no_grad()
    def materialize(self, encoder, queries, claims):
        unique_q, qmap = np.unique(queries, return_inverse=True)
        unique_c, cmap = np.unique(claims, return_inverse=True)
        items = np.r_[unique_c, unique_q + len(self.speakers)]
        rows = (
            items[:, None] * len(self.phones) + np.arange(len(self.phones))[None]
        ).reshape(-1)
        counts = self.counts[rows]
        indices = np.concatenate([self.groups[r] for r in rows])
        destinations = torch.from_numpy(np.repeat(np.arange(len(rows)), counts))
        sums = torch.zeros(len(rows), 128)
        encoder.eval()
        for start in range(0, len(indices), 8192):
            values = torch.from_numpy(
                np.array(self.features[indices[start : start + 8192]])
            )
            embedded = torch.nn.functional.normalize(
                encoder.projection(values), dim=1, eps=encoder.l2_epsilon
            )
            sums.index_add_(0, destinations[start : start + 8192], embedded)
        means = sums / torch.from_numpy(counts.clip(1))[:, None]
        norm = means.norm(dim=1)
        meta = torch.from_numpy(self.static[rows].copy())
        meta[:, 2] = torch.where(torch.from_numpy(counts > 0), 1 - norm, 0)
        means[: len(unique_c) * len(self.phones)] = torch.nn.functional.normalize(
            means[: len(unique_c) * len(self.phones)], dim=1, eps=1e-12
        )
        means = means.reshape(len(items), len(self.phones), 128)
        meta = meta.reshape(len(items), len(self.phones), 4)
        mask = torch.from_numpy(counts.reshape(len(items), len(self.phones)) > 0)
        return {
            "enrollment": means[: len(unique_c)],
            "queries": means[len(unique_c) :],
            "emeta": meta[: len(unique_c)],
            "qmeta": meta[len(unique_c) :],
            "emask": mask[: len(unique_c)],
            "qmask": mask[len(unique_c) :],
            "qmap": torch.from_numpy(qmap),
            "cmap": torch.from_numpy(cmap),
        }

    def features_for(self, arrays, positions, shared_masks=None):
        q, c = arrays["qmap"][positions], arrays["cmap"][positions]
        f, components = backend.pair_features(
            arrays["enrollment"][c],
            arrays["emeta"][c],
            arrays["queries"][q],
            arrays["qmeta"][q],
        )
        mask = arrays["emask"][c] & arrays["qmask"][q] & torch.from_numpy(self.allowed)
        if shared_masks is not None:
            mask &= shared_masks[positions]
        return f, components, mask


class Evaluation:
    def __init__(self, bank, rows, allowed_speakers=None):
        self.bank = bank
        allowed = set(bank.speakers if allowed_speakers is None else allowed_speakers)
        n = len(bank.speakers)
        keep = [
            i
            for i, r in enumerate(rows)
            if r["speaker_id"] in allowed
            and r["claimed_speaker_id"] in allowed
            and r["role"] == "verification"
        ]
        self.rows = [rows[i] for i in keep]
        self.total_genuine = sum(r["is_genuine"] for r in self.rows)
        self.total_impostor = len(self.rows) - self.total_genuine
        scorable = [i for i in keep if rows[i]["score"] is not None]
        self.queries = np.array([i // n for i in scorable], dtype=np.int64)
        self.claims = np.array([i % n for i in scorable], dtype=np.int64)
        self.labels = np.array([rows[i]["is_genuine"] for i in scorable], dtype=bool)
        self.query_speakers = bank.owners[self.queries]
        if not self.labels.any() or self.labels.all():
            raise ValueError("Evaluation requires genuine and impostor trials")

    @torch.no_grad()
    def scores(self, student, positions=None):
        pos = (
            np.arange(len(self.labels)) if positions is None else np.asarray(positions)
        )
        arrays = self.bank.materialize(
            student.encoder, self.queries[pos], self.claims[pos]
        )
        student.fusion.eval()
        values = []
        for start in range(0, len(pos), 256):
            features = self.bank.features_for(arrays, slice(start, start + 256))
            if not features[2][:, :5].all():
                raise ValueError("Frozen five-vowel eligibility changed")
            values.append(student.fusion(*features)[0].numpy())
        return np.concatenate(values)

    def proxy_positions(self, seed, count):
        rng = np.random.default_rng(seed)
        # Match each genuine query to an impostor for the same query.
        genuine = np.flatnonzero(self.labels)
        genuine = rng.choice(genuine, min(count, len(genuine)), replace=False)
        negative = []
        for index in genuine:
            choices = np.flatnonzero(
                (self.queries == self.queries[index]) & ~self.labels
            )
            if not len(choices):
                raise ValueError("Missing matched proxy impostor")
            negative.append(rng.choice(choices))
        return np.r_[genuine, negative]


def evaluate(student, evaluations, save_scores=False):
    results, scores = {}, {}
    for corpus, data in evaluations.items():
        values = data.scores(student)
        results[corpus] = float(metrics.roc_eer(values, data.labels)["eer"])
        if save_scores:
            scores[corpus] = values
    return (results, scores) if save_scores else results


def macro(values):
    return float(np.mean(list(values.values())))


def proxy_loss(student, evaluations, config, seed):
    losses = []
    for corpus, data in evaluations.items():
        pos = data.proxy_positions(
            seed_for(seed, corpus), config["proxy_queries_per_corpus"]
        )
        score = torch.from_numpy(data.scores(student, pos))
        # Current training-only calibration is frozen in both virtual encoder branches.
        logits = (
            student.fusion.log_scale.detach().exp().clamp(1, 100) * score
            + student.fusion.bias.detach()
        )
        target = torch.from_numpy(data.labels[pos].astype(np.float32))
        losses.append(
            float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
        )
    return float(np.mean(losses))


def training_pairs(bank, seed, steps, count):
    from runtime import corpus

    rng = np.random.default_rng(seed)
    by_speaker = [np.flatnonzero(bank.owners == i) for i in range(len(bank.speakers))]
    corpora = [corpus(s) for s in bank.speakers]
    negatives = [
        np.array([j for j, c in enumerate(corpora) if c == corpora[i] and i != j])
        for i in range(len(bank.speakers))
    ]
    owner = rng.integers(len(bank.speakers), size=steps * count)
    q = np.array([rng.choice(by_speaker[s]) for s in owner])
    impostor = np.array([rng.choice(negatives[s]) for s in owner])
    q = np.stack((q, q), 1).reshape(steps, count * 2)
    claims = np.stack((owner, impostor), 1).reshape(steps, count * 2)
    return q, claims


def fit_fusion(student, bank, config, seed, steps):
    q, c = training_pairs(bank, seed, steps, config["fusion_queries_per_update"])
    arrays = bank.materialize(student.encoder, q.reshape(-1), c.reshape(-1))
    width = q.shape[1]
    # Genuine/impostor share the same available inventory, as in the source learner.
    qm, cm = arrays["qmap"], arrays["cmap"]
    masks = arrays["qmask"][qm] & arrays["emask"][cm]
    shared = masks.reshape(-1, 2, len(bank.phones)).all(1).repeat_interleave(2, 0)
    if not shared[:, :5].all():
        raise ValueError("Fusion training eligibility changed")
    labels = torch.tensor([1.0, 0.0]).repeat(width // 2)
    student.fusion.train()
    losses = []
    for i in range(steps):
        torch.manual_seed(seed_for(seed, "fusion-dropout", i))
        features = bank.features_for(arrays, slice(i * width, (i + 1) * width), shared)
        student.fusion_optimizer.zero_grad(set_to_none=True)
        loss = student.fusion.loss(
            *features, labels, regularization=config["fusion_uniform_kl"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            student.fusion.parameters(),
            config["gradient_clip"],
            error_if_nonfinite=True,
        )
        student.fusion_optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def load_evaluations(config, prepared, split):
    from runtime import ROOT

    info = read(prepared / "inputs.json")
    partitions = read(prepared / "partitions.json")
    result = {}
    origin = "validation" if split in ("reward", "selection") else split
    for corpus in config["corpora"]:
        item = info[f"{corpus}-{origin}"]
        path = ROOT / item["features"]
        raw = (
            np.load(path, mmap_mode="r", allow_pickle=False)
            if path.suffix == ".npy"
            else np.memmap(path, mode="r", dtype="<f4", shape=(item["count"], 128))
        )
        metadata = np.load(
            prepared / f"{corpus}-{origin}-metadata.npy", allow_pickle=False
        )
        bank = Bank(read(ROOT / item["inputs"]), raw, metadata, info["phones"])
        allowed = (
            partitions[corpus][split] if split in ("reward", "selection") else None
        )
        result[corpus] = Evaluation(bank, read(ROOT / item["trials"]), allowed)
    return result
