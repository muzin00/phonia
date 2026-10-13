"""Same initialization and real-token schedule for both representation arms."""

import hashlib
import math
import time
from collections import defaultdict

from pair_common import (
    ARMS,
    ROOT,
    atomic_torch,
    check,
    model_hash,
    np,
    pin,
    rank,
    read,
    sha,
    state,
    torch,
    write,
)
from pair_model import PairEncoder, supcon
from phase3_train.losses import AAMSoftmax
from phase3_train.training import seed_everything


def schedule(units, pairs, updates, seed, speakers):
    groups = defaultdict(list)
    for i, u in enumerate(units):
        groups[u["pair"], u["speaker_id"]].append(i)
    lookup = {s: i for i, s in enumerate(speakers)}
    ids = np.empty((updates * 5, 20), dtype=np.int64)
    labels = np.empty_like(ids)
    for position, pair in enumerate(pairs):
        eligible = {
            s: np.asarray(v)
            for (p, s), v in groups.items()
            if p == pair and len(v) >= 2
        }
        owners = sorted(eligible, key=lambda s: (rank(seed, f"speaker/{pair}/{s}"), s))
        assert len(owners) >= 10
        slots = np.arange(position, updates * 5, len(pairs))
        chosen = (np.arange(len(slots))[:, None] * 10 + np.arange(10)[None]) % len(
            owners
        )
        for i, s in enumerate(owners):
            rr, cc = np.where(chosen == i)
            rng = np.random.default_rng(int(rank(seed, f"tokens/{pair}/{s}")[:16], 16))
            pool = eligible[s]
            first = rng.integers(len(pool), size=len(rr))
            second = rng.integers(len(pool) - 1, size=len(rr))
            second += second >= first
            for offset, chosen_ids in ((0, first), (1, second)):
                ids[slots[rr], cc * 2 + offset] = pool[chosen_ids]
                labels[slots[rr], cc * 2 + offset] = lookup[s]
    ids, labels = ids.reshape(updates, 100), labels.reshape(updates, 100)
    digest = hashlib.sha256(ids.tobytes() + labels.tobytes()).hexdigest()
    return ids, labels, digest


def learning_rate(config, update, total):
    if update < config["warmup_updates"]:
        return config["learning_rate"] * (update + 1) / config["warmup_updates"]
    progress = min(
        1,
        max(
            0, (update - config["warmup_updates"]) / (total - config["warmup_updates"])
        ),
    )
    return config["minimum_learning_rate"] + 0.5 * (
        config["learning_rate"] - config["minimum_learning_rate"]
    ) * (1 + math.cos(math.pi * progress))


def optimize(
    config,
    directory,
    arm,
    seed,
    features,
    ids,
    labels,
    speaker_count,
    identity,
    stop_after=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = PairEncoder(arm)
    head = AAMSoftmax(speaker_count)
    initial = {
        "encoder": model_hash(model.state_dict()),
        "head": model_hash(head.state_dict()),
        "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    parameters = [*model.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    completed, history = 0, []
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert (
            saved["identity"] == identity
            and saved["initial"] == initial
            and saved["arm"] == arm
        )
        model.load_state_dict(saved["model"])
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        completed, history = saved["update"], saved["history"]
    total = len(ids)
    target = total if stop_after is None else min(total, stop_after)
    start = time.monotonic()
    model.train()
    head.train()
    for u in range(completed, target):
        lr = learning_rate(config, u, total)
        for group in optimizer.param_groups:
            group["lr"] = lr
        embeddings = model(features[ids[u]])
        aam = head(embeddings, labels[u])
        contrastive = supcon(embeddings)
        loss = aam + 0.5 * contrastive
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (u + 1) % 1000 == 0 or u + 1 == total:
            history.append(
                {
                    "update": u + 1,
                    "loss": float(loss.detach()),
                    "aam": float(aam.detach()),
                    "supcon": float(contrastive.detach()),
                    "gradient_norm": float(gradient),
                    "learning_rate": lr,
                }
            )
        if (u + 1) % config["checkpoint_interval"] == 0 or u + 1 == target:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(config, u + 1, total)
            atomic_torch(
                checkpoint,
                {
                    "identity": identity,
                    "arm": arm,
                    "initial": initial,
                    "update": u + 1,
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                    "history": history,
                },
            )
            state(
                directory,
                "training",
                update=u + 1,
                total=total,
                elapsed=time.monotonic() - start,
                loss=float(loss.detach()),
            )
    return model.eval(), initial


def train_seed(config, run, seed):
    frozen = read(run / "design-freeze.json")
    check(frozen["files"])
    pairs = read(run / "pair-selection.json")["selected"]
    units = read(run / "data/train-train/units.json")
    speakers = sorted({u["speaker_id"] for u in units})
    source_speakers = read(ROOT / config["source_run"] / "preparation-report.json")[
        "training_speakers"
    ]
    assert len(speakers) >= 2 and set(speakers) <= set(source_speakers)
    groups = defaultdict(int)
    for unit in units:
        groups[unit["speaker_id"], unit["pair"]] += 1
    assert {speaker for (speaker, _), count in groups.items() if count >= 2} == set(
        speakers
    )
    updates = config["updates_per_pair_type"] * len(pairs)
    ids, labels, digest = schedule(units, pairs, updates, seed, speakers)
    seed_dir = run / f"seed-{seed}"
    seed_dir.mkdir(exist_ok=True)
    schedule_path = seed_dir / "schedule.npz"
    if schedule_path.exists():
        with np.load(schedule_path, allow_pickle=False) as old:
            np.testing.assert_array_equal(old["ids"], ids)
            np.testing.assert_array_equal(old["labels"], labels)
    else:
        temp = schedule_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, ids=ids, labels=labels)
        temp.replace(schedule_path)
    cache = run / "data/train-train/features.npy"
    features = torch.from_numpy(np.load(cache, allow_pickle=False))
    ids, labels = torch.from_numpy(ids), torch.from_numpy(labels)
    identity = {
        "design_sha256": sha(run / "design-freeze.json"),
        "schedule_sha256": digest,
        "feature_sha256": sha(cache),
        "units_sha256": sha(run / "data/train-train/units.json"),
        "updates": updates,
    }
    initial = None
    for arm in ARMS:
        directory = seed_dir / arm
        if (directory / "complete.json").exists():
            receipt = read(directory / "complete.json")
            check(receipt["files"])
            assert receipt["identity"] == identity
            current = read(directory / "summary.json")["initial"]
        else:
            model, current = optimize(
                config,
                directory,
                arm,
                seed,
                features,
                ids,
                labels,
                len(speakers),
                identity,
            )
            atomic_torch(
                directory / "encoder.pt",
                {
                    "model": model.state_dict(),
                    "arm": arm,
                    "update": updates,
                    "seed": seed,
                },
            )
            restored = PairEncoder(arm).eval()
            restored.load_state_dict(
                torch.load(directory / "encoder.pt", weights_only=True)["model"]
            )
            with torch.no_grad():
                assert torch.equal(model(features[:100]), restored(features[:100]))
            saved = torch.load(directory / "last.pt", weights_only=False)
            assert saved["update"] == updates
            assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {
                updates
            }
            write(directory / "history.json", saved["history"], immutable=True)
            write(
                directory / "summary.json",
                {
                    "arm": arm,
                    "seed": seed,
                    "updates": updates,
                    "training_tokens": updates * 100,
                    "source_training_labels": len(source_speakers),
                    "active_training_labels": len(speakers),
                    "labels_without_selected_tokens": len(
                        set(source_speakers) - set(speakers)
                    ),
                    "initial": current,
                    "final_encoder_sha256": model_hash(model.state_dict()),
                    "encoder_parameters": sum(p.numel() for p in model.parameters()),
                    "identity": identity,
                },
                immutable=True,
            )
            write(
                directory / "complete.json",
                {
                    "identity": identity,
                    "files": pin(
                        [
                            directory / n
                            for n in (
                                "encoder.pt",
                                "last.pt",
                                "history.json",
                                "summary.json",
                            )
                        ]
                    ),
                },
                immutable=True,
            )
            state(directory, "complete", updates=updates)
        if initial is not None:
            assert initial == current
        initial = current
