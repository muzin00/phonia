"""Train the original MLP on cached nonlearned pooling, with phone-local groups."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict

from common import (
    checked,
    create_encoder,
    expanded,
    json_sha256,
    np,
    ordered,
    rank,
    read_json,
    rows,
    sha256_file,
    torch,
    write_json,
)
from phase3_train.losses import AAMSoftmax
from phase3_train.training import WarmupCosine, _restore_rng, _rng_state


def schedule(groups, phones, updates, seed, speaker_to_class):
    """Complete positive pairs, unique speakers, and balanced phone exposure.

    Rare phones remain in the search. Their groups contain up to ten available
    speakers, rather than requiring that all 140 speakers utter every phone.
    Padding IDs are masked out of both losses and every contrastive denominator.
    """
    matrix = np.full((updates * 5, 20), -1, dtype=np.int64)
    labels = np.zeros_like(matrix)
    valid = np.zeros_like(matrix, dtype=bool)
    for position, phone in enumerate(phones):
        eligible = {
            s: np.asarray(ids, dtype=np.int64)
            for (s, p), ids in groups.items()
            if p == phone and len(ids) >= 2
        }
        if not eligible:
            raise ValueError(f"no positive pair for {phone}")
        speakers = sorted(
            eligible, key=lambda s: (rank(seed, f"speaker/{phone}/{s}"), s)
        )
        width = min(10, len(speakers))
        slots = np.arange(position, updates * 5, len(phones))
        selected = (
            np.arange(len(slots))[:, None] * width + np.arange(width)[None]
        ) % len(speakers)
        for i, speaker in enumerate(speakers):
            rr, cc = np.where(selected == i)
            generator = np.random.default_rng(
                int(rank(seed, f"segments/{phone}/{speaker}")[:16], 16)
            )
            ids = eligible[speaker]
            first = generator.integers(len(ids), size=len(rr))
            second = generator.integers(len(ids) - 1, size=len(rr))
            second += second >= first
            for offset, choices in ((0, first), (1, second)):
                matrix[slots[rr], cc * 2 + offset] = ids[choices]
                labels[slots[rr], cc * 2 + offset] = speaker_to_class[speaker]
                valid[slots[rr], cc * 2 + offset] = True
    if not valid.any(axis=1).all() or np.any(valid.sum(axis=1) % 2):
        raise ValueError("incomplete scheduled positive groups")
    return (
        matrix.reshape(updates, 100),
        labels.reshape(updates, 100),
        valid.reshape(updates, 100),
    )


def grouped_supcon(embeddings, valid):
    """The original within-phone loss, vectorized over five complete groups."""
    values = torch.nn.functional.normalize(
        embeddings.reshape(5, 20, 128), dim=2, eps=1e-12
    )
    mask = valid.reshape(5, 20)
    eye = torch.eye(20, dtype=torch.bool)
    logits = (values @ values.transpose(1, 2)) / 0.07
    logits = logits.masked_fill(eye[None] | ~mask[:, None, :], -torch.inf)
    denominator = torch.logsumexp(logits, dim=2)
    positive_index = (torch.arange(20) ^ 1)[None, :, None].expand(5, -1, -1)
    positive_logits = logits.gather(2, positive_index).squeeze(2)
    loss = torch.where(mask, denominator - positive_logits, 0)
    return (loss.sum(dim=1) / mask.sum(dim=1)).mean()


def grouped_aam(head, embeddings, labels, valid):
    cosine = head.cosine(embeddings)
    target = cosine.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
    logits = (
        cosine.scatter(1, labels[:, None], torch.cos(torch.acos(target) + head.margin))
        * head.scale
    )
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").reshape(
        5, 20
    )
    mask = valid.reshape(5, 20)
    return ((loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()


class TrainingPool:
    def __init__(self, config, run):
        report = read_json(run / "preparation-report.json")
        self.speakers = report["training_speakers"]
        self.speaker_to_class = {s: i for i, s in enumerate(self.speakers)}
        self.groups = {}
        self.phones = []
        self.source_counts = {}
        self.group_count = {}
        for row in rows(run / "train-segments.jsonl"):
            key = (row["speaker_id"], row["phoneme"])
            self.groups.setdefault(key, []).append(row["cache_index"])
            self.group_count[row["phoneme"]] = (
                self.group_count.get(row["phoneme"], 0) + 1
            )
        matrix = np.memmap(
            run / "train-features.f32",
            mode="r",
            dtype="<f4",
            shape=(report["train_feature_count"], 128),
        )
        self.features = torch.from_numpy(np.array(matrix))
        if not torch.isfinite(self.features).all():
            raise ValueError("nonfinite frozen feature cache")
        self.manifest_sha256 = sha256_file(run / "train-segments.jsonl")
        self.feature_sha256 = sha256_file(run / "train-features.f32")
        self.statistics_sha256 = config["feature_statistics_sha256"]


def load_model(trial):
    summary = read_json(trial / "training-summary.json")
    checked(trial / "encoder.pt", summary["encoder_sha256"])
    bundle = torch.load(trial / "encoder.pt", map_location="cpu", weights_only=True)
    model = create_encoder("statistics_mlp").eval()
    model.load_state_dict(bundle["model"], strict=True)
    if expanded.model_sha256(model) != summary["final_encoder_sha256"]:
        raise ValueError("exported encoder state mismatch")
    return model, bundle


def train_trial(config, run, trial, phones, pool):
    if (trial / "training-summary.json").exists():
        model, _ = load_model(trial)
        return model
    trial.mkdir(exist_ok=True)
    phones = ordered(phones)
    settings = expanded.ExpansionSettings(
        encoder=config["encoder"],
        cohort=140,
        rms_enabled=False,
        supcon_enabled=True,
        seed=config["training_seed"],
        maximum_updates=config["maximum_updates"],
        warmup_updates=config["warmup_updates"],
        validation_interval=config["checkpoint_interval"],
    )
    expanded.seed_everything(settings.seed)
    model = create_encoder(settings.encoder)
    head = AAMSoftmax(len(pool.speakers))
    initial_model = expanded.model_sha256(model)
    initial_head = expanded.model_sha256(head)
    training_data = {
        "phonemes": phones,
        "settings": asdict(settings),
        "initial_encoder_sha256": initial_model,
        "initial_head_sha256": initial_head,
        "all_pool_manifest_sha256": pool.manifest_sha256,
        "feature_cache_sha256": pool.feature_sha256,
        "normalization_sha256": pool.statistics_sha256,
        "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
    }
    if (trial / "training-freeze.json").exists():
        if read_json(trial / "training-freeze.json") != training_data:
            raise ValueError("trial protocol changed during resume")
    else:
        write_json(trial / "training-freeze.json", training_data)
    planned, classes, valid = schedule(
        pool.groups,
        phones,
        settings.maximum_updates,
        settings.seed,
        pool.speaker_to_class,
    )
    scheduling_sha = json_sha256(
        {
            "planned": __import__("hashlib").sha256(planned.tobytes()).hexdigest(),
            "labels": __import__("hashlib").sha256(classes.tobytes()).hexdigest(),
            "valid": __import__("hashlib").sha256(valid.tobytes()).hexdigest(),
        }
    )
    planned = torch.from_numpy(np.maximum(planned, 0))
    classes = torch.from_numpy(classes)
    valid = torch.from_numpy(valid)
    parameters = [*model.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings.weight_decay,
    )
    scheduler = WarmupCosine(optimizer, settings)
    start_update = 0
    checkpoint = trial / "last.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        if (
            state["training_freeze_sha256"]
            != sha256_file(trial / "training-freeze.json")
            or state["schedule_sha256"] != scheduling_sha
        ):
            raise ValueError("checkpoint trial/schedule mismatch")
        model.load_state_dict(state["model"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        start_update = state["update"]
    history = trial / "history.jsonl"
    if history.exists():
        kept = [r for r in rows(history) if r["update"] <= start_update]
        history.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in kept))
    model.train()
    head.train()
    start = time.perf_counter()
    examples = 0
    with history.open("a") as stream:
        for u in range(start_update, settings.maximum_updates):
            inputs = pool.features[planned[u]]
            mask = valid[u]
            embeddings = torch.nn.functional.normalize(
                model.projection(inputs), dim=1, eps=model.l2_epsilon
            )
            aam = grouped_aam(head, embeddings, classes[u], mask)
            contrastive = grouped_supcon(embeddings, mask)
            loss = aam + 0.5 * contrastive
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, settings.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            examples += int(mask.sum())
            if (u + 1) % 1000 == 0:
                entry = {
                    "update": u + 1,
                    "loss": float(loss.detach()),
                    "aam": float(aam.detach()),
                    "supcon": float(contrastive.detach()),
                    "gradient_norm": float(norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - start,
                }
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps({"trial": trial.name, **entry}), flush=True)
            if (
                u + 1
            ) % settings.validation_interval == 0 or u + 1 == settings.maximum_updates:
                temporary = trial / ".last.tmp"
                torch.save(
                    {
                        "update": u + 1,
                        "model": model.state_dict(),
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": _rng_state(),
                        "training_freeze_sha256": sha256_file(
                            trial / "training-freeze.json"
                        ),
                        "schedule_sha256": scheduling_sha,
                    },
                    temporary,
                )
                temporary.replace(checkpoint)
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": settings.encoder,
            "embedding_dimension": 128,
            "phonemes": phones,
            "checkpoint_update": settings.maximum_updates,
            "run_seed": settings.seed,
        },
        trial / "encoder.pt",
    )
    exported = create_encoder(settings.encoder).eval()
    exported.load_state_dict(
        torch.load(trial / "encoder.pt", weights_only=True)["model"]
    )
    probe = pool.features[:32]
    with torch.inference_mode():
        if not torch.equal(model.projection(probe), exported.projection(probe)):
            raise ValueError("export reload changes projection")
    counts = {
        p: 20
        * int(((torch.arange(settings.maximum_updates * 5) % len(phones)) == i).sum())
        for i, p in enumerate(phones)
    }
    actual_examples = int(valid.sum())
    summary = {
        "status": "completed",
        "phonemes": phones,
        "completed_updates": settings.maximum_updates,
        "encoder_parameters": sum(p.numel() for p in model.parameters()),
        "training_examples": actual_examples,
        "maximum_examples_per_update": 100,
        "phone_slot_maximum_example_counts": counts,
        "sparse_phone_masked_slots": settings.maximum_updates * 100 - actual_examples,
        "initial_encoder_sha256": initial_model,
        "initial_head_sha256": initial_head,
        "final_encoder_sha256": expanded.model_sha256(model),
        "encoder_sha256": sha256_file(trial / "encoder.pt"),
        "training_freeze_sha256": sha256_file(trial / "training-freeze.json"),
        "schedule_sha256": scheduling_sha,
        "history_sha256": sha256_file(history),
        "elapsed_this_invocation_seconds": time.perf_counter() - start,
        "selected_checkpoint": "fixed_final_update",
        "test_used": False,
        "normalization_sha256": pool.statistics_sha256,
    }
    write_json(trial / "training-summary.json", summary)
    print(
        json.dumps(
            {
                "stage": "training_completed",
                "trial": trial.name,
                "elapsed_seconds": summary["elapsed_this_invocation_seconds"],
            }
        ),
        flush=True,
    )
    del planned, classes, valid, optimizer, parameters, exported, head
    gc.collect()
    return model
