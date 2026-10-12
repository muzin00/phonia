"""Continue a fixed acoustic endpoint without altering its training prefix."""

import hashlib
import json
import time

import scaling_learner as learner
from expansion_common import check_files, fixed, pin, read, sha

s = learner


def schedule_digest(arrays):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}


def make_schedule(config, pool, phones, seed):
    start, end = config["encoder_budgets"]
    assert start * 5 % len(phones) == 0, "continuation must preserve phone phase"
    values = s.schedule(
        pool.groups,
        phones,
        end - start,
        seed + config["encoder_schedule_seed_stride"],
        pool.speaker_to_class,
    )
    return dict(zip(("planned", "classes", "valid"), values, strict=True))


def train(config, source, target, pool, phones, seed, state):
    if (target / "complete.json").exists():
        check_files(read(target / "complete.json")["files"])
        return s.load_model(target)[0]
    target.mkdir(parents=True, exist_ok=True)
    first, last = config["encoder_budgets"]
    original = read(source / "training-summary.json")
    settings = read(source / "training-freeze.json")["settings"]
    assert phones == read(source / "training-freeze.json")["phonemes"]
    assert original["completed_updates"] == first and settings["seed"] == seed
    assert (
        settings["minimum_learning_rate"]
        == config["encoder_continuation_learning_rate"]
    )
    assert settings["cohort"] == len(pool.speakers)
    arrays = make_schedule(config, pool, phones, seed)
    fixed(
        target / "training-freeze.json",
        {
            "config": config,
            "seed": seed,
            "settings": {**settings, "maximum_updates": last},
            "files": pin(
                [
                    source / "last.pt",
                    source / "encoder.pt",
                    source / "training-summary.json",
                    source / "training-freeze.json",
                ]
            ),
            "schedule_digests": schedule_digest(arrays),
            "manifest_sha256": pool.manifest_sha256,
            "feature_sha256": pool.feature_sha256,
            "policy": "source prefix unchanged; one independent continuation sampling block; fixed terminal LR",
        },
    )
    check_files(read(target / "training-freeze.json")["files"])
    s.expanded.seed_everything(seed)
    model = s.create_encoder(settings["encoder"])
    head = s.AAMSoftmax(len(pool.speakers))
    parameters = [*model.parameters(), *head.parameters()]
    optimizer = s.torch.optim.AdamW(
        parameters,
        lr=settings["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=settings["weight_decay"],
    )
    checkpoint = target / "last.pt"
    resumed = checkpoint.exists()
    saved = s.torch.load(
        checkpoint if resumed else source / "last.pt", weights_only=False
    )
    assert saved["training_freeze_sha256"] == sha(
        (target if resumed else source) / "training-freeze.json"
    )
    if not resumed:
        assert saved["update"] == first
        exported = s.torch.load(source / "encoder.pt", weights_only=True)["model"]
        assert all(
            s.torch.equal(value, exported[key]) for key, value in saved["model"].items()
        )
    model.load_state_dict(saved["model"])
    head.load_state_dict(saved["head"])
    optimizer.load_state_dict(saved["optimizer"])
    s._restore_rng(saved["rng"])
    start = saved["update"]
    assert first <= start <= last
    planned = s.torch.from_numpy(s.np.maximum(arrays["planned"], 0))
    classes = s.torch.from_numpy(arrays["classes"])
    valid = s.torch.from_numpy(arrays["valid"])
    history = target / "history.jsonl"
    if history.exists():
        history.write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in s.rows(history)
                if row["update"] <= start
            )
        )
    began = time.monotonic()
    model.train()
    head.train()
    with history.open("a") as stream:
        for update in range(start + 1, last + 1):
            index = update - first - 1
            for group in optimizer.param_groups:
                group["lr"] = config["encoder_continuation_learning_rate"]
            embeddings = s.torch.nn.functional.normalize(
                model.projection(pool.features[planned[index]]),
                dim=1,
                eps=model.l2_epsilon,
            )
            aam = s.grouped_aam(head, embeddings, classes[index], valid[index])
            contrastive = s.grouped_supcon(embeddings, valid[index])
            loss = aam + 0.5 * contrastive
            assert s.torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = s.torch.nn.utils.clip_grad_norm_(
                parameters, settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if update % 1000 == 0 or update == last:
                row = {
                    "update": update,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - began,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(row, flush=True)
            if update % settings["validation_interval"] == 0 or update == last:
                temporary = target / "last.tmp"
                s.torch.save(
                    {
                        "update": update,
                        "model": model.state_dict(),
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": s._rng_state(),
                        "training_freeze_sha256": sha(target / "training-freeze.json"),
                    },
                    temporary,
                )
                temporary.replace(checkpoint)
                state("continuing_encoder", update=update)
    model.eval()
    s.torch.save(
        {
            "model": model.state_dict(),
            "encoder": settings["encoder"],
            "embedding_dimension": 128,
            "phonemes": phones,
            "checkpoint_update": last,
            "run_seed": seed,
        },
        target / "encoder.pt",
    )
    fixed(
        target / "training-summary.json",
        {
            "completed_updates": last,
            "initial_encoder_sha256": original["initial_encoder_sha256"],
            "final_encoder_sha256": s.expanded.model_sha256(model),
            "encoder_sha256": sha(target / "encoder.pt"),
            "source_encoder_sha256": original["encoder_sha256"],
            "continuation_examples": int(valid.sum()),
            "selected_checkpoint": "fixed_final_update",
            "test_used": False,
        },
    )
    fixed(
        target / "complete.json",
        {
            "files": pin(
                [
                    target / name
                    for name in (
                        "training-summary.json",
                        "training-freeze.json",
                        "encoder.pt",
                        "last.pt",
                        "history.jsonl",
                    )
                ]
            )
        },
    )
    return s.load_model(target)[0]
