"""Three-corpus, frozen-encoder comparison of equal, MLP and Transformer fusion."""

from __future__ import annotations

import argparse
import copy
import fcntl
import math
import os
import time

from inputs import (
    BACKEND,
    BASE,
    aggregates,
    backend,
    batch,
    make_schedule,
    r,
    s,
    training_batch,
    training_inputs,
)

CONFIG = BASE / "config.json"
ARRAYS = ("enrollment", "emeta", "emask", "queries", "qmeta", "qmask")


def freeze(config, run):
    source = s.ROOT / config["reference_run"]
    original = s.verify(source)
    s.verify_prepared(source)
    arm = source / config["encoder_arm"]
    r.checked_files(s.read(arm / "design-freeze.json")["files"])
    assert s.sha(arm / "trial/encoder.pt") == config["encoder_sha256"]
    files = {
        **original["files"],
        **s.read(source / "prepared-inputs-freeze.json")["files"],
        **s.read(arm / "design-freeze.json")["files"],
    }
    paths = [
        CONFIG,
        BASE / "experiment.py",
        BASE / "inputs.py",
        BACKEND,
        *sorted((BASE / "tests").glob("*.py")),
        arm / "trial/encoder.pt",
        arm / "trial/training-summary.json",
        source / "design-freeze.json",
        source / "results.json",
    ]
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            paths += [
                source / "evaluation-inputs" / f"{corpus}-{split}.json",
                source / "scores" / f"{corpus}-expanded280-{split}.json",
            ]
    files.update(r.pin(paths))
    r.checked_files(files)
    frozen = {
        "config": config,
        "files": files,
        "runtime": s.expanded.runtime(),
        "reference_design": original,
        "phones": original["phonemes"],
        "encoder_fixed": True,
        "all_test_previously_observed": True,
    }
    s.fixed(run / "design-freeze.json", frozen)
    return frozen


def evaluation_source(config, original, corpus, split):
    reference = s.ROOT / config["reference_run"]
    data = s.read(reference / "evaluation-inputs" / f"{corpus}-{split}.json")
    prior = original["config"]
    if corpus == "common_voice":
        path = s.ROOT / prior["cv_evaluation_run"] / split
        features = s.np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
        manifest = path / "segments.jsonl"
    else:
        path = s.ROOT / prior["source_run" if corpus == "jvs" else "corpus_run"]
        count = (
            data["feature_count"]
            if corpus == "jvs"
            else s.read(path / "preparation-report.json")["splits"][split]["segments"]
        )
        features = s.np.memmap(
            path / f"{split}-features.f32", dtype="<f4", mode="r", shape=(count, 128)
        )
        manifest = path / f"{split}-segments.jsonl"
    return data, features, manifest


def score_inputs(data, arrays, phones, baseline):
    count = len(data["queries"]) * len(data["speakers"])
    assert count == len(baseline)
    components = s.np.zeros((count, len(phones)), dtype=s.np.float64)
    masks = s.np.zeros_like(components, dtype=bool)
    allowed = set(data["universally_registered"]) & set(phones)
    for q, query in enumerate(data["queries"]):
        for c, speaker in enumerate(data["speakers"]):
            i = q * len(data["speakers"]) + c
            row = baseline[i]
            assert (
                row["query_id"] == query["query_id"]
                and row["claimed_speaker_id"] == speaker
            )
            for p, phone in enumerate(phones):
                masks[i, p] = (
                    phone in allowed and arrays["qmask"][q, p] and arrays["emask"][c, p]
                )
                if masks[i, p]:
                    components[i, p] = float(
                        arrays["queries"][q, p] @ arrays["enrollment"][c, p]
                    )
            assert bool(masks[i, :5].all()) == (row["score"] is not None)
            if row["score"] is not None:
                assert abs(components[i, masks[i]].mean() - row["score"]) < 1e-12
    return components, masks


def prepare(config, run, frozen, state):
    if (run / "prepared-freeze.json").exists():
        r.checked_files(s.read(run / "prepared-freeze.json")["files"])
        return
    source = s.ROOT / config["reference_run"]
    phones = frozen["phones"]
    model, _ = s.load_model(source / config["encoder_arm"] / "trial")
    evaluation_speakers, evaluation_hashes = set(), set()
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            data, _, manifest = evaluation_source(
                config, frozen["reference_design"], corpus, split
            )
            evaluation_speakers.update(data["speakers"])
            evaluation_hashes.update(row["source_sha256"] for row in s.rows(manifest))
    state("preparing_train")
    train_source = source / config["encoder_arm"]
    train, metadata = training_inputs(
        config,
        train_source / "train-segments.jsonl",
        phones,
        evaluation_speakers,
        evaluation_hashes,
    )
    s.fixed(run / "train-inputs.json", train)
    features = s.np.memmap(
        train_source / "train-features.f32",
        dtype="<f4",
        mode="r",
        shape=(train["feature_count"], 128),
    )
    vectors = s.model_embeddings(model, features)
    arrays = aggregates(train, vectors, metadata, phones)
    s.np.savez(run / "train.npz", **arrays)
    del vectors, features, arrays, metadata
    for seed in config["seeds"]:
        s.np.savez(run / f"schedule-{seed}.npz", **make_schedule(config, train, seed))
    paths = [
        run / "train-inputs.json",
        run / "train.npz",
        *(run / f"schedule-{seed}.npz" for seed in config["seeds"]),
    ]
    for corpus in s.CORPORA:
        for split in ("validation", "test"):
            state("preparing_evaluation", corpus=corpus, split=split)
            data, features, manifest = evaluation_source(
                config, frozen["reference_design"], corpus, split
            )
            metadata = s.np.array(
                [
                    [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
                    for row in s.rows(manifest)
                ],
                dtype=s.np.float32,
            )
            assert len(metadata) == len(features)
            vectors = s.model_embeddings(model, features)
            arrays = aggregates(data, vectors, metadata, phones)
            baseline = s.read(source / "scores" / f"{corpus}-expanded280-{split}.json")
            components, masks = score_inputs(data, arrays, phones, baseline)
            path = run / f"{corpus}-{split}.npz"
            s.np.savez(path, **arrays, components=components, masks=masks)
            s.fixed(run / f"{corpus}-{split}-inputs.json", data)
            s.fixed(run / f"{corpus}-{split}-baseline.json", baseline)
            paths += [
                path,
                run / f"{corpus}-{split}-inputs.json",
                run / f"{corpus}-{split}-baseline.json",
            ]
            del vectors, features, arrays, metadata, components, masks
    s.fixed(
        run / "prepared-freeze.json",
        {
            "files": r.pin(paths),
            "eligible_fusion_labels": len(train["speakers"]),
            "label_counts": train["training_label_counts"],
            "excluded_labels": train["excluded_labels"],
            "queries": len(train["queries"]),
            "baseline_maximum_score_difference_tolerance": 1e-12,
        },
    )


def load_arrays(path):
    with s.np.load(path, allow_pickle=False) as archive:
        return {
            k: s.torch.from_numpy(archive[k].copy())
            if k in ARRAYS
            else archive[k].copy()
            for k in archive.files
        }


def create_model(config, kind):
    return backend.Fusion(
        kind,
        dimension=config["dimension"],
        layers=config["transformer_layers"],
        heads=config["attention_heads"],
    )


def score(model, arrays, baseline):
    model.eval()
    result = copy.deepcopy(baseline)
    active = s.np.array(
        [i for i, row in enumerate(result) if row["score"] is not None],
        dtype=s.np.int64,
    )
    speakers = len(arrays["enrollment"])
    with s.torch.inference_mode():
        for start in range(0, len(active), 128):
            ids = active[start : start + 128]
            f, c, mask = batch(
                arrays,
                s.torch.from_numpy(ids // speakers),
                s.torch.from_numpy(ids % speakers),
                s.torch.from_numpy(arrays["masks"][ids]),
            )
            _, weights = model(f, c, mask)
            weights = weights.numpy().astype(s.np.float64)
            weights /= weights.sum(axis=1, keepdims=True)
            values = (weights * arrays["components"][ids]).sum(axis=1)
            for i, value in zip(ids, values, strict=True):
                result[int(i)]["score"] = float(value)
    return result


def eer(records):
    rows = [
        row
        for row in records
        if row["role"] == "verification" and row["score"] is not None
    ]
    value = s.roc_eer(
        [row["score"] for row in rows], [row["is_genuine"] for row in rows]
    )["eer"]
    assert value is not None
    return value


def train(config, run, state):
    r.checked_files(s.read(run / "prepared-freeze.json")["files"])
    train = s.read(run / "train-inputs.json")
    arrays = load_arrays(run / "train.npz")
    validation = {
        c: (
            load_arrays(run / f"{c}-validation.npz"),
            s.read(run / f"{c}-validation-baseline.json"),
        )
        for c in s.CORPORA
    }
    for seed in config["seeds"]:
        with s.np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as archive:
            schedule = {k: archive[k].copy() for k in archive.files}
        for kind in config["architectures"]:
            directory = run / f"{kind}-{seed}"
            if (directory / "training-summary.json").exists():
                assert (
                    s.sha(directory / "fusion.pt")
                    == s.read(directory / "training-summary.json")["model_sha256"]
                )
                continue
            directory.mkdir(exist_ok=True)
            s.expanded.seed_everything(seed)
            model = create_model(config, kind)
            optimizer = s.torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            best, selected, start, history = math.inf, 0, 0, []
            if (directory / "last.pt").exists():
                saved = s.torch.load(directory / "last.pt", weights_only=False)
                assert saved["design_sha256"] == s.sha(run / "design-freeze.json")
                model.load_state_dict(saved["model"])
                optimizer.load_state_dict(saved["optimizer"])
                s.torch.set_rng_state(saved["rng"])
                best, selected, start, history = (
                    saved["best"],
                    saved["selected"],
                    saved["update"] + 1,
                    saved["history"],
                )
            began = time.monotonic()
            for update in range(start, config["updates"] + 1):
                if update:
                    model.train()
                    lr = config["learning_rate"] * (
                        0.1
                        + 0.9
                        * (1 + math.cos(math.pi * (update - 1) / config["updates"]))
                        / 2
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    f, c, mask, labels = training_batch(
                        arrays, train, schedule, update - 1
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(
                        f, c, mask, labels, config["uniform_kl_coefficient"]
                    )
                    assert s.torch.isfinite(loss)
                    loss.backward()
                    s.torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 5, error_if_nonfinite=True
                    )
                    optimizer.step()
                if update % config["validation_every"]:
                    continue
                values = {c: score(model, a, b) for c, (a, b) in validation.items()}
                eers = {c: eer(v) for c, v in values.items()}
                macro = sum(eers.values()) / len(eers)
                if macro < best - 1e-12:
                    best, selected = macro, update
                    s.torch.save(
                        {
                            "kind": kind,
                            "seed": seed,
                            "update": update,
                            "state_dict": model.state_dict(),
                        },
                        directory / "fusion.pt",
                    )
                item = {
                    "update": update,
                    "validation_eer": eers,
                    "macro_eer": macro,
                    "selected_update": selected,
                    "loss": float(loss.detach()) if update else None,
                    "elapsed_seconds": time.monotonic() - began,
                }
                history.append(item)
                s.atomic(directory / "history.json", history)
                state("training", kind=kind, seed=seed, **item)
                print({"variant": directory.name, **item}, flush=True)
                temporary = directory / "last.tmp"
                s.torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": s.torch.get_rng_state(),
                        "best": best,
                        "selected": selected,
                        "update": update,
                        "history": history,
                        "design_sha256": s.sha(run / "design-freeze.json"),
                    },
                    temporary,
                )
                temporary.replace(directory / "last.pt")
            s.fixed(
                directory / "training-summary.json",
                {
                    "completed_updates": config["updates"],
                    "selected_update": selected,
                    "validation_macro_eer": best,
                    "kind": kind,
                    "seed": seed,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "model_sha256": s.sha(directory / "fusion.pt"),
                    "encoder_sha256": config["encoder_sha256"],
                    "encoder_updated": False,
                    "schedule_sha256": s.sha(run / f"schedule-{seed}.npz"),
                    "history_sha256": s.sha(directory / "history.json"),
                },
            )


def load_model(config, directory):
    summary = s.read(directory / "training-summary.json")
    assert s.sha(directory / "fusion.pt") == summary["model_sha256"]
    bundle = s.torch.load(directory / "fusion.pt", weights_only=True)
    model = create_model(config, bundle["kind"])
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model


def evaluate(config, run, state):
    variants = [
        f"{kind}-{seed}" for seed in config["seeds"] for kind in config["architectures"]
    ]
    models = {name: load_model(config, run / name) for name in variants}
    points = {}
    for corpus in s.CORPORA:
        state("calibrating", corpus=corpus)
        arrays = load_arrays(run / f"{corpus}-validation.npz")
        baseline = s.read(run / f"{corpus}-validation-baseline.json")
        points[corpus] = {}
        for name in ["equal", *variants]:
            records = (
                baseline if name == "equal" else score(models[name], arrays, baseline)
            )
            rows = [
                row
                for row in records
                if row["role"] == "verification" and row["score"] is not None
            ]
            value, _ = s.far_target_threshold(
                [row["score"] for row in rows],
                [row["is_genuine"] for row in rows],
                0.01,
            )
            points[corpus][name] = s.json_threshold(value)
            s.fixed(run / "scores" / f"{corpus}-{name}-validation.json", records)
    s.fixed(
        run / "selection-freeze.json",
        {
            "models": r.pin(
                run / name / file
                for name in variants
                for file in ("fusion.pt", "training-summary.json")
            ),
            "thresholds": points,
            "all_six_models_fixed_before_test_scoring": True,
            "test_previously_observed": True,
        },
    )
    results = {}
    for corpus in s.CORPORA:
        state("evaluating", corpus=corpus)
        arrays = load_arrays(run / f"{corpus}-test.npz")
        baseline = s.read(run / f"{corpus}-test-baseline.json")
        results[corpus] = {}
        for name in ["equal", *variants]:
            records = (
                baseline if name == "equal" else score(models[name], arrays, baseline)
            )
            s.fixed(run / "scores" / f"{corpus}-{name}-test.json", records)
            results[corpus][name] = {
                role: s.operating(
                    [row for row in records if row["role"] == role],
                    points[corpus][name],
                )
                for role in sorted({row["role"] for row in records})
            }
    s.fixed(
        run / "results.json",
        {
            "conditions": results,
            "config": config,
            "selection_sha256": s.sha(run / "selection-freeze.json"),
            "comparison": "one frozen encoder; mean over fusion seeds, not acoustic encoder seeds",
        },
    )
    lines = [
        "# 3コーパスencoderとTransformer統合",
        "",
        "音響encoderは初回280条件の1モデルを固定。MLP/Transformerは統合層3seedの平均±標本SD。EERは採点可能分。",
        "",
        "| コーパス | 統合 | EER | FAR | FRR |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for corpus, values in results.items():
        for kind in ("equal", *config["architectures"]):
            names = (
                ["equal"]
                if kind == "equal"
                else [f"{kind}-{seed}" for seed in config["seeds"]]
            )
            metrics = []
            for metric in ("eer", "far", "frr"):
                x = (
                    s.np.array([values[name]["verification"][metric] for name in names])
                    * 100
                )
                metrics.append(
                    f"{x.mean():.3f}%"
                    if len(x) == 1
                    else f"{x.mean():.3f}±{x.std(ddof=1):.3f}%"
                )
            lines.append(f"| {corpus} | {kind} | {' | '.join(metrics)} |")
    lines += [
        "",
        "3コーパスの校正EER平均でcheckpointを選び、最終モデルと校正FAR1%の閾値を固定後にtest採点。全testは既観測。採点不能を拒否としてFRRに含む。",
    ]
    (run / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "prepare", "execute", "status"))
    args = parser.parse_args()
    s.torch.set_num_threads(1)
    config = s.read(CONFIG)
    run = s.ROOT / config["run_directory"]
    if args.phase == "status":
        print(s.read(run / "execution-state.json"))
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def state(status, **details):
            s.atomic(
                run / "execution-state.json",
                {
                    "status": status,
                    "pid": os.getpid(),
                    "updated_at_epoch": time.time(),
                    **details,
                },
            )

        try:
            frozen = freeze(config, run)
            if args.phase in ("prepare", "execute"):
                prepare(config, run, frozen, state)
            if args.phase == "execute":
                train(config, run, state)
                r.checked_files(frozen["files"])
                evaluate(config, run, state)
                state("complete", results=str((run / "results.md").relative_to(s.ROOT)))
        except BaseException as error:
            state("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
