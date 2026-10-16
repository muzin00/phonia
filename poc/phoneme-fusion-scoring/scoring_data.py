"""Audit source provenance and freeze one shared mined schedule per encoder."""

import sys

from mining import cv_score_table, make_schedule
from scoring import FrozenModel
from shared import (
    ARMS,
    BASE,
    ROOT,
    batch,
    encoder_path,
    evaluation_prepared,
    load_encoder,
    rt,
    source,
)

sys.path.insert(0, str(BASE.parent / "phoneme-training-quality-ablation"))
from quality_data import load_training

sys.path.insert(0, str(BASE))


def training_bank(config, seed):
    previous = rt.read(source(config) / "design-freeze.json")["config"]
    return load_training(previous, "original", seed)[1]


def audit_table(table, arrays, bank):
    positions = rt.np.flatnonzero(rt.np.isfinite(table).any(1))
    positions = positions[:: max(1, len(positions) // 100)][:100]
    largest = 0.0
    for q in positions:
        claims = rt.np.flatnonzero(rt.np.isfinite(table[q]))[:8]
        mask = (
            arrays["qmask"][q]
            & arrays["emask"][bank.owners[q]]
            & arrays["emask"][claims]
        )
        components = (
            arrays["enrollment"][claims].double() * arrays["queries"][q].double()
        ).sum(-1)
        expected = (components * mask).sum(1) / mask.sum(1)
        largest = max(
            largest, float(rt.np.max(rt.np.abs(expected.numpy() - table[q, claims])))
        )
    if largest > 1e-12:
        raise AssertionError("Independent mining score calculation differs")
    return largest


def prepare(config, run):
    directory = run / "prepared"
    seal = directory / "freeze.json"
    if seal.exists():
        saved = rt.read(seal)
        if saved["config"] != config:
            raise ValueError("Prepared configuration changed")
        rt.check(saved["files"])
        return
    directory.mkdir(parents=True, exist_ok=True)
    previous = rt.read(source(config) / "design-freeze.json")
    rt.check(previous["files"])
    rt.check(rt.read(source(config) / "complete.json")["files"])
    if (
        previous["config"]["evaluation_seeds"][: len(config["evaluation_seeds"])]
        != config["evaluation_seeds"]
    ):
        raise ValueError("Source seed identities changed")
    bank = training_bank(config, config["evaluation_seeds"][0])
    files = dict(previous["files"])
    files.update(
        rt.pin(
            [source(config) / "design-freeze.json", source(config) / "complete.json"]
        )
    )
    info = rt.read(evaluation_prepared(config) / "inputs.json")
    for corpus in config["corpora"]:
        for split in ("validation", "test"):
            people = rt.read(ROOT / info[f"{corpus}-{split}"]["inputs"])["speakers"]
            if set(people) & set(bank.speakers):
                raise ValueError("Training and evaluation speakers overlap")
    corpora = rt.np.array([rt.corpus(s) for s in bank.speakers])
    report = {
        "training_speakers": len(bank.speakers),
        "training_queries": len(bank.info["queries"]),
        "phonemes": bank.phones,
        "train_evaluation_speakers_disjoint": True,
        "encoders_frozen": True,
        "per_seed": {},
    }
    for seed in config["evaluation_seeds"]:
        current = directory / f"seed-{seed}"
        current.mkdir(exist_ok=True)
        encoder = load_encoder(config, seed)
        arrays = bank.materialize(
            encoder, rt.np.arange(len(bank.owners)), rt.np.arange(len(bank.speakers))
        )
        table = cv_score_table(arrays, bank.owners, corpora)
        audit_error = audit_table(table, arrays, bank)
        origin = source(config) / "prepared/schedules" / f"schedule-{seed}.npz"
        with rt.np.load(origin, allow_pickle=False) as archive:
            original = {k: archive[k].copy() for k in archive.files}
        schedule, diagnostics = make_schedule(
            original, table, bank.owners, corpora, config, seed
        )
        rt.np.testing.assert_array_equal(
            schedule["queries"], original["queries"][: len(schedule["queries"])]
        )
        rt.np.savez_compressed(current / "schedule.npz", **schedule)
        rt.np.savez_compressed(current / "mining-candidates.npz", **diagnostics)
        features = batch(
            bank, arrays, schedule["queries"][0], schedule["negative_claims"][0]
        )
        initial, counts, hashes = {}, {}, {}
        for arm in ARMS:
            model = FrozenModel(config, arm, seed)
            model.fusion.eval()
            with rt.torch.no_grad():
                initial[arm] = model.fusion(*features[:3])[0]
            counts[arm] = sum(p.numel() for p in model.fusion.parameters())
            hashes[arm] = rt.tree_hash(model.fusion.state_dict())
        if len(set(counts.values())) != 1 or len(set(hashes.values())) != 1:
            raise AssertionError("Paired initialization/parameter counts differ")
        delta = max(
            float((v - initial["weighted"]).abs().max()) for v in initial.values()
        )
        if delta > 1e-7:
            raise AssertionError("Initial scores differ")
        audit = {
            "encoder_sha256": rt.sha(encoder_path(config, seed)),
            "encoder_state_sha256": rt.tree_hash(encoder.state_dict()),
            "schedule_sha256": rt.sha(current / "schedule.npz"),
            "mined_cv_slots": len(diagnostics["positions"]),
            "changed_negatives": int(
                (
                    schedule["negative_claims"]
                    != original["negative_claims"][: len(schedule["queries"])]
                ).sum()
            ),
            "independent_mining_max_error": audit_error,
            "initial_score_max_difference": delta,
            "parameter_counts": counts,
            "initial_model_hashes": hashes,
        }
        rt.write(current / "audit.json", audit)
        report["per_seed"][str(seed)] = audit
        files.update(rt.pin([encoder_path(config, seed), origin, *current.iterdir()]))
        del arrays, encoder, table
    rt.write(directory / "audit.json", report)
    files.update(rt.pin([directory / "audit.json"]))
    rt.write(seal, {"config": config, "files": files}, immutable=True)
