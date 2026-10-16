"""Count-matched random controls and shared fusion eligibility, without EER feedback."""

from shared_quality import (
    BASE,
    ROOT,
    Bank,
    Pool,
    configuration,
    evaluation_prepared,
    pool_paths,
    prepared,
    rt,
    variant,
)

# isort: split
from ablation_data import make_schedule

filters = rt.module(
    "quality_filter_data", BASE.parent / "phoneme-training-quality/data.py"
)


def matched_random_keep(pool, screened_keep, seed):
    screened_keep = rt.np.asarray(screened_keep, dtype=bool)
    if screened_keep.shape != (len(pool.features),):
        raise ValueError("Screen mask and original pool differ")
    keep = rt.np.zeros(len(screened_keep), bool)
    for g, ids in enumerate(pool.groups):
        count = int(screened_keep[ids].sum())
        rng = rt.np.random.default_rng(
            rt.seed_for(
                seed,
                "random-quality-control",
                pool.speakers[pool.group_speaker[g]],
                pool.phones[pool.group_phone[g]],
            )
        )
        keep[rng.choice(ids, count, replace=False)] = True
        if keep[ids].sum() != count:
            raise AssertionError("Speaker/phone counts differ")
    return keep


def common_banks(banks):
    common_speakers = set.intersection(*(set(b["speakers"]) for b in banks.values()))
    lookups = {}
    for name, bank in banks.items():
        lookup = {
            q["query_id"]: q
            for q in bank["queries"]
            if q["speaker_id"] in common_speakers
        }
        if len(lookup) != sum(
            q["speaker_id"] in common_speakers for q in bank["queries"]
        ):
            raise ValueError("Duplicate query ID")
        lookups[name] = lookup
    query_ids = sorted(set.intersection(*(set(v) for v in lookups.values())))
    first = next(iter(banks.values()))
    owners = {lookups[next(iter(banks))][q]["speaker_id"] for q in query_ids}
    speakers = [s for s in first["speakers"] if s in common_speakers & owners]
    result = {}
    for name, bank in banks.items():
        for qid in query_ids:
            if (
                lookups[name][qid]["speaker_id"]
                != lookups[next(iter(banks))][qid]["speaker_id"]
            ):
                raise ValueError("Shared query has inconsistent owner")
        result[name] = {
            "speakers": speakers,
            "profiles": {s: bank["profiles"][s] for s in speakers},
            "queries": [lookups[name][qid] for qid in query_ids],
        }
    if not speakers or not query_ids:
        raise ValueError("No common fusion training inputs")
    return result


def load_training(config, arm, seed):
    source, directory = pool_paths(config, arm, seed)
    pool = Pool(source, directory)
    info = rt.read(variant(config, arm, seed) / "train-inputs.json")
    metadata = rt.np.load(directory / "train-metadata.npy", allow_pickle=False)
    return pool, Bank(info, pool.features, metadata, pool.phones)


def prepare(config):
    config = configuration(False)  # Full data design is shared by smoke and full runs.
    base = prepared(config)
    base.mkdir(parents=True, exist_ok=True)
    seal = base / "freeze.json"
    if seal.exists():
        saved = rt.read(seal)
        if saved["config"] != config:
            raise ValueError("Prepared configuration changed; use a new run")
        rt.check(saved["files"])
        return
    if len(config["random_filter_seeds"]) != len(config["evaluation_seeds"]):
        raise ValueError("One random filter draw is required per model seed")
    reference = rt.read(ROOT / config["reference_run"] / "design-freeze.json")
    rt.check(reference["files"])
    quality = ROOT / config["quality_run"]
    qseal = rt.read(quality / "complete.json")
    rt.check(qseal["inputs"])
    rt.check(qseal["outputs"])
    files = {**reference["files"], **qseal["inputs"], **qseal["outputs"]}
    original = evaluation_prepared(config)
    info = rt.read(original / "pool.json")
    pool = Pool(ROOT / config["source_run"], original)
    with rt.np.load(original / "pool.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    metadata = rt.np.load(original / "train-metadata.npy", allow_pickle=False)
    source_bank = rt.read(ROOT / rt.read(original / "inputs.json")["training_inputs"])
    with rt.np.load(quality / "decisions.npz", allow_pickle=False) as archive:
        keep_quality = archive["status"] == 0
    if len(keep_quality) != len(pool.features):
        raise ValueError("Quality population changed")
    banks = {
        "original": source_bank,
        "screened": rt.read(quality / "screened/train-inputs.json"),
    }
    variants = {"original": ("original", None), "screened": ("screened", None)}
    report = {"original": info, "screened": rt.read(quality / "screened/pool.json")}
    random_report = {}
    for seed, draw in zip(
        config["evaluation_seeds"], config["random_filter_seeds"], strict=True
    ):
        directory = variant(config, "random_matched", seed)
        directory.mkdir(parents=True, exist_ok=True)
        keep = matched_random_keep(pool, keep_quality, draw)
        counts, values, ids, mapping = filters.subset_pool(info, arrays, keep)
        # These invariants establish exact count matching, including empty/singleton groups.
        quality_counts = rt.np.array([keep_quality[g].sum() for g in pool.groups])
        random_counts = rt.np.array([keep[g].sum() for g in pool.groups])
        if not rt.np.array_equal(quality_counts, random_counts):
            raise ValueError("Random control is not count matched")
        if counts["speakers"] != report["screened"]["speakers"]:
            raise ValueError("Random control changed speaker identities")
        bank, dropped = filters.subset_bank(source_bank, mapping, info["phones"][:5])
        filters.validate_bank(bank, counts, values)
        rt.np.savez(directory / "pool.npz", **values)
        rt.np.save(directory / "source-segment-indices.npy", ids)
        rt.np.save(directory / "train-metadata.npy", metadata[ids])
        rt.write(directory / "pool.json", counts)
        features = rt.np.memmap(
            directory / "train-features.f32",
            mode="w+",
            dtype="<f4",
            shape=(len(ids), 128),
        )
        for start in range(0, len(ids), 65536):
            features[start : start + 65536] = pool.features[ids[start : start + 65536]]
        features.flush()
        del features
        loaded = rt.np.memmap(
            directory / "train-features.f32",
            mode="r",
            dtype="<f4",
            shape=(len(ids), 128),
        )
        for start in range(0, len(ids), 65536):
            if not rt.np.array_equal(
                loaded[start : start + 65536], pool.features[ids[start : start + 65536]]
            ):
                raise ValueError("Random feature mapping failed")
        name = f"random_matched/{seed}"
        banks[name] = bank
        variants[name] = ("random_matched", seed)
        random_report[str(seed)] = {
            "filter_seed": draw,
            "segments": len(ids),
            "removed": int((~keep).sum()),
            "overlap_with_quality_removed": int((~keep & ~keep_quality).sum()),
            "exact_speaker_phone_counts_matched": True,
            "native_fusion_speakers": len(bank["speakers"]),
            "native_fusion_queries": len(bank["queries"]),
            "bank_items_removed": dropped,
        }
        report["random_matched"] = counts
    common = common_banks(banks)
    common_audits = {}
    for name, bank in common.items():
        arm, seed = variants[name]
        directory = variant(config, arm, seed)
        directory.mkdir(parents=True, exist_ok=True)
        _source, pool_dir = pool_paths(config, arm, seed)
        current_info = rt.read(pool_dir / "pool.json")
        with rt.np.load(pool_dir / "pool.npz", allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        common_audits[name] = filters.validate_bank(bank, current_info, values)
        common_audits[name]["queries_removed_for_common_protocol"] = len(
            banks[name]["queries"]
        ) - len(bank["queries"])
        rt.write(directory / "train-inputs.json", bank)
    schedule_path = base / "schedules"
    schedule_path.mkdir(exist_ok=True)
    schedule_counts = {}
    for seed in config["evaluation_seeds"]:
        schedule, counts = make_schedule(
            common["original"],
            config["final_fusion_updates"],
            config["fusion_queries_per_update"],
            seed,
        )
        rt.np.savez_compressed(schedule_path / f"schedule-{seed}.npz", **schedule)
        schedule_counts[str(seed)] = counts
    report["comparison_design"] = {
        "random_controls": random_report,
        "common_banks": common_audits,
        "shared_speaker_and_query_order": True,
        "shared_fusion_schedule_counts": schedule_counts,
        "filtering_used_training_data_only": True,
        "historical_models_reused": False,
    }
    rt.write(base / "audit.json", report)
    files.update(
        rt.pin(
            [
                BASE / "config.json",
                quality / "complete.json",
                *base.rglob("*.json"),
                *base.rglob("*.npz"),
                *base.rglob("*.npy"),
                *base.rglob("*.f32"),
            ]
        )
    )
    rt.write(
        seal,
        {"config": config, "files": files, "evaluation_unchanged": True},
        immutable=True,
    )
