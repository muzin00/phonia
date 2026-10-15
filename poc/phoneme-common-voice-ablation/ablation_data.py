"""Remove CV from every training view and freeze new, corpus-neutral schedules."""

from collections import Counter

from shared_ablation import (
    ARMS,
    ROOT,
    Bank,
    Pool,
    configuration,
    evaluation_prepared,
    prepared,
    rt,
)


def subset_pool(pool, keep_speakers):
    keep_speakers = set(keep_speakers)
    speakers = [s for s in pool.speakers if s in keep_speakers]
    speaker_map = rt.np.array(
        [speakers.index(s) if s in keep_speakers else -1 for s in pool.speakers]
    )
    source_ids = rt.np.flatnonzero(speaker_map[pool.segment_speaker] >= 0)
    index_map = rt.np.full(len(pool.features), -1, dtype=rt.np.int64)
    index_map[source_ids] = rt.np.arange(len(source_ids))
    groups = [i for i, s in enumerate(pool.group_speaker) if speaker_map[s] >= 0]
    indices = [index_map[pool.groups[g]] for g in groups]
    arrays = {
        "indices": rt.np.concatenate(indices),
        "offsets": rt.np.r_[0, rt.np.cumsum([len(v) for v in indices])],
        "group_speaker": speaker_map[pool.group_speaker[groups]],
        "group_phone": pool.group_phone[groups],
        "segment_speaker": speaker_map[pool.segment_speaker[source_ids]].astype(
            rt.np.int16
        ),
        "segment_phone": pool.segment_phone[source_ids],
        "recording": pool.recording[source_ids],
        "metadata": pool.metadata[source_ids],
    }
    info = {
        "phones": pool.phones,
        "speakers": speakers,
        "segments": len(source_ids),
        "pair_eligible_segments": sum(len(v) for v in indices if len(v) >= 2),
        "singleton_segments": sum(len(v) == 1 for v in indices),
        "speaker_phone_groups": len(indices),
        "corpus_labels": dict(Counter(rt.corpus(s) for s in speakers)),
    }
    return info, arrays, source_ids, index_map


def subset_bank(info, keep_speakers, index_map):
    keep = set(keep_speakers)

    def remap(groups):
        result = {}
        for phone, ids in groups.items():
            mapped = index_map[rt.np.asarray(ids, dtype=rt.np.int64)]
            if (mapped < 0).any():
                raise ValueError("Retained training item refers to an excluded segment")
            result[phone] = mapped.tolist()
        return result

    return {
        "speakers": [s for s in info["speakers"] if s in keep],
        "profiles": {
            s: remap(info["profiles"][s]) for s in info["speakers"] if s in keep
        },
        "queries": [
            dict(q, groups=remap(q["groups"]))
            for q in info["queries"]
            if q["speaker_id"] in keep
        ],
    }


def audit_bank(info, pool, exclude_cv=False):
    speaker_index = {s: i for i, s in enumerate(pool.speakers)}
    phone_index = {p: i for i, p in enumerate(pool.phones)}
    if exclude_cv and any(rt.corpus(s) == "common_voice" for s in pool.speakers):
        raise ValueError("CV remains in acoustic training")
    if not set(info["speakers"]) <= set(pool.speakers):
        raise ValueError("Fusion speaker missing from acoustic pool")
    items = [(s, info["profiles"][s]) for s in info["speakers"]]
    items.extend((q["speaker_id"], q["groups"]) for q in info["queries"])
    for speaker, groups in items:
        if speaker not in info["speakers"]:
            raise ValueError("Query owner is not enrolled")
        if any(not groups.get(p) for p in pool.phones[:5]):
            raise ValueError("Training five-vowel gate changed")
        for phone, ids in groups.items():
            ids = rt.np.asarray(ids, dtype=rt.np.int64)
            if ((ids < 0) | (ids >= len(pool.features))).any():
                raise ValueError("Training segment index out of bounds")
            if not (pool.segment_speaker[ids] == speaker_index[speaker]).all():
                raise ValueError("Training speaker remapping mismatch")
            if not (pool.segment_phone[ids] == phone_index[phone]).all():
                raise ValueError("Training phoneme remapping mismatch")
    return {
        "fusion_speakers": len(info["speakers"]),
        "fusion_queries": len(info["queries"]),
        "fusion_labels_by_corpus": dict(
            Counter(rt.corpus(s) for s in info["speakers"])
        ),
        "fusion_queries_by_corpus": dict(
            Counter(rt.corpus(q["speaker_id"]) for q in info["queries"])
        ),
        "all_segment_owners_and_phones_verified": True,
    }


def make_schedule(info, updates, width, seed):
    speakers = info["speakers"]
    lookup = {s: i for i, s in enumerate(speakers)}
    owners = rt.np.array([lookup[q["speaker_id"]] for q in info["queries"]])
    corpora = rt.np.array([rt.corpus(s) for s in speakers])
    # Three independent uniform draws per slot; same random stream across arms.
    rng = rt.np.random.default_rng(rt.seed_for(seed, "ablation-fusion-schedule"))
    draws = rng.random((updates, width, 3))
    selected = (draws[:, :, 0] * len(speakers)).astype(rt.np.int64)
    queries = rt.np.empty((updates, width), rt.np.int32)
    negatives = rt.np.empty((updates, width), rt.np.int16)
    for s in range(len(speakers)):
        available = rt.np.flatnonzero(owners == s)
        impostors = rt.np.flatnonzero(
            (corpora == corpora[s]) & (rt.np.arange(len(speakers)) != s)
        )
        if not len(available) or not len(impostors):
            raise ValueError(
                "Speaker lacks training queries or within-corpus impostors"
            )
        rows = selected == s
        queries[rows] = available[(draws[:, :, 1][rows] * len(available)).astype(int)]
        negatives[rows] = impostors[(draws[:, :, 2][rows] * len(impostors)).astype(int)]
    if not (owners[queries] == selected).all() or (selected == negatives).any():
        raise AssertionError("Schedule label mismatch")
    if not (corpora[selected] == corpora[negatives]).all():
        raise AssertionError("Cross-corpus negative")
    counts = dict(Counter(corpora[selected].reshape(-1)))
    return {"queries": queries, "negative_claims": negatives}, counts


def paths_for(config, arm):
    base = prepared(config)
    if arm == "with_cv":
        return ROOT / config["source_run"], evaluation_prepared(config)
    return base / arm, base / arm


def load_training(config, arm):
    source, pool_path = paths_for(config, arm)
    pool = Pool(source, pool_path)
    info = rt.read(prepared(config) / arm / "train-inputs.json")
    metadata = rt.np.load(pool_path / "train-metadata.npy", allow_pickle=False)
    return pool, Bank(info, pool.features, metadata, pool.phones)


def prepare(config):
    base = prepared(config)
    base.mkdir(parents=True, exist_ok=True)
    seal = base / "freeze.json"
    source_config = configuration(False)
    if seal.exists():
        saved = rt.read(seal)
        if saved["preparation_config"] != source_config:
            raise ValueError("Preparation configuration changed; use a new run")
        rt.check(saved["files"])
        return
    original = evaluation_prepared(config)
    reference = rt.read(ROOT / config["reference_run"] / "design-freeze.json")
    rt.check(reference["files"])
    pool = Pool(ROOT / config["source_run"], original)
    inputs = rt.read(original / "inputs.json")
    train = rt.read(ROOT / inputs["training_inputs"])
    files, report = dict(reference["files"]), {}
    for arm in ARMS:
        directory = base / arm
        directory.mkdir(exist_ok=True)
        if arm == "with_cv":
            info = rt.read(original / "pool.json")
            bank_info = subset_bank(
                train, pool.speakers, rt.np.arange(len(pool.features))
            )
            current = pool
        else:
            keep = [s for s in pool.speakers if rt.corpus(s) != "common_voice"]
            info, arrays, source_ids, mapping = subset_pool(pool, keep)
            bank_info = subset_bank(train, keep, mapping)
            rt.np.savez(directory / "pool.npz", **arrays)
            rt.np.save(directory / "source-segment-indices.npy", source_ids)
            rt.write(directory / "pool.json", info)
            metadata = rt.np.load(original / "train-metadata.npy", allow_pickle=False)
            rt.np.save(directory / "train-metadata.npy", metadata[source_ids])
            output = rt.np.memmap(
                directory / "train-features.f32",
                mode="w+",
                dtype="<f4",
                shape=(len(source_ids), 128),
            )
            for start in range(0, len(source_ids), 65536):
                output[start : start + 65536] = pool.features[
                    source_ids[start : start + 65536]
                ]
            output.flush()
            del output
            current = Pool(directory, directory)
        report[arm] = {**info, **audit_bank(bank_info, current, arm == "without_cv")}
        # These are non-learned raw features; no pretrained weights are loaded.
        rt.write(directory / "train-inputs.json", bank_info)
        report[arm]["fusion_scheduled_queries_by_corpus"] = {}
        for seed in source_config["evaluation_seeds"]:
            schedule, counts = make_schedule(
                bank_info,
                source_config["final_fusion_updates"],
                config["fusion_queries_per_update"],
                seed,
            )
            rt.np.savez_compressed(directory / f"schedule-{seed}.npz", **schedule)
            report[arm]["fusion_scheduled_queries_by_corpus"][str(seed)] = counts
        files.update(rt.pin(p for p in directory.iterdir() if p.is_file()))
    if (
        report["with_cv"]["segments"] != 1930593
        or report["without_cv"]["segments"] != 1264413
    ):
        raise ValueError("Expected source population changed")
    rt.write(base / "audit.json", report)
    files.update(rt.pin([base / "audit.json", original / "freeze.json"]))
    rt.write(
        seal,
        {
            "preparation_config": source_config,
            "files": files,
            "evaluation_partitions_unchanged": True,
            "fresh_models_only": True,
        },
        immutable=True,
    )
