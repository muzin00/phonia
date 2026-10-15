"""Audit source bytes and build compact indices; never regenerate frozen audio."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from runtime import ROOT, check, corpus, np, pin, read, sha, write


def manifest_rows(path):
    with path.open() as stream:
        yield from map(json.loads, stream)


def prepare(config, run):
    prepared = run / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    seal = prepared / "freeze.json"
    if seal.exists():
        frozen = read(seal)
        if (
            frozen["source_run"] != config["source_run"]
            or frozen["partition_source"] != config["partition_source"]
        ):
            raise ValueError("Prepared source changed")
        check(frozen["files"])
        return
    source = ROOT / config["source_run"]
    design = read(source / "design-freeze.json")
    old = design["reference_design"]["config"]
    family = source / f"acoustic-{config['source_seed']}/fusion"
    report = read(source / "preparation-report.json")
    phones, speakers = design["phones"], report["training_speakers"]
    phone_index, speaker_index = (
        {p: i for i, p in enumerate(phones)},
        {s: i for i, s in enumerate(speakers)},
    )
    paths = [
        source / "design-freeze.json",
        source / "prepared-freeze.json",
        source / "preparation-report.json",
        source / "train-segments.jsonl",
        source / "train-features.f32",
        ROOT / config["partition_source"],
        family / "train-inputs.json",
        source / f"acoustic-{config['source_seed']}/continued/schedule-20261015.npz",
    ]
    groups, metadata, segment_phone, segment_speaker, recording = (
        defaultdict(list),
        [],
        [],
        [],
        [],
    )
    recordings, train_hashes = {}, set()
    for i, row in enumerate(manifest_rows(source / "train-segments.jsonl")):
        if row["cache_index"] != i or row["split"] != "train" or row["quality_flags"]:
            raise ValueError("Training cache identity/QC changed")
        s, p = speaker_index[row["speaker_id"]], phone_index[row["phoneme"]]
        groups[s, p].append(i)
        metadata.append(
            [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
        )
        segment_phone.append(p)
        segment_speaker.append(s)
        recording.append(recordings.setdefault(row["source_sha256"], len(recordings)))
        train_hashes.add(row["source_sha256"])
    if len(metadata) != report["train_feature_count"]:
        raise ValueError("Training feature count changed")
    metadata = np.asarray(metadata, np.float32)
    np.save(prepared / "train-metadata.npy", metadata)
    keys = sorted(groups)
    np.savez(
        prepared / "pool.npz",
        indices=np.concatenate([groups[k] for k in keys]),
        offsets=np.r_[0, np.cumsum([len(groups[k]) for k in keys])],
        group_speaker=np.array([k[0] for k in keys]),
        group_phone=np.array([k[1] for k in keys]),
        segment_speaker=np.asarray(segment_speaker, np.int16),
        segment_phone=np.asarray(segment_phone, np.int8),
        recording=np.asarray(recording, np.int32),
        metadata=np.stack((np.log(metadata[:, 0]), metadata[:, 1] / 50), 1),
    )
    counts = {
        "phones": phones,
        "speakers": speakers,
        "segments": len(metadata),
        "pair_eligible_segments": sum(len(v) for v in groups.values() if len(v) >= 2),
        "singleton_segments": sum(len(v) == 1 for v in groups.values()),
        "speaker_phone_groups": len(groups),
        "corpus_labels": dict(Counter(corpus(s) for s in speakers)),
    }
    write(prepared / "pool.json", counts)
    partitions = read(ROOT / config["partition_source"])
    write(prepared / "partitions.json", partitions)
    inputs = {
        "phones": phones,
        "training_inputs": str((family / "train-inputs.json").relative_to(ROOT)),
    }
    split_speakers, split_hashes = {"train": set(speakers)}, {"train": train_hashes}
    for c in config["corpora"]:
        for split in ("validation", "test"):
            info_path = family / f"{c}-{split}-inputs.json"
            trials_path = family / f"{c}-{split}-baseline.json"
            data = read(info_path)
            if c == "common_voice":
                directory = ROOT / old["cv_evaluation_run"] / split
                features, manifest = (
                    directory / "features.npy",
                    directory / "segments.jsonl",
                )
                count = len(np.load(features, mmap_mode="r", allow_pickle=False))
            else:
                directory = ROOT / old["source_run" if c == "jvs" else "corpus_run"]
                features, manifest = (
                    directory / f"{split}-features.f32",
                    directory / f"{split}-segments.jsonl",
                )
                count = features.stat().st_size // (128 * 4)
            meta, hashes, source_speakers = [], set(), set()
            for i, row in enumerate(manifest_rows(manifest)):
                if row["cache_index"] != i:
                    raise ValueError("Evaluation cache indexing changed")
                meta.append(
                    [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
                )
                hashes.add(row["source_sha256"])
                source_speakers.add(row["speaker_id"])
            if len(meta) != count or not set(data["speakers"]) <= source_speakers:
                raise ValueError("Evaluation source mapping changed")
            if set(speakers) & source_speakers or train_hashes & hashes:
                raise ValueError("Training/evaluation speaker or recording overlap")
            split_speakers.setdefault(split, set()).update(source_speakers)
            split_hashes.setdefault(split, set()).update(hashes)
            if split == "validation":
                reward, selection = map(
                    set, (partitions[c]["reward"], partitions[c]["selection"])
                )
                if reward & selection or reward | selection != set(data["speakers"]):
                    raise ValueError("Reward/selection partition changed")
            np.save(
                prepared / f"{c}-{split}-metadata.npy", np.asarray(meta, np.float32)
            )
            inputs[f"{c}-{split}"] = {
                "features": str(features.relative_to(ROOT)),
                "count": count,
                "inputs": str(info_path.relative_to(ROOT)),
                "trials": str(trials_path.relative_to(ROOT)),
            }
            paths.extend((features, manifest, info_path, trials_path))
    if (
        split_speakers["validation"] & split_speakers["test"]
        or split_hashes["validation"] & split_hashes["test"]
    ):
        raise ValueError("Validation/test speaker or recording overlap")
    # Check reused bytes against their existing recorded provenance before pinning anew.
    known = {**design["files"], **read(source / "prepared-freeze.json")["files"]}
    for path in paths:
        name = str(path.relative_to(ROOT))
        if name in known and sha(path) != known[name]:
            raise ValueError(f"Source integrity failure: {name}")
    write(prepared / "inputs.json", inputs)
    paths.extend(p for p in prepared.iterdir() if p.is_file() and p != seal)
    write(
        seal,
        {
            "source_run": config["source_run"],
            "partition_source": config["partition_source"],
            "files": pin(paths),
            "counts": counts,
            "speaker_and_recording_disjoint": True,
        },
        immutable=True,
    )


def load_training_bank(config, prepared, pool):
    from verification import Bank

    inputs = read(prepared / "inputs.json")
    return Bank(
        read(ROOT / inputs["training_inputs"]),
        pool.features,
        np.load(prepared / "train-metadata.npy", allow_pickle=False),
        pool.phones,
    )
