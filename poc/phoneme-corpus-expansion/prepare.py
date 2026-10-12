"""Re-extract all 36 phones from the existing SRC4VC train alignment only."""

import hashlib
import json
import shutil
import wave
from collections import Counter

from shared import (
    BASE,
    CONFIG,
    ROOT,
    check_files,
    d,
    fixed,
    pin,
    primitives,
    read,
    s,
    sha,
)


def pcm_hash(path):
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            24000,
            1,
            2,
        )
        return hashlib.sha256(wav.readframes(wav.getnframes())).hexdigest()


def prepare():
    config = read(CONFIG)
    run = ROOT / config["run_directory"]
    run.mkdir(parents=True, exist_ok=True)
    if (run / "prepared-freeze.json").exists():
        check_files(read(run / "prepared-freeze.json")["files"])
        check_files(read(run / "source-freeze.json")["files"])
        return
    source, baseline = ROOT / config["source_run"], ROOT / config["baseline_data"]
    design = read(ROOT / config["baseline_run"] / "design-freeze.json")
    phones = design["phones"]
    extraction = read(ROOT / config["extraction_reference"])["configuration"]
    assert len(phones) == 36
    check_files(read(baseline / "design-freeze.json")["files"])
    previous = read(source / "preparation-report.json")
    for name, digest in previous["outputs_sha256"].items():
        assert sha(source / name) == digest
    split = read(source / "speaker-splits.json")["splits"]
    train_labels = {"ext_" + v.lower() for v in split["train"]}
    held_labels = {
        "ext_" + v.lower()
        for k, values in split.items()
        if k != "train"
        for v in values
    }
    inputs = list(s.rows(source / "additional-utterances.jsonl"))
    assert len(inputs) == 3500 and len(train_labels) == 70 and len(held_labels) == 30
    assert {v["speaker_id"] for v in inputs} == train_labels
    paths = [
        CONFIG,
        BASE / "prepare.py",
        BASE / "shared.py",
        ROOT / config["extraction_reference"],
        source / "additional-utterances.jsonl",
        source / "speaker-splits.json",
        source / "selection.json",
        source / "preparation-report.json",
        source / "source/manifest.json",
        ROOT / extraction["feature_statistics"],
        BASE.parent / "phoneme-greedy-selection/prepare.py",
        BASE.parent / "phoneme-greedy-selection/common.py",
    ]
    files = pin(paths)
    all_sources = {}
    baseline_manifest = baseline / "train-segments.jsonl"
    for row in s.rows(baseline_manifest):
        assert row["speaker_id"] not in held_labels | train_labels
        all_sources[row["source_file"]] = row["source_sha256"]
    evaluation_labels = set()
    for c in s.CORPORA:
        for split_name in ("validation", "test"):
            data, _, manifest = d.e.evaluation_source(
                config, design["reference_design"], c, split_name
            )
            evaluation_labels.update(data["speakers"])
            files.update(pin([manifest]))
            for row in s.rows(manifest):
                all_sources[row["source_file"]] = row["source_sha256"]
    assert not evaluation_labels & train_labels
    d.state(
        run / "preparation-state.json",
        "checking_audio_overlap",
        sources=len(all_sources),
    )
    known_pcm = set()
    for name, digest in all_sources.items():
        assert sha(ROOT / name) == digest, name
        known_pcm.add(pcm_hash(ROOT / name))
    duplicates, accepted_pcm = [], set()
    for row in inputs:
        assert (
            row["split"] == row["source_split"] == "train"
            and row["evaluation_role"] == "training"
        )
        path = ROOT / row["source_file"]
        assert sha(path) == row["source_sha256"]
        files.update(pin([path, source / "prepared" / (row["utterance_id"] + ".json")]))
        alignment = (
            source
            / "alignments/raw"
            / row["speaker_id"]
            / (row["utterance_id"] + ".json")
        )
        if alignment.exists():
            files.update(pin([alignment]))
        digest = pcm_hash(path)
        if digest in known_pcm or digest in accepted_pcm:
            duplicates.append(row["utterance_id"])
        accepted_pcm.add(digest)
    fixed(
        run / "source-freeze.json",
        {
            "config": config,
            "files": files,
            "phones": phones,
            "extraction": extraction,
            "duplicate_utterances": duplicates,
            "existing_wavs_checked": len(all_sources),
            "preserved_split": split,
            "selected_utterances": len(inputs),
            "review_omission_authorized": config["user_authorization"],
            "human_quality_measurement": None,
        },
    )
    pipe = s.pipeline(extraction)
    prepared = run / "extracted"
    prepared.mkdir(exist_ok=True)
    for index, meta in enumerate(inputs):
        seal = prepared / (meta["utterance_id"] + ".json")
        if seal.exists():
            check_files(read(seal)["files"])
            continue
        path = (
            source
            / "alignments/raw"
            / meta["speaker_id"]
            / (meta["utterance_id"] + ".json")
        )
        qc, rows, values = Counter(), [], []
        status = "processed"
        raw = []
        if meta["utterance_id"] in duplicates:
            status = "duplicate_pcm"
        elif not path.exists():
            prior = read(source / "prepared" / (meta["utterance_id"] + ".json"))
            assert prior["failure"] is not None
            status = "previous_alignment_failure"
        else:
            for row, clip in primitives.extract(path, meta, extraction, {}, qc):
                reasons = row["quality_flags"] + (
                    [] if row["phoneme"] in phones else ["outside_inventory"]
                )
                raw.append({**row, "exclusion_reasons": reasons})
                if not reasons:
                    rows.append({**row, "cache_index": len(rows), "corpus": "src4vc"})
                    values.append(s.pool_features(pipe, clip, row["segment_id"]))
        raw_path = prepared / (meta["utterance_id"] + "-raw.json")
        rows_path = prepared / (meta["utterance_id"] + "-segments.json")
        features_path = prepared / (meta["utterance_id"] + ".npy")
        fixed(raw_path, raw)
        fixed(rows_path, rows)
        s.np.save(
            features_path, s.np.asarray(values, dtype=s.np.float32).reshape(-1, 128)
        )
        fixed(
            seal,
            {
                "status": status,
                "files": pin([raw_path, rows_path, features_path]),
                "qc": dict(qc),
                "source_sha256": meta["source_sha256"],
            },
        )
        if (index + 1) % 100 == 0 or index + 1 == len(inputs):
            d.state(
                run / "preparation-state.json",
                "extracting",
                processed=index + 1,
                total=len(inputs),
            )
            print({"processed": index + 1, "total": len(inputs)}, flush=True)
    # Preserve every baseline row and feature byte as an exact prefix.
    manifest, cache = run / "train-segments.jsonl", run / "train-features.f32"
    shutil.copyfile(baseline_manifest, manifest)
    shutil.copyfile(baseline / "train-features.f32", cache)
    prior_report = read(baseline / "preparation-report.json")
    count = prior_report["train_feature_count"]
    support, status, excluded = Counter(), Counter(), Counter()
    with manifest.open("a") as stream, cache.open("ab") as output:
        for meta in inputs:
            uid = meta["utterance_id"]
            receipt = read(prepared / (uid + ".json"))
            check_files(receipt["files"])
            status[receipt["status"]] += 1
            for row in read(prepared / (uid + "-raw.json")):
                excluded.update(row["exclusion_reasons"])
            rows = read(prepared / (uid + "-segments.json"))
            matrix = s.np.load(prepared / (uid + ".npy"), allow_pickle=False)
            assert matrix.shape == (len(rows), 128) and s.np.isfinite(matrix).all()
            for row in rows:
                stream.write(
                    json.dumps({**row, "cache_index": count}, ensure_ascii=False) + "\n"
                )
                count += 1
                support[row["speaker_id"], row["phoneme"]] += 1
            output.write(matrix.astype("<f4").tobytes())
    labels = {v for v, _ in support}
    assert labels == train_labels and all(
        support[v, p] >= 2 for v in labels for p in s.VOWELS
    )
    phone_counts = Counter(prior_report["phone_counts"])
    for (_, phone), n in support.items():
        phone_counts[phone] += n
    fixed(
        run / "preparation-report.json",
        {
            "training_speakers": sorted(
                prior_report["training_speakers"] + sorted(labels)
            ),
            "train_feature_count": count,
            "baseline_features": prior_report["train_feature_count"],
            "added_features": count - prior_report["train_feature_count"],
            "phone_counts": dict(phone_counts),
            "utterance_status": dict(status),
            "exclusions": dict(excluded),
            "added_phone_support": {
                v: {p: support[v, p] for p in phones} for v in sorted(labels)
            },
            "baseline_prefix_bytes": {
                "manifest": baseline_manifest.stat().st_size,
                "features": (baseline / "train-features.f32").stat().st_size,
            },
            "source_sha256": sha(run / "source-freeze.json"),
        },
    )
    paths = [
        manifest,
        cache,
        run / "preparation-report.json",
        run / "source-freeze.json",
        *prepared.glob("*"),
    ]
    fixed(
        run / "prepared-freeze.json",
        {"files": pin(paths), "runtime": s.expanded.runtime()},
    )
    d.state(
        run / "preparation-state.json",
        "complete",
        added_features=count - prior_report["train_feature_count"],
        training_labels=490,
    )


if __name__ == "__main__":
    s.torch.set_num_threads(1)
    prepare()
