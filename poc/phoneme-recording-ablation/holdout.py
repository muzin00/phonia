"""A single preregistered, speaker-disjoint evaluation of relaxed vowel support."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import train as e

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-all-missing-vowel-evaluation"))
import shared as s


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


primitives = module(
    "holdout_primitives", BASE.parent / "phoneme-greedy-selection/prepare.py"
)
preparation = module(
    "holdout_preparation",
    BASE.parent / "phoneme-training-data-expansion/prepare_data.py",
)
bootstrap = module(
    "holdout_bootstrap",
    BASE.parent / "phoneme-all-missing-vowel-evaluation/auditing.py",
)


CONFIG = BASE / "holdout-config.json"


def digest(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def verify(config, run):
    frozen = s.read_json(run / "design-freeze.json")
    if frozen["config"] != config:
        raise ValueError("protocol changed after freeze")
    for path, checksum in frozen["files"].items():
        s.checked(ROOT / path, checksum)
    return frozen


def freeze(config, run):
    from pyarrow import parquet

    if run.exists():
        raise ValueError("unused run required")
    source = ROOT / config["cv_source"]
    selection = s.read_json(source / "selection.json")
    trained_clients = {r["client_id"] for r in selection}
    trained_speakers = {r["speaker_id"] for r in selection}
    if len(trained_clients) != 70:
        raise ValueError("training client list changed")
    prior = ROOT / config["prior_holdout"]
    prior_selection = s.read_json(prior / "selection.json")
    trained_clients.update(r["client_id"] for r in prior_selection)
    trained_speakers.update(r["speaker_id"] for r in prior_selection)
    assert len(trained_clients) == 130
    dev = ROOT / config["development_run"]
    screen = s.read_json(dev / "results.json")["candidate_screen"]
    assert [r["arm"] for r in screen if r["passed"]] == ["single5"]
    files = {}
    paths = [
        dev / "results.json",
        dev / "design-freeze.json",
        BASE / "holdout-design.md",
        prior / "selection.json",
    ]
    paths += [prior / split / "utterances.jsonl" for split in ("validation", "test")]
    paths += [
        dev / f"{arm}-{seed}" / "encoder.pt"
        for arm in config["arms"]
        for seed in config["seeds"]
    ]
    paths += [
        CONFIG,
        BASE / "README.md",
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        source / "selection.json",
        source / "source/manifest.json",
        source / "additional-utterances.jsonl",
    ]
    paths.extend(
        [
            Path(s.__file__),
            Path(bootstrap.__file__),
            Path(primitives.__file__),
            Path(preparation.__file__),
        ]
    )
    source_manifest = s.read_json(source / "source/manifest.json")
    by_client = defaultdict(list)
    for shard, checksum in source_manifest["files"].items():
        path = source / "source" / shard
        s.checked(path, checksum)
        paths.append(path)
        if not shard.endswith(".parquet"):
            continue
        for index, row in enumerate(
            parquet.read_table(
                path, columns=["client_id", "sentence", "locale"]
            ).to_pylist()
        ):
            if (
                row["client_id"] not in trained_clients
                and row["client_id"]
                and row["sentence"]
                and row["locale"] == "ja"
            ):
                by_client[row["client_id"]].append(
                    {**row, "shard": shard, "shard_row": index}
                )
    clients = sorted(
        [p for p, rows in by_client.items() if len(rows) >= config["minimum_clips"]],
        key=lambda p: digest(config["speaker_seed"], p),
    )
    count = config["calibration_speakers"] + config["test_speakers"]
    if len(clients) < count:
        raise ValueError("insufficient unused speakers")
    selected = []
    for number, client in enumerate(clients[:count]):
        split = "validation" if number < config["calibration_speakers"] else "test"
        speaker = "cv17_" + preparation.stable_hash("speaker_id", client)[:16]
        if speaker in trained_speakers:
            raise ValueError("training speaker overlap")
        clips = sorted(
            by_client[client],
            key=lambda r: digest(
                config["clip_seed"], f"{r['shard']}/{r['shard_row']}/{r['sentence']}"
            ),
        )
        for index, row in enumerate(
            clips[: config["enrollment_clips"] + config["query_clips"]]
        ):
            selected.append(
                {
                    **row,
                    "speaker_id": speaker,
                    "split": split,
                    "role": "enrollment"
                    if index < config["enrollment_clips"]
                    else "verification",
                }
            )
    encoder = (
        ROOT / config["encoder_run"] / "trials" / config["encoder_trial"] / "encoder.pt"
    )
    s.checked(encoder, config["encoder_sha256"])
    model, bundle = s.original.load_model(encoder.parent)
    del model
    source_design = s.read_json(ROOT / config["encoder_run"] / "design-freeze.json")
    extract_config = {
        **source_design["config"],
        "minimum_frames": 720,
        "maximum_frames": 6000,
        "minimum_rms_dbfs": -50,
    }
    paths.extend(
        [
            encoder,
            encoder.parent / "training-summary.json",
            ROOT / config["encoder_run"] / "design-freeze.json",
            ROOT / extract_config["feature_statistics"],
        ]
    )
    # Preserve the complete original source provenance; verification includes old audio hashes.
    files.update(source_design["files"])
    for path in paths:
        s.original.pin(files, path)
    for loaded in tuple(sys.modules.values()):
        filename = getattr(loaded, "__file__", None)
        if (
            filename
            and filename.endswith(".py")
            and Path(filename).resolve().is_relative_to(ROOT / "poc")
        ):
            s.original.pin(files, Path(filename))
    jvs_points = {}
    prior = s.read_json(
        ROOT / "poc/phoneme-all-missing-vowel-evaluation/config/protocol.json"
    )
    path = ROOT / prior["run_directory"] / "threshold-freeze.json"
    s.original.pin(files, path)
    jvs_points = s.read_json(path)["thresholds"]["equal"]
    # Reserve and hash-check all selected compressed audio without scoring it.
    needed = {(r["shard"], r["shard_row"]): r for r in selected}
    hashes = defaultdict(list)
    for shard in sorted({r["shard"] for r in selected}):
        for index, audio in enumerate(
            parquet.read_table(source / "source" / shard, columns=["audio"]).to_pylist()
        ):
            key = (shard, index)
            if key not in needed:
                continue
            row = needed[key]
            row["audio_path"] = audio["audio"]["path"]
            row["original_audio_sha256"] = hashlib.sha256(
                audio["audio"]["bytes"]
            ).hexdigest()
            hashes[row["original_audio_sha256"]].append(key)
    old_raw = {
        r["original_audio_sha256"]
        for r in s.rows(source / "additional-utterances.jsonl")
    }
    old_raw.update(r["original_audio_sha256"] for r in prior_selection)
    duplicates = {h for h, keys in hashes.items() if len(keys) > 1 or h in old_raw}
    for row in selected:
        row["excluded_duplicate"] = row["original_audio_sha256"] in duplicates
    run.mkdir(parents=True)
    s.write_json(run / "selection.json", selected)
    s.original.pin(files, run / "selection.json")
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "files": files,
            "phones": bundle["phonemes"],
            "extract_config": extract_config,
            "jvs_thresholds": jvs_points,
            "eligible_unused_clients": len(clients),
            "selected_speakers": count,
            "duplicate_excluded_clips": sum(r["excluded_duplicate"] for r in selected),
            "status": "selection_protocol_and_model_fixed_before_new_audio_processing_or_scoring",
        },
    )
    print(
        f"Frozen {count} new speakers, {len(selected)} clips, {len(duplicates)} duplicate hashes",
        flush=True,
    )


def prepare_split(config, run, split, design):
    import pyopenjtalk
    from pyarrow import parquet

    destination = run / split
    if destination.exists():
        raise ValueError("split already exists; use a fresh run")
    destination.mkdir()
    selected = [
        dict(r) for r in s.read_json(run / "selection.json") if r["split"] == split
    ]
    needed = {(r["shard"], r["shard_row"]): r for r in selected}
    for shard in sorted({r["shard"] for r in selected}):
        path = ROOT / config["cv_source"] / "source" / shard
        for i, audio in enumerate(
            parquet.read_table(path, columns=["audio"]).to_pylist()
        ):
            if (shard, i) in needed:
                row = needed[shard, i]
                row["audio_bytes"] = audio["audio"]["bytes"]
                if (
                    hashlib.sha256(row["audio_bytes"]).hexdigest()
                    != row["original_audio_sha256"]
                ):
                    raise ValueError("compressed audio changed")
    dictionary = pyopenjtalk.OPEN_JTALK_DICT_DIR
    if isinstance(dictionary, bytes):
        dictionary = dictionary.decode()
    version = importlib.metadata.version("pyopenjtalk")

    def process(row):
        value = preparation.prepare_clip(row, destination, version, dictionary)
        # The upstream helper is for training; override metadata in this new evaluation run.
        meta = value["utterance"]
        meta.update(split=split, evaluation_role=row["role"], learning_curve_cohorts=[])
        s.write_json(destination / "prepared" / f"{meta['utterance_id']}.json", value)
        return row, value

    prepared = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(process, row) for row in selected]
        for i, future in enumerate(as_completed(pending), 1):
            prepared.append(future.result())
            if i % 100 == 0 or i == len(pending):
                print(f"{split} alignment {i}/{len(pending)}", flush=True)
    prepared.sort(key=lambda pair: (pair[0]["shard"], pair[0]["shard_row"]))
    # Check decoded PCM as well as compressed bytes for old-source and split overlap.
    old_hashes = {
        r["source_sha256"]
        for which in ("train", "validation", "test")
        for r in s.rows(ROOT / config["encoder_run"] / f"{which}-segments.jsonl")
    }
    old_hashes.update(
        r["source_sha256"]
        for previous_split in ("validation", "test")
        for r in s.rows(
            ROOT / config["prior_holdout"] / previous_split / "utterances.jsonl"
        )
    )
    if split == "test":
        old_hashes.update(
            r["source_sha256"] for r in s.rows(run / "validation/utterances.jsonl")
        )
    pcm_counts = Counter(v["utterance"]["source_sha256"] for _, v in prepared)
    pipe = primitives.pipeline(design["extract_config"])
    segments, features, utterances, excluded = [], [], [], []
    groups_by_source = {}
    files, qc = {}, Counter()
    for row, value in prepared:
        meta = value["utterance"]
        meta["role"] = row["role"]
        meta["excluded_duplicate"] = (
            row["excluded_duplicate"]
            or pcm_counts[meta["source_sha256"]] > 1
            or meta["source_sha256"] in old_hashes
        )
        utterances.append(meta)
        groups = {p: [] for p in design["phones"]}
        groups_by_source[meta["source_file"]] = groups
        alignment = (
            destination
            / "alignments/raw"
            / meta["speaker_id"]
            / f"{meta['utterance_id']}.json"
        )
        if meta["excluded_duplicate"] or not alignment.exists():
            excluded.append(
                {
                    "utterance_id": meta["utterance_id"],
                    "reason": "duplicate_audio"
                    if meta["excluded_duplicate"]
                    else "alignment_failed",
                }
            )
            continue
        for segment, clip in primitives.extract(
            alignment, meta, design["extract_config"], files, qc
        ):
            if segment["quality_flags"] or segment["phoneme"] not in groups:
                continue
            segment["cache_index"] = len(segments)
            groups[segment["phoneme"]].append(len(segments))
            features.append(primitives.pool_features(pipe, clip, segment["segment_id"]))
            segments.append(segment)
    s.write_rows(destination / "utterances.jsonl", utterances)
    s.write_rows(destination / "segments.jsonl", segments)
    s.np.save(
        destination / "features.npy",
        s.np.asarray(features, dtype=s.np.float32),
        allow_pickle=False,
    )
    speakers = sorted({r["speaker_id"] for r in selected})
    profiles = {}
    for speaker in speakers:
        candidates = [
            r
            for r in segments
            if r["speaker_id"] == speaker and r["role"] == "enrollment"
        ]
        profiles[speaker] = {
            p: [
                r["cache_index"]
                for r in primitives.diverse(
                    [r for r in candidates if r["phoneme"] == p],
                    config["enrollment_limit"],
                    config["clip_seed"],
                )
            ]
            for p in design["phones"]
        }
    universal = [
        p for p in design["phones"] if all(profiles[speaker][p] for speaker in speakers)
    ]
    queries = [
        {
            "query_id": u["utterance_id"],
            "speaker_id": u["speaker_id"],
            "source_file": u["source_file"],
            "role": "verification",
            "groups": groups_by_source[u["source_file"]],
        }
        for u in utterances
        if u["role"] == "verification"
    ]
    data = {
        "split": split,
        "speakers": speakers,
        "profiles": profiles,
        "universally_registered": universal,
        "queries": queries,
    }
    s.write_json(destination / "inputs.json", data)
    s.write_json(
        destination / "preparation.json",
        {
            "files": files,
            "qc": dict(qc),
            "excluded": excluded,
            "utterances": len(utterances),
            "queries": len(queries),
            "intervals": len(segments),
            "universal_phones": universal,
            "missing_enrollment_vowels": {
                speaker: [p for p in s.VOWELS if not profiles[speaker][p]]
                for speaker in speakers
            },
            "recording_seconds": sum(u["duration_sec"] for u in utterances),
            "enrollment_seconds": sum(
                u["duration_sec"] for u in utterances if u["role"] == "enrollment"
            ),
        },
    )
    model, _ = s.original.load_model(
        ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
    )
    embeddings = s.original.model_embeddings(
        model, s.np.asarray(features, dtype=s.np.float32)
    )
    s.np.save(destination / "embeddings.npy", embeddings, allow_pickle=False)
    return data, embeddings


def candidates(data, vectors):
    use = data["universally_registered"]
    profiles = {}
    for speaker in data["speakers"]:
        for p in use:
            mean = vectors[data["profiles"][speaker][p]].mean(axis=0)
            profiles[speaker, p] = mean / s.np.linalg.norm(mean)
    result = []
    for query in data["queries"]:
        available = sorted(p for p in use if query["groups"][p])
        means = {p: vectors[query["groups"][p]].mean(axis=0) for p in available}
        for speaker in data["speakers"]:
            components = {p: float(means[p] @ profiles[speaker, p]) for p in available}
            result.append(
                {
                    "query_id": query["query_id"],
                    "speaker_id": query["speaker_id"],
                    "claimed_speaker_id": speaker,
                    "is_genuine": speaker == query["speaker_id"],
                    "role": "verification",
                    "split": data["split"],
                    "status": "scored" if available else "no_score",
                    "score": float(s.np.mean(list(components.values())))
                    if available
                    else None,
                    "phone_scores": components,
                    "used_phones": available,
                }
            )
    return result


def names(config):
    return [f"{arm}-{seed}" for arm in config["arms"] for seed in config["seeds"]]


def load_model(config, name):
    # The original factory is shared with the development experiment.
    state = s.torch.load(
        ROOT / config["development_run"] / name / "encoder.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = e.learner.create_encoder("statistics_mlp")
    model.load_state_dict(state["model"])
    return model.eval()


def evaluate(config, run, split):
    design = verify(config, run)
    if split == "test":
        frozen = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", frozen["design_sha256"])
        for path, digest in frozen["files"].items():
            s.checked(ROOT / path, digest)
    data, _ = prepare_split(config, run, split, design)
    features = s.np.load(run / split / "features.npy", allow_pickle=False)
    points = {}
    for name in names(config):
        model = load_model(config, name)
        vectors = s.original.model_embeddings(model, features)
        s.np.save(run / split / f"{name}-embeddings.npy", vectors, allow_pickle=False)
        values = candidates(data, vectors)
        s.write_rows(run / split / f"{name}-scores.jsonl", values)
        if split == "validation":
            points[name] = {
                p["name"]: s.original.thresholds(s.apply_policy(values, p))
                for p in config["policies"]
            }
        print(f"Completed fresh {split}/{name}", flush=True)
    verify(config, run)
    if split == "validation":
        files = {
            str(p.relative_to(ROOT)): s.sha256_file(p)
            for p in (run / split).glob("*")
            if p.is_file()
        }
        s.write_json(
            run / "threshold-freeze.json",
            {
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "thresholds": points,
                "files": files,
                "status": "all_six_models_and_thresholds_frozen_before_test_audio_processing",
            },
        )


def main():
    # Load shared scoring code before freeze so its source hashes are included.
    assert e.ROOT == ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "validation", "test"))
    args = parser.parse_args()
    config = s.read_json(CONFIG)
    run = ROOT / config["run_directory"]
    s.torch.set_num_threads(1)
    if args.stage == "freeze":
        freeze(config, run)
    else:
        evaluate(config, run, args.stage)


if __name__ == "__main__":
    main()
