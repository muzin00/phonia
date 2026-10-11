"""Disjoint enrollment/query audio and fixed-encoder per-phone aggregates."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-greedy-selection"))
from audit import audit_trial, independent_eer, speaker_draws, weighted_eers
from common import (
    VOWELS,
    checked,
    expanded,
    model_embeddings,
    np,
    pin,
    rank,
    read_json,
    rows,
    sha256_file,
    torch,
    write_json,
    write_rows,
)
from evaluate import metrics, scores_for, thresholds
from learner import load_model
from prepare import diverse

__all__ = [
    "audit_trial",
    "independent_eer",
    "metrics",
    "speaker_draws",
    "thresholds",
    "weighted_eers",
    "write_rows",
]

CONFIG = BASE / "config/protocol.json"
ROLES = ("verification", "cross_text_verification")


def aggregate(indices, vectors, interval_metadata, normalize=False):
    if not indices:
        return np.zeros(128, np.float32), np.zeros(4, np.float32), False
    mean = vectors[indices].mean(axis=0)
    norm = np.linalg.norm(mean)
    metadata = np.array(
        [
            np.log1p(len(indices)),
            np.log(interval_metadata[indices, 0].mean()),
            1 - norm,
            interval_metadata[indices, 1].mean() / 50,
        ],
        dtype=np.float32,
    )
    return mean / norm if normalize else mean, metadata, True


def aggregates(data, vectors, interval_metadata, phones):
    speakers = data["speakers"]
    enrollment = np.zeros((len(speakers), len(phones), 128), np.float32)
    emeta = np.zeros((len(speakers), len(phones), 4), np.float32)
    emask = np.zeros((len(speakers), len(phones)), bool)
    for s, speaker in enumerate(speakers):
        for p, phone in enumerate(phones):
            enrollment[s, p], emeta[s, p], emask[s, p] = aggregate(
                data["profiles"][speaker].get(phone, []),
                vectors,
                interval_metadata,
                True,
            )
    queries = np.zeros((len(data["queries"]), len(phones), 128), np.float32)
    qmeta = np.zeros((len(data["queries"]), len(phones), 4), np.float32)
    qmask = np.zeros((len(data["queries"]), len(phones)), bool)
    for q, query in enumerate(data["queries"]):
        for p, phone in enumerate(phones):
            queries[q, p], qmeta[q, p], qmask[q, p] = aggregate(
                query["groups"].get(phone, []), vectors, interval_metadata
            )
    return {
        "enrollment": enrollment,
        "emeta": emeta,
        "emask": emask,
        "queries": queries,
        "qmeta": qmeta,
        "qmask": qmask,
    }


def source_catalog(path):
    catalog = defaultdict(dict)
    for i, row in enumerate(rows(path)):
        if row["cache_index"] != i or row["split"] != "train" or row["quality_flags"]:
            raise ValueError("training identity or QC changed")
        item = {"sha256": row["source_sha256"], "corpus": row["corpus"]}
        previous = catalog[row["speaker_id"]].setdefault(row["source_file"], item)
        if previous != item:
            raise ValueError("inconsistent source identity")
    return dict(catalog)


def training_inputs(config, path, phones, evaluation_data):
    catalog = source_catalog(path)
    speakers = sorted(catalog)
    if len(speakers) != 140:
        raise ValueError("expected 140 original training speakers")
    eval_speakers = {s for d in evaluation_data for s in d["speakers"]}
    if set(speakers) & eval_speakers:
        raise ValueError("training/evaluation speaker overlap")
    enrollment_sources, query_sources = set(), set()
    enrollment_hashes, query_hashes = set(), set()
    seen_hashes = {}
    for speaker in speakers:
        sources = catalog[speaker]
        if len({v["corpus"] for v in sources.values()}) != 1:
            raise ValueError("speaker corpus changed")
        unique = {}
        for name in sorted(
            sources,
            key=lambda n: ("/parallel100/" not in n, rank(config["data_seed"], n)),
        ):
            checksum = sources[name]["sha256"]
            owner = seen_hashes.setdefault(checksum, speaker)
            if owner != speaker:
                raise ValueError("identical training WAV belongs to different speakers")
            unique.setdefault(checksum, name)
        names = list(unique.values())
        if len(names) <= config["training_enrollment_wavs"]:
            raise ValueError("not enough distinct training source WAVs")
        selected = set(names[: config["training_enrollment_wavs"]])
        selected_hashes = {sources[n]["sha256"] for n in selected}
        # Duplicate paths with the same audio belong to a single side only.
        for name, item in sources.items():
            if item["sha256"] in selected_hashes:
                enrollment_sources.add(name)
                enrollment_hashes.add(item["sha256"])
            else:
                query_sources.add(name)
                query_hashes.add(item["sha256"])
    if enrollment_sources & query_sources or enrollment_hashes & query_hashes:
        raise ValueError("training enrollment/query leakage")
    pools, queries = defaultdict(list), {}
    metadata = []
    for row in rows(path):
        metadata.append(
            [(row["end_frame"] - row["start_frame"]) / 24000, row["rms_dbfs"]]
        )
        if row["source_file"] in enrollment_sources:
            pools[row["speaker_id"], row["phoneme"]].append(
                {k: row[k] for k in ("segment_id", "source_file", "cache_index")}
            )
        else:
            query = queries.setdefault(
                row["source_file"],
                {
                    "query_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "source_file": row["source_file"],
                    "source_sha256": row["source_sha256"],
                    "corpus": row["corpus"],
                    "groups": {p: [] for p in phones},
                },
            )
            query["groups"][row["phoneme"]].append(row["cache_index"])
    profiles = {
        s: {
            p: [
                r["cache_index"]
                for r in diverse(
                    pools[s, p], config["enrollment_limit"], config["data_seed"]
                )
            ]
            for p in phones
        }
        for s in speakers
    }
    eligible = [
        q for _, q in sorted(queries.items()) if all(q["groups"][p] for p in VOWELS)
    ]
    if {q["speaker_id"] for q in eligible} != set(speakers):
        raise ValueError("training speaker has no scorable query")
    return {
        "split": "train",
        "speakers": speakers,
        "profiles": profiles,
        "queries": eligible,
        "catalog": catalog,
        "enrollment_sources": sorted(enrollment_sources),
        "query_sources": sorted(query_sources),
        "excluded_missing_vowels": len(queries) - len(eligible),
        "feature_count": len(metadata),
    }, np.asarray(metadata, dtype=np.float32)


def make_schedule(config, data, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    speakers = data["speakers"]
    by_speaker = [
        [i for i, q in enumerate(data["queries"]) if q["speaker_id"] == s]
        for s in speakers
    ]
    corpora = [next(iter(data["catalog"][s].values()))["corpus"] for s in speakers]
    negatives = [
        np.array([j for j, c in enumerate(corpora) if c == corpora[i] and j != i])
        for i in range(len(speakers))
    ]
    shape = (config["updates"], config["queries_per_update"])
    selected = rng.integers(len(speakers), size=shape)
    qids = np.empty(shape, dtype=np.int32)
    claims = np.empty(shape, dtype=np.int16)
    for update in range(shape[0]):
        for column, speaker in enumerate(selected[update]):
            qids[update, column] = rng.choice(by_speaker[speaker])
            claims[update, column] = rng.choice(negatives[speaker])
    return {"queries": qids, "negative_claims": claims}


def prepare(config, run):
    if run.exists():
        raise ValueError("prepare requires unused directory")
    source = ROOT / config["source_run"]
    evaluation = ROOT / config["evaluation_run"]
    frozen = read_json(source / "design-freeze.json")
    if frozen["runtime"] != expanded.runtime():
        raise ValueError("original numerical runtime changed")
    print("Verifying original audio, alignment and feature hashes", flush=True)
    for name, checksum in frozen["files"].items():
        checked(ROOT / name, checksum)
    for name, checksum in read_json(evaluation / "completion-verification.json")[
        "output_sha256"
    ].items():
        checked(ROOT / name, checksum)
    model, bundle = load_model(source / "trials" / config["source_trial"])
    checked(
        source / "trials" / config["source_trial"] / "encoder.pt",
        config["encoder_sha256"],
    )
    phones = bundle["phonemes"]
    if len(phones) != 36 or config["encoder_trainable"]:
        raise ValueError("expected frozen all36 encoder")
    evaluation_data = [
        read_json(evaluation / "enrollment-30" / f"{s}-inputs.json")
        for s in ("validation", "test")
    ]
    train, metadata = training_inputs(
        config, source / "train-segments.jsonl", phones, evaluation_data
    )
    eval_hashes = {
        r["source_sha256"]
        for s in ("validation", "test")
        for r in rows(source / f"{s}-segments.jsonl")
    }
    train_hashes = {
        v["sha256"] for src in train["catalog"].values() for v in src.values()
    }
    if eval_hashes & train_hashes:
        raise ValueError("training/evaluation audio hash overlap")
    run.mkdir(parents=True)
    files = {}
    for path in (
        CONFIG,
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
        source / "design-freeze.json",
        evaluation / "design-freeze.json",
        evaluation / "completion-verification.json",
        source / "trials" / config["source_trial"] / "encoder.pt",
    ):
        pin(files, path)
    write_json(run / "train-inputs.json", train)
    raw = np.memmap(
        source / "train-features.f32",
        mode="r",
        dtype="<f4",
        shape=(train["feature_count"], 128),
    )
    print(
        f"Embedding {len(raw)} training intervals with the frozen encoder", flush=True
    )
    vectors = model_embeddings(model, raw)
    np.savez(
        run / "train-aggregates.npz", **aggregates(train, vectors, metadata, phones)
    )
    for seed in config["seeds"]:
        np.savez(run / f"schedule-{seed}.npz", **make_schedule(config, train, seed))
    for split, data in zip(("validation", "test"), evaluation_data):
        segments = list(rows(source / f"{split}-segments.jsonl"))
        metadata = np.array(
            [
                [(r["end_frame"] - r["start_frame"]) / 24000, r["rms_dbfs"]]
                for r in segments
            ],
            np.float32,
        )
        vectors = np.load(
            evaluation / "enrollment-30" / f"{split}-embeddings.npy", allow_pickle=False
        )
        write_json(run / f"{split}-inputs.json", data)
        np.savez(
            run / f"{split}-aggregates.npz",
            **aggregates(data, vectors, metadata, phones),
        )
        if scores_for(data, vectors, phones) != list(
            rows(evaluation / "enrollment-30" / f"{split}-scores.jsonl")
        ):
            raise ValueError(
                "equal-weight enrollment30 baseline not exactly reproduced"
            )
        for path in (evaluation / "enrollment-30").glob(f"{split}-*"):
            pin(files, path)
        pin(files, source / f"{split}-segments.jsonl")
    for path in run.glob("*"):
        pin(files, path)
    write_json(
        run / "design-freeze.json",
        {
            "status": "all_inputs_schedules_code_fixed_before_learned_training",
            "config": config,
            "runtime": expanded.runtime(),
            "phones": phones,
            "source_design_sha256": sha256_file(source / "design-freeze.json"),
            "files": files,
            "train_speakers": len(train["speakers"]),
            "train_queries": len(train["queries"]),
            "train_intervals": train["feature_count"],
            "missing_vowel_training_queries_excluded": train["excluded_missing_vowels"],
            "equal_baseline_exactly_reproduced": True,
            "training_audio_disjoint_from_enrollment_queries_and_evaluation": True,
        },
    )
    print(
        f"Prepared: {len(train['queries'])} queries, 140 training speakers", flush=True
    )


def load_arrays(run, split):
    with np.load(run / f"{split}-aggregates.npz", allow_pickle=False) as archive:
        return {k: torch.from_numpy(archive[k]) for k in archive.files}


def verify(run, full=False):
    frozen = read_json(run / "design-freeze.json")
    if frozen["config"] != read_json(CONFIG) or frozen["runtime"] != expanded.runtime():
        raise ValueError("configuration or runtime changed after freeze")
    for name, checksum in frozen["files"].items():
        checked(ROOT / name, checksum)
    if full:
        for name, checksum in read_json(
            ROOT / frozen["config"]["source_run"] / "design-freeze.json"
        )["files"].items():
            checked(ROOT / name, checksum)
    return frozen
