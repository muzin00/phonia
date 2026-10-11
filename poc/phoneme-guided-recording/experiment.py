"""Frozen prompt selection and coverage-only stopping on recorded utterances."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE.parent / "phoneme-recording-quality"))
import report as qr

t, s = qr.t, qr.t.s
np = s.np
CONFIG = BASE / "config.json"


def select_prompts(metadata, config):
    from scipy.optimize import linear_sum_assignment

    grouped = defaultdict(list)
    for row in metadata:
        name = Path(row["source_file"]).stem
        if (
            row["split"] == "train"
            and name.startswith("VOICEACTRESS100_")
            and 51 <= int(name.rsplit("_", 1)[1]) <= 100
        ):
            grouped[name].append(row)
    inventory = []
    for name, rows in sorted(grouped.items()):
        transcripts = {r["transcript"] for r in rows}
        assert len(transcripts) == 1
        counts = Counter(p.lower() for p in rows[0]["raw_phonemes"])
        inventory.append(
            {
                "id": name,
                "transcript": rows[0]["transcript"],
                "train_speakers": len(rows),
                "median_seconds": float(np.median([r["duration_sec"] for r in rows])),
                "text_vowels": {p: counts[p] for p in s.VOWELS},
            }
        )
    eligible = [
        r
        for r in inventory
        if r["median_seconds"] <= config["prompt_training_median_max_seconds"]
        and min(r["text_vowels"].values()) >= config["prompt_minimum_text_vowels"]
    ]
    chosen = sorted(eligible, key=lambda r: (r["median_seconds"], r["id"]))[
        : config["prompt_count"]
    ]
    assert len(chosen) == config["prompt_count"]
    remainder = [r for r in inventory if r["id"] not in {x["id"] for x in chosen}]
    cost = np.array(
        [
            [abs(a["median_seconds"] - b["median_seconds"]) for b in remainder]
            for a in chosen
        ]
    )
    ri, ci = linear_sum_assignment(cost)
    return {
        "inventory": inventory,
        "pairs": [
            {
                "guided": chosen[i],
                "control": remainder[j],
                "absolute_train_seconds_gap": float(cost[i, j]),
            }
            for i, j in zip(ri, ci, strict=True)
        ],
    }


def verify(config, run):
    design = s.read_json(run / "design-freeze.json")
    assert design["config"] == config
    for name, digest in design["files"].items():
        s.checked(ROOT / name, digest)
    return design


def freeze(config, run):
    if run.exists():
        raise ValueError("fresh run required")
    previous = t.verify(ROOT / config["quality_run"])
    files = dict(previous["files"])
    old = previous["config"]
    meta_path = (
        ROOT / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
    )
    metadata = list(s.rows(meta_path))
    prompts = select_prompts(metadata, config)
    paths = [
        CONFIG,
        BASE / "README.md",
        meta_path,
        *BASE.glob("*.py"),
        *BASE.glob("tests/*.py"),
    ]
    for source in config["cv_runs"].values():
        prior = s.read_json(ROOT / source / "design-freeze.json")
        for name, digest in prior["files"].items():
            s.checked(ROOT / name, digest)
        files.update(prior["files"])
        for split in ("validation", "test"):
            for name in (
                "inputs.json",
                "embeddings.npy",
                "features.npy",
                "segments.jsonl",
                "utterances.jsonl",
            ):
                paths.append(ROOT / source / split / name)
    for split in ("validation", "test"):
        paths.append(ROOT / old["encoder_run"] / f"{split}-features.f32")
    for path in paths:
        t.pin(files, path)
    run.mkdir(parents=True)
    s.write_json(run / "prompt-selection.json", prompts)
    t.pin(files, run / "prompt-selection.json")
    s.write_json(
        run / "design-freeze.json",
        {
            "config": config,
            "source_config": old,
            "extract_config": previous["extract_config"],
            "files": files,
            "status": "protocol_prompts_and_stopping_rules_frozen_before_new_scoring",
        },
    )
    print(
        "Frozen prompts:",
        [(p["guided"]["id"], p["control"]["id"]) for p in prompts["pairs"]],
        flush=True,
    )


def load(config, design, corpus, split):
    old = design["source_config"]
    if corpus == "JVS":
        data = t.jvs_data(old, split)
        data = {
            **data,
            "queries": [q for q in data["queries"] if q["role"] == "verification"],
        }
        vectors = np.load(
            ROOT / old["jvs_run"] / f"{split}-embeddings.npy", allow_pickle=False
        )
        segments = list(s.rows(ROOT / old["encoder_run"] / f"{split}-segments.jsonl"))
        metadata = list(
            s.rows(
                ROOT
                / "poc/phoneme-speaker-dataset/data/generated/expected-phonemes.jsonl"
            )
        )
        features = np.memmap(
            ROOT / old["encoder_run"] / f"{split}-features.f32", mode="r", dtype="<f4"
        ).reshape(-1, 128)
    else:
        source = ROOT / config["cv_runs"][corpus] / split
        data = s.read_json(source / "inputs.json")
        vectors = np.load(source / "embeddings.npy", allow_pickle=False)
        segments = list(s.rows(source / "segments.jsonl"))
        metadata = list(s.rows(source / "utterances.jsonl"))
        features = np.load(source / "features.npy", allow_pickle=False)
    return data, vectors, segments, {r["source_file"]: r for r in metadata}, features


def select_sequence(sequence, arm):
    target = int(arm[-1]) if arm.startswith("adaptive") else None
    total = {p: [] for p in sequence[0]["groups"]}
    selected = []
    for q in sequence:
        if selected and (
            arm == "single"
            or (arm.startswith("fixed") and len(selected) >= int(arm[-1]))
        ):
            break
        selected.append(q)
        for p, ids in q["groups"].items():
            total[p].extend(ids)
        if target and all(len(total.get(p, [])) >= target for p in s.VOWELS):
            break
    return total, selected


def variants(config, run, corpus, data, metadata):
    if corpus == "JVS":
        pairs = s.read_json(run / "prompt-selection.json")["pairs"]
        for arm in ("all50", "guided", "matched"):
            wanted = {
                p["guided" if arm == "guided" else "control"]["id"] for p in pairs
            }
            queries = (
                data["queries"]
                if arm == "all50"
                else [
                    q for q in data["queries"] if Path(q["source_file"]).stem in wanted
                ]
            )
            expected = 50 if arm == "all50" else config["prompt_count"]
            assert Counter(q["speaker_id"] for q in queries) == {
                sp: expected for sp in data["speakers"]
            }
            details = [detail(q, q["groups"], [q], metadata) for q in queries]
            yield arm, {**data, "queries": queries}, details
        return
    grouped = {
        sp: sorted(
            [q for q in data["queries"] if q["speaker_id"] == sp],
            key=lambda q: t.h.digest(config["episode_seed"], q["query_id"]),
        )
        for sp in data["speakers"]
    }
    assert all(len(qs) == 30 for qs in grouped.values())
    for arm in config["cv_arms"]:
        queries, details = [], []
        for sp, qs in grouped.items():
            for i, first in enumerate(qs):
                sequence = [
                    qs[(i + j) % len(qs)] for j in range(config["maximum_clips"])
                ]
                groups, chosen = select_sequence(sequence, arm)
                assert len({q["query_id"] for q in chosen}) == len(chosen)
                assert all(q["speaker_id"] == sp for q in chosen)
                queries.append({**first, "groups": groups})
                details.append(detail(first, groups, chosen, metadata))
        yield arm, {**data, "queries": queries}, details


def detail(first, groups, chosen, metadata):
    return {
        "query_id": first["query_id"],
        "speaker_id": first["speaker_id"],
        "source_query_ids": [q["query_id"] for q in chosen],
        "sources": [q["source_file"] for q in chosen],
        "clips": len(chosen),
        "seconds": sum(metadata[q["source_file"]]["duration_sec"] for q in chosen),
        "vowel_counts": {p: len(groups.get(p, [])) for p in s.VOWELS},
        "groups": groups,
    }


def evaluate(config, run, split):
    design = verify(config, run)
    if split == "test":
        frozen = s.read_json(run / "threshold-freeze.json")
        s.checked(run / "design-freeze.json", frozen["design_sha256"])
        for name, digest in frozen["files"].items():
            s.checked(ROOT / name, digest)
    points = {}
    for corpus in ("JVS", *config["cv_runs"]):
        data, vectors, segments, metadata, _ = load(config, design, corpus, split)
        enrollment = {
            segments[i]["source_file"]
            for ps in data["profiles"].values()
            for ids in ps.values()
            for i in ids
        }
        assert not enrollment & {q["source_file"] for q in data["queries"]}
        assert set(s.VOWELS) <= set(data["universally_registered"])
        dest = run / split / corpus
        dest.mkdir(parents=True, exist_ok=True)
        for arm, selected, details in variants(config, run, corpus, data, metadata):
            values = s.apply_policy(t.h.candidates(selected, vectors), config["policy"])
            s.write_rows(dest / f"{arm}-scores.jsonl", values)
            s.write_json(dest / f"{arm}-episodes.json", details)
            if split == "validation":
                points[f"{corpus}/{arm}"] = (
                    points["JVS/all50"]
                    if corpus == "JVS" and arm != "all50"
                    else s.original.thresholds(values)
                )
            print(
                f"Scored {split}/{corpus}/{arm}: {len(selected['queries'])} episodes",
                flush=True,
            )
    if split == "validation":
        files = {
            t.rel(p): s.sha256_file(p) for p in (run / split).rglob("*") if p.is_file()
        }
        s.write_json(
            run / "threshold-freeze.json",
            {
                "design_sha256": s.sha256_file(run / "design-freeze.json"),
                "thresholds": points,
                "files": files,
                "status": "all_calibrations_frozen_before_observed_test_scoring",
            },
        )
    verify(config, run)


def main():
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
