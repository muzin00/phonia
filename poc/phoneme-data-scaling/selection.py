"""Freeze nested data-growth candidates before decoding or training on new audio."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
CONFIG = BASE / "config.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rank(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def cv_speaker(client):
    text = json.dumps(("speaker_id", client), ensure_ascii=False, separators=(",", ":"))
    return "cv17_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def fixed(path, value):
    if path.exists():
        assert read(path) == value, f"Frozen output changed: {path}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def check_files(files):
    for path, checksum in files.items():
        assert sha(ROOT / path) == checksum, path


def pin(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def select_labels(groups, excluded, count, minimum, seed):
    eligible = [
        label
        for label, clips in groups.items()
        if label not in excluded and len(clips) >= minimum
    ]
    selected = sorted(eligible, key=lambda label: (rank(seed, label), label))[:count]
    assert len(selected) == count, f"Need {count} labels; only {len(eligible)} eligible"
    return selected, len(eligible)


def balanced_take(groups, count, seed):
    queues = [
        sorted(clips, key=lambda row: (rank(seed, row["source_id"]), row["source_id"]))
        for label, clips in sorted(
            groups.items(), key=lambda pair: (rank(seed, pair[0]), pair[0])
        )
        if clips
    ]
    assert sum(map(len, queues)) >= count, "Insufficient distinct candidate clips"
    selected, depth = [], 0
    while len(selected) < count:
        for rows in queues:
            if depth < len(rows):
                selected.append(rows[depth])
                if len(selected) == count:
                    break
        depth += 1
    assert len({v["source_id"] for v in selected}) == count
    return selected


def cv_inputs(config):
    from pyarrow import parquet

    source = ROOT / config["cv_run"]
    training = read(source / "selection.json")
    heldout = read(ROOT / config["cv_holdout"] / "selection.json")
    trained_clients = {v["client_id"] for v in training}
    heldout_clients = {v["client_id"] for v in heldout}
    assert len(trained_clients) == 70 and len(heldout_clients) == 60
    assert not trained_clients & heldout_clients
    assert all(
        cv_speaker(v["client_id"]) == v["speaker_id"] for v in training + heldout
    )
    existing_paths = {v["audio_path"] for v in training}
    manifest = read(source / "source/manifest.json")
    files = {
        str((source / "source" / name).relative_to(ROOT)): h
        for name, h in manifest["files"].items()
    }
    check_files(files)
    groups, owner = defaultdict(list), {}
    for name in sorted(manifest["files"]):
        if not name.endswith(".parquet"):
            continue
        table = parquet.read_table(
            source / "source" / name,
            columns=["client_id", "sentence", "locale", "audio.path"],
        )
        for index, row in enumerate(table.to_pylist()):
            client = row["client_id"]
            if not client or not row["sentence"] or row["locale"] != "ja":
                continue
            # Arrow flattens the projected audio.path field into a path column.
            path = row["path"]
            assert path not in owner, "Repeated CV source path; do not double-count"
            owner[path] = client
            groups[client].append(
                {
                    "corpus": "common_voice",
                    "speaker_id": cv_speaker(client),
                    "client_id": client,
                    "source_id": "cv:" + path,
                    "audio_path": path,
                    "sentence": row["sentence"],
                    "locale": row["locale"],
                    "shard": name,
                    "shard_row": index,
                    "split": "train",
                }
            )
    new_labels, eligible = select_labels(
        groups,
        trained_clients | heldout_clients,
        config["new_cv_labels"],
        config["new_cv_clips_per_label"],
        config["speaker_seed"],
    )
    new = [
        row
        for client in new_labels
        for row in sorted(
            groups[client],
            key=lambda v: (rank(config["clip_seed"], v["source_id"]), v["source_id"]),
        )[: config["new_cv_clips_per_label"]]
    ]
    old_pool = {
        client: [v for v in groups[client] if v["audio_path"] not in existing_paths]
        for client in trained_clients
    }
    extra = balanced_take(
        old_pool, config["existing_cv_extra_clips"], config["clip_seed"]
    )
    assert not {v["audio_path"] for v in extra + new} & {
        v["audio_path"] for v in training + heldout
    }
    paths = [
        source / "selection.json",
        source / "source/manifest.json",
        ROOT / config["cv_holdout"] / "selection.json",
    ]
    files.update(pin(paths))
    return (
        new,
        extra,
        files,
        {
            "new_eligible_labels": eligible,
            "existing_label_unused_clips": sum(map(len, old_pool.values())),
            "existing_labels_with_unused_clips": sum(
                bool(v) for v in old_pool.values()
            ),
            "repository": manifest["repository"],
            "revision": manifest["revision"],
            "source_split": manifest["source_split"],
            "license_from_frozen_source": "CC0-1.0",
        },
    )


def jtube_inputs(config):
    source = ROOT / config["jtube_source"]
    original = read(ROOT / config["jtube_run"] / "selection-freeze.json")
    check_files(original["files"])
    archive = read(source / "jtube-archive-index.json")
    entries = {v["path"]: v for v in archive["files"]}
    official_test = {
        p.split("/")[3]
        for p in entries
        if p.startswith("JTubeSpeech-ASV/test/") and p.endswith(".mp3")
    }
    grouped = defaultdict(list)
    for line in (source / "Japanese.lst").read_text().splitlines():
        speaker, relative = line.split()
        path = "JTubeSpeech-ASV/train/" + relative
        assert path in entries and speaker == relative.split("/")[0]
        grouped[speaker].append(path)
    assert not set(grouped) & official_test
    assert all(len(v) == len(set(v)) for v in grouped.values())
    trained = set(original["partitions"]["train"])
    heldout = set(original["partitions"]["validation"]) | set(
        original["partitions"]["test"]
    )
    assert len(trained) == 140 and len(heldout) == 40 and not trained & heldout
    old_paths = {v["path"] for v in original["clips"]}

    def row(speaker, path):
        return {
            **entries[path],
            "corpus": "jtube",
            "speaker_id": "jtube_" + speaker.lower(),
            "source_speaker_id": speaker,
            "source_id": "jtube:" + path,
            "split": "train",
        }

    new_labels, eligible = select_labels(
        grouped,
        trained | heldout | official_test,
        config["new_jtube_labels"],
        config["new_jtube_clips_per_label"],
        config["speaker_seed"],
    )
    new = [
        row(label, path)
        for label in new_labels
        for path in sorted(
            grouped[label], key=lambda p: (rank(config["clip_seed"], p), p)
        )[: config["new_jtube_clips_per_label"]]
    ]
    pools = {}
    for label in trained:
        ordered = sorted(
            grouped[label], key=lambda p: rank(original["config"]["utterance_seed"], p)
        )
        expected = {
            v["path"]
            for v in original["clips"]
            if v["source_speaker_id"] == label and v["split"] == "train"
        }
        assert set(ordered[:50]) == expected
        pools[label] = [
            row(label, path) for path in ordered[50 : config["existing_jtube_clip_cap"]]
        ]
    extra = balanced_take(
        pools, config["existing_jtube_extra_clips"], config["clip_seed"]
    )
    assert not {v["path"] for v in extra + new} & old_paths
    paths = [
        source / name
        for name in ("jtube-archive-index.json", "Japanese.lst", "README.md")
    ]
    paths += [ROOT / config["jtube_run"] / "selection-freeze.json"]
    return (
        new,
        extra,
        pin(paths),
        {
            "new_eligible_labels": eligible,
            "existing_label_unused_clips_within_cap": sum(map(len, pools.values())),
            "existing_labels_with_unused_clips": sum(bool(v) for v in pools.values()),
            "archive_url": archive["url"],
            "archive_etag": archive["etag"],
            "archive_size": archive["size"],
            "license_from_frozen_source": original["config"]["source_license"],
        },
    )


def main():
    config = read(CONFIG)
    output = ROOT / config["run_directory"]
    if (output / "selection-freeze.json").exists():
        existing = read(output / "selection-freeze.json")
        assert existing["config"] == config
        check_files(existing["files"])
        print(existing["summary"])
        return
    baseline = ROOT / config["baseline_run"]
    arm = baseline / "expanded280"
    arm_design = read(arm / "design-freeze.json")
    check_files(arm_design["files"])
    with (arm / "train-segments.jsonl").open() as f:
        original_labels = {json.loads(line)["speaker_id"] for line in f}
    assert len(original_labels) == 280
    evaluation_paths = sorted((baseline / "evaluation-inputs").glob("*.json"))
    assert len(evaluation_paths) == 6
    reserved = {
        speaker for path in evaluation_paths for speaker in read(path)["speakers"]
    }
    assert not original_labels & reserved
    cv_new, cv_extra, cv_files, cv_inventory = cv_inputs(config)
    jt_new, jt_extra, jt_files, jt_inventory = jtube_inputs(config)
    groups = {
        "existing_cv": cv_extra,
        "existing_jtube": jt_extra,
        "new_cv": cv_new,
        "new_jtube": jt_new,
    }
    for rows in groups.values():
        assert not {v["speaker_id"] for v in rows} & reserved
    assert {v["speaker_id"] for v in cv_extra + jt_extra} <= original_labels
    added_labels = {v["speaker_id"] for v in cv_new + jt_new}
    assert len(added_labels) == 140 and not added_labels & original_labels
    ids = [v["source_id"] for rows in groups.values() for v in rows]
    assert len(ids) == len(set(ids)) == 11200
    assert len(cv_extra) == len(cv_new) == 3500 and len(jt_extra) == len(jt_new) == 2100
    arms = {}
    for name, use in {
        "baseline280": [],
        "utterances280": ["existing_cv", "existing_jtube"],
        "speakers420": ["new_cv", "new_jtube"],
        "combined420": list(groups),
    }.items():
        labels = original_labels | {
            v["speaker_id"] for group in use for v in groups[group]
        }
        arms[name] = {
            "base_train_segments": str(
                (arm / "train-segments.jsonl").relative_to(ROOT)
            ),
            "base_features": str((arm / "train-features.f32").relative_to(ROOT)),
            "addition_groups": use,
            "speaker_labels": sorted(labels),
            "raw_added_clips": sum(len(groups[g]) for g in use),
            "raw_added_clips_by_corpus": dict(
                Counter(v["corpus"] for group in use for v in groups[group])
            ),
        }
    paths = [
        CONFIG,
        BASE / "selection.py",
        *sorted((BASE / "tests").glob("*.py")),
        arm / "design-freeze.json",
        arm / "train-segments.jsonl",
        arm / "train-features.f32",
        *evaluation_paths,
    ]
    files = {**arm_design["files"], **cv_files, **jt_files, **pin(paths)}
    for name, rows in groups.items():
        path = output / f"{name}.json"
        fixed(path, rows)
        files.update(pin([path]))
    summary = {
        name: {
            "labels": len(v["speaker_labels"]),
            "raw_added_clips": v["raw_added_clips"],
            "raw_added_clips_by_corpus": v["raw_added_clips_by_corpus"],
        }
        for name, v in arms.items()
    }
    fixed(
        output / "selection-freeze.json",
        {
            "config": config,
            "files": files,
            "summary": summary,
            "arms": arms,
            "inventories": {"common_voice": cv_inventory, "jtube": jt_inventory},
            "reserved_evaluation_labels": sorted(reserved),
            "group_labels": {
                name: len({v["speaker_id"] for v in rows})
                for name, rows in groups.items()
            },
            "status": "selected_awaiting_review_preparation",
            "audio_processed": False,
            "new_model_training_started": False,
            "raw_clip_count_matching_only": True,
            "waveform_hash_overlap_audit_pending": True,
            "cross_corpus_real_person_overlap": "unknown",
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
