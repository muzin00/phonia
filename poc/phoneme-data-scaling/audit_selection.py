"""Audit the selected catalog identities without decoding new candidate audio."""

from collections import Counter

import selection as s
from pyarrow import parquet


def main():
    config = s.read(s.CONFIG)
    run = s.ROOT / config["run_directory"]
    frozen = s.read(run / "selection-freeze.json")
    s.check_files(frozen["files"])
    assert frozen["status"] == "selected_awaiting_review_preparation"
    assert not frozen["audio_processed"] and not frozen["new_model_training_started"]
    groups = {name: s.read(run / f"{name}.json") for name in frozen["group_labels"]}
    ids = [v["source_id"] for rows in groups.values() for v in rows]
    assert len(ids) == len(set(ids)) == 11200
    ref = s.ROOT / config["baseline_run"] / "expanded280"
    with (ref / "train-segments.jsonl").open() as f:
        trained = {s.json.loads(line)["speaker_id"] for line in f}
    reserved = set(frozen["reserved_evaluation_labels"])
    for name, rows in groups.items():
        labels = {v["speaker_id"] for v in rows}
        assert not labels & reserved
        assert all(v["split"] == "train" for v in rows)
        assert len(labels) == frozen["group_labels"][name]
        if name.startswith("existing"):
            assert labels <= trained
        else:
            assert not labels & trained and len(labels) == 70
    assert set(Counter(v["speaker_id"] for v in groups["new_cv"]).values()) == {50}
    assert set(Counter(v["speaker_id"] for v in groups["new_jtube"]).values()) == {30}
    selected_cv = groups["existing_cv"] + groups["new_cv"]
    cv_root = s.ROOT / config["cv_run"]
    excluded_cv = s.read(cv_root / "selection.json") + s.read(
        s.ROOT / config["cv_holdout"] / "selection.json"
    )
    assert not {v["audio_path"] for v in selected_cv} & {
        v["audio_path"] for v in excluded_cv
    }
    for shard in sorted({v["shard"] for v in selected_cv}):
        original = parquet.read_table(
            cv_root / "source" / shard,
            columns=["client_id", "sentence", "locale", "audio.path"],
        ).to_pylist()
        for row in selected_cv:
            if row["shard"] != shard:
                continue
            raw = original[row["shard_row"]]
            assert raw["path"] == row["audio_path"]
            assert all(raw[k] == row[k] for k in ("client_id", "sentence", "locale"))
            assert s.cv_speaker(raw["client_id"]) == row["speaker_id"]
    archive = s.read(s.ROOT / config["jtube_source"] / "jtube-archive-index.json")
    entries = {v["path"]: v for v in archive["files"]}
    original_jt = s.read(s.ROOT / config["jtube_run"] / "selection-freeze.json")
    selected_jt = groups["existing_jtube"] + groups["new_jtube"]
    assert not {v["path"] for v in selected_jt} & {
        v["path"] for v in original_jt["clips"]
    }
    for row in selected_jt:
        assert row["path"].startswith("JTubeSpeech-ASV/train/")
        assert all(row[k] == v for k, v in entries[row["path"]].items())
        assert row["speaker_id"] == "jtube_" + row["source_speaker_id"].lower()
    for name, arm in frozen["arms"].items():
        additions = [v for group in arm["addition_groups"] for v in groups[group]]
        labels = trained | {v["speaker_id"] for v in additions}
        assert labels == set(arm["speaker_labels"])
        assert len(additions) == arm["raw_added_clips"]
        assert (
            dict(Counter(v["corpus"] for v in additions))
            == arm["raw_added_clips_by_corpus"]
        )
        assert not labels & reserved
    assert (
        frozen["arms"]["utterances280"]["raw_added_clips_by_corpus"]
        == frozen["arms"]["speakers420"]["raw_added_clips_by_corpus"]
    )
    report = {
        "verified_added_source_entries": len(ids),
        "common_voice_entries": len(selected_cv),
        "jtube_entries": len(selected_jt),
        "labels_by_group": frozen["group_labels"],
        "evaluation_label_overlap": 0,
        "original_selected_source_path_overlap": 0,
        "waveform_hash_overlap_audit": "pending audio preparation",
        "files": s.pin([run / "selection-freeze.json", s.BASE / "audit_selection.py"]),
    }
    s.fixed(run / "selection-audit.json", report)
    print({k: v for k, v in report.items() if k != "files"})


if __name__ == "__main__":
    main()
