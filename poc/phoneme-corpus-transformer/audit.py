"""Recheck split provenance, selection, thresholds and saved fusion scores."""

import experiment as e
from inputs import corpus, make_schedule

s, r = e.s, e.r


def independently_score(model, arrays, data, rows):
    speakers = {v: i for i, v in enumerate(data["speakers"])}
    queries = {v["query_id"]: i for i, v in enumerate(data["queries"])}
    maximum = 0.0
    with s.torch.inference_mode():
        for start in range(0, len(rows), 128):
            positions = [
                i
                for i in range(start, min(start + 128, len(rows)))
                if rows[i]["score"] is not None
            ]
            if not positions:
                continue
            q = s.torch.tensor([queries[rows[i]["query_id"]] for i in positions])
            c = s.torch.tensor(
                [speakers[rows[i]["claimed_speaker_id"]] for i in positions]
            )
            features, components = e.backend.pair_features(
                arrays["enrollment"][c],
                arrays["emeta"][c],
                arrays["queries"][q],
                arrays["qmeta"][q],
            )
            masks = s.torch.from_numpy(arrays["masks"][positions])
            assert (masks <= (arrays["emask"][c] & arrays["qmask"][q])).all()
            _, weights = model(features, components, masks)
            w = weights.numpy().astype(s.np.float64)
            assert (w[~masks.numpy()] == 0).all()
            w /= w.sum(axis=1, keepdims=True)
            values = (w * arrays["components"][positions]).sum(axis=1)
            maximum = max(
                maximum,
                float(
                    s.np.max(s.np.abs(values - [rows[i]["score"] for i in positions]))
                ),
            )
    assert maximum < 1e-6, maximum
    return maximum


def main():
    s.torch.set_num_threads(1)
    config = s.read(e.CONFIG)
    run = s.ROOT / config["run_directory"]
    design, prepared, selection = [
        s.read(run / name)
        for name in (
            "design-freeze.json",
            "prepared-freeze.json",
            "selection-freeze.json",
        )
    ]
    assert s.read(run / "execution-state.json")["status"] == "complete"
    for files in (design["files"], prepared["files"], selection["models"]):
        r.checked_files(files)
    train = s.read(run / "train-inputs.json")
    enroll, query = set(train["enrollment_wav_hashes"]), set(train["query_wav_hashes"])
    assert not enroll & query
    manifest = (
        s.ROOT
        / config["reference_run"]
        / config["encoder_arm"]
        / "train-segments.jsonl"
    )
    identities = [
        (row["speaker_id"], row["source_sha256"], row["phoneme"])
        for row in s.rows(manifest)
    ]
    for speaker, profile in train["profiles"].items():
        for phone, indices in profile.items():
            for i in indices:
                owner, digest, p = identities[i]
                assert owner == speaker and p == phone and digest in enroll
    for item in train["queries"]:
        for phone, indices in item["groups"].items():
            for i in indices:
                owner, digest, p = identities[i]
                assert owner == item["speaker_id"] and p == phone
                assert digest == item["query_id"] and digest in query
    models, selected = {}, {}
    for seed in config["seeds"]:
        schedule = make_schedule(config, train, seed)
        with s.np.load(run / f"schedule-{seed}.npz") as saved:
            for key, values in schedule.items():
                s.np.testing.assert_array_equal(values, saved[key])
        for kind in config["architectures"]:
            name = f"{kind}-{seed}"
            directory = run / name
            summary, history = (
                s.read(directory / "training-summary.json"),
                s.read(directory / "history.json"),
            )
            assert summary["completed_updates"] == config["updates"]
            assert summary["history_sha256"] == s.sha(directory / "history.json")
            assert summary["schedule_sha256"] == s.sha(run / f"schedule-{seed}.npz")
            assert summary["encoder_updated"] is False
            assert [v["update"] for v in history] == list(
                range(0, config["updates"] + 1, config["validation_every"])
            )
            best = history[0]
            for item in history:
                assert item["macro_eer"] == sum(item["validation_eer"].values()) / 3
                if item["macro_eer"] < best["macro_eer"] - 1e-12:
                    best = item
            assert summary["selected_update"] == best["update"]
            assert summary["validation_macro_eer"] == best["macro_eer"]
            models[name] = e.load_model(config, directory).eval()
            selected[name] = best["update"]
    result = s.read(run / "results.json")
    assert result["selection_sha256"] == s.sha(run / "selection-freeze.json")
    original = s.read(s.ROOT / config["reference_run"] / "results.json")
    count, maximum, score_files = 0, 0.0, []
    for c in s.CORPORA:
        assert (
            result["conditions"][c]["equal"]
            == original["conditions"][c][config["encoder_arm"]]
        )
        for split in ("validation", "test"):
            data, _, eval_manifest = e.evaluation_source(
                config, design["reference_design"], c, split
            )
            assert not set(train["speakers"]) & set(data["speakers"])
            assert not {row["source_sha256"] for row in s.rows(eval_manifest)} & {
                v[1] for v in identities
            }
            arrays = e.load_arrays(run / f"{c}-{split}.npz")
            baseline = s.read(run / f"{c}-{split}-baseline.json")
            for name in ("equal", *models):
                path = run / "scores" / f"{c}-{name}-{split}.json"
                rows = s.read(path)
                assert len(rows) == len(baseline)
                for a, b in zip(rows, baseline, strict=True):
                    assert {k: v for k, v in a.items() if k != "score"} == {
                        k: v for k, v in b.items() if k != "score"
                    }
                    assert (a["score"] is None) == (b["score"] is None)
                if name == "equal":
                    assert rows == baseline
                else:
                    maximum = max(
                        maximum, independently_score(models[name], arrays, data, rows)
                    )
                point = selection["thresholds"][c][name]
                if split == "validation":
                    normal = [
                        v
                        for v in rows
                        if v["role"] == "verification" and v["score"] is not None
                    ]
                    threshold, _ = s.far_target_threshold(
                        [v["score"] for v in normal],
                        [v["is_genuine"] for v in normal],
                        0.01,
                    )
                    assert s.json_threshold(threshold) == point
                else:
                    for role, metrics in result["conditions"][c][name].items():
                        assert metrics == s.operating(
                            [v for v in rows if v["role"] == role], point
                        )
                count += len(rows)
                score_files.append(path)
    report = {
        "verified_score_rows": count,
        "maximum_recomputed_score_difference": maximum,
        "selected_updates": selected,
        "fusion_training_labels": len(train["speakers"]),
        "training_label_counts": {
            c: sum(corpus(v) == c for v in train["speakers"]) for c in s.CORPORA
        },
        "encoder_sha256": config["encoder_sha256"],
        "files": r.pin(
            [
                *score_files,
                run / "results.json",
                run / "selection-freeze.json",
                e.BASE / "audit.py",
            ]
        ),
    }
    s.fixed(run / "complete-audit.json", report)
    print({k: v for k, v in report.items() if k != "files"})


if __name__ == "__main__":
    main()
