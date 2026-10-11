"""Report every seed and paired clean-control deltas; no best-test selection."""

import train as e

s, np, t, qr = e.s, e.np, e.t, e.qr
ev = t.h.module("recording_ablation_evaluation", e.BASE / "evaluate.py")


def audit(config, run, design):
    old = design["source_config"]
    source = e.ROOT / config["quality_run"]
    pipe = t.h.primitives.pipeline(design["extract_config"])
    selected = list(s.rows(run / "train-selection.jsonl"))
    feature_bank = np.load(
        run / "train-features.npy", allow_pickle=False, mmap_mode="r"
    )
    probes = 0
    # Waveform reconstruction samples both corpora and every augmentation condition.
    for corpus in ("jvs", "cv17_"):
        candidates = [
            i for i, r in enumerate(selected) if r["speaker_id"].startswith(corpus)
        ]
        for i in candidates[:: max(1, len(candidates) // 3)][:3]:
            row = selected[i]
            raw = e.a.read_wave(e.ROOT / row["source_file"])
            for mode, condition in enumerate(config["conditions"]):
                y, _ = e.a.transform(
                    raw,
                    condition,
                    e.a.seed_for(config["augmentation_seed"], row["source_file"]),
                )
                f = t.h.primitives.pool_features(
                    pipe, y[row["start_frame"] : row["end_frame"]], row["segment_id"]
                )
                np.testing.assert_array_equal(f, feature_bank[mode, i])
                probes += 1
    for seed in config["seeds"]:
        with np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as saved:
            for key, value in zip(
                ("indices", "classes", "valid", "views"),
                e.schedule(config, design, selected, seed),
                strict=True,
            ):
                np.testing.assert_array_equal(saved[key], value)
        with (
            np.load(
                e.ROOT / config["source_run"] / f"schedule-{seed}.npz",
                allow_pickle=False,
            ) as previous,
            np.load(run / f"schedule-{seed}.npz", allow_pickle=False) as current,
        ):
            for key in previous.files:
                np.testing.assert_array_equal(previous[key], current[key])
        for arm in config["arms"]:
            dest = run / f"{arm}-{seed}"
            for name, digest in s.read_json(dest / "complete.json")["files"].items():
                s.checked(e.ROOT / name, digest)
            if arm not in ("clean", "augmented"):
                with np.load(
                    run / f"schedule-{seed}.npz", allow_pickle=False
                ) as schedule:
                    np.testing.assert_array_equal(
                        np.load(dest / "views.npy"), e.arm_views(schedule["views"], arm)
                    )
            for key in ("training_examples", "initial_checkpoint_sha256"):
                assert (
                    s.read_json(dest / "complete.json")[key]
                    == s.read_json(run / f"clean-{seed}" / "complete.json")[key]
                )
        clean = s.read_json(run / f"clean-{seed}" / "complete.json")
        aug = s.read_json(run / f"augmented-{seed}" / "complete.json")
        assert (
            clean["training_examples"] == aug["training_examples"]
            and clean["initial_checkpoint_sha256"] == aug["initial_checkpoint_sha256"]
        )
        for report in (clean, aug):
            for name, digest in report["files"].items():
                s.checked(e.ROOT / name, digest)
    checks = 0
    maximum = 0.0
    baseline_matches = 0
    for split in ("validation", "test"):
        jvs = t.jvs_data(old, split)
        cv = s.read_json(e.ROOT / old["cv_run"] / split / "inputs.json")
        selected_ids = {
            r["query"]["query_id"]
            for r in s.read_json(source / "selection.json")["stress"][split]
        }
        jvs = {
            **jvs,
            "queries": [q for q in jvs["queries"] if q["query_id"] in selected_ids],
        }
        for name in ev.names(config):
            model = ev.load(config, run, name)
            dest = run / split / name
            vectors = np.load(dest / "JVS-embeddings.npy", allow_pickle=False).astype(
                np.float64
            )
            registration = t.profiles(jvs, vectors)
            jvs_features = np.memmap(
                e.ROOT / old["encoder_run"] / f"{split}-features.f32",
                mode="r",
                dtype="<f4",
            ).reshape(-1, 128)
            np.testing.assert_allclose(
                s.original.model_embeddings(model, jvs_features[:32]),
                vectors[:32],
                atol=1e-6,
                rtol=0,
            )
            for condition in config["conditions"]:
                refs = {}
                with np.load(
                    dest / f"{condition}-query-embeddings.npz", allow_pickle=False
                ) as queries:
                    for number, path in enumerate(
                        sorted((source / split / condition).glob("*/prepared.json"))
                    ):
                        item = s.read_json(path)
                        q = item["query"]
                        embedding = queries[q["query_id"]]
                        rows = t.score(
                            q,
                            jvs,
                            embedding.astype(np.float64),
                            item["fixed"],
                            registration,
                            "fixed",
                            split,
                        )
                        refs.update(
                            {(r["query_id"], r["claimed_speaker_id"]): r for r in rows}
                        )
                        if number == 0:
                            with np.load(
                                path.parent / "features.npz", allow_pickle=False
                            ) as f:
                                np.testing.assert_array_equal(
                                    s.original.model_embeddings(model, f["fixed"]),
                                    embedding,
                                )
                actual = list(s.rows(dest / f"{condition}-scores.jsonl"))
                if name == "original":
                    baseline = {
                        (r["query_id"], r["claimed_speaker_id"]): r
                        for r in s.rows(
                            source / split / condition / "fixed-scores.jsonl"
                        )
                    }
                for row in actual:
                    key = row["query_id"], row["claimed_speaker_id"]
                    ref = refs[key]
                    assert (
                        row["used_phones"] == ref["used_phones"]
                        and row["status"] == ref["status"]
                    )
                    maximum = max(maximum, abs(row["score"] - ref["score"]))
                    checks += 1
                    if name == "original":
                        assert abs(row["score"] - baseline[key]["score"]) < 2e-6
                        baseline_matches += 1
            cv_vectors = np.load(dest / "CV-embeddings.npy", allow_pickle=False).astype(
                np.float64
            )
            refs = {
                (r["query_id"], r["claimed_speaker_id"]): r
                for r in t.h.candidates(cv, cv_vectors)
            }
            cv_features = np.load(
                e.ROOT / old["cv_run"] / split / "features.npy", allow_pickle=False
            )
            np.testing.assert_allclose(
                s.original.model_embeddings(model, cv_features[:32]),
                cv_vectors[:32],
                atol=1e-6,
                rtol=0,
            )
            if name == "original":
                baseline = {
                    (r["query_id"], r["claimed_speaker_id"]): r
                    for r in s.rows(e.ROOT / old["cv_run"] / split / "candidates.jsonl")
                }
            for row in s.rows(dest / "CV-scores.jsonl"):
                key = row["query_id"], row["claimed_speaker_id"]
                ref = refs[key]
                assert (
                    row["used_phones"] == ref["used_phones"]
                    and row["status"] == ref["status"]
                )
                if row["score"] is not None:
                    maximum = max(maximum, abs(row["score"] - ref["score"]))
                checks += 1
                if name == "original":
                    if row["score"] is not None:
                        assert abs(row["score"] - baseline[key]["score"]) < 2e-6
                    baseline_matches += 1
    assert maximum < 2e-6
    return {
        "status": "passed",
        "independent_float64_score_rows": checks,
        "max_error": maximum,
        "baseline_score_replays": baseline_matches,
        "training_pcm_feature_probes": probes,
        "same_initialization_and_schedules_verified": True,
    }
