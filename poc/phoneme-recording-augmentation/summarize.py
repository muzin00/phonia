"""Report every seed and paired clean-control deltas; no best-test selection."""

import train as e

s, np, t, qr = e.s, e.np, e.t, e.qr
ev = t.h.module("recording_augmentation_evaluation", e.BASE / "evaluate.py")


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


def main():
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    s.torch.set_num_threads(1)
    frozen = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", frozen["design_sha256"])
    for name, digest in frozen["files"].items():
        s.checked(e.ROOT / name, digest)
    print(
        "Auditing feature replay, training schedules, every evaluation score",
        flush=True,
    )
    verification = audit(config, run, design)
    old = design["source_config"]
    points = frozen["thresholds"]
    reports = []
    draws = {}
    for corpus in ("JVS", "CV"):
        speakers = (
            t.jvs_data(old, "test")
            if corpus == "JVS"
            else s.read_json(e.ROOT / old["cv_run"] / "test/inputs.json")
        )["speakers"]
        counts = qr.counts_for(speakers, config)
        conditions = config["conditions"] if corpus == "JVS" else ["CV"]
        for name in ev.names(config):
            for condition in conditions:
                values = list(s.rows(run / "test" / name / f"{condition}-scores.jsonl"))
                for policy in old["policies"]:
                    chosen = s.apply_policy(values, policy)
                    p = policy["name"]
                    threshold = points[f"{name}/{corpus}"][p]["far_1pct"]
                    metrics, dd = qr.metrics(chosen, threshold, speakers, counts)
                    draws[corpus, condition, p, name] = dd
                    entry = {
                        "corpus": corpus,
                        "condition": condition,
                        "policy": p,
                        "model": name,
                        "metrics": metrics,
                    }
                    if corpus == "CV":
                        entry["jvs_transfer"] = qr.metrics(
                            chosen,
                            points[f"{name}/JVS"][p]["far_1pct"],
                            speakers,
                            counts,
                        )[0]
                    reports.append(entry)
    averages = []
    for corpus in ("JVS", "CV"):
        for condition in config["conditions"] if corpus == "JVS" else ["CV"]:
            for policy in old["policies"]:
                p = policy["name"]
                group = [
                    r
                    for r in reports
                    if (r["corpus"], r["condition"], r["policy"])
                    == (corpus, condition, p)
                ]
                original = next(r for r in group if r["model"] == "original")["metrics"]
                entry = {
                    "corpus": corpus,
                    "condition": condition,
                    "policy": p,
                    "original": original,
                }
                for arm in config["arms"]:
                    rows = [r for r in group if r["model"].startswith(arm + "-")]
                    entry[arm] = {
                        key: float(np.mean([r["metrics"][key] for r in rows]))
                        for key in (
                            "eer",
                            "far",
                            "frr",
                            "coverage",
                            "false_accepts",
                            "false_rejects",
                        )
                    }
                    entry[arm]["per_seed_far"] = [r["metrics"]["far"] for r in rows]
                    entry[arm]["per_seed_frr"] = [r["metrics"]["frr"] for r in rows]
                entry["augmented_minus_clean_ci95"] = {
                    key: qr.interval(
                        np.mean(
                            [
                                draws[corpus, condition, p, f"augmented-{seed}"][key]
                                - draws[corpus, condition, p, f"clean-{seed}"][key]
                                for seed in config["seeds"]
                            ],
                            axis=0,
                        )
                    )
                    for key in ("far", "frr")
                }
                entry["augmented_minus_original_ci95"] = {
                    key: qr.interval(
                        np.mean(
                            [
                                draws[corpus, condition, p, f"augmented-{seed}"][key]
                                - draws[corpus, condition, p, "original"][key]
                                for seed in config["seeds"]
                            ],
                            axis=0,
                        )
                    )
                    for key in ("far", "frr")
                }
                averages.append(entry)
    result = {
        "config": config,
        "training": {
            "intervals": design["intervals"],
            "phones": design["phones"],
            "omitted_from_fine_tuning": design["untrained_in_this_probe"],
            "feature_checks": s.read_json(run / "feature-freeze.json"),
        },
        "per_model": reports,
        "seed_averages": averages,
        "verification": verification,
    }
    s.write_json(run / "results.json", result)
    s.write_json(e.BASE / "evaluation-results.json", result)
    lines = [
        "# encoder録音条件拡張：追加学習の対照比較",
        "",
        "3 seedの指標の平均であり、ensembleではない。各モデルのclean validation FAR1%閾値を固定適用。観測済みtestでの探索。単位%。",
        "",
        "## 3 seed平均",
        "",
        "| 条件 | 受付 | 元EER | clean追加EER | 拡張追加EER | 元FAR | clean追加FAR | 拡張追加FAR | 元FRR | clean追加FRR | 拡張追加FRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in averages:
        row = [r["original"], r["clean"], r["augmented"]]
        numbers = [x[key] for key in ("eer", "far", "frr") for x in row]
        lines.append(
            "| "
            + r["condition"]
            + " | "
            + r["policy"]
            + " | "
            + " | ".join(f"{100 * x:.3f}" for x in numbers)
            + " |"
        )
    lines += [
        "",
        "## 全seedの結果",
        "",
        "| 条件 | 受付 | モデル | EER | FAR | FRR |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in reports:
        m = r["metrics"]
        lines.append(
            f"| {r['condition']} | {r['policy']} | {r['model']} | {100 * m['eer']:.3f} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} |"
        )
    lines += [
        "",
        "## 検証・適用範囲",
        "",
        f"学習は既存140話者・{design['intervals']:,}区間・{len(design['phones'])}音素。今回は /dy/ のpositive pairを確保できず追加学習対象外。元encoderの構造・評価支持集合は維持。各条件/seedは6,000更新。",
        "",
        f"float64スコア再計算 {verification['independent_float64_score_rows']:,}件、最大差 {verification['max_error']:.3g}。元モデル再現 {verification['baseline_score_replays']:,}件、train元PCM特徴再計算 {verification['training_pcm_feature_probes']}件、全seedのpair列と初期checkpoint一致を確認。",
        "",
        "[全数値・対応付きbootstrap・閾値](evaluation-results.json)。学習/評価音声は分離。ただし拡張条件を決める前に評価話者の結果を参照しているため、未使用testでの再現確認は別途必要。各seedの指標を平均した区間は固定した3モデルの話者再抽出によるもので、学習seed一般の不確実性を保証しない。元境界固定の診断で、強い雑音時のalignment失敗を含むend-to-end評価ではない。",
        "",
    ]
    (e.BASE / "evaluation-results.md").write_text("\n".join(lines))
    s.write_json(
        run / "completion-verification.json",
        {
            **verification,
            "files": {
                t.rel(p): s.sha256_file(p)
                for p in [
                    run / "results.json",
                    e.BASE / "evaluation-results.json",
                    e.BASE / "evaluation-results.md",
                    e.BASE / "summarize.py",
                ]
            },
        },
    )
    print("Completed all-seed report", flush=True)


if __name__ == "__main__":
    main()
