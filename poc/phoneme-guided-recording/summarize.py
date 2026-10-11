"""Audit recorded-utterance replay and report accuracy with recording cost."""

import experiment as e

s, np, t, qr = e.s, e.np, e.t, e.qr


def resources(details, speakers, counts):
    seconds = np.array([r["seconds"] for r in details])
    clips = np.array([r["clips"] for r in details])
    per_speaker = np.array(
        [
            np.mean([r["seconds"] for r in details if r["speaker_id"] == sp])
            for sp in speakers
        ]
    )
    draws = counts @ per_speaker / len(speakers)
    return {
        "mean_seconds": float(seconds.mean()),
        "median_seconds": float(np.median(seconds)),
        "p95_seconds": float(np.quantile(seconds, 0.95)),
        "mean_clips": float(clips.mean()),
        "clip_counts": {str(i): int(np.sum(clips == i)) for i in range(1, 4)},
        "mean_seconds_ci95": qr.interval(draws),
        "minimum_vowel_count_quantiles": np.quantile(
            [min(r["vowel_counts"].values()) for r in details], [0.0, 0.5, 0.95]
        ).tolist(),
    }, draws


def audit(config, run, design):
    old = design["source_config"]
    model, _ = s.original.load_model(
        e.ROOT / old["encoder_run"] / "trials" / old["encoder_trial"]
    )
    pipe = t.h.primitives.pipeline(design["extract_config"])
    checks, probes, max_error = 0, 0, 0.0
    previous_cohort_speakers = set()
    for corpus in ("JVS", *config["cv_runs"]):
        corpus_speakers = set()
        for split in ("validation", "test"):
            data, vectors, segments, metadata, features = e.load(
                config, design, corpus, split
            )
            assert not corpus_speakers & set(data["speakers"])
            corpus_speakers.update(data["speakers"])
            ids = sorted(
                {
                    i
                    for p in data["profiles"].values()
                    for indices in p.values()
                    for i in indices
                }
                | {
                    i
                    for q in data["queries"]
                    for indices in q["groups"].values()
                    for i in indices
                }
            )
            sample = np.array(ids[:: max(1, len(ids) // 30)][:30])
            np.testing.assert_allclose(
                s.original.model_embeddings(model, features[sample]),
                vectors[sample],
                atol=1e-6,
                rtol=0,
            )
            for i in sample[:10]:
                r = segments[i]
                raw = e.t.a.read_wave(e.ROOT / r["source_file"])
                f = t.h.primitives.pool_features(
                    pipe, raw[r["start_frame"] : r["end_frame"]], r["segment_id"]
                )
                np.testing.assert_array_equal(f, features[i])
                probes += 1
            original = {q["query_id"]: q for q in data["queries"]}
            by_speaker = {
                sp: sorted(
                    [q["query_id"] for q in data["queries"] if q["speaker_id"] == sp],
                    key=lambda qid: t.h.digest(config["episode_seed"], qid),
                )
                for sp in data["speakers"]
            }
            registration = {
                (sp, p): vectors[indices].astype(np.float64).mean(axis=0)
                for sp, ps in data["profiles"].items()
                for p, indices in ps.items()
                if p in data["universally_registered"]
            }
            registration = {
                key: v / np.linalg.norm(v) for key, v in registration.items()
            }
            arms = (
                ("all50", "guided", "matched") if corpus == "JVS" else config["cv_arms"]
            )
            for arm in arms:
                details = s.read_json(run / split / corpus / f"{arm}-episodes.json")
                refs = {}
                for episode in details:
                    first = original[episode["query_id"]]
                    chosen = [original[qid] for qid in episode["source_query_ids"]]
                    assert chosen[0] == first
                    assert all(q["speaker_id"] == first["speaker_id"] for q in chosen)
                    assert episode["sources"] == [q["source_file"] for q in chosen]
                    assert (
                        abs(
                            episode["seconds"]
                            - sum(
                                metadata[q["source_file"]]["duration_sec"]
                                for q in chosen
                            )
                        )
                        < 1e-10
                    )
                    reconstructed = {
                        p: [i for q in chosen for i in q["groups"][p]]
                        for p in first["groups"]
                    }
                    assert reconstructed == episode["groups"]
                    assert len(
                        [i for ids in reconstructed.values() for i in ids]
                    ) == len({i for ids in reconstructed.values() for i in ids})
                    if corpus != "JVS":
                        order = by_speaker[first["speaker_id"]]
                        start = order.index(first["query_id"])
                        assert episode["source_query_ids"] == [
                            order[(start + j) % len(order)] for j in range(len(chosen))
                        ]
                        if arm.startswith("adaptive"):
                            quota = int(arm[-1])
                            for end in range(1, len(chosen)):
                                assert not all(
                                    sum(len(q["groups"][p]) for q in chosen[:end])
                                    >= quota
                                    for p in s.VOWELS
                                )
                            assert len(chosen) == config["maximum_clips"] or all(
                                len(reconstructed[p]) >= quota for p in s.VOWELS
                            )
                        else:
                            assert len(chosen) == (
                                1 if arm == "single" else int(arm[-1])
                            )
                    phones = sorted(
                        p for p in data["universally_registered"] if reconstructed[p]
                    )
                    can_score = set(s.VOWELS) <= set(phones)
                    for claimed in data["speakers"]:
                        score = (
                            np.mean(
                                [
                                    vectors[reconstructed[p]]
                                    .astype(np.float64)
                                    .mean(axis=0)
                                    @ registration[claimed, p]
                                    for p in phones
                                ]
                            )
                            if can_score
                            else None
                        )
                        refs[episode["query_id"], claimed] = (score, phones)
                actual = list(s.rows(run / split / corpus / f"{arm}-scores.jsonl"))
                assert len(actual) == len(refs)
                for row in actual:
                    score, phones = refs[row["query_id"], row["claimed_speaker_id"]]
                    assert row["used_phones"] == phones
                    assert (row["score"] is None) == (score is None)
                    if score is not None:
                        max_error = max(max_error, abs(row["score"] - score))
                    checks += 1
                if split == "validation":
                    points = s.original.thresholds(actual)
                    frozen = s.read_json(run / "threshold-freeze.json")["thresholds"]
                    if corpus != "JVS" or arm == "all50":
                        assert points == frozen[f"{corpus}/{arm}"]
        if corpus != "JVS":
            assert not previous_cohort_speakers & corpus_speakers
            previous_cohort_speakers.update(corpus_speakers)
    assert max_error < 2e-6
    return {
        "status": "passed",
        "float64_score_rows": checks,
        "max_error": max_error,
        "pcm_feature_probes": probes,
        "stopping_and_source_order_verified": True,
        "two_cv_cohorts_disjoint_labels": True,
        "calibration_replayed": True,
    }


def main():
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    frozen = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", frozen["design_sha256"])
    for name, digest in frozen["files"].items():
        s.checked(e.ROOT / name, digest)
    s.torch.set_num_threads(1)
    verification = audit(config, run, design)
    print("Independent replay passed", verification, flush=True)
    reports, draws, duration_draws, trials = [], {}, {}, {}
    for corpus in ("JVS", *config["cv_runs"]):
        data, *_ = e.load(config, design, corpus, "test")
        counts = qr.counts_for(data["speakers"], config)
        for arm in (
            ("all50", "guided", "matched") if corpus == "JVS" else config["cv_arms"]
        ):
            path = run / "test" / corpus
            details = s.read_json(path / f"{arm}-episodes.json")
            cost, dc = resources(details, data["speakers"], counts)
            duration_draws[corpus, arm] = dc
            values = list(s.rows(path / f"{arm}-scores.jsonl"))
            trials[corpus, arm] = values
            for op in config["operating_points"]:
                threshold = frozen["thresholds"][f"{corpus}/{arm}"][op]
                metrics, dd = qr.metrics(values, threshold, data["speakers"], counts)
                metrics["unscored_queries"] = (
                    metrics["queries"] - metrics["scored_queries"]
                )
                reports.append(
                    {
                        "corpus": corpus,
                        "arm": arm,
                        "operating_point": op,
                        "metrics": metrics,
                        "recording": cost,
                    }
                )
                draws[corpus, arm, op] = dd
        print("Summarized", corpus, flush=True)
    contrasts = []

    def row(corpus, arm, op):
        return next(
            r
            for r in reports
            if (r["corpus"], r["arm"], r["operating_point"]) == (corpus, arm, op)
        )

    for corpus in ("JVS", *config["cv_runs"]):
        pairs = (
            [("guided", "matched")]
            if corpus == "JVS"
            else [(arm, "single") for arm in config["cv_arms"] if arm != "single"]
            + [
                ("adaptive1", "fixed2"),
                ("adaptive1", "fixed3"),
                ("adaptive2", "fixed3"),
            ]
        )
        for arm, control in pairs:
            for op in config["operating_points"]:
                a, b = row(corpus, arm, op), row(corpus, control, op)
                entry = {
                    "corpus": corpus,
                    "arm": arm,
                    "control": control,
                    "operating_point": op,
                    "delta": {
                        key: a["metrics"][key] - b["metrics"][key]
                        for key in ("far", "frr", "coverage", "eer")
                    },
                    "ci95": {
                        key: qr.interval(
                            draws[corpus, arm, op][key]
                            - draws[corpus, control, op][key]
                        )
                        for key in ("far", "frr")
                    },
                    "mean_seconds_delta": a["recording"]["mean_seconds"]
                    - b["recording"]["mean_seconds"],
                    "mean_seconds_delta_ci95": qr.interval(
                        duration_draws[corpus, arm] - duration_draws[corpus, control]
                    ),
                    "mean_seconds_ratio": a["recording"]["mean_seconds"]
                    / b["recording"]["mean_seconds"],
                }
                if corpus != "JVS":
                    current = {
                        r["query_id"]: r for r in trials[corpus, arm] if r["is_genuine"]
                    }
                    previous = {
                        r["query_id"]: r
                        for r in trials[corpus, control]
                        if r["is_genuine"]
                    }

                    def accepts(r, threshold):
                        return r["status"] == "scored" and r["score"] >= float(
                            threshold
                        )

                    entry["genuine_transitions"] = {
                        "previously_unscored_now_accepted": sum(
                            r["status"] != "scored"
                            and accepts(current[qid], a["metrics"]["threshold"])
                            for qid, r in previous.items()
                        ),
                        "previously_accepted_now_rejected": sum(
                            accepts(r, b["metrics"]["threshold"])
                            and not accepts(current[qid], a["metrics"]["threshold"])
                            for qid, r in previous.items()
                        ),
                    }
                contrasts.append(entry)
    gates = []
    for corpus in config["cv_runs"]:
        contrast = next(
            r
            for r in contrasts
            if (r["corpus"], r["arm"], r["control"], r["operating_point"])
            == (corpus, "adaptive1", "single", "far_0_1pct")
        )
        cost = row(corpus, "adaptive1", "far_0_1pct")["recording"]
        checks = {
            "frr_improved": contrast["ci95"]["frr"][1]
            < config["gate"]["frr_increment_upper95_strict_max"],
            "far_increment_within_limit": contrast["ci95"]["far"][1]
            <= config["gate"]["far_increment_upper95_max"],
            "mean_time_within_limit": contrast["mean_seconds_ratio"]
            <= config["gate"]["mean_duration_ratio_max"],
            "p95_time_within_limit": cost["p95_seconds"]
            <= config["gate"]["duration_p95_max_seconds"],
        }
        gates.append(
            {"corpus": corpus, "checks": checks, "passed": all(checks.values())}
        )
    result = {
        "config": config,
        "prompt_selection": s.read_json(run / "prompt-selection.json"),
        "reports": reports,
        "contrasts": contrasts,
        "primary_gates": gates,
        "primary_passed": all(g["passed"] for g in gates),
        "verification": verification,
    }
    s.write_json(run / "results.json", result)
    s.write_json(e.BASE / "evaluation-results.json", result)
    lines = [
        "# 指定文・追加録音の比較結果",
        "",
        "固定encoder、5母音必須、全入力FAR/FRR。EERは採点可能な入力のみ。既観測の録音による開発用再生。校正は採点可能な他人試行に対して行う。単位%。",
        "",
        "| 集合 | 方式 | 校正目標 | 採点率 | EER | FAR | FRR | 平均秒 | P95秒 | 平均発話数 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in reports:
        m, c = r["metrics"], r["recording"]
        lines.append(
            f"| {r['corpus']} | {r['arm']} | {r['operating_point']} | {100 * m['coverage']:.3f} | {100 * m['eer']:.3f} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} | {c['mean_seconds']:.3f} | {c['p95_seconds']:.3f} | {c['mean_clips']:.3f} |"
        )
    lines += ["", "## 主評価の事前基準", ""]
    for g in gates:
        lines.append(f"- {g['corpus']}: {g['passed']} — {g['checks']}")
    lines += [
        "",
        "## trainだけで固定した指定文と対照",
        "",
        "| 母音を含む候補 | train秒 | 対照 | train秒 |",
        "| --- | ---: | --- | ---: |",
    ]
    for pair in result["prompt_selection"]["pairs"]:
        a, b = pair["guided"], pair["control"]
        lines.append(
            f"| {a['id']}: {a['transcript']} | {a['median_seconds']:.3f} | {b['id']}: {b['transcript']} | {b['median_seconds']:.3f} |"
        )
    lines += [
        "",
        f"float64スコア再計算 {verification['float64_score_rows']:,}件、最大差 {verification['max_error']:.3g}。PCM特徴{verification['pcm_feature_probes']}件、停止位置・原区間・順序・校正・encoder出力を再検証。",
        "",
        "[全数値・対応付き区間・本人受入の遷移](evaluation-results.json)。[固定設計と制約](README.md)。",
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
                ]
            },
        },
    )
    print("Primary gates", gates, flush=True)


if __name__ == "__main__":
    main()
