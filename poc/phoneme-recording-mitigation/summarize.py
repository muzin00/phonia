"""Paired summaries and waveform/model replay for the frontend experiment."""

import experiment as e

s, np, t, qr = e.s, e.np, e.t, e.qr


def pair(original, changed, old_threshold, new_threshold, speakers, config):
    counts = qr.counts_for(speakers, config)
    before, b = qr.metrics(original, old_threshold, speakers, counts)
    after, a = qr.metrics(changed, new_threshold, speakers, counts)
    before["unscored_queries"] = before["queries"] - before["scored_queries"]
    after["unscored_queries"] = after["queries"] - after["scored_queries"]
    return {
        "original": before,
        "changed": after,
        "difference_ci95": {k: qr.interval(a[k] - b[k]) for k in ("far", "frr")},
    }


def audit(config, run, design):
    source = e.ROOT / config["quality_run"]
    old = design["source_config"]
    model, _ = s.original.load_model(
        e.ROOT / old["encoder_run"] / "trials" / old["encoder_trial"]
    )
    pipe = t.h.primitives.pipeline(design["extract_config"])
    count = 0
    maximum = 0.0
    probes = 0
    for split in ("validation", "test"):
        selected = {
            x["query"]["query_id"]
            for x in s.read_json(source / "selection.json")["stress"][split]
        }
        for corpus in ("JVS", "CV"):
            if corpus == "JVS":
                data = t.jvs_data(old, split)
                data = {
                    **data,
                    "queries": [
                        q for q in data["queries"] if q["query_id"] in selected
                    ],
                }
                segments = list(
                    s.rows(e.ROOT / old["encoder_run"] / f"{split}-segments.jsonl")
                )
            else:
                data = s.read_json(e.ROOT / old["cv_run"] / split / "inputs.json")
                segments = list(
                    s.rows(e.ROOT / old["cv_run"] / split / "segments.jsonl")
                )
            cached = np.load(
                run / split / corpus / "normalized.npz", allow_pickle=False
            )
            vectors = cached["embeddings"].astype(np.float64)
            refs = {
                (r["query_id"], r["claimed_speaker_id"]): r
                for r in t.h.candidates(data, vectors)
            }
            for actual in s.rows(run / split / corpus / "normalized-scores.jsonl"):
                reference = refs[actual["query_id"], actual["claimed_speaker_id"]]
                assert reference["used_phones"] == actual["used_phones"]
                if actual["score"] is not None:
                    maximum = max(maximum, abs(actual["score"] - reference["score"]))
                count += 1
            ids = cached["indices"][:: max(1, len(cached["indices"]) // 12)]
            np.testing.assert_allclose(
                s.original.model_embeddings(model, cached["features"][ids]),
                vectors[ids],
                atol=1e-6,
                rtol=0,
            )
            for i in ids[:3]:
                row = segments[i]
                y, _ = e.normalize(e.a.read_wave(e.ROOT / row["source_file"]), config)
                feature = t.h.primitives.pool_features(
                    pipe, y[row["start_frame"] : row["end_frame"]], row["segment_id"]
                )
                np.testing.assert_array_equal(feature, cached["features"][i])
                probes += 1
            if corpus == "JVS":
                registration = t.profiles(data, vectors)
                for condition in old["conditions"]:
                    refs = {}
                    for number, path in enumerate(
                        sorted((source / split / condition).glob("*/prepared.json"))
                    ):
                        prepared = s.read_json(path)
                        qid = prepared["query"]["query_id"]
                        cache = np.load(
                            run / split / "stress" / condition / f"{qid}.npz",
                            allow_pickle=False,
                        )
                        rr = t.score(
                            prepared["query"],
                            data,
                            cache["embeddings"].astype(np.float64),
                            prepared["fixed"],
                            registration,
                            "fixed",
                            split,
                        )
                        refs.update(
                            {(r["query_id"], r["claimed_speaker_id"]): r for r in rr}
                        )
                        if number == 0:
                            np.testing.assert_array_equal(
                                s.original.model_embeddings(model, cache["features"]),
                                cache["embeddings"],
                            )
                            y, _ = e.normalize(
                                e.a.read_wave(
                                    e.ROOT / prepared["metadata"]["source_file"]
                                ),
                                config,
                            )
                            row = prepared["fixed"][0]
                            np.testing.assert_array_equal(
                                t.h.primitives.pool_features(
                                    pipe,
                                    y[row["start_frame"] : row["end_frame"]],
                                    row["segment_id"],
                                ),
                                cache["features"][0],
                            )
                            probes += 1
                    for actual in s.rows(
                        run / split / "stress" / condition / "scores.jsonl"
                    ):
                        reference = refs[
                            actual["query_id"], actual["claimed_speaker_id"]
                        ]
                        assert reference["used_phones"] == actual["used_phones"]
                        maximum = max(
                            maximum, abs(actual["score"] - reference["score"])
                        )
                        count += 1
                # Independently select raw intervals and reconstruct every duration score.
                original_vectors = np.load(
                    e.ROOT / old["jvs_run"] / f"{split}-embeddings.npy"
                ).astype(np.float64)
                original_reg = t.profiles(data, original_vectors)
                qmap = {q["query_id"]: q for q in data["queries"]}
                for seconds in ["full", *config["duration_seconds"]]:
                    refs = {}
                    for window in s.read_json(
                        run / split / f"duration-{seconds}-windows.json"
                    ):
                        q = qmap[window["query_id"]]
                        eligible = sorted(
                            i for ids in q["groups"].values() for i in ids
                        )
                        if seconds != "full":
                            eligible = [
                                i
                                for i in eligible
                                if segments[i]["raw_start_frame"]
                                >= window["first_frame"]
                                and segments[i]["raw_end_frame"] <= window["last_frame"]
                            ]
                        assert eligible == window["indices"]
                        rr = t.score(
                            q,
                            data,
                            original_vectors[eligible],
                            [segments[i] for i in eligible],
                            original_reg,
                            "fixed",
                            split,
                        )
                        refs.update(
                            {(r["query_id"], r["claimed_speaker_id"]): r for r in rr}
                        )
                    for actual in s.rows(
                        run / split / f"duration-{seconds}-scores.jsonl"
                    ):
                        ref = refs[actual["query_id"], actual["claimed_speaker_id"]]
                        assert actual["used_phones"] == ref["used_phones"]
                        if actual["score"] is not None:
                            maximum = max(maximum, abs(actual["score"] - ref["score"]))
                        count += 1
    assert maximum < 2e-6
    return {
        "status": "passed",
        "score_reconstructions": count,
        "maximum_float64_error": maximum,
        "pcm_feature_probes": probes,
    }


def main():
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    s.torch.set_num_threads(1)
    old = design["source_config"]
    source = e.ROOT / config["quality_run"]
    frozen = s.read_json(run / "threshold-freeze.json")
    points = frozen["thresholds"]
    for name, digest in frozen["files"].items():
        s.checked(e.ROOT / name, digest)
    print("Auditing normalization and duration scores", flush=True)
    verification = audit(config, run, design)
    reports = []
    for corpus in ("JVS", "CV"):
        speakers = (
            t.jvs_data(old, "test")
            if corpus == "JVS"
            else s.read_json(e.ROOT / old["cv_run"] / "test/inputs.json")
        )["speakers"]
        conditions = old["conditions"] if corpus == "JVS" else ["clean"]
        for condition in conditions:
            before = (
                list(s.rows(source / "test" / condition / "fixed-scores.jsonl"))
                if corpus == "JVS"
                else list(s.rows(run / "test/CV/original-scores.jsonl"))
            )
            after = (
                list(s.rows(run / "test/stress" / condition / "scores.jsonl"))
                if corpus == "JVS"
                else list(s.rows(run / "test/CV/normalized-scores.jsonl"))
            )
            for policy in old["policies"]:
                p = policy["name"]
                bb = s.apply_policy(before, policy)
                aa = s.apply_policy(after, policy)
                result = pair(
                    bb,
                    aa,
                    points[f"{corpus}/original"][p]["far_1pct"],
                    points[f"{corpus}/normalized"][p]["far_1pct"],
                    speakers,
                    config,
                )
                if corpus == "CV":
                    result["jvs_transfer"] = pair(
                        bb,
                        aa,
                        points["JVS/original"][p]["far_1pct"],
                        points["JVS/normalized"][p]["far_1pct"],
                        speakers,
                        config,
                    )
                reports.append(
                    {"corpus": corpus, "condition": condition, "policy": p, **result}
                )
    durations = []
    speakers = t.jvs_data(old, "test")["speakers"]
    full = list(s.rows(run / "test/duration-full-scores.jsonl"))
    for seconds in config["duration_seconds"]:
        values = list(s.rows(run / "test" / f"duration-{seconds}-scores.jsonl"))
        for policy in old["policies"]:
            p = policy["name"]
            b = s.apply_policy(full, policy)
            a = s.apply_policy(values, policy)
            point = points["JVS/original"][p]["far_1pct"]
            scored_ids = {
                x["query_id"] for x in a if x["is_genuine"] and x["status"] == "scored"
            }
            matched = (
                pair(
                    [x for x in b if x["query_id"] in scored_ids],
                    [x for x in a if x["query_id"] in scored_ids],
                    point,
                    point,
                    speakers,
                    config,
                )
                if scored_ids
                else None
            )
            durations.append(
                {
                    "seconds": seconds,
                    "policy": p,
                    **pair(b, a, point, point, speakers, config),
                    "same_scored_queries": matched,
                }
            )
    result = {
        "config": config,
        "normalization": reports,
        "duration": durations,
        "verification": verification,
    }
    s.write_json(run / "results.json", result)
    s.write_json(e.BASE / "evaluation-results.json", result)
    lines = [
        "# 音量前処理・発話長の比較結果",
        "",
        "既存の観測済み話者での診断。方式別のclean validation閾値を固定。全入力FAR/FRRと採点可能入力EER。単位%。",
        "",
        "## 音量前処理",
        "",
        "| コーパス/条件 | 受付 | 元EER | 正規化EER | 元FAR | 正規化FAR | 元FRR | 正規化FRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for x in reports:
        b, a = x["original"], x["changed"]
        lines.append(
            f"| {x['corpus']}/{x['condition']} | {x['policy']} | {100 * b['eer']:.3f} | {100 * a['eer']:.3f} | {100 * b['far']:.3f} | {100 * a['far']:.3f} | {100 * b['frr']:.3f} | {100 * a['frr']:.3f} |"
        )
    lines += [
        "",
        "## 発話長：同一録音のnested window",
        "",
        "| 長さ | 受付 | 共通発話数 | 採点不能 | coverage | EER | FAR | FRR | 元全文FRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for x in durations:
        b, a = x["original"], x["changed"]
        eer = f"{100 * a['eer']:.3f}" if a["eer"] is not None else "—"
        lines.append(
            f"| {x['seconds']}秒 | {x['policy']} | {a['queries']} | {a['unscored_queries']} | {100 * a['coverage']:.3f} | {eer} | {100 * a['far']:.3f} | {100 * a['frr']:.3f} | {100 * b['frr']:.3f} |"
        )
    lines += [
        "",
        "## 短い窓でも採点できるqueryに限定した比較",
        "",
        "| 長さ | 受付 | 発話数 | 元全文EER | 短窓EER | 元全文FRR | 短窓FRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for x in durations:
        if x["same_scored_queries"]:
            b, a = (
                x["same_scored_queries"]["original"],
                x["same_scored_queries"]["changed"],
            )
            lines.append(
                f"| {x['seconds']}秒 | {x['policy']} | {a['queries']} | {100 * b['eer']:.3f} | {100 * a['eer']:.3f} | {100 * b['frr']:.3f} | {100 * a['frr']:.3f} |"
            )
    lines += [
        "",
        f"独立float64スコア再計算 {verification['score_reconstructions']:,}件、最大差 {verification['maximum_float64_error']:.3g}。PCM特徴再計算 {verification['pcm_feature_probes']}件とモデル再推論を確認。",
        "",
        "[区間・閾値・全数値](evaluation-results.json)。境界既知の診断で、実際の短音声再alignmentや正規化後の登録再alignmentは含まない。新規独立testの結果ではない。",
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
    print("Completed frontend and duration report", flush=True)


if __name__ == "__main__":
    main()
