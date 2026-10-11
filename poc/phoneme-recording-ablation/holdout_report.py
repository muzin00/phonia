"""Audit and report the frozen reverb-only candidate on new CV speakers."""

import train as e

s, np, t, qr = e.s, e.np, e.t, e.qr
h = t.h.module("recording_ablation_fresh_holdout", e.BASE / "holdout.py")


def main():
    config = s.read_json(h.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = h.verify(config, run)
    frozen = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", frozen["design_sha256"])
    for name, digest in frozen["files"].items():
        s.checked(e.ROOT / name, digest)
    e.torch.set_num_threads(1)
    reports, draws = [], {}
    checks, max_error, replay_checks = 0, 0.0, 0
    selected = s.read_json(run / "selection.json")
    previous = s.read_json(e.ROOT / config["prior_holdout"] / "selection.json")
    training = s.read_json(e.ROOT / config["cv_source"] / "selection.json")
    assert not {r["client_id"] for r in selected} & {
        r["client_id"] for r in previous + training
    }
    old_pcm = {
        r["source_sha256"]
        for split in ("train", "validation", "test")
        for r in s.rows(e.ROOT / config["encoder_run"] / f"{split}-segments.jsonl")
    }
    old_pcm.update(
        r["source_sha256"]
        for split in ("validation", "test")
        for r in s.rows(e.ROOT / config["prior_holdout"] / split / "utterances.jsonl")
    )
    for split in ("validation", "test"):
        data = s.read_json(run / split / "inputs.json")
        speakers = data["speakers"]
        counts = qr.counts_for(speakers, config)
        utterances = list(s.rows(run / split / "utterances.jsonl"))
        for r in utterances:
            if not r["excluded_duplicate"]:
                assert r["source_sha256"] not in old_pcm
        old_pcm.update(r["source_sha256"] for r in utterances)
        features = np.load(run / split / "features.npy", allow_pickle=False)
        segments = list(s.rows(run / split / "segments.jsonl"))
        pipe = t.h.primitives.pipeline(design["extract_config"])
        for row in segments[:: max(1, len(segments) // 20)][:20]:
            raw = e.a.read_wave(e.ROOT / row["source_file"])
            feature = t.h.primitives.pool_features(
                pipe, raw[row["start_frame"] : row["end_frame"]], row["segment_id"]
            )
            np.testing.assert_array_equal(feature, features[row["cache_index"]])
            replay_checks += 1
        for name in h.names(config):
            vectors = np.load(
                run / split / f"{name}-embeddings.npy", allow_pickle=False
            )
            np.testing.assert_array_equal(
                s.original.model_embeddings(h.load_model(config, name), features),
                vectors,
            )
            expected = {
                (r["query_id"], r["claimed_speaker_id"]): r
                for r in h.candidates(data, vectors.astype(np.float64))
            }
            values = list(s.rows(run / split / f"{name}-scores.jsonl"))
            assert len(values) == 27000
            for row in values:
                ref = expected[row["query_id"], row["claimed_speaker_id"]]
                assert (
                    row["status"] == ref["status"]
                    and row["used_phones"] == ref["used_phones"]
                )
                if row["score"] is not None:
                    max_error = max(max_error, abs(row["score"] - ref["score"]))
                checks += 1
            for policy in config["policies"]:
                p = policy["name"]
                chosen = s.apply_policy(values, policy)
                if split == "validation":
                    assert (
                        s.original.thresholds(chosen) == frozen["thresholds"][name][p]
                    )
                metrics, dd = qr.metrics(
                    chosen, frozen["thresholds"][name][p]["far_1pct"], speakers, counts
                )
                reports.append(
                    {"split": split, "model": name, "policy": p, "metrics": metrics}
                )
                draws[split, name, p] = dd
    assert max_error < 2e-6
    contrasts = []
    for policy in config["policies"]:
        p = policy["name"]
        means = {}
        for arm in config["arms"]:
            rows = [
                r["metrics"]
                for r in reports
                if r["split"] == "test"
                and r["policy"] == p
                and r["model"].startswith(arm + "-")
            ]
            means[arm] = {
                key: float(np.mean([r[key] for r in rows]))
                for key in ("eer", "far", "frr", "coverage")
            }
        cis = {
            key: qr.interval(
                np.mean(
                    [
                        draws["test", f"single5-{seed}", p][key]
                        - draws["test", f"clean-{seed}", p][key]
                        for seed in config["seeds"]
                    ],
                    axis=0,
                )
            )
            for key in ("far", "frr")
        }
        passed = (
            cis["frr"][1] < 0
            and cis["far"][1] <= config["research_gate"]["far_difference_upper95_max"]
        )
        contrasts.append(
            {
                "policy": p,
                "means": means,
                "candidate_minus_control": {
                    k: means["single5"][k] - means["clean"][k]
                    for k in ("eer", "far", "frr")
                },
                "ci95": cis,
                "gate_passed": bool(passed),
            }
        )
    verification = {
        "status": "passed",
        "independent_float64_score_rows": checks,
        "max_error": max_error,
        "pcm_feature_probes": replay_checks,
        "all_model_embeddings_replayed": True,
        "speaker_and_pcm_disjointness": True,
        "calibration_replayed": True,
    }
    result = {
        "config": config,
        "preparation": {
            split: s.read_json(run / split / "preparation.json")
            for split in ("validation", "test")
        },
        "per_model": reports,
        "test_contrasts": contrasts,
        "primary_passed": contrasts[0]["gate_passed"],
        "verification": verification,
    }
    s.write_json(run / "results.json", result)
    s.write_json(e.BASE / "holdout-results.json", result)
    lines = [
        "# 残響単独候補：新規Common Voice評価",
        "",
        "新規校正30話者・test30話者、各20登録/30照合。候補と対照は3 seed平均。モデルと閾値はtest処理前に固定。単位%。",
        "",
        "| 受付 | 学習 | EER | FAR | FRR | 採点率 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in contrasts:
        for arm, m in r["means"].items():
            lines.append(
                f"| {r['policy']} | {arm} | {100 * m['eer']:.3f} | {100 * m['far']:.3f} | {100 * m['frr']:.3f} | {100 * m['coverage']:.3f} |"
            )
    lines += [
        "",
        "| 受付 | FAR差 pp [95%区間] | FRR差 pp [95%区間] | 基準通過 |",
        "| --- | --- | --- | --- |",
    ]
    for r in contrasts:
        cells = [
            f"{100 * r['candidate_minus_control'][k]:+.3f} [{100 * r['ci95'][k][0]:+.3f}, {100 * r['ci95'][k][1]:+.3f}]"
            for k in ("far", "frr")
        ]
        lines.append(f"| {r['policy']} | {' | '.join(cells)} | {r['gate_passed']} |")
    lines += [
        "",
        f"主評価通過: {result['primary_passed']}。スコア再計算 {checks:,}件、最大差 {max_error:.3g}。PCM特徴再計算{replay_checks}件、全モデル出力・校正の再現・話者ラベルと採用PCMの非重複を検証。",
        "",
        "基準は[事前設計](holdout-design.md)、全seed・閾値・前処理結果は[JSON](holdout-results.json)。",
        "",
    ]
    (e.BASE / "holdout-results.md").write_text("\n".join(lines))
    s.write_json(
        run / "completion-verification.json",
        {
            **verification,
            "files": {
                t.rel(p): s.sha256_file(p)
                for p in [
                    run / "results.json",
                    e.BASE / "holdout-results.json",
                    e.BASE / "holdout-results.md",
                ]
            },
        },
    )
    print(contrasts, flush=True)


if __name__ == "__main__":
    main()
