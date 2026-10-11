"""All-arm reporting, paired uncertainty and preregistered development screening."""

import train as e

s, np, t, qr = e.s, e.np, e.t, e.qr
v = t.h.module("recording_ablation_verification", e.BASE / "verify.py")
ev = v.ev


def main():
    config = s.read_json(e.CONFIG)
    run = e.ROOT / config["run_directory"]
    design = e.verify(config, run)
    e.torch.set_num_threads(1)
    frozen = s.read_json(run / "threshold-freeze.json")
    s.checked(run / "design-freeze.json", frozen["design_sha256"])
    for name, digest in frozen["files"].items():
        s.checked(e.ROOT / name, digest)
    verification = v.audit(config, run, design)
    print("Score and schedule audit passed", verification, flush=True)
    old = design["source_config"]
    reports, draws, averages, contrasts = [], {}, [], []
    roc_checks = 0
    for corpus in ("JVS", "CV"):
        speakers = (
            t.jvs_data(old, "test")
            if corpus == "JVS"
            else s.read_json(e.ROOT / old["cv_run"] / "test/inputs.json")
        )["speakers"]
        lookup = {sp: i for i, sp in enumerate(speakers)}
        counts = qr.counts_for(speakers, config)
        for condition in config["conditions"] if corpus == "JVS" else ["CV"]:
            for name in ev.names(config):
                values = list(s.rows(run / "test" / name / f"{condition}-scores.jsonl"))
                for policy in old["policies"]:
                    p = policy["name"]
                    chosen = s.apply_policy(values, policy)
                    metrics, dd = qr.metrics(
                        chosen,
                        frozen["thresholds"][f"{name}/{corpus}"][p]["far_1pct"],
                        speakers,
                        counts,
                    )
                    # EER uncertainty for the primary JVS policy; CV rates include all inputs.
                    if corpus == "JVS" and p == "vowels5":
                        rows = [r for r in chosen if r["status"] == "scored"]
                        scores = np.array([r["score"] for r in rows])
                        qi = np.array([lookup[r["speaker_id"]] for r in rows])
                        ci = np.array([lookup[r["claimed_speaker_id"]] for r in rows])
                        dd["eer"] = s.original.weighted_eers(scores, qi, ci, counts)
                        for b in range(0, len(counts), len(counts) // 10):
                            weights = np.where(
                                qi == ci, counts[b, qi], counts[b, qi] * counts[b, ci]
                            )
                            assert (
                                abs(
                                    s.original.independent_eer(
                                        scores, qi == ci, weights
                                    )
                                    - dd["eer"][b]
                                )
                                < 1e-12
                            )
                            roc_checks += 1
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
                            frozen["thresholds"][f"{name}/JVS"][p]["far_1pct"],
                            speakers,
                            counts,
                        )[0]
                    reports.append(entry)
            for policy in old["policies"]:
                p = policy["name"]
                group = [
                    r
                    for r in reports
                    if (r["corpus"], r["condition"], r["policy"])
                    == (corpus, condition, p)
                ]
                point = {}
                for arm in ["original", *config["arms"]]:
                    names = (
                        ["original"]
                        if arm == "original"
                        else [f"{arm}-{seed}" for seed in config["seeds"]]
                    )
                    rows = [r for r in group if r["model"] in names]
                    point[arm] = {
                        key: float(np.mean([r["metrics"][key] for r in rows]))
                        for key in ("eer", "far", "frr", "coverage")
                    }
                    averages.append(
                        {
                            "corpus": corpus,
                            "condition": condition,
                            "policy": p,
                            "arm": arm,
                            **point[arm],
                        }
                    )
                for arm in config["arms"]:
                    if arm == "clean":
                        continue
                    for control in (
                        ["clean", "augmented"] if arm.startswith("drop") else ["clean"]
                    ):
                        cis = {}
                        for key in draws[
                            corpus, condition, p, f"{arm}-{config['seeds'][0]}"
                        ]:
                            if key not in ("eer", "far", "frr"):
                                continue
                            delta = np.mean(
                                [
                                    draws[corpus, condition, p, f"{arm}-{seed}"][key]
                                    - draws[corpus, condition, p, f"{control}-{seed}"][
                                        key
                                    ]
                                    for seed in config["seeds"]
                                ],
                                axis=0,
                            )
                            cis[key] = qr.interval(delta)
                        contrasts.append(
                            {
                                "corpus": corpus,
                                "condition": condition,
                                "policy": p,
                                "arm": arm,
                                "control": control,
                                "delta": {
                                    key: point[arm][key] - point[control][key]
                                    for key in ("eer", "far", "frr")
                                },
                                "ci95": cis,
                            }
                        )
            print(f"Summarized {corpus}/{condition}", flush=True)
    screens = []

    def row(arm, condition, policy):
        return next(
            r
            for r in averages
            if (r["arm"], r["condition"], r["policy"]) == (arm, condition, policy)
        )

    for arm in config["arms"]:
        if arm == "clean":
            continue
        checks = []
        for p in ("vowels5", "vowels4"):
            for condition, metric, bound in [
                ("clean", "far", 0.001),
                ("clean", "frr", 0.01),
                ("reverb-rt60-0.3", "frr", 0.01),
                ("CV", "far", 0.001),
                ("CV", "frr", 0.01),
                ("white-snr20", "eer", 0),
                ("white-snr10", "eer", 0),
            ]:
                delta = (
                    row(arm, condition, p)[metric] - row("clean", condition, p)[metric]
                )
                checks.append(
                    {
                        "policy": p,
                        "condition": condition,
                        "metric": metric,
                        "delta": delta,
                        "bound": bound,
                        "passed": bool(
                            delta < 0 if bound == 0 else delta <= bound + 1e-12
                        ),
                    }
                )
        screens.append(
            {"arm": arm, "passed": all(c["passed"] for c in checks), "checks": checks}
        )
    verification["independent_weighted_roc_checks"] = roc_checks
    result = {
        "config": config,
        "scope": "Previously observed development data; unadjusted exploratory confidence intervals",
        "per_model": reports,
        "seed_averages": averages,
        "paired_contrasts": contrasts,
        "candidate_screen": screens,
        "verification": verification,
    }
    s.write_json(run / "results.json", result)
    s.write_json(e.BASE / "evaluation-results.json", result)
    lines = [
        "# 録音条件の単独・除去実験",
        "",
        "観測済みデータの開発用比較。3 seed平均（ensembleではない）。各モデルのvalidation閾値を固定。EER/FAR/FRRの単位%。",
        "",
        "| 受付 | 評価条件 | 学習arm | EER | FAR | FRR |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in averages:
        lines.append(
            f"| {r['policy']} | {r['condition']} | {r['arm']} | {100 * r['eer']:.3f} | {100 * r['far']:.3f} | {100 * r['frr']:.3f} |"
        )
    lines += [
        "",
        "## 事前固定の候補スクリーニング",
        "",
        "| arm | 通過 | 不合格項目 |",
        "| --- | --- | --- |",
    ]
    for screen in screens:
        failed = [
            f"{c['policy']}/{c['condition']}/{c['metric']} {100 * c['delta']:+.3f}pp"
            for c in screen["checks"]
            if not c["passed"]
        ]
        lines.append(
            f"| {screen['arm']} | {screen['passed']} | {'; '.join(failed) or '—'} |"
        )
    lines += [
        "",
        f"float64スコア再計算 {verification['independent_float64_score_rows']:,}件（最大差 {verification['max_error']:.3g}）、重み付きROCの別計算 {roc_checks:,}件、元PCM特徴36件とpair列・加工枠の整合性を検証。",
        "",
        "全seed・対応付き信頼区間・校正閾値は[evaluation-results.json](evaluation-results.json)、設計と制約は[README](README.md)を参照。単独では加工率50%、除去では残り4条件を元の約10%ずつ保持する。区間は多重比較未補正で、採用の確証ではない。",
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
    print(
        "Candidate screening:", [(r["arm"], r["passed"]) for r in screens], flush=True
    )


if __name__ == "__main__":
    main()
