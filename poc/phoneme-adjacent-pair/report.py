"""Publish the prespecified pair-only contrast and coverage limitations."""

from pair_common import ARMS, BASE, np, read, write


def report(config, run):
    rows = read(run / "validation-metrics.json") + read(run / "test-metrics.json")
    summary = []
    for split in ("validation", "test"):
        for c in [*config["corpora"], "macro"]:
            chosen = [
                r
                for r in rows
                if r["split"] == split
                and r["role"] == "verification"
                and (c == "macro" or r["corpus"] == c)
            ]
            arms = {
                a: {
                    k: float(np.mean([r[k] for r in chosen if r["arm"] == a]))
                    for k in ("eer", "far", "frr", "coverage")
                }
                for a in ARMS
            }
            deltas = [
                float(
                    100
                    * (
                        np.mean(
                            [
                                r["eer"]
                                for r in chosen
                                if r["seed"] == seed and r["arm"] == "joined"
                            ]
                        )
                        - np.mean(
                            [
                                r["eer"]
                                for r in chosen
                                if r["seed"] == seed and r["arm"] == "split"
                            ]
                        )
                    )
                )
                for seed in config["seeds"]
            ]
            summary.append(
                {
                    "dataset_split": split,
                    "corpus": c,
                    **arms,
                    "delta_pp": float(np.mean(deltas)),
                    "seed_deltas_pp": deltas,
                    "improved": sum(v < -1e-10 for v in deltas),
                    "tied": sum(abs(v) <= 1e-10 for v in deltas),
                    "worsened": sum(v > 1e-10 for v in deltas),
                }
            )
    selection = read(run / "pair-selection.json")
    pair_rows = read(run / "validation-pairs.json") + read(run / "test-pairs.json")
    per_pair = []
    for c in config["corpora"]:
        for p in selection["selected"]:
            chosen = [
                r
                for r in pair_rows
                if r["split"] == "test" and r["corpus"] == c and r["pair"] == p
            ]
            values = {a: [r["eer"] for r in chosen if r["arm"] == a] for a in ARMS}
            means = {
                a: None if all(v is None for v in vv) else float(np.mean(vv))
                for a, vv in values.items()
            }
            assert len({r["queries"] for r in chosen}) == 1
            per_pair.append(
                {
                    "corpus": c,
                    "pair": p,
                    "queries": chosen[0]["queries"],
                    "total_queries": chosen[0]["total_queries"],
                    **means,
                }
            )
    payload = {
        "config": config,
        "pair_selection": {
            "selected": selection["selected"],
            "candidates": len(selection["support"]),
            "eligible": len(selection["eligible_ranked"]),
            "selected_support": {
                p: selection["support"][p] for p in selection["selected"]
            },
        },
        "summary": summary,
        "conditions": rows,
        "pair_summary": per_pair,
        "pair_conditions": pair_rows,
        "bootstrap": read(run / "bootstrap.json"),
        "audit": read(run / "independent-audit.json"),
        "training": {
            str(seed): {a: read(run / f"seed-{seed}/{a}/summary.json") for a in ARMS}
            for seed in config["seeds"]
        },
    }
    lines = [
        "# 隣接する子音→母音の結合比較",
        "",
        "分離＝同じC/Vを別々に特徴抽出し、共有encoderの正規化embeddingを等重み平均して再正規化。結合＝同じ連続C+V区間を一度に特徴抽出。同じ音声・初期値・学習token列・モデル規模で3 seedを比較した。",
        "",
        "対象: "
        + ", ".join(selection["selected"])
        + "。train/validationの出現数だけで事前固定。",
        "",
        "| 集合 | コーパス | 分離 EER | 結合 EER | 結合−分離 | 共通採点率 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        assert r["split"]["coverage"] == r["joined"]["coverage"]
    for r in summary:
        lines.append(
            f"| {r['dataset_split']} | {r['corpus']} | {100 * r['split']['eer']:.3f}% | {100 * r['joined']['eer']:.3f}% | {r['delta_pp']:+.3f} pp | {100 * r['split']['coverage']:.3f}% |"
        )
    lines += [
        "",
        "## testの差と対応付き話者bootstrap",
        "",
        "| コーパス | 結合−分離 | 改善 / 同値 / 悪化 seed | 95%区間 |",
        "| --- | ---: | --- | --- |",
    ]
    for r in summary:
        if r["dataset_split"] != "test":
            continue
        lo, hi = payload["bootstrap"]["eer_delta_ci95_pp"][r["corpus"]]
        lines.append(
            f"| {r['corpus']} | {r['delta_pp']:+.3f} pp | {r['improved']} / {r['tied']} / {r['worsened']} | [{lo:+.3f}, {hi:+.3f}] pp |"
        )
    lines += [
        "",
        "## validation FAR1%閾値を固定したtest性能",
        "",
        "| コーパス | 条件 | FAR | 全入力FRR |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in summary:
        if r["dataset_split"] == "test":
            for arm in ARMS:
                lines.append(
                    f"| {r['corpus']} | {arm} | {100 * r[arm]['far']:.3f}% | {100 * r[arm]['frr']:.3f}% |"
                )
    lines += [
        "",
        "## ペアごとの副評価",
        "",
        "各行の対象発話が異なるため、ペア間の絶対EERの順位を音素の優劣と解釈しない。test結果による再選別はしない。",
        "",
        "| コーパス | ペア | 対象発話 | 分離 EER | 結合 EER |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in per_pair:
        formatted = ["未採点" if r[a] is None else f"{100 * r[a]:.3f}%" for a in ARMS]
        lines.append(
            f"| {r['corpus']} | {r['pair']} | {r['queries']} / {r['total_queries']} | {formatted[0]} | {formatted[1]} |"
        )
    lines += [
        "",
        "60k固定最終モデル、学習/照合とも各条件に合わせる。統合はペア種類ごとのcosineの等重み平均で、以前の全36音素Transformerとは入力・採点対象・学習方式が異なり、過去のEERとの直接比較はしない。",
        "",
        "両音素が既存QCを通過し各30ms以上、合計250ms以下の自然な隣接区間に限定する。音声の延長・crop・別区間の接ぎ合わせは行わない。平均・標準偏差は時間順序を明示的に保持せず、差には特徴抽出とembedding集約の違いが含まれる。",
        "",
        "全testは既観測。CIは固定モデル・固定ペア集合を条件とし、学習/選択の変動を含まない。EERは採点可能分、FAR/FRRは未採点を含む。JVS別テキストとseed別の全指標はJSONへ保存。標準モデルは変更しない。",
        "",
        "[再現手順](README.md)・[全指標と監査](evaluation-results.json)",
        "",
    ]
    write(run / "results.json", payload, immutable=True)
    write(BASE / "evaluation-results.json", payload)
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)
