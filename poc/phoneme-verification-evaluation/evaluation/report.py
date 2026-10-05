"""Reproducible result tables and exported ROC/DET figures."""

from statistics import NormalDist

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .storage import load_document, read_rows


def percent(value):
    return f"{value * 100:.3f}%"


def build_report(output):
    validation = load_document(output / "metrics/validation.json")
    test = load_document(output / "metrics/test.json")
    plan = load_document(output / "evaluation-plan.json")
    primary = test["conditions"]["verification/n10/fused"]
    point = primary["operating_points"]["far_1pct"]
    ci = primary["bootstrap_95pct"]["operating_points"]["far_1pct"]
    lines = [
        "# Phase 6: JVS発話単位の本人照合評価結果",
        "",
        f"実行成果物: `{output}`。凍結した計画: `{output / 'evaluation-plan.json'}`。",
        f"計画のSHA-256: `{plan['protocol_sha256']}`（評価protocolファイル）。",
        "",
        "## 主条件",
        "",
        "登録各母音10区間、1発話の全利用可能母音区間、5母音の等重み統合。validationでFAR 1%を目標に決めた閾値をtestへ固定適用した。",
        "",
        f"- 判定閾値: `{point['threshold']}`",
        f"- test他人受入率: **{percent(point['far'])}**（{point['false_accepts']}/{point['impostor']}）。95%話者bootstrap区間: {percent(ci['far'][0])}〜{percent(ci['far'][1])}。",
        f"- test本人拒否率（scoreあり）: **{percent(point['frr'])}**（{point['false_rejects']}/{point['genuine']}）。95%区間: {percent(ci['frr'][0])}〜{percent(ci['frr'][1])}。",
        f"- score取得率: **{percent(primary['score_coverage'])}**（{primary['scored_queries']}/{primary['all_queries']}発話）。",
        f"- 入力不足も拒否に含めた本人拒否率: **{percent(point['all_input_frr'])}**（{point['false_rejects'] + point['no_score_genuine']}/{point['all_genuine']}）。95%区間: {percent(ci['all_input_frr'][0])}〜{percent(ci['all_input_frr'][1])}。",
        f"- 統合scoreのEER: **{percent(primary['pooled_eer'])}**。95%区間: {percent(primary['bootstrap_95pct']['pooled_eer'][0])}〜{percent(primary['bootstrap_95pct']['pooled_eer'][1])}。",
        "",
        "## 統合scoreの全条件",
        "",
        "すべて同じ登録数のvalidation / verificationから固定したFAR 1%動作点。EERは閾値非依存の分離診断値。",
        "",
        "| split | role | 登録区間/母音 | score取得率 | EER | 実測FAR | 条件付きFRR | 全入力FRR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, metrics in (("validation", validation), ("test", test)):
        for role in ("verification", "cross_text_verification"):
            for count in (1, 5, 10):
                item = metrics["conditions"][f"{role}/n{count}/fused"]
                p = item["operating_points"]["far_1pct"]
                lines.append(
                    f"| {split} | {role} | {count} | {percent(item['score_coverage'])} | {percent(item['pooled_eer'])} | {percent(p['far'])} | {percent(p['frr'])} | {percent(p['all_input_frr'])} |"
                )
    lines.extend(
        [
            "",
            "## 母音別の補助結果",
            "",
            "母音ごとの平均scoreを同じscoreありquery集合で評価する。母音別の単純平均EERと統合scoreのEERは別物であり、Phase 3の区間単位EERとも直接比較しない。",
            "",
            "| role | 登録区間/母音 | score種類 | validation EER | test EER | test FAR | test FRR |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role in ("verification", "cross_text_verification"):
        for count in (1, 5, 10):
            for kind in ("a", "i", "u", "e", "o"):
                key = f"{role}/n{count}/{kind}"
                item = test["conditions"][key]
                p = item["operating_points"]["far_1pct"]
                lines.append(
                    f"| {role} | {count} | {kind} | {percent(validation['conditions'][key]['pooled_eer'])} | {percent(item['pooled_eer'])} | {percent(p['far'])} | {percent(p['frr'])} |"
                )
    lines.extend(
        [
            "",
            "## 解釈と限界",
            "",
            "- 条件付きFAR/FRRはscoreあり試行だけを分母にする。全入力FAR/FRRにはno_scoreも含め、no_scoreは受け入れない。",
            "- validationの目標FARをtestで保証するものではない。testの実測値を見ても閾値を調整していない。",
            "- 95%区間は15話者を単位とする10,000回の共有bootstrap。閾値は固定しており、学習・較正・収録条件の不確かさをすべて含まない。",
            "- JVSの統制収録内で別発話を評価した。収録セッション、別日、別端末、雑音環境の変化への性能は今回確認していない。",
            "- Phase 3で同じtest話者の結果は既に観測済み。今回は固定方式の発話単位・統合scoreの後続評価であり、新しいcorpusによる独立検証ではない。",
            "- 製品向けの合否基準とベースラインは未設定。既存方式との比較はPhase 7で同じtrialを使用する。",
            "",
            "## 再現と成果物",
            "",
            "`evaluation-plan.json`、`prepared.json`、`checks.json`、`execution.json`で条件・入力・実装・環境・実行のhashを追跡する。登録profile、query、全trial、score、validation閾値、全36条件/splitの集計を保存した。",
            "",
            "`metrics/`にはFAR 0.1%とEER動作点、母音別・話者別の結果とtest信頼区間を保存している。`phase5-results/*.jsonl.gz`はscoreあり全trialのPhase 5結果JSONをchecksum付きで保存する。",
            "",
            "`curves/`には曲線データとROC・DET・score分布のPNG/PDFを保存する。DETの表示だけは0/1を有限範囲へclipし、保存した曲線データと集計は変更していない。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_figures(output)


def plot_figures(output):
    curves = list(read_rows(output / "curves/test.jsonl"))
    normal = NormalDist()
    ticks = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8])
    positions = [normal.inv_cdf(float(t)) for t in ticks]
    for role in ("verification", "cross_text_verification"):
        for name in ("roc", "det"):
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            for count in (1, 5, 10):
                curve = next(
                    c for c in curves if c["condition"] == f"{role}/n{count}/fused"
                )
                far, frr = np.asarray(curve["far"]), np.asarray(curve["frr"])
                label = f"Enrollment {count}/vowel; EER {curve['eer'] * 100:.2f}%"
                if name == "roc":
                    ax.plot(far * 100, (1 - frr) * 100, label=label)
                else:
                    x = [normal.inv_cdf(float(v)) for v in np.clip(far, 1e-5, 1 - 1e-5)]
                    y = [normal.inv_cdf(float(v)) for v in np.clip(frr, 1e-5, 1 - 1e-5)]
                    ax.plot(x, y, label=label)
            if name == "roc":
                ax.set(
                    xlabel="False acceptance rate (%)",
                    ylabel="True acceptance rate (%)",
                    xlim=(0, 10),
                    ylim=(0, 100),
                )
            else:
                labels = [f"{t * 100:g}" for t in ticks]
                ax.set_xticks(positions, labels)
                ax.set_yticks(positions, labels)
                ax.set(
                    xlabel="False acceptance rate (%)",
                    ylabel="False rejection rate (%)",
                    xlim=(positions[0], positions[-1]),
                    ylim=(positions[0], positions[-1]),
                )
            ax.set_title(f"Test {role.replace('_', ' ')} — fused score")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"curves/{role}-{name}.{extension}", dpi=180)
            plt.close(fig)
    rows = [
        r
        for r in read_rows(output / "scores/test.jsonl")
        if r["status"] == "scored" and r["enrollment_count"] == 10
    ]
    thresholds = load_document(output / "thresholds/validation.json")
    for role in ("verification", "cross_text_verification"):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for genuine, label in ((True, "Genuine"), (False, "Impostor")):
            values = [
                r["scores"]["fused"]
                for r in rows
                if r["role"] == role and r["is_genuine"] == genuine
            ]
            ax.hist(
                values,
                bins=np.linspace(-1, 1, 81),
                density=True,
                alpha=0.55,
                label=f"{label} (n={len(values)})",
            )
        ax.axvline(
            thresholds["conditions"]["n10/fused"]["far_1pct"]["threshold"],
            color="black",
            linestyle="--",
            label="Fixed validation FAR 1% threshold",
        )
        ax.set(
            xlabel="Fused cosine score",
            ylabel="Density",
            title=f"Test {role.replace('_', ' ')} — enrollment 10/vowel",
        )
        ax.legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"curves/{role}-scores.{extension}", dpi=180)
        plt.close(fig)
