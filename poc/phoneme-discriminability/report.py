"""Publish phone-level verification, score distributions and speaker diagnostics."""

import csv
import html

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import study
from support import BASE, PHONES, ROLES, ROOT, read_json, write_json


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    )


def html_table(headers, rows, heatmap=False):
    text = (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
    )
    for row in rows:
        text += "<tr>"
        for index, value in enumerate(row):
            style = ""
            if heatmap and index:
                saturation = min(float(str(value).rstrip("%")), 50) / 50
                style = f" style='background:hsl({120 * (1 - saturation):.0f} 60% 91%)'"
            text += f"<td{style}>{html.escape(str(value))}</td>"
        text += "</tr>"
    return text + "</tbody></table>"


def distribution_plot(test, role, destination):
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for phone, axis in zip(PHONES, axes.flat):
        cell = test["conditions"][f"{phone}/{role}"]
        for group, color in (("genuine", "#2563eb"), ("impostor", "#e11d48")):
            dist = cell["diagnostics"]["score_separation"][group]
            edges = np.array(dist["histogram_edges"])
            density = (
                np.array(dist["histogram_counts"]) / dist["count"] / np.diff(edges)
            )
            axis.stairs(density, edges, label=group, color=color, linewidth=1.7)
        axis.set_title(f"/{phone}/  EER {100 * cell['pooled_eer']:.2f}%")
        axis.set_xlim(-1, 1)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("Cosine score")
    figure.suptitle(f"Fixed 50 ms query, 10 x 50 ms enrollment | {role}")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    with matplotlib.rc_context({"svg.hashsalt": "phoneme-discriminability-v1"}):
        figure.savefig(destination, metadata={"Date": None})
    plt.close(figure)


def main():
    config = read_json(study.CONFIG)
    run = ROOT / config["run_directory"]
    study.frozen(config, run, "test")
    validation, test = (
        read_json(run / f"{split}-metrics.json") for split in ("validation", "test")
    )
    inputs = {
        split: read_json(run / f"{split}-inputs.json")
        for split in ("validation", "test")
    }
    audits = {}
    for split, prepared_inputs in inputs.items():
        study.load_groups(config, run, split)
        audits[split] = study.validate_plans(prepared_inputs, config)
    rows, strict, separation, speaker_rows = [], [], [], {}
    csv_rows, speaker_csv = [], []
    for split, measured in (("validation", validation), ("test", test)):
        for key, cell in measured["conditions"].items():
            phone, role = key.split("/")
            for point, rates in cell["operating_points"].items():
                csv_rows.append(
                    {
                        "split": split,
                        "phone": phone,
                        "role": role,
                        "operating_point": point,
                        "queries": cell["queries"],
                        "eer": cell["pooled_eer"],
                        **rates,
                    }
                )
            for speaker, d in cell["diagnostics"][
                "claimed_speaker_diagnostics"
            ].items():
                dist = d["separation"]
                speaker_csv.append(
                    {
                        "split": split,
                        "role": role,
                        "phone": phone,
                        "claimed_speaker": speaker,
                        "eer": d["eer"],
                        "dprime": dist["dprime"],
                        "genuine_mean": dist["genuine"]["mean"],
                        "genuine_std": dist["genuine"]["std"],
                        "impostor_mean": dist["impostor"]["mean"],
                        "impostor_std": dist["impostor"]["std"],
                        **d["fixed_far_1pct_rates"],
                    }
                )
    for filename, items in (
        ("evaluation-cells.csv", csv_rows),
        ("speaker-cells.csv", speaker_csv),
    ):
        with (BASE / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    for role in ROLES:
        label = "通常文" if role == "verification" else "別テキスト"
        for phone in PHONES:
            cell = test["conditions"][f"{phone}/{role}"]
            interval = cell["ci95"]["pooled_eer"]
            for point, target in (("far_1pct", rows), ("far_0_1pct", strict)):
                rates = cell["operating_points"][point]
                target.append(
                    [
                        label,
                        phone,
                        str(cell["queries"]),
                        f"{100 * rates['all_input_far']:.3f}%",
                        f"{100 * rates['all_input_frr']:.3f}%",
                        f"{100 * cell['pooled_eer']:.3f}%",
                        f"[{100 * interval['lower']:.3f}, {100 * interval['upper']:.3f}]%",
                    ]
                )
            d = cell["diagnostics"]["score_separation"]
            scatter = cell["diagnostics"]["embedding_scatter"]
            separation.append(
                [
                    label,
                    phone,
                    f"{d['genuine']['mean']:.4f}",
                    f"{d['genuine']['std']:.4f}",
                    f"{d['impostor']['mean']:.4f}",
                    f"{d['impostor']['std']:.4f}",
                    f"{d['mean_score_gap']:.4f}",
                    f"{d['dprime']:.3f}",
                    f"{scatter['between_within_ratio']:.3f}",
                ]
            )
        speaker_rows[role] = [
            [
                speaker,
                *[
                    f"{100 * test['conditions'][f'{p}/{role}']['diagnostics']['claimed_speaker_diagnostics'][speaker]['eer']:.2f}%"
                    for p in PHONES
                ],
            ]
            for speaker in inputs["test"]["speakers"]
        ]
        distribution_plot(test, role, BASE / f"score-distributions-{role}.svg")
    primary = []
    for key, result in test["primary_hypothesis"]["contrasts"].items():
        ci = result["ci975_bonferroni_percentage_points"]
        primary.append(
            [
                key,
                f"{result['eer_difference_percentage_points']:+.3f}",
                f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]",
                "確認" if result["supports_lower_eer_than_s"] else "未確認",
            ]
        )
    headers = ["発話", "音素", "件数", "FAR", "FRR", "EER", "EERの95% CI"]
    dheaders = [
        "発話",
        "音素",
        "本人平均",
        "本人SD",
        "他人平均",
        "他人SD",
        "平均差",
        "d′",
        "話者間/内分散",
    ]
    pheaders = ["主比較（左−右）", "EER差（pp）", "97.5% CI（pp）", "sより低EER"]
    note = """既存の母音5種＋m/n/sモデル（JVS・Common Voice計140話者、30,000更新、1 seed）を固定し、追加学習なしで評価した。
全8音素で登録10区間×50ms、照合1区間×50ms。元音素境界内のcenter cropとRMS−50dBFSの判定を全音素に同じ方法で適用する。
元のcommon8発話から、全音素が50msで使える発話を推論前に選び、全音素で同じ話者・発話を使用した。

FAR/FRRは音素別にvalidation通常文で固定した閾値による値。EERはtestスコア分布の診断値であり、運用閾値の再校正ではない。
主検証は通常文のm−sとn−sのEER差。2比較のBonferroni調整として各97.5%区間を使用する。その他の比較・別テキスト・話者別順位は探索的診断。

d′は本人/他人の平均差を両群の分散で標準化した値。大きいほどスコア分布が分かれる。
話者間/内分散は50ms区間の正規化embeddingで計算し、話者を等重みにする。大きいほど異なる話者の中心が離れ、同じ話者内でまとまる。
登録話者別EERはその人の登録プロファイルに対する本人/他人試行の診断で、話者ごとに運用閾値を調整した結果ではない。

この結果はモデルが取り出せた識別情報の比較。音素本来の個人差を測るには別モデルや音響特徴での再検証が必要。
全音素が使える発話、50msの短い区間、観測済みJVS testの15話者、1 seedに限った結果で、未知録音条件や独立holdoutでの確証ではない。
前回の複数音素・登録3秒/照合1秒の実験とは入力と発話集合が異なるため、誤り率を直接引いて改善量とは扱わない。"""
    report = {
        "status": "completed",
        "protocol": config,
        "validation": validation,
        "test": test,
        "thresholds": read_json(run / "validation-thresholds.json"),
        "preparation": {s: x["preparation"] for s, x in inputs.items()},
        "audit": audits,
        "inference": {s: read_json(run / s / "inference.json") for s in inputs},
    }
    write_json(BASE / "evaluation-results.json", report)
    sections = [
        ("test：validation目標FAR 1%", headers, rows, False),
        ("主仮説：mとnはsより低EERか", pheaders, primary, False),
        ("本人/他人の分布と話者内の安定性", dheaders, separation, False),
        ("補助：validation目標FAR 0.1%", headers, strict, False),
        ("登録話者別EER：通常文", ["話者", *PHONES], speaker_rows[ROLES[0]], True),
        ("登録話者別EER：別テキスト", ["話者", *PHONES], speaker_rows[ROLES[1]], True),
    ]
    md = "# 音素単独の識別力比較\n\n" + "\n\n".join(
        f"## {title}\n\n{table(h, r)}" for title, h, r, _ in sections
    )
    md += f"\n\n## 条件と解釈\n\n{note}\n\n[結果の解釈](interpretation.md)・[HTML比較](evaluation-results.html)・[全96セルCSV](evaluation-cells.csv)・[話者別CSV](speaker-cells.csv)\n"
    (BASE / "evaluation-results.md").write_text(md)
    document = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>音素単独の識別力</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172536;margin:0}main{max-width:1300px;margin:32px auto;padding:0 24px}h1{font-size:28px}.box{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:20px;margin:20px 0;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #dbe2ec;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#eaf0f8}tr:nth-child(even){background:#f7f9fc}img{width:100%;height:auto}a{color:#2161b1}</style></head><body><main><h1>音素単独の識別力：a/i/u/e/o/m/n/s</h1><p>同じモデル・話者・発話。全音素で登録10×50ms、照合1×50ms。</p>"""
    for title, h, r, heatmap in sections:
        document += f"<section class='box'><h2>{html.escape(title)}</h2>{html_table(h, r, heatmap)}</section>"
    for role in ROLES:
        label = "通常文" if role == "verification" else "別テキスト"
        document += f"<section class='box'><h2>スコア分布：{label}</h2><p>青：本人、赤：他人。横軸は全音素共通のcosineスコア。</p><img src='score-distributions-{role}.svg' alt='{label}の音素別本人/他人スコア分布'></section>"
    document += (
        "<section class='box'><h2>条件と解釈</h2>"
        + "".join(f"<p>{html.escape(p)}</p>" for p in note.split("\n\n"))
        + "</section><p><a href='interpretation.md'>結果の解釈</a> · <a href='evaluation-results.md'>全表</a> · <a href='evaluation-cells.csv'>全96セルCSV</a> · <a href='speaker-cells.csv'>話者別CSV</a> · <a href='evaluation-results.json'>全指標JSON</a></p></main></body></html>"
    )
    (BASE / "evaluation-results.html").write_text(document)
    write_json(
        run / "report-audit.json",
        {
            "status": "completed",
            "input_plans": audits,
            "csv_cells": len(csv_rows),
            "speaker_cells": len(speaker_csv),
            "html_result_rows": sum(len(r) for _, _, r, _ in sections),
        },
    )
    print(f"published: {BASE / 'evaluation-results.html'}", flush=True)


if __name__ == "__main__":
    main()
