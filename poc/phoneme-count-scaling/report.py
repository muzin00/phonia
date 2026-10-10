"""Publish the prespecified phoneme-count curve and every combination."""

import csv
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bridge import BASE, ROOT, checked, pin, read_json, write_json


def percent(value):
    return "—" if value is None else f"{100 * value:.3f}%"


def interval(ci):
    return f"[{100 * ci['lower']:.3f}, {100 * ci['upper']:.3f}]"


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    audit = read_json(run / "independent-audit.json")
    assert audit["status"] == "passed"
    for name, checksum in read_json(run / "audit-freeze.json")["files"].items():
        checked(ROOT / name, checksum)
    result = read_json(run / "test-results.json")
    inputs = {
        split: read_json(run / f"{split}-inputs.json")
        for split in ("validation", "test")
    }
    normal = result["count_summary"]["verification"]
    cross = result["count_summary"]["cross_text_verification"]
    document = {
        "config": config,
        "results": result,
        "preparation": {s: d["preparation"] for s, d in inputs.items()},
        "audit": audit,
        "no_direct_comparison_to_previous_1_767pct": True,
    }
    write_json(BASE / "evaluation-results.json", document)
    summary = []
    for count in config["counts"]:
        key = str(count)
        eer = normal["pooled_eer"][key]
        item = {
            "count": count,
            "combinations": eer["combinations"],
            "normal_eer": percent(eer["mean"]),
            "normal_ci95": interval(eer["ci95"]),
            "combination_range": f"{percent(eer['minimum'])}〜{percent(eer['maximum'])}",
            "normal_far": percent(normal["far_1pct/all_input_far"][key]["mean"]),
            "normal_frr": percent(normal["far_1pct/all_input_frr"][key]["mean"]),
            "cross_eer": percent(cross["pooled_eer"][key]["mean"])
            if cross["pooled_eer"][key] is not None
            else "—",
        }
        summary.append(item)
    rows = []
    for condition, phones in result["schedule"]["conditions"].items():
        for role in ("verification", "cross_text_verification"):
            cell = result["cells"][f"{condition}/{role}"]
            point = cell["operating_points"]["far_1pct"]
            rows.append(
                {
                    "condition": condition,
                    "count": len(phones),
                    "phones": " ".join(phones),
                    "role": role,
                    "queries": cell["queries"],
                    "query_speakers": cell["query_speaker_count"],
                    "enrolled_speakers": cell["enrolled_speaker_count"],
                    "EER_pct": None
                    if cell["pooled_eer"] is None
                    else 100 * cell["pooled_eer"],
                    "FAR_pct": 100 * point["all_input_far"],
                    "FRR_pct": 100 * point["all_input_frr"],
                }
            )
    with (BASE / "evaluation-cells.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    primary = result["primary"]
    delta = f"{100 * primary['difference']:+.3f} pp"
    header = "| 音素種類数 | 組合せ数 | 通常文 統合EER | 話者bootstrap95% CI | 組合せによる範囲 | FAR | FRR | 別テキストEER（参考） |"
    lines = [
        "# 音素種類数と統合EER",
        "",
        "同じ固定14音素モデル・同じ発話・同じ実使用時間・同じスコア統合方法で比較。",
        "7/10種類は事前指定した9組合せの統合EER平均。各音素EERの平均ではない。",
        "",
        header,
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in summary:
        lines.append("| " + " | ".join(str(v) for v in item.values()) + " |")
    lines += [
        "",
        f"主比較14−5種類：{delta}、95% CI {interval(primary['ci95'])} pp。",
        "",
        "## 隣接する種類数の差（探索的）",
        "",
        "| 比較 | EER差 | 対応付き95% CI |",
        "|---|---|---|",
    ]
    for key in ("7_minus_5", "10_minus_7", "14_minus_10"):
        value = result["differences"][f"{key}/pooled_eer"]
        lines.append(
            f"| {key} | {100 * value['difference']:+.3f} pp | {interval(value['ci95'])} pp |"
        )
    lines += [
        "",
        "## 条件と解釈範囲",
        "",
        "- 5母音を共通とし、9子音のhashによる並べ替えと9循環シフトから入れ子の追加順を事前固定。7種類では各子音が2回、10種類では5回出現。全組合せの網羅ではない。",
        "- 登録上限3秒・照合上限1秒。全20条件が境界内30〜250ms・RMS−50dBFS以上を満たす最大共通時間を、推論前に5ms刻みで選択。元音声の重複・pad・増幅・文脈追加なし。",
        "- 音素別の登録平均をL2正規化し、照合区間のcosineを音素内平均してから音素等重みで統合。",
        "- FAR/FRRは条件別validation通常文でFAR1%を目標に固定した閾値をtestへ適用。EERはtestの診断値で、運用閾値を再校正していない。",
        "- CIは事前固定モデル・追加順に条件付くtest15話者bootstrap2,000回。本人はn_i、他人はn_i*n_j。組合せを独立標本として扱わない。",
        "- 全14音素がある発話に限定するため、一般の短い発話や音素不足を含む入力全体の性能ではない。",
        "- 別テキストは対応発話・話者が少なく、CIや一般化の主判定を行わない。",
        "- 前回の母音＋m/n 1.767%とは発話集合・追加音素・登録候補が異なるため、直接の精度比較ではない。",
        "- 1モデル/1 seed・過去にも観測済みのJVS test。学習する音素数の効果や独立holdoutでの一般化は未検証。",
        "",
        "[再現手順](README.md)・[HTML](evaluation-results.html)・[全組合せCSV](evaluation-cells.csv)・[JSON](evaluation-results.json)",
    ]
    (BASE / "evaluation-results.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    x = np.array(config["counts"])
    y = np.array([normal["pooled_eer"][str(k)]["mean"] * 100 for k in x])
    lower = np.array([normal["pooled_eer"][str(k)]["ci95"]["lower"] * 100 for k in x])
    upper = np.array([normal["pooled_eer"][str(k)]["ci95"]["upper"] * 100 for k in x])
    ax.fill_between(
        x,
        lower,
        upper,
        color="#2563eb",
        alpha=0.15,
        label="Conditional speaker bootstrap 95% CI",
    )
    ax.plot(x, y, marker="o", color="#2563eb", label="Mean integrated EER")
    for count in (7, 10):
        values = [
            result["cells"][f"{k}/verification"]["pooled_eer"] * 100
            for k in result["schedule"]["by_count"][str(count)]
        ]
        ax.scatter([count] * len(values), values, color="#64748b", alpha=0.65, s=20)
    for a, b in zip(x, y):
        ax.annotate(
            f"{b:.3f}%", (a, b), xytext=(0, 10), textcoords="offset points", ha="center"
        )
    ax.set(
        xlabel="Number of phoneme types (five vowels + consonants)",
        ylabel="Integrated EER (%) — lower is better",
        title="Fixed14 encoder · same utterances and actual PCM time",
        xticks=x,
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(BASE / "eer-curve.svg")
    fig.savefig(run / "eer-curve.png", dpi=160)
    plt.close(fig)

    def table(headers, data, table_id):
        return (
            f'<table id="{table_id}"><thead><tr>'
            + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
                + "</tr>"
                for row in data
            )
            + "</tbody></table>"
        )

    summary_table = table(
        [
            "種類数",
            "組合せ",
            "通常文EER",
            "95% CI",
            "組合せ範囲",
            "FAR",
            "FRR",
            "別テキストEER",
        ],
        [list(s.values()) for s in summary],
        "summary",
    )
    detailed = table(
        ["条件", "種類数", "音素", "発話", "件数", "話者数", "EER", "FAR", "FRR"],
        [
            [
                r["condition"],
                r["count"],
                r["phones"],
                r["role"],
                r["queries"],
                r["query_speakers"],
                "—" if r["EER_pct"] is None else f"{r['EER_pct']:.3f}%",
                f"{r['FAR_pct']:.3f}%",
                f"{r['FRR_pct']:.3f}%",
            ]
            for r in rows
        ],
        "combinations",
    )
    notes = "".join(
        f"<li>{html.escape(line[2:])}</li>" for line in lines if line.startswith("- ")
    )
    content = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>音素種類数と統合EER</title><style>body{{font-family:system-ui,sans-serif;color:#172033;background:#f4f7fb;max-width:1200px;margin:24px auto;padding:0 20px;line-height:1.7}}section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;border-bottom:1px solid #dfe5ed;text-align:right;white-space:nowrap}}th{{background:#eaf0f9}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}img{{display:block;width:100%;max-width:850px;margin:auto}}a{{color:#1d4ed8}}li{{margin-bottom:10px}}</style><h1>音素種類数と統合EER</h1><p>同じ固定14音素モデル・同じ発話・同じ実使用時間。5母音を起点に子音を増やす比較。</p><section><h2>種類数ごとの結果</h2><p>7/10種類は、事前指定した9組合せの<strong>統合EERの平均</strong>です。</p><img src="eer-curve.svg" alt="音素種類数と統合EERの曲線"><div class="scroll">{summary_table}</div><p><strong>主比較14−5：{delta}、95% CI {interval(primary["ci95"])} pp</strong></p></section><section><h2>条件と解釈範囲</h2><ul>{notes}</ul><p>対象発話数：{html.escape(json.dumps(document["preparation"], ensure_ascii=False))}</p></section><section><h2>全組合せ</h2><div class="scroll">{detailed}</div></section><p><a href="evaluation-cells.csv">CSV</a> · <a href="evaluation-results.json">JSON</a> · <a href="evaluation-results.md">Markdown</a> · <a href="README.md">再現手順</a> · <a href="interpretation.md">解釈</a></p></html>"""
    (BASE / "evaluation-results.html").write_text(content + "\n")
    files = {}
    for path in [
        BASE / name
        for name in (
            "evaluation-results.md",
            "evaluation-results.html",
            "evaluation-results.json",
            "evaluation-cells.csv",
            "eer-curve.svg",
        )
    ]:
        pin(files, path)
    write_json(
        run / "publication-report.json",
        {"status": "published_after_independent_audit", "files": files},
    )
    print(json.dumps({"normal_summary": summary, "primary": primary}), flush=True)


if __name__ == "__main__":
    main()
