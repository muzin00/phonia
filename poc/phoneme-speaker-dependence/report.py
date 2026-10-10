"""Publish validation-frozen groups and the independent-speaker comparison."""

import csv
import html

from bridge import BASE, ROOT, pin, read_json, write_json


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    audit = read_json(run / "independent-audit.json")
    if audit["status"] != "passed":
        raise ValueError("independent audit must pass first")
    validation = read_json(run / "validation-components.json")
    test = read_json(run / "test-results.json")
    groups = test["groups"]
    primary = test["group_summary"]["verification/pooled_eer"]
    supported = test["supports_lower_normal_macro_EER_in_high_group"]
    lines = [
        "# 固定14音素の群分け検証",
        "",
        "validationの話者間・話者内分散から群分けし、別15話者のtestで単音素EERを比較した。追加学習なし。",
        "",
        "**高群の通常文macro EERが低い傾向を確認した。**"
        if supported
        else "**高群の通常文macro EERが低いという差は、95% CIでは確認できなかった。**",
        "",
        f"高群：{'・'.join(groups['high'])}。低群：{'・'.join(groups['low'])}。",
        "",
        "## 音素ごとの結果",
        "",
        "Rはvalidationの共変量調整済みREML分散比。高群所属割合はvalidation話者bootstrapの結果。",
        "",
        "| 音素 | 群 | R | Rの95% CI | 高群所属割合 | test R | 通常文EER | 別テキストEER |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    table = []
    for p in sorted(
        config["phones"], key=lambda p: validation["components"][p]["R"], reverse=True
    ):
        c = validation["components"][p]
        normal = test["cells"][f"{p}/verification"]["pooled_eer"]
        cross = test["cells"][f"{p}/cross_text_verification"]["pooled_eer"]
        row = {
            "phone": p,
            "group": "high" if p in groups["high"] else "low",
            "validation_R": c["R"],
            "R_ci_lower": c["ci95"]["lower"],
            "R_ci_upper": c["ci95"]["upper"],
            "high_membership_fraction": c["bootstrap_high_membership_fraction"],
            "test_R": test["test_components"]["components"][p]["R"],
            "normal_eer": normal,
            "cross_text_eer": cross,
        }
        table.append(row)
        lines.append(
            f"| {p} | {'高' if row['group'] == 'high' else '低'} | {c['R']:.3f} | [{c['ci95']['lower']:.3f}, {c['ci95']['upper']:.3f}] | {row['high_membership_fraction']:.1%} | {row['test_R']:.3f} | {normal * 100:.3f}% | {cross * 100:.3f}% |"
        )
    lines += [
        "",
        "## 群単位の傾向",
        "",
        "以下は単音素のEER/FAR/FRRを音素等重みで平均した値。群を融合した照合性能ではない。",
        "",
        "| 発話 | 指標 | 高群平均 | 低群平均 | 高−低 | 対応付き95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for key, item in test["group_summary"].items():
        role, field = key.split("/", 1)
        label = "通常文" if role == "verification" else "別テキスト"
        ci = item["ci95_high_minus_low"]
        lines.append(
            f"| {label} | {field} | {item['high'] * 100:.3f}% | {item['low'] * 100:.3f}% | {item['high_minus_low'] * 100:+.3f} pp | [{ci['lower'] * 100:+.3f}, {ci['upper'] * 100:+.3f}] pp |"
        )
    lines += [
        "",
        f"validation Rとtest通常文EERのSpearman相関：{test['validation_R_vs_test_EER_spearman']:.3f}。",
        f"validation Rとtest RのSpearman相関：{test['validation_R_vs_test_R_spearman']:.3f}。",
        "",
        "## 条件と解釈",
        "",
        "- 群分け：validation15話者、音素ごと180自然区間、区間長・前後文脈調整。testにも同数の自然区間。",
        "- 単音素評価：各15話者、登録5×50ms、通常文150本人/2100他人、別テキスト45本人/630他人試行（各音素）。",
        "- 高群−低群通常文EERが主1検証。その他の指標と相関は補助的。CIはvalidationで固定した群・閾値に条件付く。",
        "- 分散比は、この固定encoderが抽出した安定した話者依存性。身体形状と発音習慣、話者固定の録音差を分離した指標ではない。",
        "- 二群に分類できても、自然な二峰性は証明していない。境界付近の音素はbootstrap所属割合で判断する。",
        "- 全音素で話者・区間数・評価時間は同じだが、元発話は異なる。50ms cropと文脈調整の不完全さが結果に影響し得る。",
        "- 主結果は群内の単音素平均。音素の組み合わせによる追加効果は別に検証する必要がある。",
        "- 1モデル/1 seed、観測済みJVS test。未知録音条件や独立holdoutでの確証ではない。",
        "",
        "## 検証",
        "",
        "8単体テストとRuff、推論のbitwise反復一致、重み不変、入力・コード・群・閾値のchecksumを検査。",
        "独立監査：" + ", ".join(f"{k}={v}" for k, v in audit["checks"].items()) + "。",
        "",
        "[再現手順](README.md)・[CSV](evaluation-cells.csv)・[JSON](evaluation-results.json)・[HTML](evaluation-results.html)",
    ]
    markdown = "\n".join(lines) + "\n"
    (BASE / "evaluation-results.md").write_text(markdown)
    with (BASE / "evaluation-cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    write_json(
        BASE / "evaluation-results.json",
        {
            "status": "completed",
            "config": config,
            "validation": validation,
            "test": test,
            "audit": audit,
        },
    )
    # Tables and surrounding prose are emitted directly, without Markdown dependencies.
    body, in_table = [], False
    for line in lines:
        if line.startswith("|"):
            if line.startswith("|---"):
                continue
            cells = [x.strip() for x in line.strip("|").split("|")]
            if not in_table:
                body.append(
                    "<div class='scroll'><table><thead><tr>"
                    + "".join(f"<th>{html.escape(x)}</th>" for x in cells)
                    + "</tr></thead><tbody>"
                )
                in_table = True
            else:
                body.append(
                    "<tr>"
                    + "".join(f"<td>{html.escape(x)}</td>" for x in cells)
                    + "</tr>"
                )
            continue
        if in_table:
            body.append("</tbody></table></div>")
            in_table = False
        if line.startswith("## "):
            body.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            body.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line:
            body.append("<p>" + html.escape(line.replace("**", "")) + "</p>")
    (BASE / "evaluation-results.html").write_text(
        "<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>固定14音素の群分け検証</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#183047;line-height:1.7}h1,h2{line-height:1.4}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border:1px solid #d6e0e7;white-space:nowrap}th{background:#eaf1f7}tr:nth-child(even){background:#f7f9fc}</style><main>"
        + "\n".join(body)
        + "<p><a href='evaluation-cells.csv'>CSV</a> · <a href='evaluation-results.json'>JSON</a> · <a href='README.md'>再現手順</a></p></main></html>\n"
    )
    ledger = {}
    for path in [
        *BASE.rglob("*.py"),
        *BASE.rglob("*.json"),
        *BASE.glob("*.md"),
        *BASE.glob("*.csv"),
        *BASE.glob("*.html"),
        run / "design-freeze.json",
        run / "group-freeze.json",
        run / "test-results.json",
        run / "test-bootstrap.npz",
        run / "independent-audit.json",
    ]:
        pin(ledger, path)
    write_json(
        run / "publication-report.json", {"status": "completed", "files": ledger}
    )
    print(
        json.dumps(
            {
                "primary_difference_pp": primary["high_minus_low"] * 100,
                "supported": supported,
            }
        )
    )


if __name__ == "__main__":
    import json

    main()
