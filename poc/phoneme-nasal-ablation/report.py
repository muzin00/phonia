"""Publish every fixed condition, operating point and paired comparison."""

import csv
import html
from pathlib import Path

import numpy as np
import study

BASE = Path(__file__).resolve().parent


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )


def html_table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    config = study.read_json(study.CONFIG)
    run = study.ROOT / config["run_directory"]
    study.frozen(config, run, "test")
    test = study.read_json(run / "test-metrics.json")
    validation = study.read_json(run / "validation-metrics.json")
    plans, slices, inputs = {}, {}, {}
    for split in ("validation", "test"):
        inputs[split] = study.read_json(run / f"{split}-inputs.json")
        plans[split] = study.validate_plans(inputs[split])
        slices[split] = study.prior.audit_source_slices(inputs[split], config)
        study.load_groups(config, run, split)
    with np.load(run / "bootstrap-differences.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(test["paired_differences"]):
            raise ValueError("incomplete bootstrap pair matrix")
        for key, difference in test["paired_differences"].items():
            if difference["ci95_percentage_points"] != study.old.interval(arrays[key]):
                raise ValueError("paired CI does not match saved draws")
    cells = []
    for split, measured in (("validation", validation), ("test", test)):
        for key, cell in measured["conditions"].items():
            condition, role = key.split("/")
            for point, rates in cell["operating_points"].items():
                cells.append(
                    {
                        "split": split,
                        "condition": condition,
                        "role": role,
                        "operating_point": point,
                        "queries": cell["queries"],
                        "eer": cell["pooled_eer"],
                        **rates,
                    }
                )
    with (BASE / "evaluation-cells.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    rows, strict_rows, diff_rows = [], [], []
    header = ["発話", "条件", "件数", "FAR", "FRR", "EER"]
    for role in study.ROLES:
        for condition in study.CONDITIONS:
            cell = test["conditions"][f"{condition}/{role}"]
            for point, target in (("far_1pct", rows), ("far_0_1pct", strict_rows)):
                rates = cell["operating_points"][point]
                target.append(
                    [
                        "通常文" if role == "verification" else "別テキスト",
                        study.LABELS[condition],
                        str(cell["queries"]),
                        f"{100 * rates['all_input_far']:.3f}%",
                        f"{100 * rates['all_input_frr']:.3f}%",
                        f"{100 * cell['pooled_eer']:.3f}%",
                    ]
                )
    dheader = ["発話", "比較（左−右）", "指標", "差（pp）", "対応付き95% CI（pp）"]
    for a, b in study.PAIRS:
        for role in study.ROLES:
            for name, label in zip(config["primary_metrics"], ("FAR", "FRR", "EER")):
                value = test["paired_differences"][f"{b}_minus_{a}/{role}/{name}"]
                interval = value["ci95_percentage_points"]
                diff_rows.append(
                    [
                        "通常文" if role == "verification" else "別テキスト",
                        f"{study.LABELS[b]} − {study.LABELS[a]}",
                        label,
                        f"{value['difference_percentage_points']:+.3f}",
                        f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]",
                    ]
                )
    audit = {
        "status": "completed",
        "matched_plan_checks": plans,
        "original_source_slice_checks": slices,
        "baseline_metrics_and_ci_identical": test[
            "prior_baseline_metrics_and_ci_identical"
        ],
        "paired_ci_checks": len(test["paired_differences"]),
        "csv_cells": len(cells),
        "source_and_code_checksum_checks": len(
            study.read_json(run / "design-freeze.json")["files"]
        ),
    }
    study.write_json(run / "audit.json", audit)
    report = {
        "status": "completed",
        "protocol": config,
        "validation": validation,
        "test": test,
        "thresholds": study.read_json(run / "validation-thresholds.json"),
        "audit": audit,
        "preparation": {s: d["preparation"] for s, d in inputs.items()},
        "inference": {s: study.read_json(run / s / "inference.json") for s in inputs},
        "claim_scope": "same_existing_encoder; previously_observed_JVS_test; one_seed; six_exploratory_pairs_without_multiplicity_adjustment",
    }
    study.write_json(BASE / "evaluation-results.json", report)
    note = """全条件で同じ学習済み7音素encoder、同じ登録候補WAV、同じcommon7発話を使用した。
登録最大3秒・照合最大1秒で、各登録・各発話の実使用PCMフレーム数を4条件で完全一致させた。
validation通常文525件で条件ごとの閾値を固定し、test・別テキストでは再校正していない。
FAR/FRRは固定閾値での値、EERはtestのスコア分布から計算した診断指標であり運用閾値ではない。

5母音とm/n併用の区間・スコア・閾値・test指標・CIは前回common7と一致する。
両基準条件を新しいバッチで再推論して誤差が事前規定1e-6以内であることも検査し、
表と比較には保存済みの元スコアをそのまま再利用した。新規学習は行っていない。

差は左側−右側、負のエラー率差は改善。15話者の同じ10,000 bootstrap drawを全6組で共有した。
全6組を事前固定して掲載し、CIは多重比較補正をしていない探索的な区間である。
既に観測済みのJVS test・1学習seedであり、独立holdout・未知録音条件での確証ではない。

mだけ対nだけは同じ6音素で種類の違いを比較する。併用対各単独は追加効果を比較する。
同じ総時間でも、各音素の時間・区間数・選択PCM・等重み統合における母音の重みは変わる。
したがって音素種類だけの因果効果や「どの音素でも種類数を増やせば改善する」は確定しない。
その確認には別の子音組と、良質データでの追加学習・独立評価が必要である。"""
    md = f"""# /m/・/n/ の個別追加と併用の切り分け

## test：validation目標FAR 1%の固定閾値

{table(header, rows)}

## test：validation目標FAR 0.1%の固定閾値（補助）

{table(header, strict_rows)}

## 全6組の対応付き差（主指標）

{table(dheader, diff_rows)}

## 条件と解釈の範囲

{note}

## 監査

{sum(plans.values()):,}個のplanの時間・音素・重複・重なり・登録/照合分離と、
{sum(slices.values()):,}個の選択区間が元の音素区間内center cropであることを確認した。
validation {len(inputs["validation"]["queries"])}発話、test {len(inputs["test"]["queries"])}発話、
各15 claimed話者×4条件の全試行を検査した。前回基準条件との指標・CI一致も確認済み。
全{len(cells)}セル（validation/test×4条件×2発話種×3動作点）はCSV、
全{len(test["paired_differences"])}組の指標差はJSONとartifacts内bootstrap配列へ保存した。

[実験条件と再現手順](README.md)・[結果の解釈](interpretation.md)・[比較HTML](evaluation-results.html)・
[全セルCSV](evaluation-cells.csv)・[閾値・件数・CIのJSON](evaluation-results.json)
"""
    (BASE / "evaluation-results.md").write_text(md, encoding="utf-8")
    document = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>m/n 個別追加と併用の比較</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172536;margin:0}main{max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:28px}.box{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:20px;margin:20px 0;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dbe2ec;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#eaf0f8}tr:nth-child(even){background:#f7f9fc}a{color:#2161b1}</style></head><body><main><h1>/m/・/n/ 個別追加と併用の比較</h1><p>同じモデル・同じ発話・同じ使用音声量。4条件を比較。</p>"""
    for title, headers, values in (
        ("test：validation目標FAR 1%", header, rows),
        ("補助：validation目標FAR 0.1%", header, strict_rows),
        ("全6組の差と対応付き95% CI", dheader, diff_rows),
    ):
        document += f"<section class='box'><h2>{html.escape(title)}</h2>{html_table(headers, values)}</section>"
    document += (
        "<section class='box'><h2>比較条件と解釈</h2>"
        + "".join(f"<p>{html.escape(p)}</p>" for p in note.split("\n\n"))
        + "</section><p><a href='interpretation.md'>結果の解釈</a> · <a href='evaluation-results.md'>監査と結果</a> · <a href='evaluation-cells.csv'>全48セルCSV</a> · <a href='evaluation-results.json'>全指標JSON</a></p></main></body></html>"
    )
    (BASE / "evaluation-results.html").write_text(document, encoding="utf-8")
    outputs = {
        str(p.relative_to(run)): study.sha256_file(p)
        for p in run.rglob("*")
        if p.is_file() and p.name != "study-report.json"
    }
    study.write_json(
        run / "study-report.json",
        {"status": "completed", "outputs_sha256": outputs, "audit": audit},
    )
    print(f"published: {BASE / 'evaluation-results.html'}", flush=True)


if __name__ == "__main__":
    main()
