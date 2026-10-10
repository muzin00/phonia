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
        slices[split] = study.audit_source_slices(inputs[split], config)
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
        "shared_new_eight_phone_model": test["shared_new_eight_phone_model"],
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
        "claim_scope": "same_new_eight_phone_encoder; common8; previously_observed_JVS_test; one_seed; six_exploratory_pairs_without_multiplicity_adjustment",
    }
    study.write_json(BASE / "evaluation-results.json", report)
    note = """全条件で同じ新規学習済み8音素encoder、同じ登録候補WAV、同じcommon8発話を使用した。
JVS70話者＋Common Voice70話者、母音5種＋m/n/sを1 seed・30,000更新で学習し、最後の重みを固定した。
登録最大3秒・照合最大1秒で、各登録・各発話の実使用PCMフレーム数を4条件で完全一致させた。
全8音素があり、4条件とも同じ時間を使える発話をスコアの推論前に固定した。

m+n・m+s・n+sはすべて同じ7音素で、子音の組み合わせを比べる。
5母音との比較では音素の種類数も変わる。過去の7音素モデルとはモデルと対象発話が異なるため、
前回の数値との直接差を今回の子音追加効果とは扱わない。今回の4条件はすべて再推論した。

validation通常文で条件ごとの閾値を固定し、test・別テキストでは再校正していない。
FAR/FRRは固定閾値での値、EERはtestのスコア分布から計算した診断指標であり運用閾値ではない。
差は左側−右側、負のエラー率差は改善。15話者の同じ10,000 bootstrap drawを全6組で共有した。
CIは多重比較補正をしていない探索的な区間である。

既に観測済みのJVS test・1学習seedであり、独立holdout・未知録音条件での確証ではない。
全8音素を含む発話に絞った比較であり、音素不足がある発話での性能は評価していない。
同じ総時間でも、各音素の時間・区間数・選択PCM・等重み統合における母音の重みは変わる。
/s/は準備済み候補と同じ最大250msのcenter crop後に音量を判定し、母音・m/nは従来の入力を再利用した。
この実験では音素の組み合わせに加え、抽出・時間配分・統合方法の影響も含む。"""
    md = f"""# 子音の組み合わせ m+n・m+s・n+s の比較

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
各15 claimed話者×4条件の全試行を検査した。全条件で新しい共通モデルを使用した。
全{len(cells)}セル（validation/test×4条件×2発話種×3動作点）はCSV、
全{len(test["paired_differences"])}組の指標差はJSONとartifacts内bootstrap配列へ保存した。

[実験条件と再現手順](README.md)・[結果の解釈](interpretation.md)・[比較HTML](evaluation-results.html)・
[全セルCSV](evaluation-cells.csv)・[閾値・件数・CIのJSON](evaluation-results.json)
"""
    (BASE / "evaluation-results.md").write_text(md, encoding="utf-8")
    document = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>子音の組み合わせ比較</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172536;margin:0}main{max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:28px}.box{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:20px;margin:20px 0;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dbe2ec;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#eaf0f8}tr:nth-child(even){background:#f7f9fc}a{color:#2161b1}</style></head><body><main><h1>子音の組み合わせ m+n・m+s・n+s</h1><p>同じモデル・同じ発話・同じ使用音声量。4条件を比較。</p>"""
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
    training_base = BASE.parent / "phoneme-consonant-combination-training"
    training_run = study.ROOT / config["training_run"]
    training_summary = study.read_json(training_run / "training/summary.json")
    preparation = study.read_json(training_run / "preparation-report.json")
    study.write_json(
        training_base / "training-results.json",
        {
            "summary": training_summary,
            "preparation": {
                k: v for k, v in preparation.items() if k != "source_inputs"
            },
        },
    )
    diagnostics = [
        [str(row["update"]), f"{100 * row['vowel_only_macro_eer']:.3f}%"]
        for row in training_summary["validation_diagnostics"]
    ]
    (training_base / "training-results.md").write_text(
        "# 8音素モデルの学習結果\n\n"
        f"JVS・Common Voice計140話者、{preparation['training_segments']:,}区間で30,000更新を完了。\n"
        "母音5種＋m/n/s、65,920 parameters、128次元、seed 20260926。\n"
        "最後の重みを採用し、8音素すべてでexport前後の出力が完全一致した。\n"
        "各音素のmicrobatchは18,750回、学習は合計3,000,000区間提示。testは学習に使っていない。\n\n"
        "## 母音単一区間のvalidation診断\n\n"
        + table(["更新", "macro EER"], diagnostics)
        + "\n\nこの診断値は重み選択や早期終了には使わず、発話照合の性能とは区別する。\n\n"
        "[同じモデルでの子音組み合わせ比較](../phoneme-consonant-combination-evaluation/evaluation-results.md)\n",
        encoding="utf-8",
    )
    print(f"published: {BASE / 'evaluation-results.html'}", flush=True)


if __name__ == "__main__":
    main()
