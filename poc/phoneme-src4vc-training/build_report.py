"""Publish compact training evidence, with JVS validation separate from test results."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    config = read(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    current = read(run / "expanded/training/summary.json")
    preparation = read(run / "preparation-report.json")
    descriptor = read(run / "expanded/run.json")
    if (
        current["status"] != "completed"
        or current["selected_update"] != 30000
        or current["test_used"]
    ):
        raise ValueError("fixed-budget, train/validation-only completion required")
    for name, digest in current["outputs_sha256"].items():
        if checksum(run / "expanded" / name) != digest:
            raise ValueError("training output changed")
    reference_path = ROOT / config["reference_training_summary"]
    if checksum(reference_path) != config["reference_training_summary_sha256"]:
        raise ValueError("reference summary changed")
    reference = read(reference_path)
    baseline = read(ROOT / config["baseline_run"] / "training/summary.json")
    rows = [
        {
            "condition": "JVS70 現行モデル（参考）",
            "speakers": 70,
            "updates": baseline["best"]["update"],
            "selection": "既存のvalidation best",
            "validation_macro_eer_pct": baseline["best"]["eer"] * 100,
        },
        {
            "condition": "JVS70 + Common Voice70",
            "speakers": 140,
            "updates": reference["selected_update"],
            "selection": "事前固定30,000",
            "validation_macro_eer_pct": reference["validation_macro_eer"] * 100,
        },
        {
            "condition": "JVS70 + SRC4VC70",
            "speakers": 140,
            "updates": current["selected_update"],
            "selection": "事前固定30,000",
            "validation_macro_eer_pct": current["validation_macro_eer"] * 100,
        },
    ]
    delta = rows[2]["validation_macro_eer_pct"] - rows[1]["validation_macro_eer_pct"]
    findings = [
        "同じJVS validation 15話者・各母音10区間登録で測る、母音別EERの平均。小さいほど良い。",
        f"SRC4VC版とCommon Voice版の差は {delta:+.3f} percentage points。1 seedでの観測値で、統計的な優位性は未検証。",
        "発話全体を集約したtest FAR/FRRは今回は測定していない。過去の発話単位評価との数値の直接比較はできない。",
        "SRC4VC予約30話者は区間抽出・特徴統計・学習・validationに使っていない。JVSのtrain/validation/test分割も保持した。",
        "追加データの量、話し方、収録条件、特徴統計が同時に変わるため、年齢・性別の偏りだけを切り分ける実験ではない。",
        "追加コーパスとJVSの実在人物の重複は未確認。異なる話者ラベルだけでは同一人物でないことを保証できない。",
    ]
    data = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "seed": config["seed"],
        "training_conditions": rows,
        "src4vc_minus_cv_validation_eer_percentage_points": delta,
        "preparation": preparation,
        "exposure": current["exposure"],
        "encoder_parameters": descriptor["encoder_parameters"],
        "embedding_dimension": 128,
        "initial_encoder_sha256": descriptor["initial_encoder_sha256"],
        "export_reload_bitwise_equal_validation_segments": current[
            "export_reload_bitwise_equal_validation_segments"
        ],
        "test_used": False,
        "findings": findings,
        "artifact_directory": config["run_directory"],
        "source_url": config["source_project_url"],
        "evidence_sha256": {
            str(p.relative_to(ROOT)): checksum(p)
            for p in (
                BASE / "config/protocol.json",
                run / "training-freeze.json",
                run / "expanded/training/summary.json",
                run / "preparation-report.json",
                run / "expanded/run.json",
                reference_path,
            )
        },
    }
    (BASE / "training-results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    with (BASE / "training-comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        "学習条件",
        "学習話者ラベル",
        "採用update",
        "採用基準",
        "JVS validation EER",
    ]
    values = [
        [
            r["condition"],
            str(r["speakers"]),
            f"{r['updates']:,}",
            r["selection"],
            f"{r['validation_macro_eer_pct']:.3f}%",
        ]
        for r in rows
    ]
    md_table = (
        "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    )
    md_table += "\n".join("| " + " | ".join(row) + " |" for row in values)
    info = (
        f"SRC4VC train {preparation['speakers']}話者・{preparation['selected_clips']:,}発話、"
        f"抽出成功{preparation['aligned_clips']:,}発話。学習可能な母音{preparation['eligible_segments']:,}区間、"
        f"除外{preparation['excluded_segments']:,}区間、失敗{len(preparation['failures'])}発話。"
        f"原音声{preparation['source_hours']:.3f}時間、母音合計{preparation['eligible_vowel_hours']:.3f}時間。"
    )
    attrs = f"選定したtrainの申告属性：性別{preparation['train_gender_labels']}、年齢{preparation['train_age_years']['min']:g}–{preparation['train_age_years']['max']:g}歳。"
    md = (
        "# SRC4VCでの追加学習結果\n\n"
        + md_table
        + "\n\n"
        + info
        + "\n\n"
        + attrs
        + "\n\n"
        + "\n".join("- " + f for f in findings)
        + "\n\n"
    )
    md += "encoderは65,920 parameters・128次元、1 seed・1新規条件。同一初期化・同一scheduleで30,000 updateを完了した。\n\n"
    md += f"JVS各話者のbatch選択回数は{current['exposure']['jvs_label_batch_count_min']}–{current['exposure']['jvs_label_batch_count_max']}回。現行JVSモデルとの差は最大{current['exposure']['maximum_jvs_difference_from_baseline']}回。保存と再読み込みで8区間のembeddingがbit単位で一致した。\n\n"
    md += f"[SRC4VC公式配布]({config['source_project_url']})。音声と重みは`{config['run_directory']}`に保存し、Gitには含めない。\n"
    (BASE / "training-results.md").write_text(md)
    head = "".join("<th>" + html.escape(c) + "</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>"
        for row in values
    )
    document = (
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SRC4VC学習結果</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#172033;background:#f6f8fc}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d5dce6;padding:12px;text-align:left}th{background:#e9eef7}td:last-child{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}li{margin:10px 0}a{color:#175dc6}</style><main><h1>SRC4VCでの追加学習結果</h1><p>学習完了・JVS validationまで。発話単位test FAR/FRRは未測定。</p><div class="scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div><p>"
        + html.escape(info)
        + "</p><p>"
        + html.escape(attrs)
        + "</p><ul>"
        + "".join("<li>" + html.escape(f) + "</li>" for f in findings)
        + '</ul><p><a href="training-results.json">集計JSON</a> · <a href="training-comparison.csv">CSV</a> · <a href="'
        + html.escape(config["source_project_url"], quote=True)
        + '">SRC4VC公式配布</a></p></main></html>'
    )
    (BASE / "training-results.html").write_text(document)
    (BASE / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": data["evidence_sha256"],
                "outputs": {
                    name: checksum(BASE / name)
                    for name in (
                        "training-results.json",
                        "training-results.md",
                        "training-results.html",
                        "training-comparison.csv",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(md)


if __name__ == "__main__":
    main()
