"""Publish the one full-phone model comparison after numerical checks pass."""

from __future__ import annotations

import html
import json
from pathlib import Path

import study
from common import read_json, relative, sha256_file, torch, write_json

BASE = Path(__file__).resolve().parent


def percent(value):
    return f"{100 * value:.3f}%"


def table(headers, records):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in record) + "</tr>"
        for record in records
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def main():
    torch.set_num_threads(1)
    config = read_json(study.CONFIG)
    run = study.ROOT / config["run_directory"]
    study.verify(config, run)
    audit = read_json(run / "independent-audit.json")
    if audit["status"] != "passed":
        raise ValueError("publish only after numerical audit passes")
    design = read_json(run / "design-freeze.json")
    frozen = read_json(run / "selection-freeze.json")
    test = read_json(run / "final-test.json")
    if frozen["selected_phonemes"] != design["phonemes"] or test[
        "selection_freeze_sha256"
    ] != sha256_file(run / "selection-freeze.json"):
        raise ValueError("published selection or test provenance changed")
    bootstrap = read_json(run / "bootstrap-results.json")
    prep = read_json(run / "preparation-report.json")
    validation = {
        key: read_json(run / "trials" / config[name] / "validation-metrics.json")
        for key, name in (("baseline", "baseline_trial"), ("final", "final_trial"))
    }
    summary = read_json(
        run / "trials" / config["final_trial"] / "training-summary.json"
    )
    data = {
        "protocol": config,
        "run": relative(run),
        "phonemes": design["phonemes"],
        "unavailable_phonemes": design["unavailable_phonemes"],
        "validation": validation,
        "test": test,
        "training": summary,
        "bootstrap": bootstrap,
        "audit": audit,
    }
    write_json(BASE / "evaluation-results.json", data)
    headers = [
        "集合",
        "発話",
        "全発話",
        "scoreあり",
        "5母音 EER",
        "全36音素 EER",
        "EER差",
        "5母音 FAR",
        "全36音素 FAR",
        "5母音 FRR",
        "全36音素 FRR",
    ]
    records = []
    for split, results in (("validation", validation), ("test", test)):
        for role, label in (
            ("verification", "通常文"),
            ("cross_text_verification", "別テキスト"),
        ):
            a, b = (
                results["baseline"]["metrics"][role],
                results["final"]["metrics"][role],
            )
            ap, bp = (
                a["operating_points"]["far_1pct"],
                b["operating_points"]["far_1pct"],
            )
            records.append(
                [
                    split,
                    label,
                    a["all_queries"],
                    a["scored_queries"],
                    percent(a["eer"]),
                    percent(b["eer"]),
                    f"{100 * (b['eer'] - a['eer']):+.3f} pp",
                    percent(ap["all_input_far"]),
                    percent(bp["all_input_far"]),
                    percent(ap["all_input_frr"]),
                    percent(bp["all_input_frr"]),
                ]
            )
    phone_records = []
    val_inputs, test_inputs = (
        read_json(run / f"{split}-inputs.json") for split in ("validation", "test")
    )
    for p in [*design["phonemes"], *design["unavailable_phonemes"]]:
        support = prep["phone_support"][p]
        phone_records.append(
            [
                p,
                support["eligible_intervals"],
                support["speakers_with_two_intervals"],
                audit["real_training_examples"].get(p, 0),
                "学習済み" if p in design["phonemes"] else "データ不足",
                "可" if p in val_inputs["universally_registered"] else "不可",
                "可" if p in test_inputs["universally_registered"] else "不可",
            ]
        )
    baseline, full = (
        test["baseline"]["metrics"]["verification"]["eer"],
        test["final"]["metrics"]["verification"]["eer"],
    )
    cross_base, cross_full = (
        test["baseline"]["metrics"]["cross_text_verification"]["eer"],
        test["final"]["metrics"]["cross_text_verification"]["eer"],
    )
    direction = "低下" if full < baseline else ("同値" if full == baseline else "上昇")
    lead = f"test通常文EERは{percent(baseline)} → {percent(full)}（{direction}）、別テキストは{percent(cross_base)} → {percent(cross_full)}。"
    note = "FAR/FRRはvalidation通常文で目標FAR1%に校正した閾値を固定適用した全入力値。EERはスコアのある共通発話から算出。"
    markdown = [
        "# 全36音素学習の評価結果",
        "",
        lead,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown.extend("| " + " | ".join(map(str, r)) + " |" for r in records)
    markdown.extend(
        [
            "",
            note,
            "",
            "新規学習は1モデル・1 seed・216,000更新。既存5母音モデルは30,000更新。各母音への延べ投入は両モデル600,000区間。",
            "学習可能な36音素を入力だけで固定。tyは適合学習区間が0件のため未学習。",
            "",
            "[HTML比較表](evaluation-results.html)・[結果の解釈](interpretation.md)・[JSON](evaluation-results.json)・[再現手順](README.md)",
            "",
        ]
    )
    (BASE / "evaluation-results.md").write_text("\n".join(markdown))
    ci, cross_ci = (
        bootstrap["verification"]["eer_difference95"],
        bootstrap["cross_text_verification"]["eer_difference95"],
    )
    interpretation = f"""# 全音素をまとめて学習した結果

**{lead}**

5母音＋追加31音素の全36音素を、EERによる音素選択をせず学習前に固定した。
tyは学習区間0件で未学習。JVS70話者＋Common Voice70話者、適合学習区間957,271件、SRC4VCなし。
新規学習は1モデル、seed {config["training_seed"]}、最終{config["maximum_updates"]:,}更新の重みを使用した。
基準5母音モデルと同じ初期重み・正規化・損失・サンプル選択規則を使用した。
各母音の延べ投入は両モデル600,000区間。全36音素モデルの実投入合計は{summary["training_examples"]:,}区間。
dyは区間ペアがある1話者の実サンプルだけを使用し、空slotを実投入に数えていない。

通常文735発話・別テキスト392発話は両モデル共通。入力母音不足は全入力FAR/FRRへ含めた。
登録可能な追加音素が照合発話にある場合に使い、音素間は等重みで統合する。
全36音素を学習するが、各発話に全音素が揃うことは要求しない。
元の母音切片を短くせず、追加音素分の登録・照合音声を使う。

全音素セットは学習前、最終重みとvalidation閾値は新モデルのtest推論前に固定した。
validation・testで音素セットやcheckpointを選び直していない。
通常文test EER差の対応付き話者bootstrap95%区間は[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] pp。
別テキストは[{100 * cross_ci[0]:+.3f}, {100 * cross_ci[1]:+.3f}] pp。1モデルを固定した2000回の話者bootstrapで、学習seed間の変動は含まない。
総更新数と学習率scheduleの期間も増やした比較であり、音素数だけの因果効果ではなく、各母音への投入を維持した全音素構成の性能を測る。
既存JVS testを使用しており、新しい独立holdoutではない。

[HTML比較表](evaluation-results.html)・[測定表](evaluation-results.md)・[再現手順](README.md)
"""
    (BASE / "interpretation.md").write_text(interpretation)
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全36音素学習のEER比較</title>
<style>body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#172337}}main{{max-width:1440px;margin:auto;padding:32px}}p{{line-height:1.8}}.table-wrap{{overflow:auto;background:white;border:1px solid #d8dfe8;border-radius:8px;margin:20px 0}}table{{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}}th,td{{padding:12px;border-bottom:1px solid #e4e9ef;text-align:right}}th{{background:#eaf0f6}}td:first-child,th:first-child{{text-align:left}}a{{color:#1768a6}}</style>
<main><h1>全36音素を1モデルで学習した結果</h1><p><b>{html.escape(lead)}</b></p>
<p>音素選択なし・新規学習1モデル・216,000更新。各母音への投入は5母音モデルと同じ600,000区間。最終重みを固定してtest評価。</p>
<h2>5母音モデルとの比較</h2>{table(headers, records)}<p>{html.escape(note)}</p>
<h2>全音素の学習・登録可否</h2>{table(["音素", "適合学習区間", "2区間以上ある話者", "延べ学習投入", "学習", "validation全話者登録", "test全話者登録"], phone_records)}
<p>学習音素：{html.escape(" / ".join(design["phonemes"]))}<br>tyは区間0件のため未学習。各照合発話では5母音必須、追加音素は発話内にあり、全登録話者にprofileがある場合に使用。</p>
<p>JVS＋Common Voice、SRC4VCなし。追加音素の音声時間も使用。基準モデルのtest結果は再利用。既存JVS test・1 seed。</p>
<p>数値再計算：{audit["scores"]:,}スコア、{audit["eer_checks"]} EER、{audit["rate_checks"]} FAR/FRR値を照合。</p>
<p><a href="evaluation-results.json">JSON</a> · <a href="evaluation-results.md">測定表</a> · <a href="interpretation.md">解釈</a> · <a href="README.md">再現手順</a></p></main></html>
"""
    (BASE / "evaluation-results.html").write_text(page)
    files = [
        BASE / name
        for name in (
            "evaluation-results.json",
            "evaluation-results.md",
            "evaluation-results.html",
            "interpretation.md",
        )
    ]
    write_json(
        run / "publication-freeze.json",
        {
            "files": {relative(p): sha256_file(p) for p in files},
            "audit_sha256": sha256_file(run / "independent-audit.json"),
            "source_sha256": sha256_file(BASE / "report.py"),
            "summary_rows": len(records),
            "phone_rows": len(phone_records),
        },
    )
    print(
        json.dumps(
            {
                "stage": "published",
                "summary": lead,
                "html": relative(BASE / "evaluation-results.html"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
