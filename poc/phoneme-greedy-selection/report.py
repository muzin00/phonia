"""Publish the complete search, selected encoder and final comparison tables."""

from __future__ import annotations

import csv
import html
import json
import xml.etree.ElementTree as ET

from common import (
    BASE,
    CONFIG,
    ROOT,
    np,
    read_json,
    relative,
    sha256_file,
    write_json,
)


def percent(value):
    return "—" if value is None else f"{100 * value:.3f}%"


def pp(value):
    return "—" if value is None else f"{100 * value:+.3f} pp"


def table(headers, records, classes=None):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for i, record in enumerate(records):
        cls = classes[i] if classes else ""
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in record)
        body.append(f'<tr class="{cls}">{cells}</tr>')
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def report():
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    audit = read_json(run / "independent-audit.json")
    if audit["status"] != "passed":
        raise ValueError("independent audit must pass before publication")
    state = read_json(run / "search-state.json")
    frozen = read_json(run / "selection-freeze.json")
    final = read_json(run / "final-test.json")
    preparation = read_json(run / "preparation-report.json")
    bootstrap = read_json(run / "bootstrap-results.json")
    baseline_validation = read_json(
        run / "trials/trial-000-baseline/validation-metrics.json"
    )
    chosen_validation = read_json(
        run / "trials" / frozen["final_trial"] / "validation-metrics.json"
    )
    baseline_eer = final["baseline"]["metrics"]["verification"]["eer"]
    chosen_eer = final["final"]["metrics"]["verification"]["eer"]
    delta = chosen_eer - baseline_eer
    numerical = {
        "protocol": config,
        "run": relative(run),
        "selected_phonemes": state["accepted"],
        "search_order": state["order"],
        "attempts": state["attempts"],
        "unique_training_trials": audit["training_trials"],
        "unavailable_candidates": frozen["unavailable_candidates"],
        "baseline_validation": baseline_validation,
        "selected_validation": chosen_validation,
        "test": final,
        "bootstrap": bootstrap,
        "independent_audit": audit,
    }
    write_json(BASE / "evaluation-results.json", numerical)
    statuses = {"accepted": "採用", "rejected": "不採用", "unavailable": "データ不足"}
    search_rows = []
    classes = []
    for entry in state["attempts"]:
        search_rows.append(
            [
                entry["step"],
                entry["pass"],
                entry["phoneme"],
                " + ".join(entry["phonemes"][5:]),
                percent(entry["previous_best_eer"]),
                percent(entry["validation_eer"]),
                pp(entry["eer_delta"]),
                statuses[entry["status"]],
                percent(entry["best_eer_after"]),
            ]
        )
        classes.append(entry["status"])
    summary_rows = []
    for split, before, after in (
        ("validation", baseline_validation, chosen_validation),
        ("test", final["baseline"], final["final"]),
    ):
        for role, name in (
            ("verification", "通常文"),
            ("cross_text_verification", "別テキスト"),
        ):
            a, b = before["metrics"][role], after["metrics"][role]
            point_a, point_b = (
                a["operating_points"]["far_1pct"],
                b["operating_points"]["far_1pct"],
            )
            summary_rows.append(
                [
                    split,
                    name,
                    a["all_queries"],
                    a["scored_queries"],
                    percent(a["eer"]),
                    percent(b["eer"]),
                    pp(b["eer"] - a["eer"]),
                    percent(point_a["all_input_far"]),
                    percent(point_b["all_input_far"]),
                    percent(point_a["all_input_frr"]),
                    percent(point_b["all_input_frr"]),
                ]
            )
    inventory_rows = []
    for phone in preparation["candidates"]:
        coverage = preparation["phone_support"][phone]
        inventory_rows.append(
            [
                phone,
                coverage["eligible_intervals"],
                coverage["speakers_with_two_intervals"],
                "可能" if coverage["trainable"] else "データ不足",
                "採用" if phone in state["accepted"] else "未採用",
            ]
        )
    with (BASE / "search-cells.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "step",
                "pass",
                "phoneme",
                "additional_phonemes",
                "previous_best_eer",
                "validation_eer",
                "eer_delta",
                "status",
                "best_eer_after",
            ]
        )
        for entry in state["attempts"]:
            writer.writerow(
                [
                    entry["step"],
                    entry["pass"],
                    entry["phoneme"],
                    "/".join(entry["phonemes"][5:]),
                    entry["previous_best_eer"],
                    entry["validation_eer"],
                    entry["eer_delta"],
                    entry["status"],
                    entry["best_eer_after"],
                ]
            )
    headers = [
        "手順",
        "巡",
        "追加候補",
        "試した子音セット",
        "直前の最良EER",
        "候補EER",
        "差",
        "採否",
        "採否後の最良EER",
    ]
    summary_headers = [
        "集合",
        "発話",
        "全発話",
        "scoreあり",
        "5母音 EER",
        "選定後 EER",
        "EER差",
        "5母音 FAR",
        "選定後 FAR",
        "5母音 FRR",
        "選定後 FRR",
    ]
    outcome = "改善" if delta < 0 else ("同値" if delta == 0 else "上昇")
    selected = "5母音" + (
        " + " + " / ".join(state["accepted"][5:]) if len(state["accepted"]) > 5 else ""
    )
    confidence = bootstrap["verification"]["eer_difference95"]
    note = f"test通常文EERは{percent(baseline_eer)} → {percent(chosen_eer)}（{pp(delta)}、{outcome}）。採用音素は{selected}。"
    unavailable_note = "、".join(frozen["unavailable_candidates"]) or "なし"
    interpretation = f"""# ランダム音素追加と再試験の結果

**{note}**

全{preparation["candidate_count"]}候補を箱に入れ、{audit["training_trials"]}モデルを学習した。
5母音の基準と各候補は同じ初期値から30,000更新。探索中の{len(state["attempts"])}回の採否にはvalidation通常文EERだけを使用した。
validation通常文EERは{percent(baseline_validation["metrics"]["verification"]["eer"])} → {percent(chosen_validation["metrics"]["verification"]["eer"])}。
採用セットが変わった後、不採用候補を元のランダム順で再試験した。
区間不足の候補（{unavailable_note}）は性能による不採用とは区別し、候補一覧に残した。

testは選定と閾値固定後、5母音モデルと最終モデルだけを評価した。
通常文750発話のうち735発話、別テキスト450発話のうち392発話が全モデル共通でscoreを持つ。
5母音不足はFAR/FRRの全入力分母に含め、EERはscoreのある発話で計算した。
通常文test EER差の対応付き話者bootstrap95%区間は[{100 * confidence[0]:+.3f}, {100 * confidence[1]:+.3f}] pp。
この区間は選択済みモデルを固定したもので、探索・学習の変動を含まない。

音素を追加すると、その音素の登録・照合音声も追加される。既存の母音切片は短くしない。
今回測るのは選択した音素セットで学習・登録・照合する構成の性能であり、固定総時間当たりの性能ではない。
元ラベルv/tyのアライメント用acoustic modelはb/chで、音素の真の境界を保証するものではない。
JVS testは過去にも使用した集合であり、新規の独立holdoutではない。

[HTML比較表](evaluation-results.html)・[全手順CSV](search-cells.csv)・[JSON](evaluation-results.json)・[再現手順](README.md)
"""
    (BASE / "interpretation.md").write_text(interpretation)
    markdown = [
        "# 全音素ランダム追加の実測結果",
        "",
        note,
        "",
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in summary_rows)
    markdown.extend(
        [
            "",
            "## 探索の全手順",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in search_rows)
    markdown.extend(
        [
            "",
            f"データ不足として保持した候補: {unavailable_note}。",
            "",
            "採否はvalidation通常文EER、最終確認はtest。FAR/FRRはvalidation通常文でFAR1%を目標に校正した閾値を固定適用した全入力値。",
            "",
            "[結果の解釈](interpretation.md)・[HTML](evaluation-results.html)",
            "",
        ]
    )
    (BASE / "evaluation-results.md").write_text("\n".join(markdown))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    steps = [0, *[e["step"] for e in state["attempts"]]]
    best = [
        baseline_validation["metrics"]["verification"]["eer"],
        *[e["best_eer_after"] for e in state["attempts"]],
    ]
    ax.step(
        steps,
        np.array(best) * 100,
        where="post",
        color="#16704a",
        linewidth=2,
        label="Best validation EER",
    )
    for status, color in (("accepted", "#16704a"), ("rejected", "#8893a1")):
        chosen = [e for e in state["attempts"] if e["status"] == status]
        ax.scatter(
            [e["step"] for e in chosen],
            [e["validation_eer"] * 100 for e in chosen],
            color=color,
            label=status,
            s=24,
        )
    ax.set(
        xlabel="Search step (including retry passes)",
        ylabel="Normal-utterance validation EER (%)",
        title="Random phoneme additions with retry",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(run / "search-curve.png", dpi=160)
    fig.savefig(BASE / "search-curve.svg", metadata={"Date": None})
    plt.close(fig)
    svg = BASE / "search-curve.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    ET.parse(svg)
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全音素ランダム追加・実測結果</title>
<style>body{{font-family:system-ui,sans-serif;background:#f4f6f8;color:#172337;margin:0}}main{{max-width:1440px;margin:auto;padding:32px}}h1{{font-size:28px}}p{{line-height:1.8}}.cards{{display:flex;flex-wrap:wrap;gap:16px}}.card{{background:white;padding:20px;border:1px solid #d8dfe8;border-radius:10px;min-width:180px}}.card strong{{display:block;font-size:28px;margin:8px 0}}.table-wrap{{overflow:auto;border:1px solid #d8dfe8;border-radius:8px;background:white;margin:16px 0 28px}}table{{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}}th,td{{padding:11px 14px;border-bottom:1px solid #e4e9ef;text-align:right}}th{{background:#eaf0f6;position:sticky;top:0}}th:first-child,td:first-child{{text-align:left}}tr.accepted{{background:#e9f6ee}}tr.unavailable{{background:#fff5df}}img{{width:100%;max-width:1000px;background:white;border-radius:8px}}small{{color:#586679}}a{{color:#1768a6}}</style>
<main><h1>全音素からのランダム追加・再試験</h1><p>{html.escape(note)}</p>
<div class="cards"><div class="card">5母音のtest EER<strong>{percent(baseline_eer)}</strong><small>通常文・同じ735発話</small></div><div class="card">選定encoderのtest EER<strong>{percent(chosen_eer)}</strong><small>5母音からの差 {pp(delta)}</small></div><div class="card">探索した全候補<strong>{preparation["candidate_count"]}音素</strong><small>{audit["training_trials"]}モデル学習・{len(state["attempts"])}回の採否</small></div></div>
<p>採用セット：<b>{html.escape(selected)}</b><br>採否はvalidation通常文EER。改善した候補だけを採用し、採用セット変更後に再試験。testは最後の比較だけに使用。<br>データ不足として候補に保持：{html.escape(unavailable_note)}</p>
<h2>基準モデルと選定encoder</h2>{table(summary_headers, summary_rows)}<p><small>FAR/FRRは母音不足を含む全入力値。閾値は各モデルのvalidation通常文で目標FAR1%を校正して固定。追加音素が発話内にある場合に使用し、元の母音を短くせず、その音声時間が追加される。</small></p>
<h2>探索の推移</h2><img src="search-curve.svg" alt="候補EERと採用後の最良EERの推移">
<h2>全手順（緑＝採用、黄＝データ不足）</h2>{table(headers, search_rows, classes)}
<h2>全候補の学習区間数</h2>{table(["音素", "適合区間", "2区間以上ある話者", "学習", "最終セット"], inventory_rows)}
<p><a href="search-cells.csv">全手順CSV</a> · <a href="evaluation-results.json">JSON</a> · <a href="interpretation.md">解釈</a> · <a href="README.md">再現手順</a></p><p><small>JVS＋Common Voice、SRC4VCなし。1探索順・1学習seed。既存JVS test。独立監査：{audit["training_trials"]}学習、{audit["score_checks"]:,}スコア、{audit["eer_checks"]} EER、{audit["rate_checks"]} FAR/FRR値の照合が成功。</small></p></main></html>"""
    (BASE / "evaluation-results.html").write_text(page)
    if page.count("<table>") != 3 or page.count("<tbody>") != 3:
        raise ValueError("publication table count mismatch")
    paths = [
        BASE / name
        for name in (
            "evaluation-results.json",
            "evaluation-results.md",
            "evaluation-results.html",
            "interpretation.md",
            "search-cells.csv",
            "search-curve.svg",
        )
    ]
    write_json(
        run / "publication-freeze.json",
        {
            "files": {relative(p): sha256_file(p) for p in paths},
            "audit_sha256": sha256_file(run / "independent-audit.json"),
            "source_sha256": sha256_file(BASE / "report.py"),
            "search_rows": len(search_rows),
            "summary_rows": len(summary_rows),
            "candidate_rows": len(inventory_rows),
        },
    )
    print(
        json.dumps(
            {
                "stage": "published",
                "test_normal_baseline": baseline_eer,
                "test_normal_final": chosen_eer,
                "selected": state["accepted"],
                "html": relative(BASE / "evaluation-results.html"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    report()
