"""Publish the audited fixed-all36 enrollment comparison as tables."""

from __future__ import annotations

import html
import json
from html.parser import HTMLParser

import study
from common import read_json, relative, sha256_file, write_json

BASE = study.BASE


def pct(value):
    return f"{100 * value:.3f}%"


def markdown_table(headers, records):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in records],
        ]
    )


def html_table(headers, records):
    head = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>"
        for row in records
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag == "td":
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == "td":
            self.row.append(self.cell)
            self.cell = None
        if tag == "tr" and self.row:
            self.records.append(self.row)


def main():
    config = read_json(study.CONFIG)
    run = study.ROOT / config["run_directory"]
    frozen = study.verify(config, run)
    completion = read_json(run / "completion-verification.json")
    if completion["status"] != "passed":
        raise ValueError("publish after independent numerical audit passes")
    for name, sha in completion["output_sha256"].items():
        study.checked(study.ROOT / name, sha)
    data = {
        "protocol": config,
        "phonemes": frozen["phonemes"],
        "untrained_phonemes": frozen["untrained_phonemes"],
        "enrollment": frozen["enrollment"],
        "bootstrap": read_json(run / "bootstrap-results.json"),
        "audit": read_json(run / "independent-audit.json"),
        **{
            split: {
                str(n): read_json(study.condition(run, n) / f"{split}-metrics.json")
                for n in config["enrollment_limits"]
            }
            for split in study.SPLITS
        },
    }
    write_json(BASE / "evaluation-results.json", data)
    headers = [
        "集合",
        "発話",
        "登録上限",
        "全発話",
        "scoreあり",
        "EER",
        "10との差",
        "FAR",
        "FRR",
        "本人平均",
        "他人平均",
        "平均差",
        "登録音声秒/人中央値",
    ]
    records = []
    for split in study.SPLITS:
        for role, label in zip(study.ROLES, ("通常文", "別テキスト"), strict=True):
            baseline = data[split]["10"]["metrics"][role]["eer"]
            for n in config["enrollment_limits"]:
                report = data[split][str(n)]
                m, dist = report["metrics"][role], report["score_distribution"][role]
                point = m["operating_points"]["far_1pct"]
                records.append(
                    [
                        split,
                        label,
                        n,
                        m["all_queries"],
                        m["scored_queries"],
                        pct(m["eer"]),
                        f"{100 * (m['eer'] - baseline):+.3f} pp",
                        pct(point["all_input_far"]),
                        pct(point["all_input_frr"]),
                        f"{dist['genuine']['mean']:.6f}",
                        f"{dist['impostor']['mean']:.6f}",
                        f"{dist['mean_separation']:.6f}",
                        f"{data['enrollment'][split][str(n)]['used_audio_seconds_per_speaker_median']:.3f}",
                    ]
                )
    ci_headers, ci_records = (
        ["test発話", "登録上限差", "EER差", "95%下限", "95%上限"],
        [],
    )
    for role, label in zip(study.ROLES, ("通常文", "別テキスト"), strict=True):
        for pair in ("20-10", "30-10", "30-20"):
            cell = data["bootstrap"][f"{pair}/{role}"]
            ci_records.append(
                [
                    label,
                    pair,
                    *[
                        f"{100 * x:+.3f} pp"
                        for x in [cell["eer_difference"], *cell["eer_difference95"]]
                    ],
                ]
            )
    phone_headers = [
        "集合",
        "音素",
        "照合に使用",
        "上限10 実数/人",
        "上限20 実数/人",
        "上限30 実数/人",
        "上限30未満の話者",
    ]
    phone_records = []
    for split in study.SPLITS:
        for p in [*frozen["phonemes"], *frozen["untrained_phonemes"]]:
            supports = [
                data["enrollment"][split][str(n)]["per_phone"].get(p)
                for n in config["enrollment_limits"]
            ]
            sizes = [f"{x['minimum']}–{x['maximum']}" if x else "0–0" for x in supports]
            phone_records.append(
                [
                    split,
                    p,
                    "可" if supports[0] and supports[0]["used_for_scoring"] else "不可",
                    *sizes,
                    supports[-1]["speakers_below_limit"] if supports[-1] else 15,
                ]
            )
    normal, cross = (data["test"]["30"]["metrics"][r]["eer"] for r in study.ROLES)
    normal10, cross10 = (data["test"]["10"]["metrics"][r]["eer"] for r in study.ROLES)
    lead = f"登録上限10→30でtest EERは通常文{pct(normal10)}→{pct(normal)}、別テキスト{pct(cross10)}→{pct(cross)}。"
    note = "FAR/FRRは登録上限ごとにvalidation通常文で目標FAR1%へ校正した閾値をtestへ固定適用した全入力値。EERは同じscoreあり発話から算出。"
    design_note = "学習済み全36音素モデルを固定し、再学習なし。音素の種類・照合発話・音素間の等重みは共通。登録10は20/30に含まれる。実区間が不足する音素は上限まで水増ししない。"
    tables = [
        markdown_table(headers, records),
        markdown_table(ci_headers, ci_records),
        markdown_table(phone_headers, phone_records),
    ]
    markdown = f"# 全音素モデルの登録区間数比較\n\n{lead}\n\n{design_note}\n\n{tables[0]}\n\n{note}\n\n## 対応付き話者bootstrap\n\n差が負ならEER改善。2,000回、同じ話者抽出を全条件に共有。固定モデル・既存15 test話者に条件付けた区間。\n\n{tables[1]}\n\n## 実際の登録区間数\n\n各セルは15話者の最小–最大。tyは未学習。登録音声が全話者で揃わない音素は前回と同じく照合に使わない。\n\n{tables[2]}\n\n[HTML比較表](evaluation-results.html)・[JSON](evaluation-results.json)・[解釈](interpretation.md)・[再現手順](README.md)\n"
    (BASE / "evaluation-results.md").write_text(markdown)
    statements = []
    for role, label in zip(study.ROLES, ("通常文", "別テキスト"), strict=True):
        cell = data["bootstrap"][f"30-10/{role}"]
        ci = cell["eer_difference95"]
        eers = [
            pct(data["test"][str(n)]["metrics"][role]["eer"])
            for n in config["enrollment_limits"]
        ]
        d0, d1 = (data["test"][str(n)]["score_distribution"][role] for n in (10, 30))
        statements.append(
            f"- {label}test EER（上限10・20・30）：{' → '.join(eers)}。30−10差{100 * cell['eer_difference']:+.3f} pp、95%区間[{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] pp。本人平均{d0['genuine']['mean']:.6f}→{d1['genuine']['mean']:.6f}、他人平均{d0['impostor']['mean']:.6f}→{d1['impostor']['mean']:.6f}、平均差{d0['mean_separation']:.6f}→{d1['mean_separation']:.6f}。"
        )
    interpretation = (
        f"# 全音素モデルの登録区間数を増やした結果\n\n**{lead}**\n\n"
        + "\n".join(statements)
        + f"\n\n{design_note}\n\n各上限のvalidation閾値を全て固定してからtestを採点し、testで音素・モデル・閾値を選び直していない。通常文750発話中735、別テキスト450発話中392を全条件共通に評価。登録10の全36,000 trial・embedding・指標・閾値は前回と完全一致した。\n\n登録上限30でも全音素に30件あるわけではない。希少音素を捨てず、同じ登録候補音声から取得可能な区間だけを使用した。増えた登録音声量と音素別実数は測定表に記録している。\n\nEERは本人と他人のスコア分布の分離を評価する。本人平均が上がっただけで精度向上とは判定せず、EERとFAR/FRRも併記した。今回の既存JVS testは以前から観測済みで、未観測話者への確証ではない。\n\n[測定表](evaluation-results.md)・[HTML](evaluation-results.html)・[再現手順](README.md)\n"
    )
    (BASE / "interpretation.md").write_text(interpretation)
    page = f"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全音素モデル：登録10・20・30比較</title>
<style>body{{font-family:system-ui,sans-serif;background:#f4f6f8;color:#172337;margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}p{{line-height:1.8}}.table-wrap{{overflow:auto;background:white;border:1px solid #d8dfe8;border-radius:8px;margin:20px 0}}table{{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}}th,td{{padding:12px;border-bottom:1px solid #e4e9ef;text-align:right}}th{{background:#eaf0f6}}td:first-child,th:first-child{{text-align:left}}a{{color:#1768a6}}</style>
<main><h1>全音素モデル：登録区間数10・20・30</h1><p><b>{html.escape(lead)}</b></p><p>{html.escape(design_note)}</p>
<h2>EERと本人・他人スコア</h2>{html_table(headers, records)}<p>{html.escape(note)}</p>
<h2>対応付き話者bootstrapによるtest EER差</h2>{html_table(ci_headers, ci_records)}<p>差が負なら改善。固定モデル・15話者・2,000回。既存testを使った追加評価。</p>
<h2>音素ごとの実際の登録区間数</h2>{html_table(phone_headers, phone_records)}<p>数値は15話者の最小–最大。全音素を残し、実在区間だけを使用。tyは未学習。</p>
<p><a href="evaluation-results.json">JSON</a> · <a href="evaluation-results.md">測定表</a> · <a href="interpretation.md">解釈</a> · <a href="README.md">再現手順</a></p></main></html>
"""
    (BASE / "evaluation-results.html").write_text(page)
    reader = TableReader()
    reader.feed(page)
    expected = [[str(x) for x in r] for r in [*records, *ci_records, *phone_records]]
    markdown_rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in markdown.splitlines()
        if line.startswith("| ")
    ]
    markdown_values = [
        row
        for row in markdown_rows
        if row not in (headers, ci_headers, phone_headers)
        and not all(cell == "---" for cell in row)
    ]
    if reader.records != expected or markdown_values != expected:
        raise ValueError("HTML/Markdown table values differ from JSON-derived rows")
    for name in (
        "evaluation-results.json",
        "evaluation-results.md",
        "interpretation.md",
        "README.md",
    ):
        if not (BASE / name).is_file():
            raise ValueError("missing report link")
    files = {
        relative(BASE / n): sha256_file(BASE / n)
        for n in (
            "evaluation-results.json",
            "evaluation-results.md",
            "evaluation-results.html",
            "interpretation.md",
        )
    }
    write_json(
        run / "publication-freeze.json",
        {
            "files": files,
            "source_sha256": sha256_file(BASE / "report.py"),
            "audit_sha256": sha256_file(run / "independent-audit.json"),
            "table_rows": [len(records), len(ci_records), len(phone_records)],
            "html_markdown_values_verified": True,
            "links_verified": True,
        },
    )
    print(
        json.dumps(
            {
                "stage": "published",
                "summary": lead,
                "table_rows": [len(records), len(ci_records), len(phone_records)],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
