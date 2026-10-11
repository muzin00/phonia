"""Generate auditable HTML/Markdown tables from measured results only."""

from __future__ import annotations

import html
import json
from html.parser import HTMLParser

import shared as s

CASES = {
    "clean": "clean",
    "missing-half-consonants": "子音50%欠損",
    "noise-20pct-intervals-snr10": "20%区間に10dBノイズ",
}
ROLES = {s.ROLES[0]: "通常", s.ROLES[1]: "別文"}
GROUPS = {
    "equal": "等重み",
    "previous-mlp": "前回MLP",
    "augmented-mlp": "追加学習MLP",
    "previous-transformer": "前回Transformer",
    "augmented-transformer": "追加学習Transformer",
}


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.links = [], []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attributes):
        if tag == "table":
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""
        elif tag == "a":
            self.links.extend(v for k, v in attributes if k == "href")

    def handle_data(self, value):
        if self.cell is not None:
            self.cell += value

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.table.append(self.row)
            self.row = None
        elif tag == "table":
            self.tables.append(self.table)
            self.table = None


def percent(value):
    return f"{value * 100:.3f}%"


def summarize(results, seeds):
    summary = {}
    for case, variants in results["scenarios"].items():
        summary[case] = {}
        for group in GROUPS:
            names = (
                ["equal"] if group == "equal" else [f"{group}-{seed}" for seed in seeds]
            )
            summary[case][group] = {}
            for role in s.ROLES:
                values = [variants[n][role] for n in names]
                eers = [v["eer"] for v in values]
                summary[case][group][role] = {
                    "eer_mean": float(s.np.mean(eers)),
                    "eer_min": min(eers),
                    "eer_max": max(eers),
                    "all_input_far_mean": float(
                        s.np.mean(
                            [
                                v["operating_points"]["far_1pct"]["all_input_far"]
                                for v in values
                            ]
                        )
                    ),
                    "all_input_frr_mean": float(
                        s.np.mean(
                            [
                                v["operating_points"]["far_1pct"]["all_input_frr"]
                                for v in values
                            ]
                        )
                    ),
                }
    return summary


def make_tables(results, summary, bootstrap):
    comparison, detail, rates, intervals, selected, history = [], [], [], [], [], []
    for case, groups in summary.items():
        for role in s.ROLES:
            control = groups["previous-transformer"][role]["eer_mean"]
            augmented = groups["augmented-transformer"][role]["eer_mean"]
            comparison.append(
                [
                    CASES[case],
                    ROLES[role],
                    *[
                        percent(groups[g][role]["eer_mean"])
                        for g in (
                            "equal",
                            "previous-transformer",
                            "augmented-transformer",
                        )
                    ],
                    f"{(augmented - control) * 100:+.3f}",
                ]
            )
        for group, roles in groups.items():
            for role, values in roles.items():
                rates.append(
                    [
                        CASES[case],
                        GROUPS[group],
                        ROLES[role],
                        percent(values["eer_mean"]),
                        f"{percent(values['eer_min'])} ～ {percent(values['eer_max'])}",
                        percent(values["all_input_far_mean"]),
                        percent(values["all_input_frr_mean"]),
                    ]
                )
        for name, roles in results["scenarios"][case].items():
            detail.append(
                [CASES[case], name, *[percent(roles[r]["eer"]) for r in s.ROLES]]
            )
        for role in s.ROLES:
            for control in ("previous", "equal"):
                value = bootstrap["differences"][
                    f"{case}/augmented-transformer-mean-minus-{control}/{role}"
                ]
                intervals.append(
                    [
                        CASES[case],
                        ROLES[role],
                        "前回Transformer" if control == "previous" else "等重み",
                        f"{value['difference'] * 100:+.3f}",
                        f"[{value['ci95'][0] * 100:+.3f}, {value['ci95'][1] * 100:+.3f}]",
                    ]
                )
    for name, values in results["training"].items():
        chosen = next(
            h for h in values["history"] if h["update"] == values["selected_update"]
        )
        selected.append(
            [
                name,
                str(values["parameters"]),
                str(values["completed_updates"]),
                str(values["selected_update"]),
                percent(chosen["normal_eer"]),
                percent(chosen["cross_text_eer"]),
                "/".join(str(n) for n in values["presented_queries_by_mode"]),
            ]
        )
        history.extend(
            [
                name,
                str(h["update"]),
                percent(h["normal_eer"]),
                percent(h["cross_text_eer"]),
            ]
            for h in values["history"]
        )
    return [
        {
            "title": "Transformer test EER（3 seed平均）",
            "headers": [
                "入力条件",
                "評価",
                "等重み",
                "前回clean学習",
                "ノイズ・欠損追加学習",
                "前回との差 pp",
            ],
            "rows": comparison,
        },
        {
            "title": "全方式の平均・範囲と固定閾値FAR/FRR",
            "headers": [
                "入力条件",
                "方式",
                "評価",
                "平均EER",
                "EER範囲",
                "全入力FAR",
                "全入力FRR",
            ],
            "rows": rates,
        },
        {
            "title": "全seedのtest EER",
            "headers": ["入力条件", "方式 / seed", "通常EER", "別文EER"],
            "rows": detail,
        },
        {
            "title": "話者bootstrap：追加学習Transformer平均との差",
            "headers": ["入力条件", "評価", "比較先", "平均差 pp", "95%区間 pp"],
            "rows": intervals,
        },
        {
            "title": "学習完了とclean validationでの採用",
            "headers": [
                "方式 / seed",
                "追加パラメータ",
                "完了更新",
                "採用更新",
                "採用通常EER",
                "採用別文EER",
                "clean/noise/missing/both提示数",
            ],
            "rows": selected,
        },
        {
            "title": "clean validation学習曲線",
            "headers": ["方式 / seed", "更新", "通常EER", "別文EER"],
            "rows": history,
        },
    ]


def interpretation(results, summary):
    paragraphs = [
        "全36音素encoderと登録上限30区間/音素を固定し、統合モデルの学習入力にだけノイズ・子音欠損を追加した。clean / noise / missing / 両方を各25%で提示し、前回と同じ3 seed・本人/他人ペア列・4,000更新・optimizer・モデル構造を使った。",
        "各モデルはclean validationの通常EERが最小になるcheckpointを採用した。同値は早い更新を採用し、初期の等重みcheckpointも候補に含めた。6モデルとclean validation由来の閾値をすべて固定してからtestを計算した。",
    ]
    for case, groups in summary.items():
        values = []
        for role in s.ROLES:
            old = groups["previous-transformer"][role]["eer_mean"]
            new = groups["augmented-transformer"][role]["eer_mean"]
            equal = groups["equal"][role]["eer_mean"]
            values.append(
                f"{ROLES[role]} {percent(old)}→{percent(new)}（等重み {percent(equal)}、前回比{(new - old) * 100:+.3f} pp）"
            )
        paragraphs.append(
            f"{CASES[case]}のTransformer平均EER: " + "、".join(values) + "。"
        )
    zero = [
        name
        for name, values in results["training"].items()
        if values["selected_update"] == 0
    ]
    if zero:
        paragraphs.append(
            "初期checkpoint採用: "
            + ", ".join(zero)
            + "。これらのtest値は等重みであり、学習後の重みの性能ではない。全6本の学習そのものは4,000更新まで完了した。"
        )
    noise = results["training_noise"]
    paragraphs.extend(
        [
            f"学習ノイズは実波形の {noise['corrupted_intervals']:,}/{noise['eligible_training_query_intervals']:,} 区間に白色ノイズを加えて固定encoderで再抽出した。{noise['clean_waveform_replay_probes']}区間のclean波形再計算は既存embeddingと完全一致。クリップされたsampleは {noise['clipped_samples']:,}、実測SNRは {noise['achieved_snr_db_minimum']:.3f}～{noise['achieved_snr_db_maximum']:.3f} dB。登録側と5母音の支持は維持した。",
            "testには前回と同じ欠損マスク・ノイズ区間・noise seedを使用した。学習noise seedは別で、trainとvalidation/testは話者・音声hashが非重複。本人/他人ペアには同じ加工を使用し、全方式で同じ照合対象と音素別cosineを採点した。",
            "FAR/FRRは各方式のclean validation FAR 1%閾値を固定し、母音不足による採点不能も拒否として含めた全入力率。EERは採点可能な入力のROC全体から計算している。",
            "平均は3モデルのEER平均で、ensembleではない。bootstrapは同じ15 test話者を2,000回再抽出した対応差で、今回の固定モデルに対する話者不確実性を表す。学習seed全般の信頼区間ではない。前回参照済みのtestを使う追加検証で、独立した未知データへの確証とは扱わない。",
            "音素境界は元の切り出し位置を固定し、ノイズ後の再alignment・追加QCは行っていない。学習ノイズは各queryに固定した1変種で、今回はノイズと欠損をまとめて追加したため、どちらの加工が効いたかは分離していない。",
        ]
    )
    return paragraphs


def publish(config, run):
    s.verify(config, run)
    completion = s.read_json(run / "completion-verification.json")
    if completion["status"] != "passed":
        raise ValueError("independent audit required")
    for name, checksum in completion["output_sha256"].items():
        s.checked(s.ROOT / name, checksum)
    results = s.read_json(run / "results.json")
    bootstrap = s.read_json(run / "bootstrap-results.json")
    seeds = s.read_json(run / "design-freeze.json")["training_protocol"]["seeds"]
    summary = summarize(results, seeds)
    tables = make_tables(results, summary, bootstrap)
    paragraphs = interpretation(results, summary)
    markdown = [
        "# ノイズ・欠損を追加した音素統合学習の結果",
        "",
        *[p + "\n" for p in paragraphs],
    ]
    fragments = []
    for table in tables:
        headers, cells = table["headers"], table["rows"]
        if any(len(row) != len(headers) for row in cells):
            raise ValueError("table width mismatch")
        markdown.extend(
            [
                "## " + table["title"],
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *["| " + " | ".join(row) + " |" for row in cells],
                "",
            ]
        )
        fragments.append(
            "<h2>"
            + html.escape(table["title"])
            + "</h2><div class='scroll'><table><thead><tr>"
            + "".join("<th>" + html.escape(h) + "</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join("<td>" + html.escape(c) + "</td>" for c in row)
                + "</tr>"
                for row in cells
            )
            + "</tbody></table></div>"
        )
    markdown.extend(
        ["[HTML比較表](evaluation-results.html) · [設計・再実行](README.md)", ""]
    )
    document = (
        "<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ノイズ・欠損追加学習の結果</title><style>body{max-width:1320px;margin:32px auto;padding:0 20px;font-family:system-ui;color:#172033;line-height:1.8;background:#f8fafc}h1{font-size:1.8rem}h2{font-size:1.25rem;margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #cbd5e1;padding:9px 12px;text-align:left;white-space:nowrap}th{background:#e2e8f0}tbody tr:nth-child(even){background:#f1f5f9}a{color:#155e75}</style><h1>ノイズ・欠損追加学習の結果</h1>"
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs)
        + "".join(fragments)
        + "<p><a href='evaluation-results.md'>Markdown</a> · <a href='evaluation-results.json'>JSON</a> · <a href='README.md'>設計・再実行</a></p></html>"
    )
    (s.BASE / "evaluation-results.md").write_text("\n".join(markdown), encoding="utf-8")
    (s.BASE / "evaluation-results.html").write_text(document, encoding="utf-8")
    s.write_json(
        s.BASE / "evaluation-results.json",
        {
            "source_run": str(run.relative_to(s.ROOT)),
            "summary": summary,
            "tables": tables,
            "paragraphs": paragraphs,
            "bootstrap": bootstrap,
            "audit": {
                k: v
                for k, v in completion.items()
                if k not in ("checks", "output_sha256")
            },
            "source_sha256": {
                n: s.sha256_file(run / n)
                for n in (
                    "results.json",
                    "bootstrap-results.json",
                    "independent-audit.json",
                    "completion-verification.json",
                )
            },
        },
    )
    parser = TableParser()
    parser.feed(document)
    expected = [[table["headers"], *table["rows"]] for table in tables]
    if parser.tables != expected:
        raise ValueError("HTML differs from source table cells")
    for link in parser.links:
        if not (s.BASE / link).is_file():
            raise ValueError(f"broken report link: {link}")
    source_json = s.read_json(s.BASE / "evaluation-results.json")
    if source_json["tables"] != tables:
        raise ValueError("JSON table differs")
    md_rows = [
        line
        for line in (s.BASE / "evaluation-results.md").read_text().splitlines()
        if line.startswith("| ") and not line.startswith("| ---")
    ]
    if md_rows != [
        "| " + " | ".join(row) + " |" for table in expected for row in table
    ]:
        raise ValueError("Markdown table differs")
    s.write_json(
        run / "publication-freeze.json",
        {
            "status": "passed",
            "script_sha256": s.sha256_file(__file__),
            "html_tables": len(tables),
            "data_rows": sum(len(t["rows"]) for t in tables),
            "checked_cells": sum(len(r) for table in expected for r in table),
            "html_markdown_json_cells_identical": True,
            "report_links_exist": True,
            "files": {
                str((s.BASE / n).relative_to(s.ROOT)): s.sha256_file(s.BASE / n)
                for n in (
                    "README.md",
                    "evaluation-results.html",
                    "evaluation-results.md",
                    "evaluation-results.json",
                )
            },
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    s.torch.set_num_threads(1)
    settings = s.read_json(s.CONFIG)
    publish(settings, s.ROOT / settings["run_directory"])
