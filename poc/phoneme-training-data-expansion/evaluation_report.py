"""Publish measured utterance metrics and paired intervals as reviewable tables."""

import csv
import html
import shutil
from html.parser import HTMLParser

from evaluation_study import (
    BASE,
    METRICS,
    MODELS,
    POINTS,
    ROLES,
    read_json,
    sha256_file,
    write_json,
)

LABELS = {"jvs70": "JVSのみ", "jvs70_cv70": "JVS＋Common Voice"}
ROLE_LABELS = {"verification": "通常文", "cross_text_verification": "別テキスト"}


def percent(value):
    return f"{100 * value:.3f}%"


def bounds(value):
    return f"[{value['lower']:.3f}, {value['upper']:.3f}]"


def table(headers, rows):
    markdown = (
        "|" + "|".join(headers) + "|\n|" + "|".join("---" for _ in headers) + "|\n"
    )
    markdown += "".join("|" + "|".join(map(str, row)) + "|\n" for row in rows)
    markup = (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
    )
    markup += "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return markdown, markup + "</tbody></table></div>"


def records(run):
    rows = []
    for split in ("validation", "test"):
        metrics = read_json(run / f"{split}-metrics.json")
        for model in MODELS:
            for role in ROLES:
                cell = metrics["conditions"][f"{model}/{role}"]
                for point, _ in POINTS:
                    p = cell["operating_points"][point]
                    row = {
                        "split": split,
                        "model": model,
                        "role": role,
                        "operating_point": point,
                        "pooled_eer": cell["pooled_eer"],
                        "query_coverage": cell["query_coverage"],
                        "queries": cell["queries"],
                        "scored_queries": cell["scored_queries"],
                        **p,
                    }
                    for name in METRICS:
                        ci = (
                            cell.get("ci95", {})
                            .get("operating_points", {})
                            .get(point, {})
                            .get(name)
                        )
                        row[f"{name}_ci95_lower"] = ci["lower"] if ci else None
                        row[f"{name}_ci95_upper"] = ci["upper"] if ci else None
                    rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report_tables(run):
    metrics = read_json(run / "test-metrics.json")
    tables = []
    for role in ROLES:
        headers = [
            "条件",
            "全入力FAR",
            "全入力FRR",
            "条件付きFAR",
            "条件付きFRR",
            "EER",
            "coverage",
            "他人誤受入/全入力",
            "本人誤拒否＋no_score/全入力",
        ]
        rows = []
        for model in MODELS:
            cell = metrics["conditions"][f"{model}/{role}"]
            p = cell["operating_points"]["far_1pct"]
            rows.append(
                [
                    LABELS[model],
                    percent(p["all_input_far"]),
                    percent(p["all_input_frr"]),
                    percent(p["far"]),
                    percent(p["frr"]),
                    percent(cell["pooled_eer"]),
                    percent(cell["query_coverage"]),
                    f"{p['false_accepts']}/{p['all_impostor']}",
                    f"{p['false_rejects']}＋{p['no_score_genuine']}/{p['all_genuine']}",
                ]
            )
        tables.append((f"{ROLE_LABELS[role]}（validation目標FAR 1%）", headers, rows))
    headers = ["test", "指標", "追加−現行 (pp)", "差の95% CI (pp)"]
    rows = []
    for role in ROLES:
        for name, label in (
            ("far_1pct/all_input_far", "全入力FAR"),
            ("far_1pct/all_input_frr", "全入力FRR"),
            ("pooled_eer", "EER"),
        ):
            difference = metrics["paired_differences"][f"{role}/{name}"]
            rows.append(
                [
                    ROLE_LABELS[role],
                    label,
                    f"{difference['difference_percentage_points']:+.3f}",
                    bounds(difference["ci95_percentage_points"]),
                ]
            )
    tables.append(("同じ話者抽選による対応付き差", headers, rows))
    headers = [
        "test",
        "条件",
        "動作点",
        "全入力FAR",
        "全入力FRR",
        "条件付きFAR",
        "条件付きFRR",
    ]
    rows = []
    for role in ROLES:
        for model in MODELS:
            cell = metrics["conditions"][f"{model}/{role}"]
            for point in ("far_0_1pct", "eer_operating"):
                p = cell["operating_points"][point]
                rows.append(
                    [
                        ROLE_LABELS[role],
                        LABELS[model],
                        point,
                        *[
                            percent(p[name])
                            for name in ("all_input_far", "all_input_frr", "far", "frr")
                        ],
                    ]
                )
    tables.append(("補助動作点（validation目標FAR 0.1%・EER動作点）", headers, rows))
    return tables


def render(run):
    output = run / "report"
    output.mkdir(exist_ok=False)
    test = read_json(run / "test-metrics.json")
    title = "Phase 8: 学習データ追加モデルの発話単位評価"
    introduction = "比較は現行JVSとJVS＋Common Voiceの2モデル・1 seed。各母音10区間の登録、5母音必須、発話内の母音scoreの等重み統合を固定した。各モデルのvalidation通常文で閾値を校正し、test通常文・別テキストへ固定適用した。"
    interpretation = []
    for role in ROLES:
        a = test["conditions"][f"jvs70/{role}"]["operating_points"]["far_1pct"]
        b = test["conditions"][f"jvs70_cv70/{role}"]["operating_points"]["far_1pct"]
        interpretation.append(
            f"{ROLE_LABELS[role]}の全入力FRRは{percent(a['all_input_frr'])}→{percent(b['all_input_frr'])}、全入力FARは{percent(a['all_input_far'])}→{percent(b['all_input_far'])}。"
        )
    interpretation = "".join(interpretation)
    caveat = "既に観測済みのJVS testでの探索的な追加検証。追加データの匿名client_idとJVS評価話者の実在人物の重複は確認できない。1 seedで、話者ラベル数・母音区間数・収録環境が同時に増える比較。CIは固定モデル・固定閾値での10,000回の話者bootstrapで、学習・閾値校正の不確かさを含まない。独立holdout、別日・別端末、登録20/30区間・母音不足対応との組み合わせは未評価。"
    denominators = "全入力FRRはscore不足も本人拒否に数え、全入力FARはscore不足も他人試行の分母に残す。条件付きFAR/FRRとEERはscoreありの試行だけで算出する。目標FARはvalidationの校正目標であり、testの実測FARを1%に固定するものではない。"
    md = f"# {title}\n\n{introduction}\n\n{interpretation}\n\n{denominators}\n\n{caveat}\n\n"
    sections = f"<h1>{title}</h1><p>{introduction}</p><p>{interpretation}</p><p>{denominators}</p><p class='note'>{caveat}</p>"
    for label, headers, rows in report_tables(run):
        a, b = table(headers, rows)
        md += f"## {label}\n\n{a}\n"
        sections += f"<h2>{html.escape(label)}</h2>{b}"
    reproducibility = "現行モデルの全18,000 trial/split、validation閾値、test指標とCIは保存済み結果に一致。追加モデルは各splitで15 profileを再生成し、各embeddingを3回反復してbit一致、cacheなしの登録・照合との一致、学習重み・統計・音声の不変を検証した。学習時のvalidation macro EERは9.196→9.274%で、ここに示す発話単位EERとは異なる。"
    md += f"{reproducibility}\n\n[HTML比較表](evaluation-results.html)、[全条件CSV](evaluation-conditions.csv)、[対応付き差CSV](evaluation-paired-differences.csv)、[固定条件・再現手順](README.md)。\n"
    sections += f"<p>{reproducibility}</p><p><a href='evaluation-conditions.csv'>全条件CSV</a> / <a href='evaluation-paired-differences.csv'>対応付き差CSV</a> / <a href='evaluation-results.json'>結果JSON</a> / <a href='README.md'>再現手順</a></p>"
    markup = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>"
        + title
        + "</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 20px;color:#172c3c}h1{font-size:26px}h2{font-size:21px;margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #d6e1e8;padding:9px;white-space:nowrap;text-align:right}th{background:#e8f0f5}.note{padding:12px;background:#f3f6f8}</style></head><body><main>"
        + sections
        + "</main></body></html>\n"
    )
    (output / "evaluation-results.md").write_text(md, encoding="utf-8")
    (output / "evaluation-results.html").write_text(markup, encoding="utf-8")
    write_csv(output / "evaluation-conditions.csv", records(run))
    write_csv(
        output / "evaluation-paired-differences.csv",
        [
            {
                "condition": key,
                "difference_percentage_points": item["difference_percentage_points"],
                **item["ci95_percentage_points"],
            }
            for key, item in test["paired_differences"].items()
        ],
    )
    write_json(
        output / "evaluation-results.json",
        {
            "status": "completed",
            "conditions": test["conditions"],
            "paired_differences": test["paired_differences"],
            "protocol": read_json(run / "design-freeze.json")["config"],
            "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
            "evaluation_freeze_sha256": sha256_file(run / "evaluation-freeze.json"),
            "test_metrics_sha256": sha256_file(run / "test-metrics.json"),
            "test_threshold_recalibrated": False,
            "independent_holdout": False,
            "cross_corpus_person_overlap": "unknown",
        },
    )


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell, self.links = (
            [],
            None,
            None,
            None,
            [],
        )

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = ""
        elif tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
        elif tag == "table":
            self.tables.append(self.rows)


def audit_report(run):
    output = run / "report"
    parser = Tables()
    parser.feed((output / "evaluation-results.html").read_text())
    expected = [
        [headers, *[[str(v) for v in row] for row in rows]]
        for _, headers, rows in report_tables(run)
    ]
    if parser.tables != expected:
        raise ValueError("HTML table differs from measured metrics")
    with (output / "evaluation-conditions.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    expected_rows = [
        {k: str(v) if v is not None else "" for k, v in row.items()}
        for row in records(run)
    ]
    if actual != expected_rows or len(actual) != 24:
        raise ValueError("CSV numbers differ from measured metrics")
    published = read_json(output / "evaluation-results.json")
    metrics = read_json(run / "test-metrics.json")
    if (
        published["conditions"] != metrics["conditions"]
        or published["paired_differences"] != metrics["paired_differences"]
    ):
        raise ValueError("report JSON differs from measured metrics")
    return {
        "html_tables": len(expected),
        "condition_rows": len(actual),
        "paired_rows": len(metrics["paired_differences"]),
        "links": parser.links,
    }


def publish(run):
    files = {}
    for path in sorted((run / "report").iterdir()):
        destination = BASE / path.name
        if destination.exists():
            raise ValueError(f"published evaluation already exists: {destination}")
        shutil.copyfile(path, destination)
        files[path.name] = sha256_file(destination)
    write_json(
        BASE / "evaluation-manifest.json",
        {
            "status": "completed",
            "files": files,
            "study_report_sha256": sha256_file(run / "study-report.json"),
        },
    )
