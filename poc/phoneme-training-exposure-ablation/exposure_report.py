"""Publish the fixed endpoint and both reused references without retuning."""

import csv
import html
import shutil
from html.parser import HTMLParser

from exposure_evaluation import (
    BASE,
    CONDITIONS,
    POINTS,
    ROLES,
    ROOT,
    read_json,
    sha256_file,
    write_json,
)

LABELS = {
    "jvs70_u15000": "JVSのみ・15,000（既存）",
    "cv70_u15000": "JVS＋CV・15,000（既存）",
    "cv70_u30000": "JVS＋CV・30,000（今回）",
}
ROLE_LABELS = {"verification": "通常文", "cross_text_verification": "別テキスト"}


def percent(value):
    return f"{100 * value:.3f}%"


def table(headers, rows):
    md = "|" + "|".join(headers) + "|\n|" + "|".join("---" for _ in headers) + "|\n"
    md += "".join("|" + "|".join(map(str, row)) + "|\n" for row in rows)
    markup = (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
    )
    markup += "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return md, markup + "</tbody></table></div>"


def training_rows(run):
    config = read_json(run / "design-freeze.json")["config"]
    training = ROOT / config["training_run"]
    freeze = read_json(training / "training-freeze.json")
    summary = read_json(training / "expanded/training/summary.json")
    source_summary = read_json(
        ROOT / config["source_training_run"] / "expanded/training/summary.json"
    )
    original = read_json(
        ROOT / freeze["source_settings"]["baseline_run"] / "training/summary.json"
    )
    baseline = freeze["baseline_sampler_counts"]
    previous = {
        s: n for s, n in freeze["source_sampler_counts"].items() if s.startswith("jvs")
    }
    exposure = summary["exposure"]
    return [
        [
            LABELS["jvs70_u15000"],
            70,
            15000,
            f"{min(baseline.values())}～{max(baseline.values())}",
            percent(original["best"]["eer"]),
        ],
        [
            LABELS["cv70_u15000"],
            140,
            15000,
            f"{min(previous.values())}～{max(previous.values())}",
            percent(source_summary["validation_macro_eer"]),
        ],
        [
            LABELS["cv70_u30000"],
            140,
            30000,
            f"{exposure['jvs_label_batch_count_min']}～{exposure['jvs_label_batch_count_max']}",
            percent(summary["validation_macro_eer"]),
        ],
    ]


def tables(run):
    metrics = read_json(run / "test-metrics.json")
    result = [
        (
            "学習条件と実測学習回数",
            [
                "モデル",
                "話者ラベル数",
                "採用update",
                "JVS各話者のbatch選択回数",
                "単一区間validation EER",
            ],
            training_rows(run),
        )
    ]
    for role in ROLES:
        headers = [
            "モデル",
            "全入力FAR",
            "全入力FRR",
            "条件付きFAR",
            "条件付きFRR",
            "発話EER",
            "coverage",
            "他人誤受入/全入力",
            "本人誤拒否＋no_score/全入力",
        ]
        rows = []
        for condition in CONDITIONS:
            cell = metrics["conditions"][f"{condition}/{role}"]
            p = cell["operating_points"]["far_1pct"]
            rows.append(
                [
                    LABELS[condition],
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
        result.append((f"{ROLE_LABELS[role]}（validation目標FAR 1%）", headers, rows))
    headers = ["test", "指標", "30,000−15,000 (pp)", "差の95% CI (pp)"]
    rows = []
    for role in ROLES:
        for name, label in (
            ("far_1pct/all_input_far", "全入力FAR"),
            ("far_1pct/all_input_frr", "全入力FRR"),
            ("pooled_eer", "EER"),
        ):
            d = metrics["paired_differences"][
                f"cv70_u30000_minus_cv70_u15000/{role}/{name}"
            ]
            ci = d["ci95_percentage_points"]
            rows.append(
                [
                    ROLE_LABELS[role],
                    label,
                    f"{d['difference_percentage_points']:+.3f}",
                    f"[{ci['lower']:.3f}, {ci['upper']:.3f}]",
                ]
            )
    result.append(("主比較：同じ追加データで学習を延長した差", headers, rows))
    diagnostics = read_json(run / "diagnostics.json")["decision_transitions"]
    headers = [
        "test",
        "解消した他人誤受入",
        "新たな他人誤受入",
        "救済した本人発話",
        "新たに拒否した本人発話",
    ]
    rows = []
    for role in ROLES:
        d = diagnostics[f"{role}/far_1pct"]["counts"]
        rows.append(
            [
                ROLE_LABELS[role],
                d["impostor"]["accepted_before_only"],
                d["impostor"]["accepted_after_only"],
                d["genuine"]["accepted_after_only"],
                d["genuine"]["accepted_before_only"],
            ]
        )
    result.append(("判定の変化（15,000→30,000、validation目標FAR 1%）", headers, rows))
    headers = ["test", "モデル", "動作点", "全入力FAR", "全入力FRR"]
    rows = []
    for role in ROLES:
        for condition in CONDITIONS:
            cell = metrics["conditions"][f"{condition}/{role}"]
            for point in ("far_0_1pct", "eer_operating"):
                p = cell["operating_points"][point]
                rows.append(
                    [
                        ROLE_LABELS[role],
                        LABELS[condition],
                        point,
                        percent(p["all_input_far"]),
                        percent(p["all_input_frr"]),
                    ]
                )
    result.append(("補助動作点", headers, rows))
    return result


def records(run):
    rows = []
    for split in ("validation", "test"):
        metrics = read_json(run / f"{split}-metrics.json")
        for condition in CONDITIONS:
            for role in ROLES:
                cell = metrics["conditions"][f"{condition}/{role}"]
                for point, _ in POINTS:
                    rows.append(
                        {
                            "split": split,
                            "condition": condition,
                            "role": role,
                            "operating_point": point,
                            "pooled_eer": cell["pooled_eer"],
                            "query_coverage": cell["query_coverage"],
                            "queries": cell["queries"],
                            "scored_queries": cell["scored_queries"],
                            **cell["operating_points"][point],
                        }
                    )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(run):
    output = run / "report"
    output.mkdir(exist_ok=False)
    title = "Phase 8: JVSの学習回数を揃えた追加データモデルの評価"
    intro = "新規条件は1つ・1 seed。前回の追加データモデル15,000 updateから同じoptimizer・scheduler・RNGで30,000 updateまで続けた。データ・統計・encoder・loss・batchは固定。JVS各話者の学習回数を現行JVSモデルの採用時点に揃えた。主比較は同じ追加データの15,000→30,000 update。既存2モデルの結果も併記する。"
    metrics = read_json(run / "test-metrics.json")
    summary = []
    for role in ROLES:
        a = metrics["conditions"][f"cv70_u15000/{role}"]["operating_points"]["far_1pct"]
        b = metrics["conditions"][f"cv70_u30000/{role}"]["operating_points"]["far_1pct"]
        summary.append(
            f"{ROLE_LABELS[role]}の全入力FRRは{percent(a['all_input_frr'])}→{percent(b['all_input_frr'])}、全入力FARは{percent(a['all_input_far'])}→{percent(b['all_input_far'])}。"
        )
    summary = "".join(summary)
    methodology = "登録各母音10区間・5母音必須・同じ照合発話・同じ等重み統合。モデルごとのvalidation通常文で閾値を決め、testへ固定適用した。目標FARはvalidationの校正目標で、testの実測FARを揃えた比較ではない。score不足を全入力FRRの拒否と全入力FARの分母に残す。条件付き指標とEERはscoreありの試行を使う。"
    caveat = "前回のtest結果を観測してから立てた仮説の探索的な切り分け。30,000 updateの採用を新たな学習前に固定し、途中のvalidationやtestで選び直していない。既存2モデルはvalidationで採用したcheckpointであり、選択規則の違いがある。学習率は元のcosine scheduleのまま下がるため、最適な学習予算や学習率を示す実験ではない。独立holdoutは未評価。追加データの匿名話者ラベルとJVS評価話者の実在人物の重複は不明。CIは固定モデル・固定閾値による10,000回の共有話者bootstrap。"
    md = f"# {title}\n\n{intro}\n\n{summary}\n\n{methodology}\n\n{caveat}\n\n"
    sections = f"<h1>{title}</h1><p>{intro}</p><p>{summary}</p><p>{methodology}</p><p class='note'>{caveat}</p>"
    for label, headers, rows in tables(run):
        a, b = table(headers, rows)
        md += f"## {label}\n\n{a}\n"
        sections += f"<h2>{html.escape(label)}</h2>{b}"
    md += "[HTML](evaluation-results.html)、[全条件CSV](all-conditions.csv)、[対応付き差CSV](paired-differences.csv)、[判定変化の内訳](diagnostics.json)、[固定条件と再現手順](README.md)。\n"
    sections += "<p><a href='all-conditions.csv'>全条件CSV</a> / <a href='paired-differences.csv'>対応付き差CSV</a> / <a href='evaluation-results.json'>結果JSON</a> / <a href='diagnostics.json'>判定変化の内訳</a> / <a href='README.md'>再現手順</a></p>"
    markup = (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>"
        + title
        + "</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1300px;margin:40px auto;padding:0 20px;color:#172c3c}h1{font-size:26px}h2{font-size:21px;margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #d6e1e8;padding:9px;white-space:nowrap;text-align:right}th{background:#e8f0f5}.note{padding:12px;background:#f3f6f8}</style></head><body><main>"
        + sections
        + "</main></body></html>\n"
    )
    (output / "evaluation-results.md").write_text(md, encoding="utf-8")
    (output / "evaluation-results.html").write_text(markup, encoding="utf-8")
    write_csv(output / "all-conditions.csv", records(run))
    shutil.copyfile(run / "diagnostics.json", output / "diagnostics.json")
    write_csv(
        output / "paired-differences.csv",
        [
            {
                "comparison": k,
                "difference_percentage_points": v["difference_percentage_points"],
                **v["ci95_percentage_points"],
                "primary": v["primary"],
            }
            for k, v in metrics["paired_differences"].items()
        ],
    )
    config = read_json(run / "design-freeze.json")["config"]
    write_json(
        output / "evaluation-results.json",
        {
            "status": "completed",
            "conditions": metrics["conditions"],
            "paired_differences": metrics["paired_differences"],
            "protocol": config,
            "training_summary": read_json(
                ROOT / config["training_run"] / "expanded/training/summary.json"
            ),
            "design_freeze_sha256": sha256_file(run / "design-freeze.json"),
            "evaluation_freeze_sha256": sha256_file(run / "evaluation-freeze.json"),
            "test_metrics_sha256": sha256_file(run / "test-metrics.json"),
            "new_training_conditions": 1,
            "independent_holdout": False,
        },
    )


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.rows = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
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
        for _, headers, rows in tables(run)
    ]
    if parser.tables != expected:
        raise ValueError("HTML tables differ from measured metrics")
    with (output / "all-conditions.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    if (
        actual != [{k: str(v) for k, v in row.items()} for row in records(run)]
        or len(actual) != 36
    ):
        raise ValueError("CSV differs from measured metrics")
    report = read_json(output / "evaluation-results.json")
    metrics = read_json(run / "test-metrics.json")
    if (
        report["conditions"] != metrics["conditions"]
        or report["paired_differences"] != metrics["paired_differences"]
    ):
        raise ValueError("JSON differs from measured metrics")
    return {
        "html_tables": len(expected),
        "condition_rows": len(actual),
        "paired_rows": len(metrics["paired_differences"]),
    }


def publish(run):
    files = {}
    for path in sorted((run / "report").iterdir()):
        destination = BASE / path.name
        if destination.exists():
            raise ValueError("published report already exists")
        shutil.copyfile(path, destination)
        files[path.name] = sha256_file(destination)
    write_json(
        BASE / "manifest.json",
        {
            "status": "completed",
            "files": files,
            "study_report_sha256": sha256_file(run / "study-report.json"),
        },
    )
