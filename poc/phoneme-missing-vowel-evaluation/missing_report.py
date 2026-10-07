"""Static tables, complete CSVs and a scientific plot of fixed-threshold results."""

import csv
import json
import shutil
from html.parser import HTMLParser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from missing_study import BASE, METHODS, POINTS, ROLES, ROOT, sha256_file, write_json
from scaling_report import bounds, percent, table

LABELS = {"strict5": "5母音必須", "available4": "4母音以上", "available3": "3母音以上"}
ROLE_LABELS = {"verification": "通常文", "cross_text_verification": "別テキスト"}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def csv_documents(validation, test):
    condition_rows, paired_rows, transition_rows, pattern_rows = [], [], [], []
    for split, document in (("validation", validation), ("test", test)):
        for key, cell in document["conditions"].items():
            method, role = key.split("/")
            for point, _ in POINTS:
                p = cell["operating_points"][point]
                row = {
                    "split": split,
                    "method": method,
                    "role": role,
                    "point": point,
                    "coverage": cell["query_coverage"],
                    "eer": cell["pooled_eer"],
                    **p,
                }
                if split == "test":
                    for metric, ci in cell["ci95"]["operating_points"][point].items():
                        row[f"{metric}_ci95_lower"] = ci["lower"]
                        row[f"{metric}_ci95_upper"] = ci["upper"]
                condition_rows.append(row)
        transitions = (
            test["decision_transitions"]
            if split == "test"
            else test["validation_decision_transitions"]
        )
        transition_rows.extend(
            {"split": split, "condition": k, **v} for k, v in transitions.items()
        )
        diagnostics = (
            test["missing_diagnostics"]
            if split == "test"
            else test["validation_missing_diagnostics"]
        )
        for key, item in diagnostics.items():
            for point, rates in item["operating_points"].items():
                pattern_rows.append(
                    {
                        "split": split,
                        "condition": key,
                        "point": point,
                        "queries": item["queries"],
                        "speakers": item["speakers"],
                        "scored_queries": item["scored_queries"],
                        **rates,
                    }
                )
    for key, item in test["paired_differences"].items():
        paired_rows.append(
            {
                "comparison": key,
                "difference_percentage_points": item["difference_percentage_points"],
                **{f"ci95_{k}": v for k, v in item["ci95_percentage_points"].items()},
            }
        )
    return {
        "all-conditions.csv": condition_rows,
        "paired-differences.csv": paired_rows,
        "decision-transitions.csv": transition_rows,
        "missing-patterns.csv": pattern_rows,
    }


def render(run):
    output = run / "report"
    output.mkdir(exist_ok=False)
    test = json.loads((run / "test-metrics.json").read_text())
    validation = json.loads((run / "validation-metrics.json").read_text())
    title = "Phase 8: 照合時の母音不足に対応する実測比較"
    intro = "登録は各母音10区間・固定境界内母音encoderを使用。同じ発話全区間で、5母音必須・4母音以上・3母音以上を比較した。取得できた母音のsegment cosineを母音内で平均し、存在する母音のscoreだけを等重み平均する。各方式のvalidation通常文で校正した閾値をtestへ固定適用した。"
    scope = "観測済みJVS testによる探索的な追加検証。再学習・品質重み・Transformer・本番APIの切り替えは行っていない。3母音の通常文validationデータはなく、3母音以上と4母音以上の閾値は同じ。未観測holdout、別日・別端末の性能は未確認。CIは固定モデル・固定閾値の10,000回話者bootstrapで、校正の不確かさと多重比較の補正は含まない。"
    markdown = f"# {title}\n\n{intro}\n\n{scope}\n\n"
    markup = f"<h1>{title}</h1><p>{intro}</p><p class='note'>{scope}</p>"
    tables = []

    def add_table(heading, headers, rows, note=""):
        nonlocal markdown, markup
        md, html = table(headers, rows)
        markdown += f"## {heading}\n\n{note}\n\n{md}\n"
        markup += f"<h2>{heading}</h2><p>{note}</p>{html}"
        tables.append(
            {"headers": headers, "rows": [[str(x) for x in row] for row in rows]}
        )

    for role in ROLES:
        rows = []
        for method in METHODS:
            cell = test["conditions"][f"{method}/{role}"]
            p = cell["operating_points"]["far_1pct"]
            rows.append(
                [
                    LABELS[method],
                    percent(p["far"]),
                    percent(p["all_input_far"]),
                    percent(p["frr"]),
                    percent(p["all_input_frr"]),
                    bounds(
                        cell["ci95"]["operating_points"]["far_1pct"]["all_input_frr"]
                    ),
                    percent(cell["pooled_eer"]),
                    percent(cell["query_coverage"]),
                    f"{p['false_accepts']}/{p['impostor']}",
                    f"{p['false_rejects']}/{p['genuine']}",
                    f"{p['no_score_genuine']}/{p['all_genuine']}",
                ]
            )
        add_table(
            f"{ROLE_LABELS[role]}（validation目標FAR 1%）",
            [
                "方式",
                "条件付きFAR",
                "全入力FAR",
                "条件付きFRR",
                "全入力FRR",
                "全入力FRR 95% CI (%)",
                "EER",
                "coverage",
                "他人誤受入/scoreあり",
                "本人誤拒否/scoreあり",
                "本人no_score/全入力",
            ],
            rows,
            "FARとFRRの条件付き分母はscoreが出たtrial。全入力FRRにはno_scoreも拒否として含める。",
        )
    rows = []
    for role in ROLES:
        for method in METHODS:
            cell = test["conditions"][f"{method}/{role}"]
            p = cell["operating_points"]["far_0_1pct"]
            rows.append(
                [
                    ROLE_LABELS[role],
                    LABELS[method],
                    percent(p["far"]),
                    percent(p["all_input_far"]),
                    percent(p["all_input_frr"]),
                    f"{p['false_accepts']}/{p['impostor']}",
                    f"{p['false_rejects']}+{p['no_score_genuine']}/{p['all_genuine']}",
                ]
            )
    add_table(
        "厳しい動作点（validation目標FAR 0.1%）",
        [
            "発話条件",
            "方式",
            "条件付きFAR",
            "全入力FAR",
            "全入力FRR",
            "他人誤受入/scoreあり",
            "本人誤拒否+no_score/全入力",
        ],
        rows,
        "目標値はvalidationの校正条件であり、testの実測FARが同じになるとは限らない。",
    )
    rows = []
    for role in ROLES:
        for method in ("available4", "available3"):
            p = test["decision_transitions"][f"{method}/{role}/far_1pct"]
            rows.append(
                [
                    ROLE_LABELS[role],
                    LABELS[method],
                    p["missing_genuine"],
                    p["rescued_missing_genuine"],
                    p["still_rejected_missing_genuine"],
                    p["rescued_scored_genuine"],
                    p["newly_rejected_scored_genuine"],
                    p["added_false_accepts_missing"],
                    p["added_false_accepts_complete"],
                    p["removed_false_accepts"],
                ]
            )
    add_table(
        "本人救済と他人受入の内訳（主条件、対5母音必須）",
        [
            "発話条件",
            "方式",
            "元の不足本人",
            "救済した不足本人",
            "依然拒否した不足本人",
            "救済した既存score本人",
            "新たに拒否した既存本人",
            "不足発話の追加他人受入",
            "既存発話の追加他人受入",
            "減った他人受入",
        ],
        rows,
        "5母音揃う発話のscoreは全方式で同じ。既存発話の判定差は方式ごとの固定閾値の違いによる。",
    )
    rows = []
    for role in ROLES:
        for method in ("available4", "available3"):
            for metric, label in (
                ("all_input_frr", "全入力FRR"),
                ("all_input_far", "全入力FAR"),
                ("far", "条件付きFAR"),
            ):
                p = test["paired_differences"][
                    f"{role}/{method}_minus_strict5/far_1pct/{metric}"
                ]
                rows.append(
                    [
                        ROLE_LABELS[role],
                        LABELS[method],
                        label,
                        f"{p['difference_percentage_points']:.3f}",
                        bounds(p["ci95_percentage_points"], percentage=False),
                    ]
                )
    add_table(
        "方式差と共有話者bootstrap（主条件、対5母音必須）",
        ["発話条件", "方式", "指標", "差 (pp)", "95% paired CI (pp)"],
        rows,
        "負の値は誤り率の低下。同じ全入力集合で比較し、条件付き指標は方式によってscoreありの分母が異なる。",
    )
    csv_rows = csv_documents(validation, test)
    for name, rows in csv_rows.items():
        write_csv(output / name, rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, role in enumerate(ROLES):
        for row, metric, label in (
            (0, "all_input_frr", "All-input FRR (%)"),
            (1, "all_input_far", "All-input FAR (%)"),
        ):
            values, low, high = [], [], []
            for method in METHODS:
                c = test["conditions"][f"{method}/{role}"]
                values.append(100 * c["operating_points"]["far_1pct"][metric])
                ci = c["ci95"]["operating_points"]["far_1pct"][metric]
                low.append(100 * ci["lower"])
                high.append(100 * ci["upper"])
            axes[row, col].plot(range(3), values, marker="o")
            axes[row, col].fill_between(range(3), low, high, alpha=0.2)
            axes[row, col].set(
                title=("Normal text" if col == 0 else "Cross text"),
                ylabel=label,
                xticks=range(3),
                xticklabels=["5 required", ">=4 vowels", ">=3 vowels"],
                ylim=(0, None),
            )
            axes[row, col].grid(alpha=0.2)
    fig.suptitle("Validation target FAR 1%; fixed thresholds; 95% speaker bootstrap")
    fig.savefig(output / "missing-vowels.png", dpi=180)
    fig.savefig(output / "missing-vowels.pdf")
    plt.close(fig)
    markdown += "## 全条件と図\n\n![不足母音への対応とFRR/FAR](missing-vowels.png)\n\n[全条件・3動作点CSV](all-conditions.csv)、[paired差CSV](paired-differences.csv)、[判定変化CSV](decision-transitions.csv)、[欠損数・パターン別CSV](missing-patterns.csv)、[HTML比較表](evaluation-results.html)。欠損パターン別の値は小標本の診断で、閾値校正や方式選択には使っていない。\n\n"
    markup += "<h2>全条件と図</h2><img src='missing-vowels.png' alt='FRRとFARの比較'><p><a href='all-conditions.csv'>全条件CSV</a> / <a href='paired-differences.csv'>paired差CSV</a> / <a href='decision-transitions.csv'>判定変化CSV</a> / <a href='missing-patterns.csv'>欠損パターンCSV</a></p>"
    for split in ("validation", "test"):
        report = json.loads((run / split / "report.json").read_text())
        markdown += f"- {split}: {report['score_slots']:,} slots、{report['unique_embeddings']:,}固有embedding、{report['cache_free_verification_checks']}件のcacheなし照合検査、工程時間{report['elapsed_seconds']:.2f}秒。\n"
    verification = "全embeddingは3回反復して完全一致。5母音必須の全36,000 trial、validation閾値、test指標・CIは既存結果と一致。5母音揃う発話の各母音scoreと統合scoreは全方式で一致。元音声・重み・特徴統計・登録profileは変更していない。工程時間は反復推論や検査を含み、1認証のlatencyではない。"
    markdown += f"\n{verification}\n\n実行成果物: `{run.relative_to(ROOT)}`。条件と再現手順は[README](README.md)と[protocol](config/protocol.json)。\n"
    markup += f"<h2>検証</h2><p>{verification}</p>"
    (output / "evaluation-results.md").write_text(markdown, encoding="utf-8")
    style = "body{font:15px/1.7 system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#172c3c;background:#f7fafb}h1{font-size:26px}h2{margin-top:32px}.note{background:#fff2d3;padding:16px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d6e1e8;padding:10px;text-align:right;white-space:nowrap}th{background:#e8f0f5}img{max-width:100%}a{color:#176694}"
    (output / "evaluation-results.html").write_text(
        f"<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{title}</title><style>{style}</style><main>{markup}</main></html>\n",
        encoding="utf-8",
    )
    write_json(
        output / "manifest.json",
        {
            "csv_rows": {name: len(rows) for name, rows in csv_rows.items()},
            "tables": tables,
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        },
    )


def audit_report(run):
    output = run / "report"
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256_file(output / name) != checksum:
            raise ValueError("report hash mismatch")
    documents = {
        split: json.loads((run / f"{split}-metrics.json").read_text())
        for split in ("validation", "test")
    }
    for name, expected_rows in csv_documents(
        documents["validation"], documents["test"]
    ).items():
        with (output / name).open(newline="", encoding="utf-8") as stream:
            actual = list(csv.DictReader(stream))
        if (
            len(actual) != len(expected_rows)
            or len(actual) != manifest["csv_rows"][name]
        ):
            raise ValueError("CSV rows incomplete")
        for row, expected in zip(actual, expected_rows, strict=True):
            if any(
                row[k] != ("" if v is None else str(v)) for k, v in expected.items()
            ):
                raise ValueError("CSV differs from measured JSON")

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.current, self.cell, self.links = [], None, None, []

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.current = []
            elif tag in ("td", "th"):
                self.cell = []
            elif tag in ("img", "a"):
                self.links.append(dict(attrs)["src" if tag == "img" else "href"])

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self.current.append("".join(self.cell))
                self.cell = None
            elif tag == "tr":
                self.rows.append(self.current)
                self.current = None

    parser = Parser()
    parser.feed((output / "evaluation-results.html").read_text())
    expected = [row for t in manifest["tables"] for row in [t["headers"], *t["rows"]]]
    if parser.rows != expected or not all(
        (output / link).is_file() for link in parser.links
    ):
        raise ValueError("HTML tables or assets inconsistent")
    return {
        "csv_rows": manifest["csv_rows"],
        "html_tables": len(manifest["tables"]),
        "html_rows_including_headers": len(parser.rows),
        "browser_visual_inspection": False,
    }


def publish(run):
    for path in (run / "report").iterdir():
        if path.is_file():
            shutil.copyfile(path, BASE / path.name)
