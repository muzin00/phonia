"""Publish fixed enrollment-scaling tables without changing scientific results."""

import csv
import html
import json
from html.parser import HTMLParser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scaling import BASE, POINTS, ROOT, sha256_file, write_json


def percent(value):
    return "NE" if value is None else f"{100 * value:.3f}%"


def bounds(interval, *, percentage=True):
    factor = 100 if percentage else 1
    return f"[{factor * interval['lower']:.3f}, {factor * interval['upper']:.3f}]"


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


def render(run):
    output = run / "report"
    output.mkdir(exist_ok=False)
    test = json.loads((run / "test-metrics.json").read_text())
    validation = json.loads((run / "validation-metrics.json").read_text())
    title = "Phase 8: 母音の登録10・20・30区間の実測比較"
    introduction = "固定した境界内母音encoderを使い、同じ照合発話内の全利用可能母音で照合した。5母音必須・等重み統合は変更せず、登録区間数だけを変えた。閾値は登録数ごとにvalidation通常文で決定し、通常文・別テキストの評価へ固定適用した。"
    limitation = "既に観測済みのJVS testを用いた探索的な追加検証。新しい独立holdoutの評価ではなく、別日・別端末の性能は未確認。CIは固定モデル・固定閾値での話者bootstrapで、学習・閾値校正の不確かさを含まない。"
    md = f"# {title}\n\n{introduction}\n\n{limitation}\n\n"
    sections = f"<h1>{title}</h1><p>{introduction}</p><p class='note'>{limitation}</p>"
    for role, label in (
        ("verification", "通常文"),
        ("cross_text_verification", "別テキスト"),
    ):
        headers = [
            "各母音の登録区間数",
            "条件付きFAR",
            "条件付きFRR",
            "全入力FRR",
            "全入力FRR 95% CI (%)",
            "EER",
            "coverage",
            "他人誤受入/scoreあり",
            "本人誤拒否/scoreあり",
            "本人no_score/全入力",
        ]
        rows = []
        for count in (10, 20, 30):
            cell = test["conditions"][f"n{count}/{role}"]
            p = cell["operating_points"]["far_1pct"]
            rows.append(
                [
                    count,
                    percent(p["far"]),
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
        a, b = table(headers, rows)
        md += f"## {label}（validation目標FAR 1%）\n\n{a}\n"
        sections += f"<h2>{label}：validation目標FAR 1%</h2>{b}"
    rows = []
    for role, label in (
        ("verification", "通常文"),
        ("cross_text_verification", "別テキスト"),
    ):
        for pair in ("n20_minus_n10", "n30_minus_n10", "n30_minus_n20"):
            key = f"{role}/{pair}/far_1pct/all_input_frr"
            p = test["paired_differences"][key]
            rows.append(
                [
                    label,
                    pair,
                    f"{p['difference_percentage_points']:.3f}",
                    bounds(p["ci95_percentage_points"], percentage=False),
                ]
            )
    a, b = table(["発話条件", "比較", "全入力FRR差 (pp)", "95% paired CI (pp)"], rows)
    md += f"## 登録数による差\n\n負の値は本人拒否率の減少。共有した10,000回の話者抽出で算出した。\n\n{a}\n"
    sections += f"<h2>登録数による差</h2><p>負の値は本人拒否率の減少。共有10,000回の話者bootstrap。</p>{b}"
    costs = []
    for split in ("validation", "test"):
        items = json.loads((run / split / "registration-cost.json").read_text())[
            "profiles"
        ]
        for count in (10, 20, 30):
            selected = [r for r in items if r["enrollment_count"] == count]
            costs.append(
                [
                    split,
                    count,
                    f"{np.median([r['source_wavs'] for r in selected]):.0f}",
                    f"{np.median([r['acquired_source_seconds'] for r in selected]):.3f}",
                    f"{np.median([r['unique_used_seconds'] for r in selected]):.3f}",
                ]
            )
    a, b = table(
        [
            "split",
            "各母音の登録区間数",
            "元WAV数 中央値",
            "元WAV総時間 中央値 (s)",
            "内部利用時間 中央値 (s)",
        ],
        costs,
    )
    note = "元WAV総時間は今回選択した録音の合計時間で、最短の必要録音時間ではない。内部利用時間は250 ms crop後の音声の和集合。登録区間数を増やしても、照合発話の5母音不足は解消しない。"
    md += f"## 登録音声量\n\n{a}\n{note}\n\n"
    sections += f"<h2>登録音声量</h2>{b}<p>{note}</p>"
    csv_rows = []
    for split, document in (("validation", validation), ("test", test)):
        for key, cell in document["conditions"].items():
            count, role = key.split("/")
            for point, _ in POINTS:
                p = cell["operating_points"][point]
                row = {
                    "split": split,
                    "enrollment": count,
                    "role": role,
                    "point": point,
                    "threshold": p["threshold"],
                    "coverage": cell["query_coverage"],
                    "eer": cell["pooled_eer"],
                    **p,
                }
                if split == "test":
                    for metric, ci in cell["ci95"]["operating_points"][point].items():
                        row[f"{metric}_ci95_lower"] = ci["lower"]
                        row[f"{metric}_ci95_upper"] = ci["upper"]
                csv_rows.append(row)
    fields = list(dict.fromkeys(k for r in csv_rows for k in r))
    with (output / "all-conditions.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        w = csv.DictWriter(stream, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(csv_rows)
    differences = [
        {
            "comparison": k,
            "difference_percentage_points": v["difference_percentage_points"],
            **{f"ci95_{p}": n for p, n in v["ci95_percentage_points"].items()},
        }
        for k, v in test["paired_differences"].items()
    ]
    with (output / "paired-differences.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        w = csv.DictWriter(stream, list(differences[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(differences)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, role, label in zip(
        axes,
        ("verification", "cross_text_verification"),
        ("Normal text", "Cross text"),
        strict=True,
    ):
        y, low, high = [], [], []
        for count in (10, 20, 30):
            cell = test["conditions"][f"n{count}/{role}"]
            y.append(100 * cell["operating_points"]["far_1pct"]["all_input_frr"])
            ci = cell["ci95"]["operating_points"]["far_1pct"]["all_input_frr"]
            low.append(100 * ci["lower"])
            high.append(100 * ci["upper"])
        ax.plot([10, 20, 30], y, marker="o")
        ax.fill_between([10, 20, 30], low, high, alpha=0.2)
        ax.set(
            title=label,
            xlabel="Enrollment segments per vowel",
            ylabel="All-input FRR (%)",
            xticks=[10, 20, 30],
        )
        ax.grid(alpha=0.2)
    fig.savefig(output / "enrollment-scaling.png", dpi=180)
    fig.savefig(output / "enrollment-scaling.pdf")
    plt.close(fig)
    md += "## 図と全動作点\n\n![登録数と全入力FRR](enrollment-scaling.png)\n\n[全12条件・3動作点・件数・CI](all-conditions.csv)、[全72 paired差](paired-differences.csv)、[HTML比較表](evaluation-results.html)。\n\n"
    sections += "<h2>図と全動作点</h2><img src='enrollment-scaling.png' alt='登録数と全入力FRR'><p><a href='all-conditions.csv'>全12条件・3動作点CSV</a> / <a href='paired-differences.csv'>全paired差CSV</a></p>"
    for split in ("validation", "test"):
        r = json.loads((run / split / "report.json").read_text())
        md += f"- {split}: {r['profiles']} profiles、{r['score_slots']:,} slots、{r['unique_embeddings']:,}固有embedding、工程時間{r['elapsed_seconds']:.2f}秒。全embeddingを3回反復し完全一致。\n"
    md += "\n工程時間はモデル読込、反復推論、公開API検査、profile生成、全trial照合を含み、1認証のlatencyではない。\n\n登録10の全30 profileと36,000 trial、validation閾値、通常文/別テキストのtest指標・CIはPhase 7までの結果と一致。重み・特徴統計・元音声は変更していない。生score・embedding・bootstrap・凍結記録は実行成果物に保存。\n\n"
    md += f"実行成果物: `{run.relative_to(ROOT)}`。再現手順は[README](README.md)、条件は[protocol](config/protocol.json)。\n"
    sections += "<h2>検証</h2><p>登録10のprofile・score・閾値・test指標・CIは既存結果と一致。重み・特徴統計・元音声は不変。</p>"
    (output / "evaluation-results.md").write_text(md, encoding="utf-8")
    style = "body{font:15px/1.7 system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#172c3c;background:#f7fafb}h1{font-size:26px}h2{margin-top:32px}.note{background:#fff2d3;padding:16px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d6e1e8;padding:10px;text-align:right;white-space:nowrap}th{background:#e8f0f5}img{max-width:100%}a{color:#176694}"
    (output / "evaluation-results.html").write_text(
        f"<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{title}</title><style>{style}</style><main>{sections}</main></html>\n",
        encoding="utf-8",
    )
    write_json(
        output / "manifest.json",
        {
            "condition_rows": len(csv_rows),
            "paired_rows": len(differences),
            "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        },
    )


def publish(run):
    import shutil

    for p in (run / "report").iterdir():
        if p.is_file():
            shutil.copyfile(p, BASE / p.name)


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
    with (output / "all-conditions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    identities = set()
    for row in rows:
        identity = row["split"], row["enrollment"], row["role"], row["point"]
        if identity in identities:
            raise ValueError("duplicate report row")
        identities.add(identity)
        cell = documents[row["split"]]["conditions"][
            f"{row['enrollment']}/{row['role']}"
        ]
        expected = {
            "coverage": cell["query_coverage"],
            "eer": cell["pooled_eer"],
            **cell["operating_points"][row["point"]],
        }
        if row["split"] == "test":
            for metric, ci in cell["ci95"]["operating_points"][row["point"]].items():
                expected[f"{metric}_ci95_lower"] = ci["lower"]
                expected[f"{metric}_ci95_upper"] = ci["upper"]
        if any(
            row[k] != ("" if value is None else str(value))
            for k, value in expected.items()
        ):
            raise ValueError("CSV metric or CI differs from measured result")
    if len(rows) != 36 or manifest["condition_rows"] != 36:
        raise ValueError("report condition coverage mismatch")
    with (output / "paired-differences.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        pairs = list(csv.DictReader(stream))
    expected_pairs = documents["test"]["paired_differences"]
    if len(pairs) != 72 or {p["comparison"] for p in pairs} != set(expected_pairs):
        raise ValueError("paired report coverage mismatch")
    for row in pairs:
        measured = expected_pairs[row["comparison"]]
        expected = {
            "difference_percentage_points": measured["difference_percentage_points"],
            **{f"ci95_{k}": v for k, v in measured["ci95_percentage_points"].items()},
        }
        if any(row[k] != str(value) for k, value in expected.items()):
            raise ValueError("paired report differs from measured result")

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tables, self.cells, self.links = 0, 0, []

        def handle_starttag(self, tag, attrs):
            self.tables += tag == "table"
            self.cells += tag == "td"
            attributes = dict(attrs)
            if tag in ("a", "img"):
                self.links.append(attributes["href" if tag == "a" else "src"])

    parser = Parser()
    parser.feed((output / "evaluation-results.html").read_text())
    if (
        parser.tables != 4
        or parser.cells != 114
        or not all((output / link).is_file() for link in parser.links)
    ):
        raise ValueError("HTML tables or report assets incomplete")
    return {
        "condition_rows": len(rows),
        "paired_rows": len(pairs),
        "html_tables": parser.tables,
        "html_cells": parser.cells,
        "browser_visual_inspection": False,
    }
