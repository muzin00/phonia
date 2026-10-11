"""Publish complete measured tables in JSON, CSV, Markdown and standalone HTML."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import shared as s

POLICIES = {
    "vowels5": "5母音必須",
    "vowels4": "4母音以上",
    "vowels3_phones5": "3母音以上＋合計5種類",
}
ROLES = {"verification": "通常文", "cross_text_verification": "別文"}


def percent(value):
    return f"{value * 100:.3f}%"


def interval(value):
    return f"{value['difference'] * 100:+.3f} [{value['ci95'][0] * 100:+.3f}, {value['ci95'][1] * 100:+.3f}]"


class Document:
    def __init__(self):
        self.markdown = []
        self.html = []

    def title(self, text, level=2):
        self.markdown.append("#" * level + " " + text + "\n")
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def paragraph(self, text):
        self.markdown.append(text + "\n")
        self.html.append(f"<p>{html.escape(text)}</p>")

    def table(self, headers, values):
        values = [[str(c) for c in row] for row in values]
        self.markdown.append(
            "\n".join(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |",
                    *["| " + " | ".join(row) + " |" for row in values],
                ]
            )
            + "\n"
        )
        head = "".join(f"<th>{html.escape(c)}</th>" for c in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
            for row in values
        )
        self.html.append(
            f"<div class='table'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        )


def write_csv(path, values):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def publish(config, run):
    completed = s.read_json(run / "completion-verification.json")
    for path, checksum in completed["output_sha256"].items():
        s.checked(s.ROOT / path, checksum)
    results = {
        split: s.read_json(run / split / "results.json")
        for split in ("validation", "test")
    }
    bootstrap = s.read_json(run / "bootstrap-results.json")
    audit = s.read_json(run / "independent-audit.json")
    cells, patterns, changes = [], [], []
    for split, models in results.items():
        for model, policies in models.items():
            for policy, report in policies.items():
                for mode, metrics in report["metrics"].items():
                    for role, metric in metrics.items():
                        for point, rates in metric["operating_points"].items():
                            identity = {
                                "split": split,
                                "model": model,
                                "policy": policy,
                                "threshold_mode": mode,
                                "role": role,
                                "operating_point": point,
                            }
                            cells.append(
                                {
                                    **identity,
                                    "all_queries": metric["all_queries"],
                                    "scored_queries": metric["scored_queries"],
                                    "coverage": metric["coverage"],
                                    "eer": metric["eer"],
                                    **rates,
                                }
                            )
                            diagnostic = report["diagnostics"][role][mode][point]
                            for missing, counts in diagnostic["patterns"].items():
                                patterns.append(
                                    {**identity, "missing_vowels": missing, **counts}
                                )
                            for group, counts in diagnostic["changes"].items():
                                changes.append(
                                    {**identity, "previous_coverage": group, **counts}
                                )
    write_csv(s.BASE / "evaluation-cells.csv", cells)
    write_csv(s.BASE / "missing-patterns.csv", patterns)
    write_csv(s.BASE / "decision-changes.csv", changes)
    public = {
        "config": config,
        "results": results,
        "bootstrap": bootstrap,
        "audit": audit,
    }
    s.write_json(s.BASE / "evaluation-results.json", public)
    d = Document()
    d.title("全36音素モデルの母音不足対応：実測結果", 1)
    d.paragraph(
        "全36音素encoder・登録上限30区間/音素、等重みと既存clean学習Transformer 3 seedを固定。新規学習なし。主比較は等重みの4母音以上と5母音必須。3母音以上＋合計5種類は、最低5種類の比較材料を要求する補助条件であり、5母音と同じ情報量を保証しない。"
    )
    d.paragraph(
        "各条件のvalidation通常文だけでFAR 1%・FAR 0.1%・EER動作点の閾値を決め、全12条件を固定してからtestを測定した。元の5母音閾値を据え置いた診断も併記する。採点不能は全入力FAR/FRRで拒否扱い。EERは採点可能な発話だけであり、受理条件の変更でEERの評価集合も変わる。"
    )
    learned = [m for m in config["models"] if m != "equal"]
    for mode, caption in (
        ("recalibrated", "条件ごとのvalidation閾値"),
        ("original", "元の5母音閾値を据え置き"),
    ):
        d.title(f"test：{caption}（目標FAR 1%）")
        table = []
        for role in s.ROLES:
            for model_label, names in (
                ("等重み", ["equal"]),
                ("Transformer 3 seed平均", learned),
            ):
                for policy, policy_label in POLICIES.items():
                    values = [
                        results["test"][m][policy]["metrics"][mode][role] for m in names
                    ]
                    mean = lambda field, values=values: float(
                        s.np.mean(
                            [v["operating_points"]["far_1pct"][field] for v in values]
                        )
                    )
                    table.append(
                        [
                            ROLES[role],
                            model_label,
                            policy_label,
                            f"{values[0]['scored_queries']}/{values[0]['all_queries']}",
                            percent(s.np.mean([v["eer"] for v in values])),
                            percent(mean("all_input_far")),
                            percent(mean("all_input_frr")),
                        ]
                    )
        d.table(
            ["発話", "方式", "受理条件", "採点可能", "EER", "全入力FAR", "全入力FRR"],
            table,
        )
    d.paragraph(
        "Transformer平均は3モデルの指標の平均で、ensembleではない。母音不足は既存Transformerの学習条件外。元から採点できた発話のスコアは全条件で完全一致し、元の閾値を据え置くとその判定も変わらない。"
    )
    d.title("等重み：実件数と入力不足による拒否")
    table = []
    for role in s.ROLES:
        for policy, policy_label in POLICIES.items():
            metric = results["test"]["equal"][policy]["metrics"]["recalibrated"][role]
            rate = metric["operating_points"]["far_1pct"]
            table.append(
                [
                    ROLES[role],
                    policy_label,
                    metric["all_queries"] - metric["scored_queries"],
                    rate["false_rejects"],
                    rate["false_accepts"],
                    metric["all_queries"] * 14,
                ]
            )
    d.table(
        [
            "発話",
            "受理条件",
            "採点不能の本人",
            "採点後の本人拒否",
            "他人誤受入",
            "全他人試行",
        ],
        table,
    )
    d.title("5母音必須との差と対応付き95%区間（pp）")
    d.paragraph(
        "validationで条件ごとに校正したFAR 1%閾値。15 test話者を共有する2,000回bootstrap。固定モデル・固定閾値に条件付く区間で、校正・学習・発話再抽出の不確かさと多重比較補正を含まない。区間が0を含むことはFAR同等性の証明ではない。"
    )
    table = []
    for model in ("equal", "transformer-mean"):
        for role in s.ROLES:
            for policy in list(POLICIES)[1:]:
                prefix = f"{model}/{policy}/{role}/recalibrated"
                table.append(
                    [
                        model,
                        ROLES[role],
                        POLICIES[policy],
                        *[
                            interval(bootstrap["differences"][f"{prefix}/{key}"])
                            for key in ("far_1pct/far", "far_1pct/frr", "eer")
                        ],
                    ]
                )
    d.table(
        [
            "方式",
            "発話",
            "受理条件",
            "FAR差 [95%区間]",
            "FRR差 [95%区間]",
            "EER差 [95%区間]",
        ],
        table,
    )
    d.title("等重み・母音不足入力の内訳（条件別校正、目標FAR 1%）")
    table = []
    for role in s.ROLES:
        for policy in list(POLICIES)[1:]:
            pats = results["test"]["equal"][policy]["diagnostics"][role][
                "recalibrated"
            ]["far_1pct"]["patterns"]
            for missing, counts in pats.items():
                if missing == "none":
                    continue
                table.append(
                    [
                        ROLES[role],
                        POLICIES[policy],
                        missing,
                        counts["speakers"],
                        counts["queries"],
                        counts["scored_queries"],
                        counts["all_false_rejects"],
                        f"{counts['false_accepts']}/{counts['impostor_trials']}",
                    ]
                )
    d.table(
        [
            "発話",
            "受理条件",
            "欠損母音",
            "話者数",
            "本人入力",
            "採点可能",
            "本人拒否",
            "他人受入/試行",
        ],
        table,
    )
    d.title("元の判定からの変化（等重み、条件別校正、目標FAR 1%）")
    table = []
    for role in s.ROLES:
        for policy in list(POLICIES)[1:]:
            groups = results["test"]["equal"][policy]["diagnostics"][role][
                "recalibrated"
            ]["far_1pct"]["changes"]
            for group, counts in groups.items():
                table.append(
                    [
                        ROLES[role],
                        POLICIES[policy],
                        "従来採点可能"
                        if group == "previously_scored"
                        else "従来採点不能",
                        *counts.values(),
                    ]
                )
    d.table(
        [
            "発話",
            "受理条件",
            "入力群",
            "本人受入増",
            "本人受入減",
            "他人受入増",
            "他人受入減",
        ],
        table,
    )
    d.title("全モデル・全seedのtest値（条件別校正、目標FAR 1%）")
    d.table(
        ["方式", "受理条件", "発話", "EER", "全入力FAR", "全入力FRR"],
        [
            [
                c["model"],
                POLICIES[c["policy"]],
                ROLES[c["role"]],
                percent(c["eer"]),
                percent(c["all_input_far"]),
                percent(c["all_input_frr"]),
            ]
            for c in cells
            if c["split"] == "test"
            and c["threshold_mode"] == "recalibrated"
            and c["operating_point"] == "far_1pct"
        ],
    )
    d.title("低FAR動作点（条件別校正、validation目標FAR 0.1%）")
    d.table(
        ["方式", "受理条件", "発話", "test全入力FAR", "test全入力FRR"],
        [
            [
                c["model"],
                POLICIES[c["policy"]],
                ROLES[c["role"]],
                percent(c["all_input_far"]),
                percent(c["all_input_frr"]),
            ]
            for c in cells
            if c["split"] == "test"
            and c["threshold_mode"] == "recalibrated"
            and c["operating_point"] == "far_0_1pct"
        ],
    )
    d.title("監査と解釈の範囲")
    d.paragraph(
        f"監査成功。{audit['score_policy_rows']:,}件の方式別試行、{audit['metric_checks']:,}指標値、{audit['model_replay_score_checks']:,}件のモデル再推論、{audit['independent_bootstrap_checks']:,}件のbootstrap再計算を確認。float64でembeddingから再構成した最大差は{audit['max_float64_score_error']:.3g}。モデル・登録・入力・コードのhash不変を確認した。"
    )
    d.paragraph(
        "既に参照したJVS testでの探索的な追加検証で、独立holdoutの確証ではない。欠損パターンの一部は少数話者・少数発話だけである。FARの維持を保証せず、条件やseedの本番採用は行わない。全validation/test・全3動作点・両閾値条件はCSVとJSONに記録する。"
    )
    (s.BASE / "evaluation-results.md").write_text(
        "\n".join(d.markdown)
        + "\n[設計と再実行](README.md)・[解釈](interpretation.md)・[HTML](evaluation-results.html)・[全指標CSV](evaluation-cells.csv)・[欠損パターンCSV](missing-patterns.csv)・[判定変化CSV](decision-changes.csv)\n",
        encoding="utf-8",
    )
    page = (
        "<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>全36音素モデルの母音不足対応</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#172b38;background:#f6f8fb;max-width:1400px;margin:auto;padding:32px}h1,h2{line-height:1.4}h2{margin-top:40px}.table{overflow:auto;background:white;border:1px solid #dae1e7;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 14px;border-bottom:1px solid #e5e9ee;white-space:nowrap;text-align:right}th{background:#eaf0f5}td:first-child,th:first-child{text-align:left}p{max-width:1100px}</style><body>"
        + "\n".join(d.html)
        + "<p><a href='README.md'>設計</a> · <a href='interpretation.md'>解釈</a> · <a href='evaluation-cells.csv'>全指標CSV</a> · <a href='missing-patterns.csv'>欠損パターンCSV</a></p></body></html>"
    )
    (s.BASE / "evaluation-results.html").write_text(page, encoding="utf-8")
    # Ensure CSV serialization preserves the complete measured cell count/values.
    with (s.BASE / "evaluation-cells.csv").open(encoding="utf-8") as stream:
        serialized = list(csv.DictReader(stream))
    if len(serialized) != len(cells) or any(
        float(a["all_input_far"]) != b["all_input_far"]
        or float(a["all_input_frr"]) != b["all_input_frr"]
        for a, b in zip(serialized, cells)
    ):
        raise ValueError("published CSV differs from measurements")
    print(
        f"Published {len(cells)} metric rows, {len(patterns)} pattern rows, {len(changes)} change rows",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    settings = s.read_json(s.CONFIG)
    publish(settings, args.run or s.ROOT / settings["run_directory"])
