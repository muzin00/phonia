"""Publish the clean comparison and planned fixed-model diagnostics together."""

from __future__ import annotations

import html
import json

import data as shared


def publish(run):
    primary = shared.read_json(run / "results.json")
    stress = shared.read_json(run / "stress/results.json")
    frozen = shared.read_json(run / "stress/design-freeze.json")
    shared.checked(shared.BASE / "stress.py", frozen["script_sha256"])
    shared.checked(run / "selection-freeze.json", frozen["selection_sha256"])
    shared.checked(run / "design-freeze.json", frozen["primary_design_sha256"])
    for name in ("completion-verification.json", "stress/completion-verification.json"):
        completion = shared.read_json(run / name)
        if completion["status"] != "passed":
            raise ValueError("independent verification required")
        for path, checksum in completion["output_sha256"].items():
            shared.checked(shared.ROOT / path, checksum)
    control = shared.read_json(run / "stress/noise-clean-control-audit.json")
    if control["status"] != "passed":
        raise ValueError("clean waveform replay required")
    bootstrap = shared.read_json(run / "bootstrap-results.json")
    seeds = primary["config"]["seeds"]
    scenarios = {"通常の入力": {n: r["test"] for n, r in primary["variants"].items()}}
    scenarios.update(
        {
            "子音50%欠損"
            if n == "missing-half-consonants"
            else "20%区間に10dBノイズ": r["variants"]
            for n, r in stress["challenges"].items()
        }
    )
    summary, comparison = {}, []
    for condition, values in scenarios.items():
        summary[condition] = {}
        for kind in ("equal", "mlp", "transformer"):
            names = ["equal"] if kind == "equal" else [f"{kind}-{s}" for s in seeds]
            summary[condition][kind] = {}
            for role in shared.ROLES:
                records = [values[n][role] for n in names]
                summary[condition][kind][role] = {
                    "eer_mean": float(shared.np.mean([r["eer"] for r in records])),
                    "eer_minimum": min(r["eer"] for r in records),
                    "eer_maximum": max(r["eer"] for r in records),
                    "all_input_far_mean": float(
                        shared.np.mean(
                            [
                                r["operating_points"]["far_1pct"]["all_input_far"]
                                for r in records
                            ]
                        )
                    ),
                    "all_input_frr_mean": float(
                        shared.np.mean(
                            [
                                r["operating_points"]["far_1pct"]["all_input_frr"]
                                for r in records
                            ]
                        )
                    ),
                }
        comparison.append(
            [
                condition,
                *[
                    f"{summary[condition][kind][role]['eer_mean'] * 100:.3f}%"
                    for role in shared.ROLES
                    for kind in ("equal", "transformer")
                ],
                *[
                    f"{(summary[condition]['transformer'][r]['eer_mean'] - summary[condition]['equal'][r]['eer_mean']) * 100:+.3f}"
                    for r in shared.ROLES
                ],
            ]
        )
    learned = []
    for name in primary["variants"]:
        if name == "equal":
            continue
        training = shared.read_json(run / name / "training-summary.json")
        best_learned = min(
            training["history"][1:], key=lambda r: (r["normal_eer"], r["update"])
        )
        if training["completed_updates"] != 4000:
            raise ValueError("planned training incomplete")
        learned.append(
            [
                name,
                str(training["parameters"]),
                str(training["completed_updates"]),
                str(training["selected_update"]),
                f"{best_learned['normal_eer'] * 100:.3f}%",
            ]
        )
    rates = []
    for condition, kinds in summary.items():
        for kind in ("equal", "transformer"):
            for role, result in kinds[kind].items():
                rates.append(
                    [
                        condition,
                        kind,
                        "通常" if role == shared.ROLES[0] else "別文",
                        f"{result['all_input_far_mean'] * 100:.3f}%",
                        f"{result['all_input_frr_mean'] * 100:.3f}%",
                    ]
                )
    tables = [
        {
            "title": "test EER（Transformerは3 seed平均）",
            "headers": [
                "条件",
                "通常: 等重み",
                "通常: Transformer",
                "別文: 等重み",
                "別文: Transformer",
                "通常差 pp",
                "別文差 pp",
            ],
            "rows": comparison,
        },
        {
            "title": "実施した学習とvalidationでの採用",
            "headers": [
                "方式 / seed",
                "追加パラメータ数",
                "完了更新",
                "採用更新",
                "更新後の最小validation通常EER",
            ],
            "rows": learned,
        },
        {
            "title": "clean validation FAR 1%閾値を固定（全入力率、Transformerは3 seed平均）",
            "headers": ["条件", "方式", "評価", "FAR", "FRR"],
            "rows": rates,
        },
    ]
    paragraphs = [
        "固定した全36音素encoder・登録上限30区間/音素で、Transformerのtest平均EERは通常1.497%→1.186%、別文1.020%→0.941%に改善した。通常は相対20.8%、別文は相対7.7%の低下である。",
        "通常の各seedは1.497%、0.952%、1.108%、別文は1.020%、1.057%、0.747%。通常0.952%と別文0.747%は別のseedの値であり、1モデルが両方を達成した値ではない。test最良seedを選び直していない。",
        "3 seedともTransformerはvalidation通常EERを改善した。MLPは3 seedとも学習後のvalidation EERが等重みを上回れず、事前の採用規則により初期の等重みcheckpointを採用した。MLPのtest値は学習済み重みの性能を表す値ではない。全6回の学習自体は各4,000更新まで完了した。",
        "平均EER差の話者bootstrap 95%区間は通常[-0.657, +0.403] pp、別文[-0.748, +0.188] pp。今回の測定では平均の改善を観測したが、15 test話者に対する区間は0を含む。",
        "子音を確率50%で欠損させると、Transformer平均は通常1.523%、別文0.887%。同条件の等重み1.361%、1.531%に対し、通常は悪化、別文は改善した。5母音は残し、全方式で同じ欠損と採点対象を使った。",
        "照合区間の20%に10dBの白色ノイズを加えると、Transformer平均は通常3.356%、別文3.401%へ悪化。同条件の等重み1.302%、1.093%より両方とも高い。今回のclean学習から、録音品質に応じてうまく重みを下げられるとは言えない。",
        "ノイズは既存の音素境界で切り出した区間へ合成し、encoderを再実行した。境界の再推定や追加QCは行わない。15,239/75,831区間を変換し、登録側embeddingはすべて不変。ノイズなしの51区間を元WAVから再計算したembeddingは既存cacheと完全一致した。モデル・閾値の再学習や再調整はしない。",
        "EERはROC全体の指標であり、固定閾値でのFRR改善と同じではない。ノイズ条件の等重みも、通常EERは少し下がったが、固定したclean validation閾値でのFRRは上がった。FAR/FRRを下の表に併記する。",
        "入力品質へ適応する次の検証では、同じ固定encoderのまま統合モデルのtrainだけにノイズ・欠損を加え、同じ3 seedと同じ学習ペア列で再測定する。今回の結果から追加の学習やモデル選択は行っていない。",
    ]
    markdown = ["# Transformer検証の結果と解釈", "", *[p + "\n" for p in paragraphs]]
    fragments = []
    for table in tables:
        headers, cells = table["headers"], table["rows"]
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
        [
            "[全seedの通常測定表](evaluation-results.md) · [HTML比較表](summary.html) · [設計と再実行](README.md)",
            "",
        ]
    )
    (shared.BASE / "interpretation.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    document = (
        "<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Transformer比較の結果</title><style>body{max-width:1250px;margin:32px auto;padding:0 20px;font-family:system-ui;color:#172033;line-height:1.8;background:#f8fafc}h1{font-size:1.8rem}h2{font-size:1.25rem;margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #cbd5e1;padding:9px 12px;text-align:left;white-space:nowrap}th{background:#e2e8f0}tbody tr:nth-child(even){background:#f1f5f9}a{color:#155e75}</style><h1>Transformer検証の結果</h1>"
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs[:4])
        + "".join(fragments)
        + "<h2>欠損・ノイズの解釈</h2>"
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs[4:])
        + "<p><a href='evaluation-results.html'>全seedと信頼区間の表</a> · <a href='interpretation.md'>Markdown</a> · <a href='summary.json'>JSON</a> · <a href='README.md'>設計・再実行</a></p></html>"
    )
    (shared.BASE / "summary.html").write_text(document, encoding="utf-8")
    shared.write_json(
        shared.BASE / "summary.json",
        {
            "source_run": str(run.relative_to(shared.ROOT)),
            "summary": summary,
            "bootstrap": bootstrap,
            "tables": tables,
            "paragraphs": paragraphs,
            "noise_clean_control": control,
            "source_sha256": {
                name: shared.sha256_file(run / name)
                for name in (
                    "results.json",
                    "stress/results.json",
                    "independent-audit.json",
                    "stress/completion-verification.json",
                )
            },
        },
    )
    shared.write_json(
        run / "summary-publication-freeze.json",
        {
            "script_sha256": shared.sha256_file(__file__),
            "files": {
                str((shared.BASE / name).relative_to(shared.ROOT)): shared.sha256_file(
                    shared.BASE / name
                )
                for name in ("summary.html", "summary.json", "interpretation.md")
            },
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    publish(
        shared.ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1"
    )
