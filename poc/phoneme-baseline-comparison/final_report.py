"""Complete condition tables and scientific ROC/DET/duration figures."""

from __future__ import annotations

import csv
import html
import json
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot import METHODS
from smoke import ROOT, sha256_file, write_json
from validation import iter_rows

COLORS = {
    "vowel_exact": "#276fbf",
    "vowel_context20": "#e38b2c",
    "ecapa_whole": "#17905b",
}
CAPS = ("max_1s", "max_2s", "max_3s", "max_5s", "full")


def csv_file(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return "NE" if value is None else f"{100 * value:.3f}%"


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=140, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_report(run):
    inputs = json.loads((run / "inputs.json").read_text())
    validation = ROOT / inputs["config"]["validation_run"]
    test = json.loads((run / "test-metrics.json").read_text())
    vm = json.loads((validation / "validation-metrics.json").read_text())
    thresholds = json.loads((validation / "validation-thresholds.json").read_text())
    documents = {"validation": vm, "test": test}
    strata = {
        "validation": json.loads((run / "validation-duration-strata.json").read_text()),
        "test": {k: c["duration_strata"] for k, c in test["conditions"].items()},
    }
    output = run / "report"
    output.mkdir(exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    tables, duration = [], []
    for split, doc in documents.items():
        for key, c in sorted(doc["conditions"].items()):
            support, method, count, cap, role = key.split("/")
            points = c["operating_points"] or {
                p: None for p in ("far_1pct", "far_0_1pct", "eer_operating")
            }
            for point, p in points.items():
                t = thresholds["conditions"][f"{support}/{method}/{count}/{cap}"][
                    "operating_points"
                ]
                base = {
                    "split": split,
                    "support": support,
                    "method": method,
                    "enrollment": count,
                    "cap": cap,
                    "role": role,
                    "point": point,
                }
                row = {
                    **base,
                    "threshold": t[point]["threshold"] if t else None,
                    "conditional_status": c["conditional_status"],
                    "queries": c["queries"],
                    "scored_queries": c["scored_queries"],
                    "query_coverage": c["query_coverage"],
                    "support_selection_coverage": c["support_selection_coverage"],
                    "zero_scored_speakers": ";".join(c["zero_scored_speakers"]),
                    "eer": c["pooled_eer"],
                    "far_resolution": c["far_resolution"],
                }
                for field in (
                    "genuine",
                    "impostor",
                    "false_accepts",
                    "false_rejects",
                    "no_score_genuine",
                    "no_score_impostor",
                    "all_genuine",
                    "all_impostor",
                    "far",
                    "frr",
                    "all_input_far",
                    "all_input_frr",
                ):
                    row[field] = p[field] if p else None
                ci = c.get("ci95", {}).get("operating_points", {}).get(point, {})
                for field in ("far", "frr", "all_input_far", "all_input_frr"):
                    for bound in (
                        "lower",
                        "upper",
                        "valid_replicates",
                        "undefined_replicates",
                    ):
                        row[f"{field}_ci95_{bound}"] = ci.get(field, {}).get(bound)
                for field in ("query_coverage", "pooled_eer"):
                    for bound in (
                        "lower",
                        "upper",
                        "valid_replicates",
                        "undefined_replicates",
                    ):
                        row[f"{field}_ci95_{bound}"] = (
                            c.get("ci95", {}).get(field) or {}
                        ).get(bound)
                tables.append(row)
                for dimension, bins in strata[split][key].items():
                    for bin_name, b in bins.items():
                        bp = (b["operating_points"] or {}).get(point)
                        duration.append(
                            {
                                **base,
                                "dimension": dimension,
                                "bin": bin_name,
                                "conditional_status": b["conditional_status"],
                                "queries": b["queries"],
                                "scored_queries": b["scored_queries"],
                                "coverage": b["query_coverage"],
                                "eer": b["pooled_eer"],
                                "all_input_far": bp["all_input_far"] if bp else None,
                                "all_input_frr": bp["all_input_frr"] if bp else None,
                                "far": bp["far"] if bp else None,
                                "frr": bp["frr"] if bp else None,
                            }
                        )
    csv_file(output / "all-conditions.csv", tables)
    csv_file(output / "duration-strata.csv", duration)
    pairs = [
        {
            "comparison": k,
            "status": d["status"],
            "reference": d["reference"],
            "difference_pp": d.get("point_difference_percentage_points"),
            **{
                f"ci95_{b}": d.get("ci95_percentage_points", {}).get(b)
                for b in ("lower", "upper", "valid_replicates", "undefined_replicates")
            },
            "population_note": d.get("population_note"),
        }
        for k, d in test["paired_differences"].items()
    ]
    csv_file(output / "paired-differences.csv", pairs)
    nd = NormalDist()
    ppf = np.vectorize(nd.inv_cdf)
    for split, doc in documents.items():
        curve_run = validation if split == "validation" else run
        curves = {
            c["condition"]: c for c in iter_rows(curve_run / f"{split}-curves.jsonl.gz")
        }
        for support in ("native", "common"):
            for count in (1, 5, 10):
                for role in ("verification", "cross_text_verification"):
                    prefix = f"{split}-{support}-n{count}-{role}"
                    fig, axes = plt.subplots(
                        2, 5, figsize=(19, 7), constrained_layout=True
                    )
                    for j, cap in enumerate(CAPS):
                        for method in METHODS:
                            key = f"{support}/{method}/n{count}/{cap}/{role}"
                            curve = curves[key]
                            if not len(curve["far"]):
                                axes[0, j].plot(
                                    [], [], color=COLORS[method], label=f"{method} (NE)"
                                )
                                continue
                            far, frr = (
                                np.asarray(curve["far"]),
                                np.asarray(curve["frr"]),
                            )
                            idx = np.unique(
                                np.r_[
                                    np.arange(0, len(far), max(1, len(far) // 1000)),
                                    len(far) - 1,
                                ]
                            )
                            axes[0, j].plot(
                                far[idx],
                                1 - frr[idx],
                                color=COLORS[method],
                                label=method,
                            )
                            axes[1, j].plot(
                                ppf(np.clip(far[idx], 1e-5, 1 - 1e-5)),
                                ppf(np.clip(frr[idx], 1e-5, 1 - 1e-5)),
                                color=COLORS[method],
                            )
                        axes[0, j].set(
                            title=cap,
                            xlabel="FAR (scored only)",
                            ylabel="TPR",
                            xlim=(0, 0.1),
                            ylim=(0.5, 1.005),
                        )
                        ticks = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
                        axes[1, j].set(
                            xticks=[nd.inv_cdf(t) for t in ticks],
                            xticklabels=["0.01", "0.1", "1", "10", "50", "90"],
                            yticks=[nd.inv_cdf(t) for t in ticks],
                            yticklabels=["0.01", "0.1", "1", "10", "50", "90"],
                            xlabel="FAR (%)",
                            ylabel="FRR (%)",
                        )
                        for ax in axes[:, j]:
                            ax.grid(alpha=0.25)
                    axes[0, 0].legend(fontsize=7)
                    fig.suptitle(
                        f"{split} / {support} / enrollment {count} / {role}; NE retains missing speakers"
                    )
                    save_figure(fig, figures, f"roc-det-{prefix}")
                    fig, axes = plt.subplots(
                        1, 2, figsize=(11, 4), constrained_layout=True
                    )
                    for method in METHODS:
                        cells = [
                            doc["conditions"][
                                f"{support}/{method}/n{count}/{cap}/{role}"
                            ]
                            for cap in CAPS
                        ]
                        frr = [
                            100 * c["operating_points"]["far_1pct"]["all_input_frr"]
                            for c in cells
                        ]
                        axes[0].plot(
                            range(5),
                            frr,
                            marker="o",
                            color=COLORS[method],
                            label=method,
                        )
                        axes[1].plot(
                            range(5),
                            [
                                100
                                * c["support_selection_coverage"]
                                * c["query_coverage"]
                                for c in cells
                            ],
                            marker="o",
                            color=COLORS[method],
                        )
                        if split == "test":
                            cis = [
                                c["ci95"]["operating_points"]["far_1pct"][
                                    "all_input_frr"
                                ]
                                for c in cells
                            ]
                            axes[0].fill_between(
                                range(5),
                                [100 * v["lower"] for v in cis],
                                [100 * v["upper"] for v in cis],
                                color=COLORS[method],
                                alpha=0.15,
                            )
                    for ax in axes:
                        ax.set(
                            xticks=range(5),
                            xticklabels=["1 s", "2 s", "3 s", "5 s", "full"],
                            xlabel="Maximum acquired query recording",
                            ylim=(-1, 101),
                        )
                        ax.grid(alpha=0.25)
                    axes[0].set_ylabel("All-input FRR (%) at validation FAR 1%")
                    axes[1].set_ylabel("Scored queries / original queries (%)")
                    axes[0].legend(fontsize=8)
                    fig.suptitle(
                        prefix + ("; 95% speaker bootstrap" if split == "test" else "")
                    )
                    save_figure(fig, figures, f"duration-{prefix}")
                    if support == "native":
                        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
                        for method in METHODS:
                            cells = [
                                doc["conditions"][
                                    f"native/{method}/n{count}/{cap}/{role}"
                                ]["source_at_least_5s_diagnostic"]
                                for cap in CAPS
                            ]
                            ax.plot(
                                range(5),
                                [
                                    100
                                    * c["operating_points"]["far_1pct"]["all_input_frr"]
                                    for c in cells
                                ],
                                marker="o",
                                color=COLORS[method],
                                label=method,
                            )
                        ax.set(
                            xticks=range(5),
                            xticklabels=["1 s", "2 s", "3 s", "5 s", "full"],
                            xlabel="Maximum acquired query recording",
                            ylabel="All-input FRR (%)",
                            ylim=(-1, 101),
                            title=prefix + "; fixed source >= 5 s cohort",
                        )
                        ax.grid(alpha=0.25)
                        ax.legend(fontsize=8)
                        save_figure(fig, figures, f"fixed-source-duration-{prefix}")
    columns = list(tables[0])
    table = (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[c])) if row[c] is not None else 'NE'}</td>"
                for c in columns
            )
            + "</tr>"
            for row in tables
        )
        + "</tbody></table>"
    )
    gallery = "".join(
        f'<details><summary>{p.stem}</summary><a href="figures/{p.stem}.pdf">PDF</a><img loading="lazy" src="figures/{p.name}" alt="{p.stem}"></details>'
        for p in sorted(figures.glob("*.png"))
    )
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(
            '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Phase 7 results</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px;white-space:nowrap}th{position:sticky;top:0;background:#eee}img{max-width:100%}.scroll{overflow:auto;max-height:70vh}</style><h1>Phase 7: fixed-threshold baseline comparison</h1><p>360 conditions × 3 operating points. Test: 15 speakers; shared 10,000 speaker bootstrap draws. Conditional NE retains speakers. Zero error does not establish zero population risk. Common input selection varies across caps. JVS and previously inspected test limit generalization.</p><p><a href="all-conditions.csv">All conditions CSV</a> · <a href="duration-strata.csv">Duration strata CSV</a> · <a href="paired-differences.csv">Paired differences CSV</a></p><div class="scroll">'
            + table
            + "</div><h2>All condition figures</h2>"
            + gallery
            + "</html>"
        )
    lines = [
        "# Phase 7: 固定閾値test比較結果",
        "",
        "3方式・5長さ・登録数1/5/10・通常/別テキスト・native/commonを全件評価した。閾値はvalidationの通常発話で決め、testでは変更していない。",
        "",
        "主条件は登録各母音10、full、native、validation目標FAR 1%。",
        "",
        "|role|方式|coverage|条件付きFAR|条件付きFRR|全入力FRR [95% CI]|EER|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("verification", "cross_text_verification"):
        for method in METHODS:
            c = test["conditions"][f"native/{method}/n10/full/{role}"]
            p = c["operating_points"]["far_1pct"]
            ci = c["ci95"]["operating_points"]["far_1pct"]["all_input_frr"]
            lines.append(
                f"|{role}|{method}|{percent(c['query_coverage'])}|{percent(p['far'])}|{percent(p['frr'])}|{percent(p['all_input_frr'])} [{percent(ci['lower'])}, {percent(ci['upper'])}]|{percent(c['pooled_eer'])}|"
            )
    lines += [
        "",
        "## 長さ比較（native、登録10、全入力FRR）",
        "",
        "|role|方式|1秒|2秒|3秒|5秒|full|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("verification", "cross_text_verification"):
        for method in METHODS:
            vals = [
                percent(
                    test["conditions"][f"native/{method}/n10/{cap}/{role}"][
                        "operating_points"
                    ]["far_1pct"]["all_input_frr"]
                )
                for cap in CAPS
            ]
            lines.append("|" + "|".join([role, method, *vals]) + "|")
    lines += [
        "",
        "## 方式差（全入力FRR、percentage points）",
        "",
        "|role|比較|差|95% CI|",
        "|---|---|---:|---:|",
    ]
    for role in ("verification", "cross_text_verification"):
        for method in ("vowel_context20", "ecapa_whole"):
            d = test["paired_differences"][
                f"method_minus_exact/native/{method}/n10/full/{role}/far_1pct"
            ]
            ci = d["ci95_percentage_points"]
            lines.append(
                f"|{role}|{method} − vowel_exact|{d['point_difference_percentage_points']:.3f}|[{ci['lower']:.3f}, {ci['upper']:.3f}]|"
            )
    lines += [
        "",
        "## 全成果物",
        "",
        "- [全360条件・3動作点・件数・CI](all-conditions.csv)",
        "- [元発話長と実利用時間の全層](duration-strata.csv)",
        "- [全方式差・長さ差のpaired CI](paired-differences.csv)",
        "- [全条件のROC/DET・長さ曲線・固定5秒以上集合](index.html)",
        "",
        "共通入力は5母音が揃うqueryの交差集合。短いcapほど選択率が下がり、条件付き指標が改善しても全入力への改善とは限らない。全入力FRRではno_scoreを拒否として数える。各話者・roleのscore不足時は条件付きFRR/FAR・EER/ROC/DET・条件付きCIを評価不能とし、全入力値とcoverageは保持した。実利用時間はscoreなしqueryで0秒とし、利用可能なanchor量と区別した。",
        "",
        "ECAPAはVoxCeleb事前学習・20,767,552 parameters、母音方式はJVS 70話者学習・65,920 parametersで、入力範囲、目的関数、集約も異なる。方式としての比較であり、音素分割の因果効果やパラメータ数の効果は切り分けられない。",
        "",
        "JVSの管理された録音条件での結果。別日・別端末・雑音・なりすまし耐性は評価していない。Phase 3/6でtestが既に参照されており、新規の独立holdoutではない。speaker CIは学習・閾値較正の不確かさを含まない。誤り0件のbootstrap区間が[0,0]でも母集団の誤り確率0を保証しない。低FARは試行分解能・件数と併せて読む。全層の空集合・支持不足は省略せずNEとした。",
        "",
    ]
    with (output / "results.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    manifest = {
        str(p.relative_to(output)): sha256_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write_json(
        output / "manifest.json",
        {
            "files": manifest,
            "condition_cells": 360,
            "operating_point_rows": len(tables),
            "duration_rows": len(duration),
            "paired_rows": len(pairs),
            "figures_png": len(list(figures.glob("*.png"))),
            "figures_pdf": len(list(figures.glob("*.pdf"))),
        },
    )
    return {
        "condition_cells": 360,
        "operating_point_rows": len(tables),
        "figures": len(list(figures.glob("*.png"))),
    }
