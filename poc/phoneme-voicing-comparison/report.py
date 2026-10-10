"""Publish the shared-model comparison and separately scoped phone diagnostics."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from audit import digest, read, write

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
LABELS = {
    "vowels5": "5母音",
    "vowels_mn": "5母音＋m/n",
    "vowels_tk": "5母音＋t/k",
    "vowels_dg": "5母音＋d/g",
}
ROLES = {"verification": "通常文", "cross_text_verification": "別テキスト"}
DIAGNOSTIC_PAIRS = {
    "td": ("t", "d"),
    "kg": ("k", "g"),
    "sz": ("s", "z"),
    "sh": ("s", "h"),
}


def markdown_table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join("---" for _ in headers)
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )


def html_table(headers, rows):
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def measured_rows(cells, point, diagnostic=False):
    result = []

    def order(key):
        prefix, role = key.rsplit("/", 1)
        if diagnostic:
            pair, phone = prefix.split("/")
            return (
                list(ROLES).index(role),
                list(DIAGNOSTIC_PAIRS).index(pair),
                DIAGNOSTIC_PAIRS[pair].index(phone),
            )
        return list(ROLES).index(role), list(LABELS).index(prefix), 0

    for key in sorted(cells, key=order):
        cell = cells[key]
        prefix, role = key.rsplit("/", 1)
        rate = cell["operating_points"][point]
        label = (prefix.replace("/", " · /") + "/") if diagnostic else LABELS[prefix]
        result.append(
            [
                ROLES[role],
                label,
                str(cell["queries"]),
                f"{100 * rate['all_input_far']:.3f}% ({rate['false_accepts']}/{rate['impostor']})",
                f"{100 * rate['all_input_frr']:.3f}% ({rate['false_rejects']}/{rate['genuine']})",
                f"{100 * cell['pooled_eer']:.3f}%",
            ]
        )
    return result


def main():
    config = read(BASE / "config/evaluation.json")
    training = ROOT / config["training_run"]
    summary = read(training / "training/summary.json")
    preparation = read(training / "preparation-report.json")
    assert summary["status"] == "completed" and summary["selected_update"] == 30000
    assert summary["export_reload_bitwise_equal_phonemes"] == config["phonemes"]
    measured = {}
    inputs = {}
    paths = [
        BASE / "config/evaluation.json",
        BASE / "config/training.json",
        BASE / "audit.py",
        BASE / "report.py",
        training / "training/summary.json",
        training / "training-freeze.json",
        training / "bundle/encoder.pt",
        training / "bundle/feature-statistics.json",
    ]
    for name, run in (
        ("main", ROOT / config["run_directory"]),
        ("diagnostics", ROOT / config["diagnostics"]["run_directory"]),
    ):
        for freeze_name in ("design-freeze.json", "evaluation-freeze.json"):
            freeze = read(run / freeze_name)
            for path, checksum in freeze["files"].items():
                assert (
                    digest(
                        (ROOT if freeze_name == "design-freeze.json" else run) / path
                    )
                    == checksum
                ), path
            paths.append(run / freeze_name)
        assert read(run / "input-independent-audit.json")["status"] == "passed"
        assert read(run / "independent-numerical-audit.json")["status"] == "passed"
        measured[name] = {
            split: read(run / f"{split}-metrics.json")
            for split in ("validation", "test")
        }
        measured[name]["thresholds"] = read(run / "validation-thresholds.json")
        inputs[name] = {
            split: read(run / f"{split}-inputs.json")
            for split in ("validation", "test")
        }
        paths.extend(
            run / path
            for path in (
                "input-freeze.json",
                "validation-inputs.json",
                "test-inputs.json",
                "validation-metrics.json",
                "test-metrics.json",
                "validation-thresholds.json",
                "bootstrap-counts.npy",
                "bootstrap-metrics.npz",
                "bootstrap-differences.npz",
                "input-independent-audit.json",
                "independent-numerical-audit.json",
                "validation/scores.jsonl",
                "validation/inference.json",
                "validation/embedding-vectors.npz",
                "test/scores.jsonl",
                "test/inference.json",
                "test/embedding-vectors.npz",
            )
        )
    write(
        BASE / "training-results.json",
        {
            "summary": summary,
            "preparation": {
                k: v for k, v in preparation.items() if k != "source_inputs"
            },
        },
    )
    report = {
        "status": "completed",
        "protocol": config,
        "training": summary,
        "results": measured,
        "support": {
            "main": {
                split: data["preparation"] for split, data in inputs["main"].items()
            },
            "diagnostics": {
                split: {
                    key: {
                        **item["preparation"],
                        "excluded_queries": item["excluded_queries"],
                    }
                    for key, item in data["studies"].items()
                }
                for split, data in inputs["diagnostics"].items()
            },
        },
        "claim_scope": "same_frozen_fourteen_phone_model; main_common11; pair_specific_diagnostics; one_seed; previously_observed_JVS_test; labels_and_contexts_do_not_isolate_physiological_voicing",
    }
    write(BASE / "evaluation-results.json", report)
    exported = []
    for kind, result in measured.items():
        for split in ("validation", "test"):
            for key, cell in result[split]["conditions"].items():
                prefix, role = key.rsplit("/", 1)
                interval = (cell.get("ci95") or {}).get("pooled_eer") or {}
                for point, rate in cell["operating_points"].items():
                    exported.append(
                        {
                            "split": split,
                            "kind": kind,
                            "condition": prefix,
                            "role": role,
                            "operating_point": point,
                            "queries": cell["queries"],
                            "eer": cell["pooled_eer"],
                            "eer_ci95_lower": interval.get("lower"),
                            "eer_ci95_upper": interval.get("upper"),
                            **rate,
                        }
                    )
    with (BASE / "evaluation-cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(exported[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(exported)
    headers = [
        "発話",
        "条件",
        "照合発話数",
        "FAR（誤受入／他人試行）",
        "FRR（誤拒否／本人試行）",
        "EER",
    ]
    main_rows = measured_rows(measured["main"]["test"]["conditions"], "far_1pct")
    strict_rows = measured_rows(measured["main"]["test"]["conditions"], "far_0_1pct")
    diagnostic_rows = measured_rows(
        measured["diagnostics"]["test"]["conditions"], "far_1pct", True
    )
    diff_headers = ["発話", "比較（左−右）", "指標", "差（pp）", "対応付き95% CI（pp）"]
    diffs = []
    for key, entry in measured["main"]["test"]["paired_differences"].items():
        _, role, metric = key.split("/", 2)
        if metric not in config["primary_metrics"]:
            continue
        interval = entry["ci95_percentage_points"]
        label = {
            "far_1pct/all_input_far": "FAR",
            "far_1pct/all_input_frr": "FRR",
            "pooled_eer": "EER",
        }[metric]
        diffs.append(
            [
                ROLES[role],
                f"{LABELS[entry['minuend']]} − {LABELS[entry['subtrahend']]}",
                label,
                f"{entry['difference_percentage_points']:+.3f}",
                f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]",
            ]
        )
    primary_headers = [
        "単音素の比較（左−右）",
        "通常文EER差（pp）",
        "対応付き97.5% CI（pp）",
        "有声ラベル側のEER低下",
    ]
    primary_rows = []
    for key, entry in measured["diagnostics"]["test"]["primary_contrasts"].items():
        pair, contrast, _, _ = key.split("/", 3)
        interval = entry["ci_percentage_points"]
        status = (
            "支持"
            if entry["voiced_lower_eer_supported"]
            else ("逆方向の差" if interval["lower"] > 0 else "差を確定できず")
        )
        primary_rows.append(
            [
                f"{pair} · {contrast.replace('_minus_', ' − ')}",
                f"{entry['difference_percentage_points']:+.3f}",
                f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]",
                status,
            ]
        )
    support_headers = ["比較", "split", "通常文", "別テキスト"]
    support_rows = []
    for split in ("validation", "test"):
        counts = inputs["main"][split]["preparation"]["roles"]
        support_rows.append(
            [
                "主比較common11",
                split,
                str(counts["verification"]),
                str(counts["cross_text_verification"]),
            ]
        )
        for key, item in inputs["diagnostics"][split]["studies"].items():
            counts = item["preparation"]["queries_by_role"]
            support_rows.append(
                [
                    f"単音素 {key}",
                    split,
                    str(counts["verification"]),
                    str(counts["cross_text_verification"]),
                ]
            )
    notes = f"""JVS70話者＋Common Voice日本語70話者、{preparation["training_segments"]:,}区間で、14音素の共通モデルを30,000更新・1 seedで学習した。SRC4VCは含めない。65,920 parameters、128次元。最後の重みを固定し、testは学習に使用していない。

主比較は全11対象音素（5母音＋m/n/t/d/k/g）が揃う同じ発話集合common11を使用する。登録最大3秒・照合最大1秒で、4条件の実使用PCMフレーム数を完全一致させた。登録は全4条件が切り出し後のRMS条件を満たす最大の共通時間を5ms刻みで選ぶ。testは全話者3秒、validationは3話者を短くして最短2.740秒。queryは一つでも条件を満たさない区間があれば全条件共通で除外する。

母音とm/nは従来の候補を使用し、追加音素とsは境界内最大250msの中央crop後にQCする。時間調整でt/d/k/gを短くする場合は候補の終端に合わせる。破裂位置の正解アノテーションではなく、長い区間では先行の中央250ms cropの影響も残る。音声時間は等しくても、音素ごとの時間・区間数・文脈・母音の統合比率は異なる。

単音素はペアごとに対象発話を固定する。登録は最大10組の対応する区間を同数・同じ長さに揃え、照合は各音素1区間を短い方の長さに揃える。切り出し後RMSで片側が不適合なら両側を除外し、固定seedの次の候補組を使用する。各ペアは異なる発話集合なので、ペアをまたぐ数値比較は参考値になる。

閾値は各条件のvalidation通常文だけでFAR 1%・0.1%・EER動作点を固定し、test・別テキストには再校正せず適用した。表のFAR 1%／0.1%はvalidationの設定値で、testの実測FARを示す。EERは閾値を固定したFAR/FRRとは別の、test ROCの補間値。誤受入ゼロでも普遍的なFARゼロとは意味しない。

対応付き10,000回の話者bootstrapを全条件で共有した。主比較の全6組と補助診断の95% CIは探索的で多重比較補正なし。t/d・k/gの通常文EER差だけは2仮説のBonferroni補正として97.5% CIを事前固定した。

評価は過去にも観測済みのJVS test 15話者・1 seed。独立holdoutではなく、common11で除外された発話の性能は表していない。t/d・k/gの文脈を同一にしておらず、ラベルは実際の有声性を保証しない。s/zは実現様式が変わり得て、s/hは有声・無声の対立ではない。生体依存の因果効果はこの実験だけでは確定しない。前回8音素モデルとは学習音素・音素当たりの提示回数・評価対象が変わるため、直接差を追加音素だけの効果とは扱わない。"""
    sections = [
        ("主比較：validation FAR 1%の閾値", headers, main_rows),
        ("主比較：validation FAR 0.1%の閾値", headers, strict_rows),
        ("主比較：全6組の対応付き差", diff_headers, diffs),
        ("単音素診断：validation FAR 1%の閾値", headers, diagnostic_rows),
        ("事前指定した単音素の2仮説", primary_headers, primary_rows),
        ("固定した発話集合", support_headers, support_rows),
    ]
    md = (
        "# 14音素モデル：子音の組み合わせ比較\n\n"
        + notes
        + "\n\n"
        + "\n\n".join(
            "## " + title + "\n\n" + markdown_table(h, r) for title, h, r in sections
        )
        + "\n\n[全144セルCSV](evaluation-cells.csv) · [閾値・CI・設定JSON](evaluation-results.json) · [結果の解釈](interpretation.md)\n"
    )
    (BASE / "evaluation-results.md").write_text(md)
    body = "".join(
        f"<section><h2>{html.escape(title)}</h2>{html_table(h, r)}</section>"
        for title, h, r in sections
    )
    paragraphs = "".join(
        "<p>" + html.escape(paragraph) + "</p>" for paragraph in notes.split("\n\n")
    )
    (BASE / "evaluation-results.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phonia 14音素比較</title><style>body{font:15px/1.7 system-ui,sans-serif;color:#20313b;background:#f2f6f8;margin:0}main{max-width:1250px;margin:auto;padding:30px}h1{font-size:27px}section,.notes{background:white;border:1px solid #d2dce2;border-radius:12px;padding:20px;margin:22px 0}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{border-bottom:1px solid #dce4e8;padding:10px 14px;text-align:left}th{background:#edf3f6;position:sticky;top:0}tr:nth-child(even){background:#f7fafb}a{color:#106387}</style><main><h1>14音素モデル：子音の組み合わせ比較</h1><p><a href="interpretation.md">結果の解釈</a> · <a href="evaluation-cells.csv">全144セルCSV</a> · <a href="evaluation-results.json">閾値・CI・設定JSON</a></p>'
        + body
        + '<section class="notes"><h2>比較条件と解釈の範囲</h2>'
        + paragraphs
        + "</section></main></html>\n"
    )
    diagnostics_rows = [
        [str(r["update"]), f"{100 * r['vowel_only_macro_eer']:.3f}%"]
        for r in summary["validation_diagnostics"]
    ]
    phone_counts = summary["phoneme_microbatch_counts"]
    (BASE / "training-results.md").write_text(
        "# 14音素モデルの学習結果\n\n"
        + f"JVS・Common Voice計140話者、{preparation['training_segments']:,}区間（追加6音素{preparation['additional_segments']:,}区間）。\n30,000更新、3,000,000区間提示、seed 20260926、65,920 parameters、128次元。\n各音素のmicrobatchは{min(phone_counts.values()):,}〜{max(phone_counts.values()):,}回。最後の重みを採用し、14音素すべてでexport/reloadの出力が完全一致。testは学習に使用していない。\n\n"
        + markdown_table(["更新", "母音単一区間macro EER"], diagnostics_rows)
        + "\n\n診断値は重み選択や早期終了に使用していない。[発話照合の比較](evaluation-results.md)を参照。\n"
    )
    paths.extend(
        BASE / name
        for name in (
            "training-results.json",
            "training-results.md",
            "evaluation-results.json",
            "evaluation-results.md",
            "evaluation-results.html",
            "evaluation-cells.csv",
        )
    )
    write(
        ROOT / config["run_directory"] / "publication-report.json",
        {
            "status": "completed",
            "csv_cells": len(exported),
            "files": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        },
    )
    print(
        f"published {len(exported)} cells: {BASE / 'evaluation-results.html'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
