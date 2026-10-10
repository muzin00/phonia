"""Publish all measured cells, paired CIs and auditable equal-time details."""

import csv
import html
from pathlib import Path

import study

BASE = Path(__file__).resolve().parent


def percent(value):
    return f"{100 * value:.3f}%" if value is not None else "—"


def ci(value):
    return (
        f"[{value['lower']:.3f}, {value['upper']:.3f}]"
        if value["lower"] is not None
        else "—"
    )


def main():
    config = study.read_json(study.CONFIG)
    run = study.ROOT / config["run_directory"]
    study.frozen(config, run, "test")
    measured = study.read_json(run / "test-metrics.json")
    inputs = {
        split: study.read_json(run / f"{split}-inputs.json")
        for split in ("validation", "test")
    }
    checks = {split: study.validate_plans(data) for split, data in inputs.items()}
    source_checks = {
        split: study.audit_source_slices(data, config) for split, data in inputs.items()
    }
    all_cells = []
    for split in ("validation", "test"):
        metrics = study.read_json(run / f"{split}-metrics.json")
        for key, cell in metrics["conditions"].items():
            support, condition, role = key.split("/")
            for point, rates in cell["operating_points"].items():
                all_cells.append(
                    {
                        "split": split,
                        "support": support,
                        "condition": condition,
                        "role": role,
                        "operating_point": point,
                        "queries": cell["queries"],
                        "scored_queries": cell["scored_queries"],
                        "coverage": cell["query_coverage"],
                        "eer": cell["pooled_eer"],
                        **rates,
                    }
                )
    with (BASE / "evaluation-cells.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_cells[0]))
        writer.writeheader()
        writer.writerows(all_cells)
    report = {
        "status": "completed",
        "protocol": config,
        "test": measured,
        "validation": study.read_json(run / "validation-metrics.json"),
        "input_preparation": {
            split: data["preparation"] for split, data in inputs.items()
        },
        "matched_plan_checks": checks,
        "original_source_slice_checks": source_checks,
        "thresholds": study.read_json(run / "validation-thresholds.json"),
        "inference": {
            split: study.read_json(run / split / "inference.json") for split in inputs
        },
        "enrollment_seconds": {
            split: [
                data["profiles"][s]["budget_frames"] / 24000 for s in data["speakers"]
            ]
            for split, data in inputs.items()
        },
        "difference_direction": "extended_minus_vowels5; negative_error_rate_difference_is_better",
        "claim_scope": "one_seed_exploratory_on_previously_observed_JVS_test; same_seven_phone_encoder; no_claim_about_retraining_or_independent_holdout",
    }
    study.write_json(BASE / "evaluation-results.json", report)
    rows = []
    for support in study.SUPPORTS:
        for role in study.ROLES:
            for condition in study.CONDITIONS:
                cell = measured["conditions"][f"{support}/{condition}/{role}"]
                p = cell["operating_points"]["far_1pct"]
                label = (
                    "5母音"
                    if condition == study.CONDITIONS[0]
                    else "5母音＋取得できたm/n"
                )
                rows.append(
                    [
                        support,
                        "通常文" if role == "verification" else "別テキスト",
                        label,
                        f"{cell['scored_queries']}/{cell['queries']}",
                        percent(p["all_input_far"]),
                        percent(p["all_input_frr"]),
                        percent(cell["pooled_eer"]),
                    ]
                )
    header = [
        "対象",
        "発話",
        "方式",
        "照合可能／全発話",
        "全入力FAR",
        "全入力FRR",
        "条件付きEER",
    ]
    table = (
        "| "
        + " | ".join(header)
        + " |\n| "
        + " | ".join(["---"] * len(header))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in rows)
    )
    diff_rows = []
    for support in study.SUPPORTS:
        for role in study.ROLES:
            for metric in ("all_input_far", "all_input_frr", "pooled_eer"):
                suffix = f"far_1pct/{metric}" if metric != "pooled_eer" else metric
                value = measured["paired_differences"][f"{support}/{role}/{suffix}"]
                diff_rows.append(
                    [
                        support,
                        "通常文" if role == "verification" else "別テキスト",
                        metric,
                        f"{value['difference_percentage_points']:+.3f}",
                        ci(value["ci95_percentage_points"]),
                    ]
                )
    dheader = ["対象", "発話", "指標", "7音素側−5母音（point）", "差の95% CI（point）"]
    dtable = (
        "| "
        + " | ".join(dheader)
        + " |\n| "
        + " | ".join(["---"] * len(dheader))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in diff_rows)
    )
    text = f"""# /m/・/n/ 追加の使用音声量を揃えた比較

同じ7音素共通encoderで、5母音のみと5母音＋取得できたm/nを比較した。
1条件・1 seedの学習済み重みを共有し、登録・照合ごとに実際のPCMサンプル数を完全に揃えた。
登録は最大3秒、照合は最大1秒。元の母音候補が少ない場合は両方式とも同じ短い予算を使う。

## test結果（validationで目標FAR 1%の閾値を固定）

{table}

`common7`は7音素側が全7音素を使える同じ発話のみ。精度差の主比較である。
`native`は通常文750・別テキスト450発話を全て含む。追加子音がない場合は5母音を使い、
母音不足の入力は拒否して全入力FRRへ含める。条件付きEERはscoreがある入力のみの値。
条件と対象集合ごとにvalidation通常文から閾値を決め、別テキストとtestで校正し直していない。

## 方式差と不確かさ

{dtable}

差は追加子音側−5母音。エラー率の負値は改善を表す。
15話者の同じ10,000 bootstrap drawを共有して差の95% CIを求めた。
これは既に観測済みのJVS testでの探索的な比較であり、独立holdout・別日・別端末・別seedの確認ではない。
同じ7音素学習済みencoder内での入力音素の追加効果を測り、5母音のみの学習モデルとの優劣は測っていない。

## 時間とデータの監査

両splitで合計{sum(checks.values()):,}個の登録／照合planについて、正確な時間一致、30〜250ms、
元区間内のcenter crop、同じ原子区間の重複なし、同じWAV内での重なりなしを確認した。
登録の候補元WAV集合は両方式で同じ。追加子音もそのWAV集合から抽出した。
queryの候補元WAVは方式間で同じで、音素境界の外側や無音padは加えていない。
追加子音は学習と同じ30ms以上・RMS−50dBFS以上の規則を適用した。

同じ総時間でも、音素ごとの時間・区間数・実際に選んだPCM内容は異なる。
各音素の平均scoreを等重み統合するため、音素の追加と時間配分・統合重みの変化を含む結果である。

全2,400発話×15 claimed話者×2方式＝72,000 score slotsを生成し、3回の同一batch推論の完全一致、
重み不変、試行ID・話者ラベル・音素score・統合score・音声量の一致を検査した。

[固定条件と再現手順](README.md)・[全48セルのCSV](evaluation-cells.csv)・
[閾値・件数・CIのJSON](evaluation-results.json)・[ブラウザ用比較表](evaluation-results.html)を参照する。
"""
    (BASE / "evaluation-results.md").write_text(text, encoding="utf-8")

    def html_table(headers, values):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(x)}</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
                for row in values
            )
            + "</tbody></table>"
        )

    document = """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>m/n追加：使用時間を揃えた比較</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#172536;background:#f3f6fa;margin:0}main{max-width:1240px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{margin-top:32px}.box{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:20px;margin:18px 0;overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{padding:11px 14px;border-bottom:1px solid #dbe2ec;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}th{background:#eaf0f8}tr:nth-child(even){background:#f7f9fc}a{color:#2161b1}.note{color:#526174;font-size:14px}</style><main><h1>/m/・/n/ 追加：使用音声量を揃えた比較</h1><p>同じ学習済みencoder・登録最大3秒・照合最大1秒。方式間の実使用サンプル数は完全一致。</p><div class="box"><h2>test結果：validation目標FAR 1%</h2>"""
    document += (
        html_table(header, rows)
        + "<p class='note'>common7：全7音素が使える同じ発話。native：全発話、追加子音なしは5母音で照合。全入力FRRは音素不足による拒否を含みます。</p></div><div class='box'><h2>方式差と話者95%信頼区間</h2>"
        + html_table(dheader, diff_rows)
        + "<p class='note'>追加子音側−5母音。エラー率の負値は改善。話者bootstrap 10,000回。観測済みJVS testでの探索的比較です。</p></div><p><a href='evaluation-results.md'>結果の解釈・監査</a> · <a href='evaluation-cells.csv'>全48セルCSV</a> · <a href='evaluation-results.json'>全指標JSON</a></p></main></html>"
    )
    (BASE / "evaluation-results.html").write_text(document, encoding="utf-8")
    outputs = {
        str(p.relative_to(run)): study.sha256_file(p)
        for p in run.rglob("*")
        if p.is_file() and p.name != "study-report.json"
    }
    study.write_json(
        run / "study-report.json",
        {
            "status": "completed",
            "matched_plan_checks": checks,
            "test_previously_observed": True,
            "outputs_sha256": outputs,
        },
    )
    print(f"published comparison: {BASE / 'evaluation-results.html'}")


if __name__ == "__main__":
    main()
