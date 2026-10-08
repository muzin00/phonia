"""Publish the two-condition training result from checksum-verified artifacts."""

import csv
import html
import json

from training import BASE, CONFIG, ROOT, checked, read_json, sha256_file, write_json


def build():
    config = read_json(CONFIG)
    run = ROOT / config["run_directory"]
    preparation = read_json(run / "preparation-report.json")
    freeze = read_json(run / "training-freeze.json")
    summary = read_json(run / "expanded/training/summary.json")
    if (
        summary["status"] != "completed"
        or summary["new_training_runs"] != 1
        or summary["test_used"]
        or freeze["config"] != config
    ):
        raise ValueError("completed frozen single-run training required")
    for name, checksum in summary["outputs_sha256"].items():
        checked(run / "expanded" / name, checksum)
    for name, checksum in preparation["outputs_sha256"].items():
        checked(run / name, checksum)
    baseline = ROOT / config["baseline_run"]
    for path in (baseline / "run.json", baseline / "training/summary.json"):
        checked(path, freeze["files"][str(path.relative_to(ROOT))])
    old = read_json(baseline / "training/summary.json")
    old_statistics = read_json(ROOT / config["baseline_statistics"])
    rows = [
        {
            "condition": "jvs70",
            "training_speaker_labels": 70,
            "training_segments": old_statistics["segment_count"],
            "encoder_parameters": old["parameter_count"],
            "training_head_parameters": 70 * 128,
            "completed_updates": old["completed_updates"],
            "selected_update": old["best"]["update"],
            "validation_macro_eer": old["best"]["eer"],
            "test_evaluated_in_this_study": False,
        },
        {
            "condition": "jvs70_cv70",
            "training_speaker_labels": freeze["speaker_count"],
            "training_segments": freeze["training_segments"],
            "encoder_parameters": summary["encoder_parameters"],
            "training_head_parameters": summary["training_head_parameters"],
            "completed_updates": summary["completed_updates"],
            "selected_update": summary["selected_update"],
            "validation_macro_eer": summary["validation_macro_eer"],
            "test_evaluated_in_this_study": False,
        },
    ]
    if [r["condition"] for r in rows] != config["conditions"] or any(
        r["encoder_parameters"] != 65920 for r in rows
    ):
        raise ValueError("comparison scope/model size changed")
    report = {
        "status": "completed",
        "new_training_runs": 1,
        "seed": config["seed"],
        "conditions": rows,
        "additional_data": {
            k: preparation[k]
            for k in (
                "speakers",
                "selected_clips",
                "aligned_clips",
                "eligible_segments",
                "excluded_segments",
                "by_vowel",
                "source_hours",
                "eligible_vowel_hours",
            )
        },
        "preparation_failure_count": len(preparation["failures"]),
        "validation_eer_difference_percentage_points": 100
        * (rows[1]["validation_macro_eer"] - rows[0]["validation_macro_eer"]),
        "expanded_training_elapsed_this_invocation_seconds": summary[
            "elapsed_this_invocation_seconds"
        ],
        "expanded_process_max_rss_bytes": summary["process_max_rss_bytes"],
        "export_reload_bitwise_equal_validation_segments": summary[
            "export_reload_bitwise_equal_validation_segments"
        ],
        "test_evaluated": False,
        "claim_scope": config["claim_scope"],
        "cross_corpus_person_identity_overlap": config[
            "cross_corpus_person_identity_overlap"
        ],
        "source_repository": config["source_repository"],
        "source_revision": config["source_revision"],
        "training_freeze_sha256": sha256_file(run / "training-freeze.json"),
        "training_summary_sha256": sha256_file(run / "expanded/training/summary.json"),
    }
    output = run / "report"
    output.mkdir(exist_ok=True)
    write_json(output / "training-results.json", report)
    with (output / "training-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {
        "jvs70": "JVSのみ（既存run）",
        "jvs70_cv70": "JVS＋Common Voice（新規run）",
    }
    header = [
        "条件",
        "話者ラベル数",
        "学習区間数",
        "encoder parameters",
        "学習update",
        "採用update",
        "validation macro EER",
    ]
    table_rows = [
        [
            labels[r["condition"]],
            str(r["training_speaker_labels"]),
            f"{r['training_segments']:,}",
            f"{r['encoder_parameters']:,}",
            str(r["completed_updates"]),
            str(r["selected_update"]),
            f"{100 * r['validation_macro_eer']:.3f}%",
        ]
        for r in rows
    ]
    intro = f"追加学習1 runが完了。条件は2つ、seedは{config['seed']}の1つ。追加{preparation['speakers']}ラベル・{preparation['selected_clips']:,}発話から{preparation['eligible_segments']:,}母音区間を採用した。validation macro EERは{100 * rows[0]['validation_macro_eer']:.3f}→{100 * rows[1]['validation_macro_eer']:.3f}%。"
    note = "このEERは母音単一区間の照合を5母音で平均した学習時の指標で、checkpoint選択に使ったvalidationの値。発話単位のFRR/FAR、母音不足対応との組み合わせ、独立holdoutは未評価。追加データで話者ラベル数・区間数・収録環境が同時に変わるため、純粋なデータ量だけの効果ではない。"
    caveat = "JVSのtrain/validation/testラベルを分離し、追加データは配布元のtrainだけを使用。匿名client_idとJVS話者の実在人物の重複は確認できない。人物の再特定は行っていない。1 seed・既存validationによる探索的な結果で、未知話者一般への改善は保証しない。testは今回の学習・選択・追加評価で使用していない。"
    artifacts = f"採用重み: `{config['run_directory']}/expanded/bundle/encoder.pt`。特徴統計: 同bundleの`feature-statistics.json`。再読込した重みの128次元embeddingはvalidation 8区間でbit一致。encoderは65,920 parameters、追加条件の140ラベル分類headは17,920 parametersで、推論用bundleには含めない。"
    markdown = "# Phase 8: 学習データ追加の学習結果\n\n" + intro + "\n\n"
    markdown += (
        "|" + "|".join(header) + "|\n|" + "|".join(["---"] * len(header)) + "|\n"
    )
    markdown += "".join("|" + "|".join(row) + "|\n" for row in table_rows)
    markdown += f"\n{note}\n\n{caveat}\n\n{artifacts}\n\n追加データの整列成功{preparation['aligned_clips']:,}/{preparation['selected_clips']:,}発話、失敗{len(preparation['failures'])}発話。除外{preparation['excluded_segments']:,}区間。元音声は{preparation['source_hours']:.3f}時間（失敗発話も含む）、採用母音の合計は{preparation['eligible_vowel_hours']:.3f}時間（250 ms crop前）。\n\n[固定条件・再現手順](README.md)、[HTML表](training-results.html)、[全数値CSV](training-comparison.csv)、[結果JSON](training-results.json)。\n"
    (output / "training-results.md").write_text(markdown, encoding="utf-8")
    markup = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr>"
    markup += "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
        for row in table_rows
    )
    body = f"<h1>Phase 8: 学習データ追加の学習結果</h1><p>{html.escape(intro)}</p><div class='scroll'><table>{markup}</table></div><p>{html.escape(note)}</p><p>{html.escape(caveat)}</p><p><a href='training-comparison.csv'>比較CSV</a> / <a href='training-results.json'>結果JSON</a> / <a href='README.md'>再現手順</a></p>"
    style = "body{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172c3c}h1{font-size:25px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #d6e1e8;padding:10px;white-space:nowrap;text-align:right}th{background:#e8f0f5}"
    (output / "training-results.html").write_text(
        f"<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Phase 8 学習データ追加</title><style>{style}</style><main>{body}</main></html>\n",
        encoding="utf-8",
    )
    for p in output.iterdir():
        if p.is_file():
            (BASE / p.name).write_bytes(p.read_bytes())
    print(
        json.dumps(
            {
                "status": "completed",
                "conditions": rows,
                "output": str(output.relative_to(ROOT)),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    build()
