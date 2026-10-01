"""Build a standalone browser report from the completed validation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
VOWELS = ("a", "i", "u", "e", "o")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def average(values: list) -> float | None:
    if any(value is None for value in values):
        return None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite report metric")
    return statistics.mean(values)


def build(comparison: Path) -> dict:
    matrix = read(comparison / "matrix.json")
    summary = read(comparison / "results.json")
    rows = matrix["runs"]
    if len(rows) != 54 or len({row["run_id"] for row in rows}) != 54:
        raise ValueError("expected 54 unique matrix runs")
    recorded = {item["run_id"]: item for item in summary["results"]}
    if (
        len(summary["results"]) != 54
        or len(recorded) != 54
        or set(recorded) != {row["run_id"] for row in rows}
    ):
        raise ValueError("summary does not match matrix")
    if summary["matrix_sha256"] != matrix["sha256"]:
        raise ValueError("summary matrix hash differs")
    groups = {}
    for row in rows:
        run_dir = comparison / "runs" / row["run_id"]
        result = read(run_dir / "full-result.json")
        if result != recorded[row["run_id"]]:
            raise ValueError(f"summary differs: {row['run_id']}")
        if result["status"] != "completed" or not result["repeat_evaluation_identical"]:
            raise ValueError(f"incomplete run: {row['run_id']}")
        if result["matrix_sha256"] != matrix["sha256"]:
            raise ValueError("run matrix hash differs")
        update = result["selected_checkpoint_update"]
        metric_path = (
            run_dir / f"validation/update-{update:06d}/metrics/validation.json"
        )
        metric = read(metric_path)
        if metric["partial"] or metric["split"] != "validation":
            raise ValueError("full validation metrics required")
        if (
            metric["primary_metric"]
            != "validation_verification_10_enrollment_macro_eer"
        ):
            raise ValueError("unexpected ranking metric")
        if metric["macro_eer"] != result["validation_macro_eer"]:
            raise ValueError("selected metric differs from result")
        if metric["score_sha256"] != result["validation_score_sha256"]:
            raise ValueError("selected score hash differs from result")
        if metric["trial_count"] != result["validation_trial_count"]:
            raise ValueError("selected trial count differs from result")
        if not (run_dir / "checkpoints/best.pt").is_file():
            raise FileNotFoundError(run_dir / "checkpoints/best.pt")
        training = read(run_dir / "training/summary.json")
        if training["completed_updates"] != result["completed_updates"]:
            raise ValueError("training updates differ")
        if not (result["completed_updates"] == 30000 or result["early_stopped"]):
            raise ValueError("training did not reach a defined stopping condition")
        if any(
            stage.get("command") and stage["exit_code"] != 0
            for stage in result["stages"]
        ):
            raise ValueError("failed recorded command stage")
        primary = metric["roles"]["verification"]["10"]
        item = {
            "seed": row["seed"],
            "run_id": row["run_id"],
            "eer": result["validation_macro_eer"],
            "cross_text_eer": metric["roles"]["cross_text_verification"]["10"][
                "macro_eer"
            ],
            "short_eer": metric["strata"]["duration"]["30-49ms"]["macro_eer"],
            "vowels": {vowel: primary["vowels"][vowel]["eer"] for vowel in VOWELS},
            "enrollment": {
                count: metric["roles"]["verification"][count]["macro_eer"]
                for count in ("1", "5", "10")
            },
            "strata": {
                dimension: {
                    value: report["macro_eer"] for value, report in values.items()
                }
                for dimension, values in metric["strata"].items()
            },
            "completed_updates": result["completed_updates"],
            "selected_update": update,
            "early_stopped": result["early_stopped"],
            "recorded_seconds": sum(
                stage["wall_seconds"] for stage in result["stages"]
            ),
            "train_seconds": sum(
                stage["wall_seconds"]
                for stage in result["stages"]
                if stage["stage"].startswith("train")
            ),
            "peak_rss_bytes": result["sampled_peak_rss_bytes"],
            "artifact_bytes": result["artifact_bytes"],
            "trial_count": result["validation_trial_count"],
            "checkpoint_sha256": result["selected_checkpoint_sha256"],
            "score_sha256": result["validation_score_sha256"],
            "recovered": result.get("recovered_after_runner_interruption", False),
            "history": [],
        }
        for path in sorted(
            (run_dir / "validation").glob("update-*/metrics/validation.json")
        ):
            history_update = int(path.parents[1].name.removeprefix("update-"))
            if history_update == update:
                history_eer = item["eer"]
            else:
                history_eer = read(path)["macro_eer"]
            item["history"].append({"update": history_update, "eer": history_eer})
        group = groups.setdefault(
            row["config_id"],
            {
                "config_id": row["config_id"],
                "input": row["input"],
                "encoder": row["encoder"],
                "rms": row["rms_enabled"],
                "supcon": row["supcon_enabled"],
                "comparison": row["comparison"],
                "parameter_count": result["parameter_count"],
                "runs": [],
            },
        )
        if group["parameter_count"] != result["parameter_count"]:
            raise ValueError("encoder parameter counts differ between seeds")
        group["runs"].append(item)
        del metric
    expected_seeds = set(
        read(comparison / "execution-budget.json")["protocol"]["seeds"]
    )
    for group in groups.values():
        runs = sorted(group["runs"], key=lambda run: run["seed"])
        if len(runs) != 3 or {run["seed"] for run in runs} != expected_seeds:
            raise ValueError("missing seed")
        group["runs"] = runs
        group["mean_eer"] = average([run["eer"] for run in runs])
        group["std_eer"] = statistics.pstdev(run["eer"] for run in runs)
        for field in ("cross_text_eer", "short_eer", "train_seconds"):
            group[field] = average([run[field] for run in runs])
        group["vowels"] = {
            vowel: average([run["vowels"][vowel] for run in runs]) for vowel in VOWELS
        }
        group["enrollment"] = {
            count: average([run["enrollment"][count] for run in runs])
            for count in ("1", "5", "10")
        }
        group["strata"] = {
            dimension: {
                value: average([run["strata"][dimension].get(value) for run in runs])
                for value in values
            }
            for dimension, values in runs[0]["strata"].items()
        }
        group["recorded_seconds"] = sum(run["recorded_seconds"] for run in runs)
        group["peak_rss_bytes"] = max(run["peak_rss_bytes"] for run in runs)
        group["artifact_bytes"] = sum(run["artifact_bytes"] for run in runs)
    ranked = sorted(
        groups.values(), key=lambda group: (group["mean_eer"], group["config_id"])
    )
    if len(ranked) != 18:
        raise ValueError("expected 18 configurations")
    for rank, group in enumerate(ranked, 1):
        group["rank"] = rank
        group["delta_eer"] = group["mean_eer"] - ranked[0]["mean_eer"]
        group["eligible"] = group["mean_eer"] <= ranked[0]["mean_eer"] + 0.001
    return {
        "comparison_id": comparison.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": matrix["sha256"],
        "results_sha256": digest(comparison / "results.json"),
        "source_directory": str(comparison),
        "seed_count": 3,
        "run_count": 54,
        "config_count": 18,
        "eligible_count": sum(group["eligible"] for group in ranked),
        "selection_finalized": False,
        "missing_selection_evidence": [
            "共通benchmarkの推論時間・最大メモリ",
            "対応付きbootstrapの差の95%区間（報告用）",
            "採用構成の正式記録",
        ],
        "groups": ranked,
    }


def markdown_report(data: dict, comparison: Path) -> str:
    """Generate the repository review document from the same audited page data."""
    groups = data["groups"]
    selection = data.get("selection")
    chosen = next(
        (
            g
            for g in groups
            if selection and g["config_id"] == selection["selected_config_id"]
        ),
        groups[0],
    )

    def percent(value):
        return "NA" if value is None else f"{value * 100:.3f}%"

    def interval(group):
        bounds = group.get("delta_eer_ci95")
        return (
            "未計測"
            if bounds is None
            else f"{bounds[0] * 100:+.3f}〜{bounds[1] * 100:+.3f}"
        )

    lines = [
        "# Phase 3・70話者・3 seedの18設定比較結果",
        "",
        f"Issue #32。実験ID: `{data['comparison_id']}`。",
        "54/54 runが完了し、失敗・未実行・資源上限による除外は0件。全runの選択checkpointの再評価が一致した。",
        "testデータは学習・順位付け・閾値較正・今回の追加評価に使用していない。",
        "",
        "## 採用構成と根拠",
        "",
        f"採用構成: **`{chosen['config_id']}`**。"
        if selection
        else "採用構成は未確定。以下は主順位の結果。",
        f"主条件はverification・登録10区間。5母音のEERを等重みで平均し、その3 seed平均は**{percent(chosen['mean_eer'])}**、母標準偏差（ddof=0）は{percent(chosen['std_eer'])}。",
        f"最小平均EER + 0.001以下（表示上{percent(groups[0]['mean_eer'] + 0.001)}以下）の許容候補集合は{data['eligible_count']}設定。",
        "候補集合内をseed間標準偏差、cross-text、30–49 ms、encoder parameter数、推論時間、最大メモリ、config IDの順に辞書式比較する事前規則を使用した。",
        "許容候補が1設定のため、補助指標・計測値で候補を選び直していない。単一配布モデル用seedは事前指定の`20260926`。",
        "",
        f"2位との差は{groups[1]['delta_eer'] * 100:.3f} percentage point。対応付きbootstrapの95%区間は{interval(groups[1])}ポイントで0を含む。",
        f"18位との差の区間は{interval(groups[-1])}ポイント。区間は記述的なpointwise推定で、多重比較補正をした有意差・同等性検定ではない。",
        "bootstrap区間は報告にのみ使用し、順位や許容候補集合を変更していない。",
        "",
        (
            f"validationで母音別FARを約1%に抑えた運用閾値では、母音・seed平均FRRは**{percent(data['selected_operating']['far_1pct']['macro_frr'])}**。"
            f"FAR約0.1%では{percent(data['selected_operating']['far_0_1pct']['macro_frr'])}。EERとは異なる動作点で、実用性を判断する際の制約として残る。"
            if selection
            else "運用閾値の結果は採用構成確定後に報告する。"
        ),
        "",
        "## 全18設定の順位",
        "",
        "EER・標準偏差は%、差の区間はpercentage point。推論時間は1区間のms。順位は丸め前の値で計算。",
        "",
        "| 順位 | encoder | RMS | 損失 | 平均EER | 標準偏差 | Cross-text | 30–49 ms | 最良との差の95%区間 | 推論ms | benchmark RSS MiB |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['rank']} | `{g['encoder']}`{'（限定）' if g['comparison'] != 'main' else ''} | {'on' if g['rms'] else 'off'} | {'AAM+SupCon' if g['supcon'] else 'AAM'} | {percent(g['mean_eer'])} | {percent(g['std_eer'])} | {percent(g['cross_text_eer'])} | {percent(g['short_eer'])} | {interval(g)} | {g.get('inference_median_ms', 0):.3f} | {g.get('benchmark_peak_rss_bytes', 0) / 1048576:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 学習・保存コスト",
            "",
            "学習時間は記録された学習ステージの3 seed平均。4並列の競合と中断・復旧の影響を含む。RSSは実行時の観測値で、上表の専用benchmarkとは別。",
            "",
            "| 順位 | encoder / RMS / 損失 | encoder parameter数 | 平均学習時間 h | 観測最大RSS GiB | 3 seed成果物 GiB |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for g in groups:
        lines.append(
            f"| {g['rank']} | `{g['encoder']}` / {'on' if g['rms'] else 'off'} / {'A+S' if g['supcon'] else 'A'} | {g['parameter_count']:,} | {g['train_seconds'] / 3600:.2f} | {g['peak_rss_bytes'] / 1073741824:.2f} | {g['artifact_bytes'] / 1073741824:.2f} |"
        )
    total = sum(g["recorded_seconds"] for g in groups)
    lines.extend(
        [
            "",
            f"全runの記録ステージ時間合計は{total / 3600:.2f}時間（並列実行なので実際の経過時間ではない）。",
            "ランナー中断後に完了済み学習を回収した4 runは、summary・checkpoint checksum・連続した有限値の履歴を検証し、選択checkpointを再評価した。",
            "中断時に記録されなかった時間を含む厳密な総所要時間は復元できない。途中の非採用score/曲線は圧縮済みで、採用checkpointのscore/曲線・閾値は保持している。",
            "",
            "## 採用設定の3 seed",
            "",
            "| seed | EER | Cross-text | 30–49 ms | 採用update | 完了update | 推論ms | benchmark RSS MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in chosen["runs"]:
        bench = run.get("benchmark", {})
        lines.append(
            f"| {run['seed']} | {percent(run['eer'])} | {percent(run['cross_text_eer'])} | {percent(run['short_eer'])} | {run['selected_update']:,} | {run['completed_updates']:,} | {bench.get('median_ms_per_segment', 0):.3f} | {bench.get('sampled_peak_rss_bytes', 0) / 1048576:.1f} |"
        )
    lines.extend(
        [
            "",
            "## validationで固定した主閾値",
            "",
            "verification・登録10区間のseed別・母音別閾値。FAR/FRRはvalidationでの実測値で、test・実運用での達成を保証する値ではない。",
            "登録1/5区間・pooled・EER運用閾値も各seedの`thresholds.json`に保存した。",
            "",
            "| seed | 母音 | FAR 1%閾値 | 実測FAR | FRR | FAR 0.1%閾値 | 実測FAR | FRR |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in chosen["runs"]:
        path = (
            comparison
            / "runs"
            / run["run_id"]
            / f"validation/update-{run['selected_update']:06d}/selections/thresholds.json"
        )
        thresholds = read(path)["per_count"]["10"]
        for vowel in VOWELS:
            one = thresholds[vowel]["far_1pct"]
            tenth = thresholds[vowel]["far_0_1pct"]
            lines.append(
                f"| {run['seed']} | /{vowel}/ | {one['threshold']:.6f} | {percent(one['far'])} | {percent(one['frr'])} | {tenth['threshold']:.6f} | {percent(tenth['far'])} | {percent(tenth['frr'])} |"
            )
    lines.extend(
        [
            "",
            "## 追加評価の手順と制約",
            "",
            "- 対応付き話者bootstrap: 辞書順の15 validation話者をPCG64・seed `20260929`で10,000回復元抽出。異なる話者が2人未満の抽出は再抽出。同一抽出を全18設定・3学習seedへ使用。",
            "- genuineの重みはn_s、元から異なる話者のimpostor s→tの重みはn_s×n_t。scoreの同値をまとめ、重み付きROCを線形補間する。重み付き母音EER→5母音平均→3 seed平均→最良候補との差→linear補間の2.5/97.5 percentile。seed・登録区間は再抽出しない。",
            "- 元scoreのSHA-256と457,995主trialのID順を照合。各runの無重み再計算が保存済みEERと1e-12以内で一致することを確認。抽出話者ID列、seed別10,000値、実装checksumを保存。",
            '- benchmark: 主queryを`["benchmark",20260930,segment_id]`のSHA-256順にし、共通先頭1,000区間を使用。CPU・float32・batch=1、intra/inter-op各1スレッド、専用プロセス・1 runずつ。WAV読出しを除き前処理・padding・encoderを含む。',
            "- 1周warmup後5周・計5,000区間時間を計測。CPU同期実行の`perf_counter_ns`中央値をrun値、3 seedの中央値を設定値とする。",
            "- モデル読込・warmup後に観測最大値をリセットし、macOS `libproc.proc_pidinfo(PROC_PIDTASKINFO).pti_resident_size`を5 ms間隔でサンプリング。設定値は3 seedの最大。短時間peakは捕捉できない場合がある。モデル・前処理・入力キャッシュを含むプロセスRSSで、allocatorの厳密なpeakやモデル単体メモリではない。",
            "- **手順上の逸脱**: benchmark API・スレッド数は元の実行予算に事前固定されていなかった。validation結果確認後・今回の計測前に`selection-evaluation/evaluation-budget.json`で固定した。事前登録済みとは扱わず、補完による測定として報告する。許容候補集合は既に1設定で、この補完で候補を変更していない。",
            "- 15評価話者・同一登録集合に条件付けた結果。18候補を比較したvalidationへの適応、小さな差、未知の収録環境、固定した学習レシピには制約がある。採用は相対評価であり、用途別の絶対性能要件を満たしたという判断ではない。",
            "",
            "## 成果物と再現手順",
            "",
            f"成果物ルート（Git管理外）: `artifacts/phoneme-speaker-encoder/comparisons/{data['comparison_id']}/`。",
            f"matrix SHA-256: `{data['matrix_sha256']}`。",
            f"results SHA-256: `{data['results_sha256']}`。",
        ]
    )
    if selection:
        lines.extend(
            [
                f"追加評価protocol SHA-256: `{selection['evaluation_protocol_sha256']}`。",
                f"bootstrap抽出ID SHA-256: `{selection['bootstrap_draws_sha256']}`。",
                f"benchmark query SHA-256: `{selection['benchmark_queries_sha256']}`。",
                "`selection-evaluation/selection.json`が選定規則・値・ファイルchecksumの正本。",
                "`selection-evaluation/selected-bundle/<seed>/`に3 checkpoint、各seed自身の閾値、run設定、train特徴統計を複製し、元checkpointのchecksumと照合した。",
                "",
                "| seed | checkpoint SHA-256 |",
                "|---:|---|",
            ]
        )
        for run in chosen["runs"]:
            lines.append(f"| {run['seed']} | `{run['checkpoint_sha256']}` |")
    lines.extend(
        [
            "",
            "```sh",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
            "  poc/phoneme-speaker-encoder/.venv/bin/python \\",
            "  poc/phoneme-speaker-encoder/scripts/evaluate_comparison.py run",
            "poc/phoneme-speaker-encoder/.venv/bin/python \\",
            "  poc/phoneme-speaker-encoder/scripts/build_comparison_report.py",
            "```",
            "",
            "追加評価は既存の完了キャッシュをchecksum照合して再利用する。protocol・実装が変われば停止し、同じ実験へ黙って混ぜない。",
            "ブラウザ用HTML・JSONと本レポートの生成版は`browser-report/`。採用checkpoint・評価scoreを変えずに再生成できる。",
            "",
            "## 次の工程",
            "",
            "採用設定だけの10/25/50話者×3 seedの入れ子学習曲線、validationの境界・gain感度診断は別作業。",
            "その後に設定・3 checkpoint・seed別閾値・評価手順を凍結してtestを一度評価する。今回の選定bundleは、その最終test工程を実行した記録ではない。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    budget = read(BASE / "config/full-execution-budget.json")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=ROOT
        / "artifacts/phoneme-speaker-encoder/comparisons"
        / budget["comparison_id"],
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    comparison = args.comparison_dir.resolve()
    output = args.output_dir or comparison / "browser-report"
    data = build(comparison)
    evidence_path = comparison / "selection-evaluation/ranking-with-evidence.json"
    if evidence_path.exists():
        evidence = read(evidence_path)
        if (
            evidence["matrix_sha256"] != data["matrix_sha256"]
            or evidence["results_sha256"] != data["results_sha256"]
            or [(g["config_id"], g["mean_eer"]) for g in evidence["groups"]]
            != [(g["config_id"], g["mean_eer"]) for g in data["groups"]]
        ):
            raise ValueError("selection evidence differs from current results")
        selection = evidence["selection"]
        for item in selection["files"]:
            if digest(comparison / item["file"]) != item["sha256"]:
                raise ValueError("selected bundle checksum differs")
        for name, key in (
            ("evaluation-budget.json", "evaluation_protocol_sha256"),
            ("bootstrap-draws.json", "bootstrap_draws_sha256"),
            ("benchmark-queries.json", "benchmark_queries_sha256"),
        ):
            if digest(comparison / "selection-evaluation" / name) != selection[key]:
                raise ValueError("selection protocol checksum differs")
        data = evidence
        chosen = next(
            g
            for g in data["groups"]
            if g["config_id"] == selection["selected_config_id"]
        )
        operating = {
            target: {"far": [], "frr": []} for target in ("far_1pct", "far_0_1pct")
        }
        for run in chosen["runs"]:
            thresholds = read(
                comparison
                / "selection-evaluation/selected-bundle"
                / str(run["seed"])
                / "thresholds.json"
            )["per_count"]["10"]
            for target, rates in operating.items():
                for rate in ("far", "frr"):
                    rates[rate].extend(
                        thresholds[vowel][target][rate] for vowel in VOWELS
                    )
        data["selected_operating"] = {
            target: {
                f"macro_{rate}": statistics.mean(values)
                for rate, values in rates.items()
            }
            for target, rates in operating.items()
        }
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    template = (BASE / "reports/comparison.html").read_text(encoding="utf-8")
    embedded = (
        serialized.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranking.json").write_text(serialized + "\n", encoding="utf-8")
    if data["selection_finalized"]:
        (output / "results.md").write_text(
            markdown_report(data, comparison), encoding="utf-8"
        )
    (output / "index.html").write_text(
        template.replace("__REPORT_DATA__", embedded), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output / "index.html"),
                "configurations": len(data["groups"]),
                "runs": data["run_count"],
                "eligible": data["eligible_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
