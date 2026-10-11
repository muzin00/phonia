"""One measured source for Japanese Markdown, HTML tables and JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from data import (
    BASE,
    ROLES,
    ROOT,
    checked,
    np,
    read_json,
    sha256_file,
    write_json,
)


def make_report(run):
    completion = read_json(run / "completion-verification.json")
    if completion["status"] != "passed":
        raise ValueError("independent audit required")
    for name, checksum in completion["output_sha256"].items():
        checked(ROOT / name, checksum)
    results = read_json(run / "results.json")
    bootstrap = read_json(run / "bootstrap-results.json")
    tables = []
    headers = [
        "統合方式 / seed",
        "採用更新",
        "validation 通常 EER",
        "validation 別文 EER",
        "test 通常 EER",
        "test 別文 EER",
    ]
    cells = []
    for name, splits in results["variants"].items():
        selected = (
            "—"
            if name == "equal"
            else str(read_json(run / name / "training-summary.json")["selected_update"])
        )
        cells.append(
            [
                name,
                selected,
                *[
                    f"{splits[split][role]['eer'] * 100:.3f}%"
                    for split in ("validation", "test")
                    for role in ROLES
                ],
            ]
        )
    tables.append(("全seedの測定値", headers, cells))
    summary = {}
    for kind in results["config"]["architectures"]:
        names = [f"{kind}-{s}" for s in results["config"]["seeds"]]
        summary[kind] = {}
        for role in ROLES:
            numbers = [results["variants"][n]["test"][role]["eer"] for n in names]
            summary[kind][role] = {
                "mean": float(np.mean(numbers)),
                "minimum": min(numbers),
                "maximum": max(numbers),
            }
    cells = []
    for kind, roles in summary.items():
        cells.append(
            [
                kind,
                *[
                    f"{roles[r]['mean'] * 100:.3f}%（{roles[r]['minimum'] * 100:.3f}–{roles[r]['maximum'] * 100:.3f}%）"
                    for r in ROLES
                ],
            ]
        )
    tables.append(
        ("test EERの3 seed平均（最小–最大）", ["統合方式", "通常", "別文"], cells)
    )
    cells = []
    for label, result in bootstrap["differences"].items():
        cells.append(
            [
                label,
                f"{result['difference'] * 100:+.3f}",
                f"[{result['ci95'][0] * 100:+.3f}, {result['ci95'][1] * 100:+.3f}]",
            ]
        )
    tables.append(
        (
            "EER差と話者bootstrap 95%区間（pp、負が改善）",
            ["比較 / 評価", "EER差", "95%区間"],
            cells,
        )
    )
    cells = []
    for name, splits in results["variants"].items():
        for role in ROLES:
            metric = splits["test"][role]
            point = metric["operating_points"]["far_1pct"]
            cells.append(
                [
                    name,
                    "通常" if role == ROLES[0] else "別文",
                    f"{point['all_input_far'] * 100:.3f}%",
                    f"{point['all_input_frr'] * 100:.3f}%",
                    f"{metric['scored_queries']}/{metric['all_queries']}",
                ]
            )
    tables.append(
        (
            "validation FAR 1%閾値を固定したtest結果（無採点は拒否）",
            ["方式 / seed", "評価", "全入力 FAR", "全入力 FRR", "採点発話"],
            cells,
        )
    )
    design = read_json(run / "design-freeze.json")
    paragraphs = [
        "全36音素encoderを固定し、登録上限30区間/音素、同じ照合発話・同じ音素別cosineを使って、等重み、MLP、Transformerによる重み付けを比較した。",
        f"学習はJVS＋Common Voiceの140話者、{design['train_queries']:,}照合発話。登録用WAVと照合用WAVはパス・音声SHAとも分離し、validation/test話者・音声を含めない。SRC4VCは使わない。",
        "音素ごとに登録・照合embedding、絶対差、積、区間数、平均区間長、embeddingのばらつき、平均RMSを入力する。話者ID・corpus ID・正解ラベルはモデル入力に含めない。他人ペアは同じcorpusから選び、本人/他人で利用可能音素のマスクをそろえる。",
        "MLPは各音素を独立に処理し、Transformerは64次元・4 head・2層で音素間の関係を処理する。両者ともsoftmax重みで元の音素別cosineを加重平均する。音素の並び順による位置encodingは使わず、音素ラベルembeddingを使う。",
        "各方式3 seed・4,000更新。同seedでは同じ学習ペア・順序を使用する。500更新ごとのvalidation通常EERでcheckpointを選び、同点は早い更新を採用。等重みで初期化した更新0も候補とし、悪化する学習結果を強制採用しない。",
        "6モデルのcheckpointとvalidation通常発話から決めた閾値を凍結してから新しいtest採点を実行した。過去にtest結果は観測済みであり、このデータは完全な未観測の外部評価ではない。",
        "EERは採点可能な通常735/750発話、別文392/450発話で算出する。5母音不足による無採点数と試行の対応は全方式共通。FAR/FRRの全入力指標では無採点を拒否として数える。",
        "平均は3モデルのEERの算術平均で、ensembleのEERではない。95%区間は15 test話者を共有して2,000回再標本化した差の区間で、この6モデルを固定した話者の不確実性を表す。3 seedは学習のばらつきを観察するための少数反復である。",
        "学習された重みは照合への寄与であり、生体依存度や録音品質の確率として解釈しない。",
        f"独立監査: {completion['score_rows']:,}採点行、EER {completion['eer_checks']}件、率 {completion['rate_checks']}件、bootstrap {completion['independent_bootstrap_checks']}件。入力音声・特徴・encoder・コードのSHAを学習後にも検証済み。",
    ]
    lines = [
        "# 固定encoderでのTransformer音素統合",
        "",
        *[p + "\n" for p in paragraphs],
    ]
    fragments = []
    for title, headers, cells in tables:
        lines.extend(
            [
                "## " + title,
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *["| " + " | ".join(row) + " |" for row in cells],
                "",
            ]
        )
        fragments.append(
            "<h2>"
            + html.escape(title)
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
    (BASE / "evaluation-results.md").write_text("\n".join(lines), encoding="utf-8")
    document = (
        "<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Transformer音素統合の比較</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font-family:system-ui;color:#172033;line-height:1.75;background:#f8fafc}h1{font-size:1.8rem}h2{font-size:1.25rem;margin-top:32px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #cbd5e1;padding:9px 12px;text-align:left;white-space:nowrap}th{background:#e2e8f0}tbody tr:nth-child(even){background:#f1f5f9}a{color:#155e75}</style><h1>固定encoderでのTransformer音素統合</h1>"
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs[:2])
        + "".join(fragments)
        + "<h2>評価条件</h2>"
        + "".join("<p>" + html.escape(p) + "</p>" for p in paragraphs[2:])
        + "<p><a href='evaluation-results.json'>JSON</a> · <a href='evaluation-results.md'>Markdown</a> · <a href='README.md'>再実行方法</a></p></html>"
    )
    (BASE / "evaluation-results.html").write_text(document, encoding="utf-8")
    write_json(
        BASE / "evaluation-results.json",
        {
            "source_run": str(run.relative_to(ROOT)),
            "config": results["config"],
            "variants": results["variants"],
            "seed_summary": summary,
            "bootstrap": bootstrap,
            "audit": {k: v for k, v in completion.items() if k != "output_sha256"},
            "tables": [{"title": t, "headers": h, "rows": c} for t, h, c in tables],
        },
    )
    write_json(
        run / "publication-freeze.json",
        {
            "files": {
                str((BASE / n).relative_to(ROOT)): sha256_file(BASE / n)
                for n in (
                    "evaluation-results.json",
                    "evaluation-results.md",
                    "evaluation-results.html",
                )
            },
            "results_sha256": sha256_file(run / "results.json"),
            "audit_sha256": sha256_file(run / "independent-audit.json"),
        },
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "artifacts/phoneme-transformer-fusion/fixed-all36-20261011-v1",
    )
    make_report(parser.parse_args().run)
