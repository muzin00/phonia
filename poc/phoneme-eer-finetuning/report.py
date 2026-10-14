"""Report paired EER changes and fixed validation-threshold FAR/FRR."""

import csv

from common import ARMS, BASE, VARIANTS, np, read, write


def report(config, run):
    rows = read(run / "metrics.json")
    summaries = []
    for split in ("validation", "test"):
        for corpus in [*config["corpora"], "macro"]:
            chosen = [
                r
                for r in rows
                if r["split"] == split
                and r["role"] == "verification"
                and (corpus == "macro" or r["corpus"] == corpus)
            ]
            cell = {"split": split, "corpus": corpus}
            for variant in VARIANTS:
                cell[variant] = {
                    k: float(np.mean([r[k] for r in chosen if r["variant"] == variant]))
                    for k in ("eer", "far", "frr", "coverage")
                }
            cell["comparisons"] = {}
            comparisons = [
                (f"soft_eer-{mode}", f"bce-{mode}") for mode in ("selected", "final")
            ]
            comparisons += [
                (f"{arm}-{mode}", "initial")
                for arm in ARMS
                for mode in ("selected", "final")
            ]
            for treatment, control in comparisons:
                deltas = []
                for seed in config["encoder_seeds"]:
                    values = {
                        v: np.mean(
                            [
                                r["eer"]
                                for r in chosen
                                if r["variant"] == v and r["seed"] == seed
                            ]
                        )
                        for v in (treatment, control)
                    }
                    deltas.append(float(values[treatment] - values[control]))
                cell["comparisons"][f"{treatment}-versus-{control}"] = {
                    "mean_delta_pp": 100 * float(np.mean(deltas)),
                    "seed_deltas_pp": [100 * v for v in deltas],
                    "improved": sum(v < -1e-12 for v in deltas),
                    "tied": sum(abs(v) <= 1e-12 for v in deltas),
                    "worsened": sum(v > 1e-12 for v in deltas),
                }
            summaries.append(cell)
    payload = {
        "config": config,
        "summary": summaries,
        "conditions": rows,
        "bootstrap": read(run / "bootstrap.json"),
        "audit": read(run / "independent-audit.json"),
        "training": {
            f"{seed}/{arm}": read(run / f"acoustic-{seed}/{arm}/training-summary.json")
            for seed in config["encoder_seeds"]
            for arm in ARMS
        },
    }
    write(run / "results.json", payload, immutable=True)
    write(BASE / "evaluation-results.json", payload)
    with (run / "conditions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# EER近似を加えたTransformer追加学習",
        "",
        "SRC490・36音素、固定encoderの3音響seed平均。initialは追加学習前、BCEは対照、soft-EERはBCE＋KL＋EER近似。",
        "両学習条件とも同じ出発モデル、同じ128発話/256照合batch、同じ2,000更新。selectedはvalidation選定、finalは固定終点。",
        "",
        "## EER",
        "",
        "| 集合 | コーパス | 追加前 | BCE選定 | soft-EER選定 | BCE最終 | soft-EER最終 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        cells = " | ".join(f"{100 * row[v]['eer']:.3f}%" for v in VARIANTS)
        lines.append(f"| {row['split']} | {row['corpus']} | {cells} |")
    lines += [
        "",
        "## soft-EER − BCEのtest EER差",
        "",
        "| コーパス | checkpoint | 差 | 改善 / 同値 / 悪化seed | 対応付き95%区間 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    confidence = payload["bootstrap"]["eer_delta_ci95_pp"]
    for row in summaries:
        if row["split"] != "test":
            continue
        for mode in ("selected", "final"):
            key = f"soft_eer-{mode}-versus-bce-{mode}"
            value = row["comparisons"][key]
            low, high = confidence[f"{row['corpus']}/{key}"]
            lines.append(
                f"| {row['corpus']} | {mode} | {value['mean_delta_pp']:+.3f} pp | {value['improved']} / {value['tied']} / {value['worsened']} | [{low:+.3f}, {high:+.3f}] pp |"
            )
    lines += [
        "",
        "## validation FAR1%閾値を固定したtest性能",
        "",
        "| コーパス | 条件 | FAR | 全入力FRR | 採点率 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if row["split"] != "test":
            continue
        for v in VARIANTS:
            values = row[v]
            lines.append(
                f"| {row['corpus']} | {v} | {100 * values['far']:.3f}% | {100 * values['frr']:.3f}% | {100 * values['coverage']:.3f}% |"
            )
    lines += [
        "",
        "## 選択された追加更新数",
        "",
        "| 音響seed | 出発点の旧更新 | BCE | soft-EER |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in config["encoder_seeds"]:
        a, b = (payload["training"][f"{seed}/{arm}"] for arm in ARMS)
        lines.append(
            f"| {seed} | {a['source_selected_update']} | {a['selected_update']} | {b['selected_update']} |"
        )
    lines += [
        "",
        "EERは採点可能分、FAR/FRRは未採点を拒否として含む。testのFARを同じ値に揃えた比較ではない。全条件で採点対象を固定。",
        "",
        "近似損失の温度0.03・係数1・追加2,000更新の単一条件。論文EERLossの再現ではない。3つは音響seedで、追加学習seedは1つ。全testは既観測。CIは固定モデルを条件とした話者bootstrapで、学習や選択の変動を含まない。今回の結果だけで標準モデルを置き換えない。",
        "",
        f"成果物: `{config['run_directory']}`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。",
        "",
    ]
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)
