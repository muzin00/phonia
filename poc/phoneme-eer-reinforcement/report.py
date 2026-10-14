"""Report paired EER changes and fixed validation-threshold FAR/FRR."""

import csv

from common import ARMS, BASE, VARIANTS, comparisons, np, read, write


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
            for treatment, control in comparisons():
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
        "calibration": {
            str(seed): read(run / f"calibration-{seed}.json")
            for seed in config["encoder_seeds"]
        },
        "score_diagnostics": read(run / "score-audit.json"),
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
        "# 実測EERを報酬にしたTransformerの強化学習",
        "",
        "SRC490・36音素、固定encoderの3音響seed平均。BCEは対照、weightは重み付け方策、subsetは音素選択方策。",
        "各条件512発話/1,024照合batch・500更新。RLは各batchで4回行動を抽選。selectedはvalidation選定、finalは固定終点。",
        "",
        "## EER",
        "",
        "| 集合 | コーパス | 追加前 | BCE選定 | 重みRL選定 | 選択RL選定 | BCE最終 | 重みRL最終 | 選択RL最終 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        cells = " | ".join(f"{100 * row[v]['eer']:.3f}%" for v in VARIANTS)
        lines.append(f"| {row['split']} | {row['corpus']} | {cells} |")
    lines += [
        "",
        "## RL − BCEのtest EER差",
        "",
        "| コーパス | 方策 | checkpoint | 差 | 改善 / 同値 / 悪化seed | 対応付き95%区間 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    confidence = payload["bootstrap"]["eer_delta_ci95_pp"]
    for row in summaries:
        if row["split"] != "test":
            continue
        for arm in ("weight", "subset"):
            for mode in ("selected", "final"):
                key = f"{arm}-{mode}-versus-bce-{mode}"
                value = row["comparisons"][key]
                low, high = confidence[f"{row['corpus']}/{key}"]
                lines.append(
                    f"| {row['corpus']} | {arm} | {mode} | {value['mean_delta_pp']:+.3f} pp | {value['improved']} / {value['tied']} / {value['worsened']} | [{low:+.3f}, {high:+.3f}] pp |"
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
        "## validationで選択された追加更新数",
        "",
        "| 音響seed | 出発点の旧更新 | BCE | 重みRL | 選択RL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in config["encoder_seeds"]:
        values = [payload["training"][f"{seed}/{arm}"] for arm in ARMS]
        cells = " | ".join(str(v["selected_update"]) for v in values)
        lines.append(f"| {seed} | {values[0]['source_selected_update']} | {cells} |")
    lines += [
        "",
        "## 学習中の行動探索",
        "",
        "| seed | 方策 | 平均報酬EER | rollout間EER標準偏差 | 非ゼロadvantage率 | 平均選択音素数 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for seed in config["encoder_seeds"]:
        for arm in ("weight", "subset"):
            history = read(run / f"acoustic-{seed}/{arm}/history.json")
            means = [r["training_mean"] for r in history if r["training_mean"]]
            avg = {k: float(np.mean([r[k] for r in means])) for k in means[0]}
            lines.append(
                f"| {seed} | {arm} | {100 * avg['reward_macro_eer']:.3f}% | {100 * avg['reward_rollout_std']:.3f} pp | {100 * avg['nonzero_advantage_fraction']:.1f}% | {avg['sampled_mean_phones']:.2f} |"
            )
    lines += [
        "",
        "報酬は学習batchの実測EER。正解ラベルは報酬計算だけに用い、方策の入力には含めない。探索中の報酬EERと決定的推論のEERは別の量。選択方策は音素数を直接減らす報酬を持たない。",
        "EERは採点可能分、FAR/FRRは未採点を拒否として含む。採点対象は全条件固定。test FARは同値に揃えていない。",
        "3音響seed・1方策seed・単一設定。全testは既観測。CIは固定モデルを条件とした話者bootstrapで学習・選択の変動を含まない。計算量と構造も条件間で異なる。標準モデルは置き換えない。",
        "",
        f"成果物: `{config['run_directory']}`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。",
        "",
    ]
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)
