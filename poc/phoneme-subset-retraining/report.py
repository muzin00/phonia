"""Publish the fixed three-condition comparison without selecting on test."""

import csv

from common import BASE, np, read, write
from evaluation import ARMS, MODES


def report(config, run):
    rows = read(run / "metrics.json")
    summaries = []
    for split in ("validation", "test"):
        for mode in MODES:
            for corpus in [*config["corpora"], "macro"]:
                chosen = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["mode"] == mode
                    and r["role"] == "verification"
                    and (corpus == "macro" or r["corpus"] == corpus)
                ]
                cell = {"split": split, "mode": mode, "corpus": corpus}
                for arm in ARMS:
                    cell[arm] = {
                        key: float(np.mean([r[key] for r in chosen if r["arm"] == arm]))
                        for key in ("eer", "far", "frr", "coverage")
                    }
                cell["comparisons"] = {}
                for control in ARMS[:2]:
                    deltas = []
                    for seed in config["encoder_seeds"]:
                        values = {
                            arm: np.mean(
                                [
                                    r["eer"]
                                    for r in chosen
                                    if r["arm"] == arm and r["seed"] == seed
                                ]
                            )
                            for arm in (control, "train34_score34")
                        }
                        deltas.append(
                            float(values["train34_score34"] - values[control])
                        )
                    cell["comparisons"][control] = {
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
        "training": {
            str(seed): read(run / f"acoustic-{seed}/training-summary.json")
            for seed in config["encoder_seeds"]
        },
        "audit": read(run / "independent-audit.json"),
    }
    write(run / "results.json", payload, immutable=True)
    write(BASE / "evaluation-results.json", payload)
    with (run / "conditions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 34音素に合わせたTransformer再学習",
        "",
        "36→36＝36音素学習・36音素照合、36→34＝照合だけch/k除外、34→34＝学習・照合ともch/k除外。固定encoderの3音響seed平均。",
        "",
        "## EER",
        "",
        "| 集合 | checkpoint | コーパス | 36→36 | 36→34 | 34→34 | 再学習−照合時除外 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = [100 * row[arm]["eer"] for arm in ARMS]
        delta = row["comparisons"]["train36_score34"]["mean_delta_pp"]
        lines.append(
            f"| {row['split']} | {row['mode']} | {row['corpus']} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% | {delta:+.3f} pp |"
        )
    lines += [
        "",
        "## validationで選んだtestモデルの固定閾値性能",
        "",
        "| コーパス | 条件 | FAR | 全入力FRR | 採点率 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if row["split"] != "test" or row["mode"] != "selected":
            continue
        for arm, label in zip(ARMS, ("36→36", "36→34", "34→34"), strict=True):
            value = row[arm]
            lines.append(
                f"| {row['corpus']} | {label} | {100 * value['far']:.3f}% | {100 * value['frr']:.3f}% | {100 * value['coverage']:.3f}% |"
            )
    lines += [
        "",
        "## 再学習のEER差と対応付き95%区間",
        "",
        "| checkpoint | コーパス | 比較対象 | 差 | 改善 / 同値 / 悪化 seed | 95%区間 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    confidence = payload["bootstrap"]["eer_delta_ci95_pp"]
    for row in summaries:
        if row["split"] != "test":
            continue
        for control in ARMS[:2]:
            c = row["comparisons"][control]
            lo, hi = confidence[f"{row['corpus']}/{row['mode']}-versus-{control}"]
            lines.append(
                f"| {row['mode']} | {row['corpus']} | {control} | {c['mean_delta_pp']:+.3f} pp | {c['improved']} / {c['tied']} / {c['worsened']} | [{lo:+.3f}, {hi:+.3f}] pp |"
            )
    lines += [
        "",
        "## 選択更新",
        "",
        "| 音響seed | 36音素学習 | 34音素学習 |",
        "| --- | ---: | ---: |",
    ]
    for seed in config["encoder_seeds"]:
        before = next(
            r["update"]
            for r in rows
            if r["seed"] == seed
            and r["arm"] == "train36_score36"
            and r["mode"] == "selected"
        )
        after = payload["training"][str(seed)]["selected_update"]
        lines.append(f"| {seed} | {before} | {after} |")
    lines += [
        "",
        "学習ペア・初期値・更新数・LR・encoderは固定。モデルとvalidation閾値を凍結してからtest推論。selectedでは各学習条件のvalidation最良モデルを使うため、学習とcheckpoint選択の効果を含む。36→34の対照は元の36音素validationで選ばれたcheckpointを維持し、34音素向けに選び直していない。finalは全条件64k固定。",
        "",
        "34音素は前実験で固定された候補で、今回は再探索していない。全testは既観測。前回の探索/確認validation話者も今回はcheckpoint選択に使うため、独立確認とは扱わない。CIは固定モデル・固定音素集合を条件とした対応付き話者bootstrapで、学習・選択の変動を含まない。EERは採点可能分、FAR/FRRは未採点を含む。",
        "",
        f"実行成果物: `{config['run_directory']}`。条件・再現方法は[README](README.md)。全seed・JVS別テキスト・監査は[evaluation-results.json](evaluation-results.json)。",
        "",
    ]
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)
