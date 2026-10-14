"""Report paired EER changes and fixed selection-threshold FAR/FRR."""

import csv

from common import ARMS, BASE, VARIANTS, comparisons, np, read, write


def report(config, run):
    rows = read(run / "metrics.json")
    summaries = []
    for split in ("selection", "test"):
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
        "# EER改善を報酬にした学習データ選択",
        "",
        "学習済みの音響encoderを固定し、Transformerを同じ未学習状態から学習。3音響seed平均。",
        "uniformは均等抽選、randomはランダムな音素重視、rlは報酬に基づく音素重視。各条件8,000更新、32発話/64照合。",
        "100更新ごとに2行動を抽選し、同じモデル・AdamW状態から分岐した均等抽選との差で報酬を計算。最初の行動を必ず継続。",
        "selectedは独立したselection話者で選定、finalは8,000更新の終点。推論時は全条件で利用可能な全音素を使う。",
        "",
        "## EER",
        "",
    ]

    def table(headers):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    table(["集合", "コーパス", *VARIANTS])
    for row in summaries:
        cells = " | ".join(f"{100 * row[v]['eer']:.3f}%" for v in VARIANTS)
        lines.append(f"| {row['split']} | {row['corpus']} | {cells} |")
    lines += ["", "## Test EERの対応差", ""]
    table(["コーパス", "比較", "差", "改善/同値/悪化seed", "対応付き95%区間"])
    confidence = payload["bootstrap"]["eer_delta_ci95_pp"]
    for row in summaries:
        if row["split"] != "test":
            continue
        for treatment, control in comparisons():
            key = f"{treatment}-versus-{control}"
            value = row["comparisons"][key]
            low, high = confidence[f"{row['corpus']}/{key}"]
            lines.append(
                f"| {row['corpus']} | {key} | {value['mean_delta_pp']:+.3f} pp | {value['improved']} / {value['tied']} / {value['worsened']} | [{low:+.3f}, {high:+.3f}] pp |"
            )
    lines += ["", "## Selection FAR1%閾値を固定したtest性能", ""]
    table(["コーパス", "条件", "FAR", "全入力FRR", "採点率"])
    for row in summaries:
        if row["split"] != "test":
            continue
        for v in VARIANTS:
            values = row[v]
            lines.append(
                f"| {row['corpus']} | {v} | {100 * values['far']:.3f}% | {100 * values['frr']:.3f}% | {100 * values['coverage']:.3f}% |"
            )
    lines += ["", "## 学習・探索", ""]
    table(
        [
            "音響seed",
            "条件",
            "選定更新",
            "実際のstudent更新",
            "正/負のrollout報酬数",
            "負報酬でも継続した回数",
        ]
    )
    for seed in config["encoder_seeds"]:
        for arm in ARMS:
            value = payload["audit"]["training"][f"{seed}/{arm}"]
            lines.append(
                f"| {seed} | {arm} | {value['selected_update']} | {value['actual_student_updates']} | {value['positive_rollout_rewards']} / {value['negative_rollout_rewards']} | {value['negative_reward_branches_committed']} |"
            )
    lines += ["", "## RLの学習終了時の行動確率", ""]
    table(["seed", "上位5行動（進捗1・最終reward EERで算出）"])
    from common import torch
    from selection import Teacher

    for seed in config["encoder_seeds"]:
        saved = torch.load(run / f"acoustic-{seed}/rl/last.pt", weights_only=False)
        teacher = Teacher(len(saved["actions"]), config["teacher_exploration"])
        teacher.load_state_dict(saved["teacher"])
        probabilities = (
            teacher.probabilities(
                [1, 1, sum(saved["current_reward"].values()) / len(config["corpora"])]
            )
            .detach()
            .numpy()
        )
        phones = read(run / "design-freeze.json")["phones"]
        ranked = np.argsort(-probabilities)[:5]
        cells = []
        for i in ranked:
            phone = saved["actions"][i]
            cells.append(
                f"{'uniform' if phone == -1 else phones[phone]}: {100 * probabilities[i]:.2f}%"
            )
        lines.append(f"| {seed} | {', '.join(cells)} |")
    lines += [
        "",
        "報酬用話者はteacherの学習データ。selection/testとは話者を分離し、queryとclaimの両側で制限した。",
        "音素は学習データの抽選にのみ利用し、入力音素を削除しない。5母音は候補すべてに存在するため抽選行動から除外。",
        "EERは採点可能分、FAR/FRRは未採点を拒否として含む。全条件で採点対象は固定。test FARは同値に揃えていない。",
        "RLは単一ブロックの改善を学ぶcontextual banditで、長期最適化や大域最適を保証しない。対照より追加の分岐計算を使う。",
        "3音響seed・1初期化/方策seed・単一設定。既存64kモデルへの優位性を調べる実験ではない。全testは既観測。",
        "CIは固定モデルを条件とした話者bootstrapで学習・選択の変動を含まない。既存encoderの選定に旧validationが使われた可能性も残る。標準モデルは置き換えない。",
        "",
        f"成果物: `{config['run_directory']}`。[設計と再現方法](README.md)・[全seed、別文、監査](evaluation-results.json)。",
        "",
    ]
    text = "\n".join(lines)
    (run / "results.md").write_text(text)
    (BASE / "evaluation-results.md").write_text(text)
