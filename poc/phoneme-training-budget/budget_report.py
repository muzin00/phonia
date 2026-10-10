"""Publish the training curves and identically scored utterance comparison tables."""

from __future__ import annotations

import csv
import html
from html.parser import HTMLParser

import budget_evaluation as study
import budget_training as training
import numpy as np

BASE, ROOT = study.BASE, study.ROOT


def collect_training():
    config = training.read_json(training.CONFIG)
    run = ROOT / config["run_directory"]
    training.frozen(config, run)
    curves, rows = {}, []
    descriptors = []
    for corpus in config["corpora"]:
        output = run / corpus
        summary = training.read_json(output / "training/summary.json")
        if summary["status"] != "completed" or summary["completed_updates"] != 60000:
            raise ValueError("both complete 60k trajectories required")
        for name, digest in summary["outputs_sha256"].items():
            training.checked(output / name, digest)
        descriptor = training.read_json(output / "run.json")
        descriptors.append(descriptor)
        history = list(training.cv.read_rows(output / "training/history.jsonl"))
        if [r["update"] for r in history] != list(range(1, 60001)) or not all(
            np.isfinite(r[k])
            for r in history
            for k in ("loss", "accuracy", "gradient_norm", "learning_rate")
        ):
            raise ValueError("invalid finite consecutive training history")
        points = []
        for path in sorted(
            (output / "validation").glob("update-*/metrics/validation.json")
        ):
            update = int(path.parents[1].name.split("-")[1])
            metrics = training.read_json(path)
            if metrics["split"] != "validation" or metrics["partial"]:
                raise ValueError("full validation curve required")
            points.append(
                {
                    "update": update,
                    "validation_macro_eer_pct": 100 * metrics["macro_eer"],
                    "learning_rate": history[update - 1]["learning_rate"],
                }
            )
        if [p["update"] for p in points] != list(range(1000, 60001, 1000)):
            raise ValueError("incomplete validation curve")
        blocks = []
        for stop in range(5000, 60001, 5000):
            block = history[stop - 5000 : stop]
            blocks.append(
                {
                    "last_update": stop,
                    "mean_training_loss": float(np.mean([r["loss"] for r in block])),
                    "mean_training_accuracy": float(
                        np.mean([r["accuracy"] for r in block])
                    ),
                }
            )
        curves[corpus] = {
            "validation": points,
            "training_blocks": blocks,
            "diagnostic_best": min(points, key=lambda p: p["validation_macro_eer_pct"]),
            "descriptor": descriptor,
            "segment_count": training.read_json(run / "training-freeze.json")[
                "sources"
            ][corpus]["segment_count"],
        }
        for update in config["snapshot_updates"]:
            snapshot = training.read_json(
                output / f"snapshots/update-{update:06d}/snapshot.json"
            )
            point = next(p for p in points if p["update"] == update)
            if (
                point["validation_macro_eer_pct"]
                != 100 * snapshot["validation_macro_eer"]
            ):
                raise ValueError("snapshot and curve EER disagree")
            checkpoint = training.cv.torch.load(
                output / f"snapshots/update-{update:06d}/checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            export = training.cv.torch.load(
                output / f"snapshots/update-{update:06d}/bundle/encoder.pt",
                map_location="cpu",
                weights_only=True,
            )
            if (
                checkpoint["update"] != update
                or checkpoint["scheduler"]["completed_steps"] != update
                or checkpoint["config_sha256"] != descriptor["configuration_sha256"]
                or checkpoint["sampler"]["speaker_counts"] != snapshot["speaker_counts"]
                or any(
                    not training.cv.torch.equal(tensor, export["model"][name])
                    for name, tensor in checkpoint["model"].items()
                )
                or any(
                    not training.cv.torch.isfinite(t).all()
                    for key in ("model", "head")
                    for t in checkpoint[key].values()
                )
            ):
                raise ValueError("full checkpoint, export or sampler snapshot mismatch")
            rows.append(
                {
                    "corpus": corpus,
                    "update": update,
                    "schedule_maximum_updates": 60000,
                    "training_segments": curves[corpus]["segment_count"],
                    "encoder_parameters": 65920,
                    "validation_macro_eer_pct": point["validation_macro_eer_pct"],
                    "learning_rate": point["learning_rate"],
                }
            )
    if (
        descriptors[0]["settings"] != descriptors[1]["settings"]
        or descriptors[0]["initial_encoder_sha256"]
        != descriptors[1]["initial_encoder_sha256"]
        or descriptors[0]["initial_head_sha256"]
        != descriptors[1]["initial_head_sha256"]
    ):
        raise ValueError("corpus settings or initialization differ")
    for update in config["snapshot_updates"]:
        counts = [
            training.read_json(
                run / c / f"snapshots/update-{update:06d}/snapshot.json"
            )["speaker_counts"]
            for c in config["corpora"]
        ]
        jvs = {s for s in counts[0] if s.startswith("jvs")}
        if jvs != {s for s in counts[1] if s.startswith("jvs")} or any(
            abs(counts[0][s] - counts[1][s]) > 1 for s in jvs
        ):
            raise ValueError("JVS per-speaker exposure differs across corpora")
    return {
        "status": "completed",
        "test_used": False,
        "curves": curves,
        "snapshot_rows": rows,
        "history_updates_checked": 120000,
        "validation_points_checked": 120,
        "full_snapshots_checked": 6,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_training(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "phase8-budget-60k-v1"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for corpus, color, label in (
        ("cv", "#2563eb", "JVS + Common Voice"),
        ("src", "#e87916", "JVS + SRC4VC"),
    ):
        points = data["curves"][corpus]["validation"]
        x = [p["update"] for p in points]
        y = [p["validation_macro_eer_pct"] for p in points]
        for axis in axes:
            axis.plot(x, y, color=color, label=label, linewidth=1.8)
            for update in (30000, 45000, 60000):
                point = next(p for p in points if p["update"] == update)
                axis.scatter(
                    update,
                    point["validation_macro_eer_pct"],
                    color=color,
                    s=30,
                    zorder=3,
                )
            axis.set_xlabel("Training updates")
            axis.set_ylabel("Mean vowel validation EER (%)")
            axis.grid(alpha=0.25)
    axes[0].set_title("Full training trajectories")
    axes[0].set_xlim(0, 61000)
    axes[1].set_title("Late training: 30k to 60k")
    axes[1].set_xlim(29000, 61000)
    late = [
        p["validation_macro_eer_pct"]
        for corpus in ("cv", "src")
        for p in data["curves"][corpus]["validation"]
        if p["update"] >= 30000
    ]
    margin = max(0.05, (max(late) - min(late)) * 0.1)
    axes[1].set_ylim(min(late) - margin, max(late) + margin)
    axes[0].legend(fontsize=9)
    figure.suptitle("Same 60k schedule, one seed; diagnostic validation only")
    figure.savefig(destination / "learning-curves.svg", metadata={"Date": None})
    figure.savefig(destination / "learning-curves.png", dpi=160)
    plt.close(figure)


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def training_table(data):
    return [
        [
            {"cv": "Common Voice", "src": "SRC4VC"}[r["corpus"]],
            f"{r['update']:,}",
            f"{r['validation_macro_eer_pct']:.3f}%",
            f"{r['learning_rate']:.8f}",
        ]
        for r in data["snapshot_rows"]
    ]


def condition_label(condition):
    if condition == study.BASELINE:
        return "JVSのみ・15,000（参考）"
    corpus, update = study.specification(condition)
    return f"JVS＋{ {'cv': 'Common Voice', 'src': 'SRC4VC'}[corpus] }・{update:,}"


def comparison_label(key):
    comparison, role, name = key.split("/", 2)
    a, b = comparison.split("_minus_")
    corpus, later = study.specification(a)
    _, earlier = study.specification(b)
    metric = {
        "far_1pct/all_input_far": "全入力FAR",
        "far_1pct/all_input_frr": "全入力FRR",
        "pooled_eer": "発話EER（scoreあり）",
    }[name]
    role_label = "通常文" if role == "verification" else "別テキスト"
    return f"{ {'cv': 'Common Voice', 'src': 'SRC4VC'}[corpus] } {later:,}−{earlier:,} / {role_label} / {metric}"


def document(body):
    return (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>60,000回の学習量比較</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033;background:#f6f8fc}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{border:1px solid #d5dce6;padding:10px;text-align:left}th{background:#e9eef7}td{font-variant-numeric:tabular-nums}.scroll{overflow:auto}img{max-width:100%;height:auto}a{color:#175dc6}h2{margin-top:32px}</style></head><body><main>'
        + body
        + "</main></body></html>"
    )


def publish_training():
    data = collect_training()
    training.write_json(BASE / "training-results.json", data)
    write_csv(BASE / "training-snapshots.csv", data["snapshot_rows"])
    plot_training(data, BASE)
    rows = training_table(data)
    body = '<h1>60,000回の学習量比較：学習・validation</h1><p>Common Voice版とSRC4VC版を初期化から各60,000回学習。同一の60,000回用schedule内で30,000・45,000・60,000回を比較する。母音単一区間のvalidation EERであり、発話単位FAR/FRRとは異なる。</p><img src="learning-curves.svg" alt="同一60,000回scheduleでの2コーパスのvalidation学習曲線"><div class="scroll">'
    body += table(["追加コーパス", "update", "validation EER", "学習率"], rows)
    body += '</div><p>1 seed。途中の最良値は診断用で、checkpointの採用を変更しない。過去の30,000回用scheduleとの比較は参考に限る。</p><p><a href="evaluation-results.html">発話単位評価</a> · <a href="training-results.json">学習JSON</a> · <a href="training-snapshots.csv">CSV</a></p>'
    (BASE / "training-results.html").write_text(document(body), encoding="utf-8")
    markdown = "# 60,000回の学習量比較：学習・validation\n\n同じ60,000回用schedule内の途中経過。1 seed・母音単一区間validation EER。\n\n| コーパス | update | validation EER | 学習率 |\n|---|---:|---:|---:|\n"
    markdown += (
        "\n".join("| " + " | ".join(row) + " |" for row in rows)
        + "\n\n[学習曲線](learning-curves.svg)・[発話単位評価](evaluation-results.html)。途中の最良値で採用時点を変えない。\n"
    )
    (BASE / "training-results.md").write_text(markdown, encoding="utf-8")
    print(
        "training histories, snapshots, exposure and curves audited; training report published",
        flush=True,
    )


def condition_rows(run):
    thresholds = study.read_json(run / "validation-thresholds.json")["conditions"]
    result = []
    for split in ("validation", "test"):
        cells = study.read_json(run / f"{split}-metrics.json")["conditions"]
        for condition in study.CONDITIONS:
            for role in study.ROLES:
                cell = cells[f"{condition}/{role}"]
                for point, _ in study.POINTS:
                    rates = cell["operating_points"][point]
                    result.append(
                        {
                            "split": split,
                            "condition": condition,
                            "role": role,
                            "operating_point": point,
                            "threshold": thresholds[condition]["operating_points"][
                                point
                            ]["threshold"],
                            "all_input_far_pct": 100 * rates["all_input_far"],
                            "all_input_frr_pct": 100 * rates["all_input_frr"],
                            "scored_far_pct": 100 * rates["far"],
                            "scored_frr_pct": 100 * rates["frr"],
                            "pooled_eer_pct": 100 * cell["pooled_eer"],
                            "query_coverage_pct": 100 * cell["query_coverage"],
                            "false_accepts": rates["false_accepts"],
                            "all_impostor_trials": rates["all_impostor"],
                            "false_rejects_scored": rates["false_rejects"],
                            "no_score_genuine": rates["no_score_genuine"],
                            "all_genuine_trials": rates["all_genuine"],
                        }
                    )
    return result


def render(run):
    report = run / "report"
    report.mkdir(exist_ok=True)
    for name in ("learning-curves.svg", "learning-curves.png"):
        if (BASE / name).exists():
            (report / name).write_bytes((BASE / name).read_bytes())
    cells = study.read_json(run / "test-metrics.json")
    validation = study.read_json(run / "validation-metrics.json")
    data = {
        "status": "completed",
        "training": study.read_json(BASE / "training-results.json"),
        "validation": validation["conditions"],
        "test": cells["conditions"],
        "paired_differences": cells["paired_differences"],
        "condition_rows": condition_rows(run),
        "test_threshold_recalibrated": False,
        "test_previously_observed": True,
        "independent_holdout": False,
    }
    study.write_json(report / "evaluation-results.json", data)
    write_csv(report / "all-conditions.csv", data["condition_rows"])
    paired_rows = [
        {
            "comparison_metric": k,
            "difference_percentage_points": v["difference_percentage_points"],
            "ci95_low_pp": v["ci95_percentage_points"]["lower"],
            "ci95_high_pp": v["ci95_percentage_points"]["upper"],
            "primary": v["primary"],
        }
        for k, v in data["paired_differences"].items()
    ]
    write_csv(report / "paired-differences.csv", paired_rows)
    headers = [
        "条件",
        "全入力FAR",
        "全入力FRR",
        "発話EER（scoreあり）",
        "score coverage",
    ]
    body = '<h1>60,000回の学習量比較：発話単位評価</h1><p>CV / SRCは同じ60,000回用schedule内の30,000・45,000・60,000回checkpoint。JVS70は従来モデルの参考値。登録各母音10区間・照合は発話中の母音全区間・5母音必須。閾値はモデルごとのvalidation通常文で固定し、testで再調整していない。</p><img src="learning-curves.svg" alt="2コーパスのvalidation学習曲線">'
    md = "# 60,000回の学習量比較：発話単位評価\n\n同じ60,000回schedule内の固定checkpoint比較。1 seed・観測済みJVS test。母音不足を拒否として数える全入力FAR/FRR。\n\n"
    for split in ("validation", "test"):
        for role in study.ROLES:
            selected = [
                r
                for r in data["condition_rows"]
                if r["split"] == split
                and r["role"] == role
                and r["operating_point"] == "far_1pct"
            ]
            values = [
                [
                    condition_label(r["condition"]),
                    *[
                        f"{r[k]:.3f}%"
                        for k in (
                            "all_input_far_pct",
                            "all_input_frr_pct",
                            "pooled_eer_pct",
                            "query_coverage_pct",
                        )
                    ],
                ]
                for r in selected
            ]
            title = f"{split} / {'通常文' if role == 'verification' else '別テキスト'}：validation目標FAR 1%"
            body += (
                f'<h2>{html.escape(title)}</h2><div class="scroll">'
                + table(headers, values)
                + "</div>"
            )
            md += (
                f"## {title}\n\n| "
                + " | ".join(headers)
                + " |\n|"
                + "|".join(["---"] * len(headers))
                + "|\n"
            )
            md += "\n".join("| " + " | ".join(row) + " |" for row in values) + "\n\n"
    values = []
    for key, value in sorted(data["paired_differences"].items()):
        if "/far_1pct/all_input_" in key or key.endswith("/pooled_eer"):
            ci = value["ci95_percentage_points"]
            values.append(
                [
                    comparison_label(key),
                    f"{value['difference_percentage_points']:+.3f}",
                    f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]",
                    "主比較" if value["primary"] else "補助",
                ]
            )
    body += (
        '<h2>30,000回との差と話者bootstrapの95% CI（pp）</h2><div class="scroll">'
        + table(["比較 / 指標", "差", "95% CI", "位置付け"], values)
        + '</div><p>負の差は誤り率の低下。CIは固定モデル・固定閾値の話者変動を表し、seed変動や閾値校正の不確かさは含まない。学習回数と学習率が一緒に推移する1本の学習曲線であり、最適な予算やデータ量のスケーリング則を確定する比較ではない。</p><p><a href="all-conditions.csv">全動作点CSV</a> · <a href="paired-differences.csv">差のCSV</a> · <a href="evaluation-results.json">JSON</a> · <a href="interpretation.md">解釈</a></p>'
    )
    md += (
        "## 30,000回との差（pp）\n\n| 比較 / 指標 | 差 | 95% CI | 位置付け |\n|---|---:|---|---|\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in values)
        + "\n"
    )
    (report / "evaluation-results.html").write_text(document(body), encoding="utf-8")
    (report / "evaluation-results.md").write_text(md, encoding="utf-8")


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
            self.row = None
        elif tag == "table":
            self.tables.append(self.rows)
            self.rows = None


def audit_report(run):
    report = run / "report"
    published = study.read_json(report / "evaluation-results.json")
    if (
        published["condition_rows"] != condition_rows(run)
        or published["test"] != study.read_json(run / "test-metrics.json")["conditions"]
        or published["paired_differences"]
        != study.read_json(run / "test-metrics.json")["paired_differences"]
    ):
        raise ValueError("published JSON differs from audited metrics")
    with (report / "all-conditions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 84:
        raise ValueError("all-condition CSV population differs")
    for row, expected in zip(rows, published["condition_rows"], strict=True):
        for key, value in expected.items():
            actual = float(row[key]) if isinstance(value, float) else row[key]
            wanted = value if isinstance(value, float) else str(value)
            if actual != wanted:
                raise ValueError("CSV metric differs")
    with (report / "paired-differences.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        paired_rows = list(csv.DictReader(stream))
    if len(paired_rows) != 112:
        raise ValueError("paired CSV population differs")
    for row in paired_rows:
        expected = published["paired_differences"][row["comparison_metric"]]
        if (
            float(row["difference_percentage_points"])
            != expected["difference_percentage_points"]
            or float(row["ci95_low_pp"]) != expected["ci95_percentage_points"]["lower"]
            or float(row["ci95_high_pp"]) != expected["ci95_percentage_points"]["upper"]
        ):
            raise ValueError("paired CSV metric differs")
    parser = Tables()
    parser.feed((report / "evaluation-results.html").read_text(encoding="utf-8"))
    if len(parser.tables) != 5 or [len(t) - 1 for t in parser.tables] != [
        7,
        7,
        7,
        7,
        24,
    ]:
        raise ValueError("HTML table populations differ")
    for index, (split, role) in enumerate(
        (s, r) for s in ("validation", "test") for r in study.ROLES
    ):
        expected = [
            r
            for r in published["condition_rows"]
            if r["split"] == split
            and r["role"] == role
            and r["operating_point"] == "far_1pct"
        ]
        for row, value in zip(parser.tables[index][1:], expected, strict=True):
            wanted = [
                condition_label(value["condition"]),
                *[
                    f"{value[k]:.3f}%"
                    for k in (
                        "all_input_far_pct",
                        "all_input_frr_pct",
                        "pooled_eer_pct",
                        "query_coverage_pct",
                    )
                ],
            ]
            if row != wanted:
                raise ValueError("HTML metric differs")
    expected_pairs = []
    for key, value in sorted(published["paired_differences"].items()):
        if "/far_1pct/all_input_" in key or key.endswith("/pooled_eer"):
            ci = value["ci95_percentage_points"]
            expected_pairs.append(
                [
                    comparison_label(key),
                    f"{value['difference_percentage_points']:+.3f}",
                    f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]",
                    "主比較" if value["primary"] else "補助",
                ]
            )
    if parser.tables[4][1:] != expected_pairs:
        raise ValueError("HTML paired difference differs")
    return {
        "html_tables": 5,
        "condition_rows": len(rows),
        "paired_rows": len(paired_rows),
    }


def publish(run):
    names = (
        "evaluation-results.html",
        "evaluation-results.md",
        "evaluation-results.json",
        "all-conditions.csv",
        "paired-differences.csv",
    )
    for name in names:
        (BASE / name).write_bytes((run / "report" / name).read_bytes())
    names += (
        "training-results.html",
        "training-results.md",
        "training-results.json",
        "training-snapshots.csv",
        "learning-curves.svg",
        "learning-curves.png",
    )
    study.write_json(
        BASE / "manifest.json",
        {
            "status": "completed",
            "study_report_sha256": study.sha256_file(run / "study-report.json"),
            "files": {name: study.sha256_file(BASE / name) for name in names},
        },
    )
