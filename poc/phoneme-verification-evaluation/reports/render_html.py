"""Render saved, checksum-verified metrics into a self-contained HTML report."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_document(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    checksum = document.pop("sha256", None)
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if hashlib.sha256(canonical.encode()).hexdigest() != checksum:
        raise ValueError(f"document checksum mismatch: {path}")
    return document


def render(run, output):
    run = run.resolve()
    output = output.resolve()
    if output.suffix != ".html" or output.is_relative_to(run):
        raise ValueError("write the HTML outside the immutable evaluation run")
    execution = load_document(run / "execution.json")
    if execution["status"] != "completed":
        raise ValueError("evaluation is not completed")

    def verified_path(name):
        path = run / name
        if execution["files"][name] != sha256_file(path):
            raise ValueError(f"artifact checksum mismatch: {name}")
        return path

    metrics = {}
    for split in ("validation", "test"):
        document = load_document(verified_path(f"metrics/{split}.json"))
        metrics[split] = {
            name: {
                key: value for key, value in condition.items() if key != "by_speaker"
            }
            for name, condition in document["conditions"].items()
        }
    thresholds = load_document(verified_path("thresholds/validation.json"))
    images = {}
    for role in ("verification", "cross_text_verification"):
        images[role] = {}
        for chart in ("scores", "det", "roc"):
            raw = verified_path(f"curves/{role}-{chart}.png").read_bytes()
            images[role][chart] = (
                "data:image/png;base64," + base64.b64encode(raw).decode()
            )
    data = {
        "metrics": metrics,
        "thresholds": thresholds["conditions"],
        "images": images,
        "run_id": run.name,
        "completed_at": execution["completed_at"],
        "execution_sha256": sha256_file(run / "execution.json"),
        "plan_sha256": sha256_file(verified_path("evaluation-plan.json")),
    }
    primary = metrics["test"]["verification/n10/fused"]
    point = primary["operating_points"]["far_1pct"]
    ci = primary["bootstrap_95pct"]
    interval = ci["operating_points"]["far_1pct"]
    cross = metrics["test"]["cross_text_verification/n10/fused"]
    cross_point = cross["operating_points"]["far_1pct"]

    def percent(value):
        return f"{100 * value:.3f}%"

    def limits(values):
        return "〜".join(percent(value) for value in values)

    values = {
        "FAR": percent(point["far"]),
        "FAR_COUNT": f"{point['false_accepts']:,} / {point['impostor']:,}",
        "FAR_CI": limits(interval["far"]),
        "FRR": percent(point["frr"]),
        "FRR_COUNT": f"{point['false_rejects']:,} / {point['genuine']:,}",
        "FRR_CI": limits(interval["frr"]),
        "ALL_FRR": percent(point["all_input_frr"]),
        "ALL_FRR_COUNT": f"{point['false_rejects'] + point['no_score_genuine']:,} / {point['all_genuine']:,}",
        "ALL_FRR_CI": limits(interval["all_input_frr"]),
        "EER": percent(primary["pooled_eer"]),
        "EER_CI": limits(ci["pooled_eer"]),
        "THRESHOLD": str(point["threshold"]),
        "CROSS_SCORED": str(cross["scored_queries"]),
        "CROSS_ALL": str(cross["all_queries"]),
        "CROSS_FRR": percent(cross_point["frr"]),
        "CROSS_ALL_FRR": percent(cross_point["all_input_frr"]),
        "ACCEPTED": str(point["genuine"] - point["false_rejects"]),
        "REJECTED": str(point["false_rejects"]),
        "NO_SCORE": str(point["no_score_genuine"]),
        "ACCEPTED_WIDTH": str(
            100 * (point["genuine"] - point["false_rejects"]) / point["all_genuine"]
        ),
        "REJECTED_WIDTH": str(100 * point["false_rejects"] / point["all_genuine"]),
        "NO_SCORE_WIDTH": str(100 * point["no_score_genuine"] / point["all_genuine"]),
        "PRIMARY_IMAGE": images["verification"]["scores"],
        "DATA": json.dumps(
            data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).replace("<", "\\u003c"),
    }
    html = (Path(__file__).parent / "report-template.html").read_text(encoding="utf-8")
    for name, value in values.items():
        html = html.replace("{{" + name + "}}", value)
    if "{{" in html:
        raise ValueError("unresolved HTML template value")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=BASE / "evaluation-results.html")
    args = parser.parse_args()
    render(args.run, args.output)


if __name__ == "__main__":
    main()
