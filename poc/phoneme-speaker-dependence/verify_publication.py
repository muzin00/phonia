"""Check original labels, Cholesky REML, and exact CSV/HTML display values."""

import csv
import json
import re
from collections import Counter
from html.parser import HTMLParser

import dependence_math as dm
import numpy as np
from bridge import BASE, ROOT, pin, read_json, write_json


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.row.append("".join(self.cell))
            self.cell = None
        if tag == "tr":
            self.tables[-1].append(self.row)
            self.row = None


def cholesky(data, vectors, components):
    count = 0
    for p, c in components.items():
        rows = [r for r in data["variance"] if r["vowel"] == p]
        design = dm.design(rows)
        u, s, _ = np.linalg.svd(design, full_matrices=False)
        x = u[:, s * s > max(s[0] * s[0] * 1e-10, 1e-12)]
        z = np.array(
            [
                [r["speaker_id"] == speaker for speaker in data["speakers"]]
                for r in rows
            ],
            float,
        )
        y = np.stack([vectors[r["segment_id"]] for r in rows])

        def objective(lam, rows=rows, x=x, y=y, z=z):
            v = np.eye(len(rows)) + lam * z @ z.T
            lower = np.linalg.cholesky(v)
            inverse = np.linalg.solve(
                lower.T, np.linalg.solve(lower, np.eye(len(rows)))
            )
            info = x.T @ inverse @ x
            lower_info = np.linalg.cholesky(info)
            residual = y - x @ np.linalg.solve(info, x.T @ inverse @ y)
            w = np.sum(residual * (inverse @ residual)) / (
                y.shape[1] * (len(rows) - x.shape[1])
            )
            likelihood = y.shape[1] * (
                (len(rows) - x.shape[1]) * np.log(w)
                + 2 * np.log(np.diag(lower)).sum()
                + 2 * np.log(np.diag(lower_info)).sum()
            )
            assert np.isfinite(likelihood)
            return float(likelihood), float(w)

        base, w = objective(c["lambda"])
        assert abs(w * y.shape[1] - c["W_trace"]) < 1e-8
        for lam in (
            [c["lambda"] * 0.99, c["lambda"] * 1.01] if c["lambda"] else [0.0001]
        ):
            assert objective(lam)[0] >= base - 1e-7
        count += 1
    return count


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    output = read_json(BASE / "evaluation-results.json")
    raw_cache, source_checks, likelihood_checks = {}, 0, 0
    for split in ("validation", "test"):
        data = read_json(run / f"{split}-inputs.json")
        for category, expected in (
            ("variance", {"verification": 12}),
            ("enrollment", {"enrollment": 5}),
            ("queries", {"verification": 10, "cross_text_verification": 3}),
        ):
            counts = Counter(
                (r["speaker_id"], r["vowel"], r["role"]) for r in data[category]
            )
            for speaker in data["speakers"]:
                for phone in config["phones"]:
                    for role, number in expected.items():
                        assert counts[speaker, phone, role] == number
            for r in data[category]:
                if r["source_file"] not in raw_cache:
                    filename = r["segment_id"].split("--")[0] + ".json"
                    raw_cache[r["source_file"]] = read_json(
                        ROOT
                        / "poc/phoneme-speaker-dataset/data/generated/alignments/raw"
                        / r["speaker_id"]
                        / filename
                    )
                raw = raw_cache[r["source_file"]]
                interval = raw["intervals"][r["raw_index"]]
                assert interval["phoneme"] == r["vowel"]
                assert (
                    raw["source_file"] == r["source_file"]
                    and raw["source_sha256"] == r["source_sha256"]
                )
                assert round(interval["start_sec"] * 24000) == r["raw_start_frame"]
                assert round(interval["end_sec"] * 24000) == r["raw_end_frame"]
                lo, hi = r["raw_start_frame"], r["raw_end_frame"]
                size = min(hi - lo, 6000) if category == "variance" else 1200
                first = (
                    hi - size
                    if category != "variance" and r["vowel"] in ("t", "d", "k", "g")
                    else lo + (hi - lo - size) // 2
                )
                assert (r["start_frame"], r["end_frame"]) == (first, first + size)
                source_checks += 1
        components = (
            output["validation"]["components"]
            if split == "validation"
            else output["test"]["test_components"]["components"]
        )
        likelihood_checks += cholesky(
            data, np.load(run / split / "embedding-vectors.npz"), components
        )
    with (BASE / "evaluation-cells.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 14
    parser = Tables()
    parser.feed((BASE / "evaluation-results.html").read_text())
    assert len(parser.tables) == 2 and [len(t) - 1 for t in parser.tables] == [14, 10]
    for row, displayed in zip(csv_rows, parser.tables[0][1:], strict=True):
        p = row["phone"]
        c = output["validation"]["components"][p]
        normal = output["test"]["cells"][f"{p}/verification"]["pooled_eer"]
        cross = output["test"]["cells"][f"{p}/cross_text_verification"]["pooled_eer"]
        assert (
            float(row["validation_R"]) == c["R"]
            and float(row["normal_eer"]) == normal
            and float(row["cross_text_eer"]) == cross
        )
        assert (
            displayed[0] == p
            and displayed[2] == f"{c['R']:.3f}"
            and displayed[6:] == [f"{normal * 100:.3f}%", f"{cross * 100:.3f}%"]
        )
    for displayed, (key, cell) in zip(
        parser.tables[1][1:], output["test"]["group_summary"].items(), strict=True
    ):
        assert displayed[1] == key.split("/", 1)[1]
        assert displayed[2:5] == [
            f"{cell['high'] * 100:.3f}%",
            f"{cell['low'] * 100:.3f}%",
            f"{cell['high_minus_low'] * 100:+.3f} pp",
        ]
    for path in BASE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not target.startswith("https://"):
                assert (path.parent / target.split("#")[0]).is_file()
    result = {
        "status": "passed",
        "raw_label_boundaries_and_crops": source_checks,
        "cholesky_REML_checks": likelihood_checks,
        "csv_rows": 14,
        "html_tables": 2,
        "html_result_rows": 24,
        "local_markdown_links_valid": True,
        "browser_rendering_checked": False,
    }
    verification_path = run / "publication-verification.json"
    if verification_path.exists():
        assert read_json(verification_path) == result
    else:
        write_json(verification_path, result)
    ledger = read_json(run / "publication-report.json")
    archive = run / "initial-publication-ledger.json"
    if not archive.exists():
        write_json(archive, ledger)
    documentation = str((BASE / "README.md").relative_to(ROOT))
    previous = ledger["files"].pop(documentation, None)
    ledger["files"].pop(str((BASE / "verify_publication.py").relative_to(ROOT)), None)
    for p in [
        ROOT / "README.md",
        BASE / "verify_publication.py",
        BASE / "README.md",
        BASE / "interpretation.md",
        run / "publication-verification.json",
    ]:
        pin(ledger["files"], p)
    if previous:
        ledger["documentation_revision"] = {
            "initial_ledger": str(archive.relative_to(ROOT)),
            "note": "README completion notes and interpretation added after measurements; numerical publications unchanged",
        }
    (run / "publication-report.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
