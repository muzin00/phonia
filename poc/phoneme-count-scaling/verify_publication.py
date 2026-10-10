"""Check published numeric cells, links, SVG and immutable audit sources."""

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from bridge import BASE, ROOT, checked, pin, read_json, write_json
from report import interval, percent


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.table = None
        self.cell = None
        self.row = []
        self.tables = {}
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.table = attrs["id"]
            self.tables[self.table] = []
        elif tag == "tr":
            self.row = []
        elif tag == "td":
            self.cell = ""
        elif tag in ("a", "img"):
            self.links.append(attrs.get("href", attrs.get("src")))

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag == "td":
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row:
            self.tables[self.table].append(self.row)
        elif tag == "table":
            self.table = None


def main():
    config = read_json(BASE / "config/protocol.json")
    run = ROOT / config["run_directory"]
    for name in (
        "design-freeze.json",
        "evaluation-freeze.json",
        "audit-freeze.json",
        "publication-report.json",
    ):
        for path, checksum in read_json(run / name)["files"].items():
            checked(ROOT / path, checksum)
    document = read_json(BASE / "evaluation-results.json")
    assert document["results"] == read_json(run / "test-results.json")
    parser = Tables()
    parser.feed((BASE / "evaluation-results.html").read_text())
    assert (
        len(parser.tables["summary"]) == 4 and len(parser.tables["combinations"]) == 40
    )
    normal = document["results"]["count_summary"]["verification"]
    cross = document["results"]["count_summary"]["cross_text_verification"]
    for row, count in zip(parser.tables["summary"], config["counts"], strict=True):
        key = str(count)
        value = normal["pooled_eer"][key]
        expected = [
            key,
            str(value["combinations"]),
            percent(value["mean"]),
            interval(value["ci95"]),
            f"{percent(value['minimum'])}〜{percent(value['maximum'])}",
            percent(normal["far_1pct/all_input_far"][key]["mean"]),
            percent(normal["far_1pct/all_input_frr"][key]["mean"]),
            percent(cross["pooled_eer"][key]["mean"])
            if cross["pooled_eer"][key] is not None
            else "—",
        ]
        assert row == expected, (row, expected)
    with (BASE / "evaluation-cells.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    for row, displayed in zip(rows, parser.tables["combinations"], strict=True):
        cell = document["results"]["cells"][f"{row['condition']}/{row['role']}"]
        phones = document["results"]["schedule"]["conditions"][row["condition"]]
        assert row["phones"] == " ".join(phones) and int(row["count"]) == len(phones)
        assert (
            int(row["queries"]) == cell["queries"]
            and int(row["query_speakers"]) == cell["query_speaker_count"]
        )
        values = [
            cell["pooled_eer"],
            cell["operating_points"]["far_1pct"]["all_input_far"],
            cell["operating_points"]["far_1pct"]["all_input_frr"],
        ]
        for key, v in zip(("EER_pct", "FAR_pct", "FRR_pct"), values, strict=True):
            if v is not None:
                assert math.isclose(float(row[key]), 100 * v, abs_tol=1e-12)
        assert displayed[-3:] == [percent(v) for v in values]
    assert ET.parse(BASE / "eer-curve.svg").getroot().tag.endswith("svg")
    for link in parser.links:
        assert link and (BASE / link).is_file(), link
    for name in ("README.md", "interpretation.md", "evaluation-results.md"):
        for target in re.findall(r"\]\(([^)]+)\)", (BASE / name).read_text()):
            if not target.startswith(("http:", "https:")):
                assert (BASE / target).is_file(), target
    result = {
        "status": "passed",
        "html_tables": 2,
        "html_summary_rows": 4,
        "html_combination_rows": 40,
        "csv_rows": 40,
        "all_local_links_valid": True,
        "svg_parse_valid": True,
        "browser_rendering_checked": False,
    }
    write_json(run / "publication-verification.json", result)
    files = {}
    for path in [
        ROOT / "README.md",
        *[
            BASE / name
            for name in (
                "README.md",
                "interpretation.md",
                "evaluation-results.md",
                "evaluation-results.html",
                "evaluation-results.json",
                "evaluation-cells.csv",
                "eer-curve.svg",
            )
        ],
        *[
            run / name
            for name in (
                "design-freeze.json",
                "evaluation-freeze.json",
                "audit-freeze.json",
                "publication-report.json",
                "publication-verification.json",
            )
        ],
    ]:
        pin(files, path)
    write_json(run / "completion-freeze.json", {"status": "completed", "files": files})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
