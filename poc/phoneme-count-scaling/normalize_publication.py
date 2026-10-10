"""Normalize generated-file whitespace after checking publication semantics."""

import csv
import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def xml_semantics(element):
    return (
        element.tag,
        tuple(
            sorted(
                (key, " ".join(value.split())) for key, value in element.attrib.items()
            )
        ),
        " ".join((element.text or "").split()),
        tuple(xml_semantics(child) for child in element),
    )


def main():
    config = json.loads((BASE / "config/protocol.json").read_text())
    run = ROOT / config["run_directory"]
    archive = run / "publication-before-git-formatting"
    if archive.exists():
        raise ValueError("publication formatting was already applied")
    ledger_path = run / "publication-report.json"
    ledger = json.loads(ledger_path.read_text())
    for name, expected in ledger["files"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    originals = {
        name: (BASE / name).read_bytes()
        for name in ("eer-curve.svg", "evaluation-cells.csv")
    }
    normalized = {
        "eer-curve.svg": (
            "\n".join(
                line.rstrip()
                for line in originals["eer-curve.svg"].decode().splitlines()
            )
            + "\n"
        ).encode(),
        "evaluation-cells.csv": originals["evaluation-cells.csv"].replace(
            b"\r\n", b"\n"
        ),
    }
    assert xml_semantics(ET.fromstring(originals["eer-curve.svg"])) == xml_semantics(
        ET.fromstring(normalized["eer-curve.svg"])
    )
    assert list(
        csv.DictReader(io.StringIO(originals["evaluation-cells.csv"].decode()))
    ) == list(csv.DictReader(io.StringIO(normalized["evaluation-cells.csv"].decode())))
    archive.mkdir()
    shutil.copy2(ledger_path, archive / ledger_path.name)
    for name, content in originals.items():
        (archive / name).write_bytes(content)
    for name in ("publication-verification.json", "completion-freeze.json"):
        path = run / name
        if path.exists():
            path.rename(archive / name)
    revisions = {}
    for name, content in normalized.items():
        path = BASE / name
        path.write_bytes(content)
        relative = str(path.relative_to(ROOT))
        ledger["files"][relative] = sha(content)
        revisions[name] = {
            "original_sha256": sha(originals[name]),
            "normalized_sha256": sha(content),
        }
    helper = str(Path(__file__).resolve().relative_to(ROOT))
    ledger["files"][helper] = sha(Path(__file__).read_bytes())
    ledger["formatting_revision"] = {
        "originals": str(archive.relative_to(ROOT)),
        "SVG_semantics_equal": True,
        "CSV_cells_equal": True,
        "files": revisions,
    }
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(ledger["formatting_revision"]), flush=True)


if __name__ == "__main__":
    main()
