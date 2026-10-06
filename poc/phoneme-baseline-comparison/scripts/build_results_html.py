"""Build the offline table report from published, fixed Phase 7 result CSVs."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import BASE, sha256_file

TEXT_COLUMNS = {
    "split",
    "support",
    "method",
    "enrollment",
    "cap",
    "role",
    "point",
    "conditional_status",
    "zero_scored_speakers",
    "comparison",
    "status",
    "reference",
    "population_note",
    "dimension",
    "bin",
}


def pack_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        records = []
        for row in reader:
            values = []
            for column in columns:
                value = row[column]
                if value == "":
                    value = None
                elif column not in TEXT_COLUMNS and value not in ("+inf", "-inf"):
                    number = float(value)
                    value = int(number) if number.is_integer() else number
                values.append(value)
            records.append(values)
    return {"columns": columns, "rows": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "evaluation-results.html")
    args = parser.parse_args()
    files = {
        "conditions": "all-conditions.csv",
        "differences": "paired-differences.csv",
        "strata": "duration-strata.csv",
    }
    payload = {key: pack_csv(BASE / name) for key, name in files.items()}
    if tuple(len(payload[key]["rows"]) for key in files) != (1080, 792, 9720):
        raise ValueError("published condition table is incomplete")
    payload["sources"] = {name: sha256_file(BASE / name) for name in files.values()}
    data = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    template = BASE / "report/results-template.html"
    text = template.read_text(encoding="utf-8")
    if text.count("__REPORT_DATA__") != 1:
        raise ValueError("expected one embedded data placeholder")
    args.output.write_text(text.replace("__REPORT_DATA__", data), encoding="utf-8")
    print(
        f"Built {args.output}: 360 conditions, 792 paired differences, 9720 duration rows"
    )


if __name__ == "__main__":
    main()
