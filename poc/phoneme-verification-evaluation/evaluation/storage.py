"""Checksum-bearing, immutable JSON and atomic JSONL publication."""

import gzip
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from phase3_data.manifest import json_sha256
from verification.scoring import write_document


def save_document(path: Path, value: dict) -> None:
    if "sha256" in value:
        raise ValueError("document already has an envelope checksum")
    write_document(path, {**value, "sha256": json_sha256(value)})


def load_document(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.pop("sha256", None) != json_sha256(value):
        raise ValueError(f"document checksum mismatch: {path}")
    return value


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


@contextmanager
def row_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as raw:
            if path.suffix == ".gz":
                stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
            else:
                stream = raw

            def write(row):
                payload = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write((payload + "\n").encode("utf-8"))

            yield write
            if stream is not raw:
                stream.close()
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def save_rows(path: Path, rows) -> None:
    with row_writer(path) as write:
        for row in rows:
            write(row)
