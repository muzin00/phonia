"""Download a fixed public ECAPA revision without sending any JVS audio."""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import CONFIG, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(CONFIG.read_text())["model"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "model_id": model["id"],
        "revision": model["revision"],
        "files": {},
    }
    for name in model["files"]:
        url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{name}"
        temporary = args.output_dir / f".{name}.partial"
        destination = args.output_dir / name
        with urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if sha256_file(temporary) != model["file_sha256"][name]:
            raise ValueError(f"download checksum mismatch: {name}")
        temporary.rename(destination)
        manifest["files"][name] = {
            "url": url,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        print(f"Downloaded {name}: {destination.stat().st_size} bytes", flush=True)
    write_json(args.output_dir / "model-manifest.json", manifest)


if __name__ == "__main__":
    main()
