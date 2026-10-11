"""Loopback-only pilot UI with same-origin writes and immutable run identity."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import store

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def handler(database, token, port):
    lock = threading.Lock()
    allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, value, mime="application/json; charset=utf-8"):
            content = (
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
                if mime.startswith("application/json")
                else value
            )
            self.send_response(status)
            for key, val in {
                "Content-Type": mime,
                "Content-Length": str(len(content)),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'",
            }.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(content)

        def host_ok(self):
            return self.headers.get("Host") in allowed_hosts

        def do_GET(self):
            if not self.host_ok():
                self.send(403, {"error": "Local access only"})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/api/info":
                self.send(
                    200,
                    {
                        "token": token,
                        "purpose": database.config["purpose"],
                        "threshold": None,
                        "prompts": database.config["prompts"],
                    },
                )
                return
            if parsed.path == "/api/episode":
                try:
                    self.send(
                        200,
                        database.public(database.load(parse_qs(parsed.query)["id"][0])),
                    )
                except (ValueError, KeyError, FileNotFoundError):
                    self.send(404, {"error": "録音が見つかりません。"})
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript"),
                "/audio.js": ("audio.js", "text/javascript"),
                "/capture.js": ("capture.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if parsed.path not in assets:
                self.send(404, {"error": "Not found"})
                return
            name, mime = assets[parsed.path]
            self.send(200, (BASE / "web" / name).read_bytes(), mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.host_ok()
                or self.headers.get("X-Phonia-Token") != token
                or (
                    origin is not None
                    and origin not in {f"http://{h}" for h in allowed_hosts}
                )
            ):
                self.send(
                    403,
                    {"error": "ページをこのローカルサーバーから開き直してください。"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 3_000_000:
                    raise ValueError("入力サイズが上限を超えています。")
                value = json.loads(self.rfile.read(length))
                with lock:
                    if self.path == "/api/episode":
                        result = database.create(value)
                    elif self.path == "/api/clip":
                        result = database.submit(
                            value["episode_id"],
                            base64.b64decode(value["wav_base64"], validate=True),
                            value,
                        )
                    elif self.path == "/api/abort":
                        result = database.abort(value["episode_id"])
                    else:
                        self.send(404, {"error": "Not found"})
                        return
                self.send(200, result)
            except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
                self.send(400, {"error": str(error)[:500]})
            except Exception as error:  # noqa: BLE001 - report local processing failure without crashing server.
                self.log_error("%s", error)
                self.send(
                    500,
                    {
                        "error": "処理に失敗しました。保存記録とサーバーログを確認してください。"
                    },
                )

    return Handler


def main():
    import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "artifacts/phoneme-live-recording/pilot-20261011-v2",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--purpose", choices=("human_pilot", "replay_smoke"), default="human_pilot"
    )
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / "artifacts/phoneme-live-recording"):
        raise ValueError("Run must be under artifacts/phoneme-live-recording")
    config = {**store.read(BASE / "protocol.json"), "purpose": args.purpose}
    engine = runtime.Runtime(config)
    files = engine.fingerprints()
    files.update(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in BASE.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix in (".py", ".js", ".html", ".css")
        }
    )
    frozen = {
        "config": config,
        "files": files,
        "prompt_counts": engine.prompt_counts,
        "purpose": args.purpose,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "torch", "pyopenjtalk")
        },
    }
    run.mkdir(parents=True, exist_ok=True)
    freeze = run / "protocol-freeze.json"
    if freeze.exists():
        if store.read(freeze) != frozen:
            raise ValueError(
                "Code/protocol changed: use a new run directory; existing recordings preserved"
            )
    else:
        store.write(freeze, frozen)
    database = store.Store(run, config, engine, runtime.decode_wav)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler(database, token, args.port)
    )
    print(
        f"Local recording pilot: http://127.0.0.1:{args.port} | purpose={args.purpose} | authentication threshold unset",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
