"""Bounded, append-only recording episodes; no authentication threshold."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

VOWELS = ("a", "i", "u", "e", "o")
ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root, config, runtime, decoder):
        self.root, self.config, self.runtime, self.decoder = (
            Path(root),
            config,
            runtime,
            decoder,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "episodes").mkdir(exist_ok=True)
        for path in (self.root / "episodes").glob("*/episode.json"):
            episode = read(path)
            if any(c["status"] == "processing" for c in episode["clips"]):
                for clip in episode["clips"]:
                    if clip["status"] == "processing":
                        clip.update(
                            status="interrupted", error="process interrupted", counts={}
                        )
                self.update(episode)
            if (
                episode["role"] == "enrollment"
                and episode["status"] == "complete"
                and not (path.parent / "profile.json").exists()
            ):
                episode.update(
                    status="insufficient",
                    profile_error="Profile was not saved before interruption",
                )
                write(path, episode)

    def path(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("録音IDが不正です。")
        return self.root / "episodes" / identifier

    def create(self, request):
        for key in ("participant", "device", "session"):
            if not isinstance(request.get(key), str) or not ID.fullmatch(request[key]):
                raise ValueError(
                    f"{key} は半角英数字・ハイフン・下線で入力してください。"
                )
        role = request.get("role")
        if role not in ("practice", "enrollment", "query"):
            raise ValueError("録音の種類が不正です。")
        if request.get("consent") is not True:
            raise ValueError("ローカル保存への同意が必要です。")
        recorded_on = request.get("recorded_on", "")
        date.fromisoformat(recorded_on)
        episode = {
            "id": uuid.uuid4().hex,
            "participant": request["participant"],
            "device": request["device"],
            "session": request["session"],
            "recorded_on": recorded_on,
            "role": role,
            "purpose": self.config["purpose"],
            "created_at": now(),
            "status": "collecting",
            "clips": [],
            "counts": {v: 0 for v in VOWELS},
            "seconds": 0.0,
            "consent": True,
            "threshold": None,
        }
        self.path(episode["id"]).mkdir()
        self.update(episode)
        return self.public(episode)

    def load(self, identifier):
        return read(self.path(identifier) / "episode.json")

    def update(self, episode):
        counts = {
            v: sum(
                c.get("counts", {}).get(v, 0)
                for c in episode["clips"]
                if c["status"] == "processed"
            )
            for v in VOWELS
        }
        episode["counts"] = counts
        episode["seconds"] = sum(c["seconds"] for c in episode["clips"])
        n = len(episode["clips"])
        if episode["status"] not in ("aborted",):
            if episode["role"] == "enrollment":
                if n >= self.config["enrollment_clips"]:
                    episode["status"] = (
                        "complete"
                        if min(counts.values())
                        >= self.config["enrollment_minimum_per_vowel"]
                        else "insufficient"
                    )
            elif min(counts.values()) >= 1:
                episode["status"] = "complete"
            elif (
                n >= self.config["maximum_query_clips"]
                or episode["seconds"] >= self.config["maximum_query_seconds"] - 0.05
            ):
                episode["status"] = "insufficient"
        episode["updated_at"] = now()
        write(self.path(episode["id"]) / "episode.json", episode)

    def public(self, episode):
        enrollment = episode["role"] == "enrollment"
        remaining = (
            self.config["maximum_clip_seconds"]
            if enrollment
            else min(
                self.config["maximum_clip_seconds"],
                max(0, self.config["maximum_query_seconds"] - episode["seconds"]),
            )
        )
        index = len(episode["clips"]) % len(self.config["prompts"])
        return {
            **episode,
            "prompt": self.config["prompts"][index],
            "remaining_clip_seconds": remaining,
            "maximum_clips": self.config["enrollment_clips"]
            if enrollment
            else self.config["maximum_query_clips"],
            "missing_vowels": [
                v
                for v, n in episode["counts"].items()
                if n
                < (self.config["enrollment_minimum_per_vowel"] if enrollment else 1)
            ],
            "authentication": "not_calibrated",
        }

    def submit(self, identifier, content, request):
        episode = self.load(identifier)
        if episode["status"] != "collecting":
            raise ValueError("この録音は終了しています。新しい試行を開始してください。")
        public = self.public(episode)
        if request.get("prompt_id") != public["prompt"]["id"]:
            raise ValueError("表示文が更新されています。ページを再読込してください。")
        origin = request.get("origin")
        allowed = (
            ("replay",)
            if self.config["purpose"] == "replay_smoke"
            else ("microphone", "file")
        )
        if origin not in allowed:
            raise ValueError("この実行では使用できない音声入力です。")
        pcm, audio = self.decoder(content, public["remaining_clip_seconds"])
        pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
        clip_id = uuid.uuid4().hex
        dest = self.path(identifier) / clip_id
        dest.mkdir()
        (dest / "original.wav").write_bytes(content)
        clip = {
            "id": clip_id,
            "prompt_id": public["prompt"]["id"],
            "origin": origin,
            "seconds": audio["seconds"],
            "audio": audio,
            "sha256": hashlib.sha256(content).hexdigest(),
            "pcm_sha256": pcm_hash,
            "status": "processing",
            "counts": {},
            "created_at": now(),
            "browser_audio_settings": request.get("audio_settings", {}),
        }
        duplicate = any(
            c.get("pcm_sha256") == pcm_hash
            for p in (self.root / "episodes").glob("*/episode.json")
            for c in read(p)["clips"]
        )
        episode["clips"].append(clip)
        write(self.path(identifier) / "episode.json", episode)
        started = time.monotonic()
        try:
            if duplicate:
                raise ValueError(
                    "同一音声の再利用を検出しました。この試行は不採用として記録します。"
                )
            metadata = {
                "utterance_id": clip_id,
                "speaker_id": episode["participant"],
                "session_id": episode["session"],
                "transcript": public["prompt"]["text"],
                "role": episode["role"],
                "origin": origin,
                "recorded_on": episode["recorded_on"],
                "device": episode["device"],
            }
            result = self.runtime.process(pcm, dest, metadata)
            clip.update(status=result["status"], counts=result["counts"])
        except Exception as error:  # noqa: BLE001 - failed attempts remain part of the episode.
            clip.update(status="failed", error=str(error)[:500], counts={})
        clip["processing_seconds"] = time.monotonic() - started
        self.update(episode)
        if episode["role"] == "enrollment" and episode["status"] == "complete":
            try:
                self.runtime.profile(self.path(identifier), episode)
            except Exception as error:  # noqa: BLE001 - do not claim a saved profile after failure.
                episode.update(status="insufficient", profile_error=str(error)[:500])
                write(self.path(identifier) / "episode.json", episode)
        return self.public(episode)

    def abort(self, identifier):
        episode = self.load(identifier)
        if episode["status"] == "collecting":
            episode["status"] = "aborted"
            self.update(episode)
        return self.public(episode)
