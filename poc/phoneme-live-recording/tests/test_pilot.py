import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from pilot_report import summarize
from runtime import decode_wav
from store import Store, read, write


def wav(value=1, seconds=1, rate=24000, channels=1):
    out = io.BytesIO()
    with wave.open(out, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(
            np.full(round(seconds * rate) * channels, value, dtype="<i2").tobytes()
        )
    return out.getvalue()


class FakeRuntime:
    def __init__(self):
        self.counts = {v: 1 for v in "aiueo"}
        self.fail = False

    def process(self, pcm, destination, metadata):
        if self.fail:
            raise RuntimeError("test failure")
        return {"status": "processed", "counts": self.counts}

    def profile(self, directory, episode):
        write(directory / "profile.json", {"threshold": None})


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = read(BASE / "protocol.json")
        self.engine = FakeRuntime()
        self.db = Store(self.root, self.config, self.engine, decode_wav)
        self.request = {
            "participant": "p001",
            "device": "a",
            "session": "day1",
            "recorded_on": "2026-10-11",
            "role": "query",
            "consent": True,
        }

    def tearDown(self):
        self.temp.cleanup()

    def create(self, **kwargs):
        return self.db.create({**self.request, **kwargs})

    def submit(self, episode, value=1, seconds=1, origin="microphone"):
        return self.db.submit(
            episode["id"],
            wav(value, seconds),
            {"prompt_id": episode["prompt"]["id"], "origin": origin},
        )

    def test_stop_on_first_covered_clip_no_authentication(self):
        e = self.submit(self.create())
        self.assertEqual(e["status"], "complete")
        self.assertIsNone(e["threshold"])
        self.assertEqual(e["authentication"], "not_calibrated")
        with self.assertRaises(ValueError):
            self.submit(e, 2)

    def test_count_limit_preserves_failures(self):
        self.engine.fail = True
        e = self.create()
        for i in range(3):
            e = self.submit(e, i + 1)
        self.assertEqual(e["status"], "insufficient")
        self.assertEqual(len(e["clips"]), 3)
        self.assertTrue(all(c["status"] == "failed" for c in e["clips"]))
        self.assertEqual(e["seconds"], 3)
        self.assertEqual(len(list(self.root.rglob("original.wav"))), 3)

    def test_twenty_second_budget(self):
        self.engine.counts = {"a": 2}
        e = self.create()
        e = self.submit(e, 1, 8)
        e = self.submit(e, 2, 8)
        self.assertEqual(e["remaining_clip_seconds"], 4)
        with self.assertRaises(ValueError):
            self.submit(e, 3, 4.01)
        e = self.submit(e, 3, 4)
        self.assertEqual((e["status"], e["seconds"]), ("insufficient", 20))

    def test_duplicates_count_as_failed_attempts_across_episodes(self):
        self.submit(self.create(), 1)
        e = self.submit(self.create(), 1)
        self.assertEqual(e["clips"][0]["status"], "failed")
        self.assertEqual(e["counts"], dict.fromkeys("aiueo", 0))

    def test_interrupted_attempt_recovered_without_retrying(self):
        self.engine.fail = True
        e = self.submit(self.create())
        path = self.db.path(e["id"]) / "episode.json"
        saved = read(path)
        saved["clips"][0]["status"] = "processing"
        write(path, saved)
        db = Store(self.root, self.config, self.engine, decode_wav)
        e = db.load(e["id"])
        self.assertEqual(e["clips"][0]["status"], "interrupted")
        self.assertEqual(e["seconds"], 1)

    def test_enrollment_requires_twenty_clips_and_profile(self):
        e = self.create(role="enrollment")
        for i in range(19):
            e = self.submit(e, i + 1)
        self.assertEqual(e["status"], "collecting")
        e = self.submit(e, 20)
        self.assertEqual(e["status"], "complete")
        self.assertTrue((self.db.path(e["id"]) / "profile.json").exists())

    def test_missing_profile_after_interruption_not_complete(self):
        e = self.create(role="enrollment")
        saved = self.db.load(e["id"])
        saved["status"] = "complete"
        write(self.db.path(e["id"]) / "episode.json", saved)
        db = Store(self.root, self.config, self.engine, decode_wav)
        self.assertEqual(db.load(e["id"])["status"], "insufficient")

    def test_consent_ids_and_origin(self):
        for values in (
            {"consent": False},
            {"participant": "../p"},
            {"recorded_on": "invalid"},
        ):
            with self.assertRaises(ValueError):
                self.create(**values)
        with self.assertRaises(ValueError):
            self.db.path("../bad")
        e = self.create()
        with self.assertRaises(ValueError):
            self.submit(e, origin="replay")
        with self.assertRaises(ValueError):
            self.db.submit(
                e["id"], wav(), {"prompt_id": "wrong", "origin": "microphone"}
            )

    def test_inventory_excludes_imports_replay_and_unstarted_episodes(self):
        self.create()
        self.submit(self.create(), origin="file")
        report = summarize(self.root)
        self.assertEqual(report["microphone_episodes"], 0)
        self.assertEqual(report["state"], "awaiting_new_recordings")
        self.submit(self.create(role="practice"), 2)
        report = summarize(self.root)
        self.assertEqual(report["practice_episodes"], 1)
        self.assertEqual(report["query_episodes"], 0)
        self.assertIsNone(report["far"])
        self.assertIsNone(report["frr"])

    def test_abort_is_terminal(self):
        e = self.create()
        self.db.abort(e["id"])
        with self.assertRaises(ValueError):
            self.submit(e)

    def test_wave_format_duration_truncation(self):
        for content in (b"bad", wav(channels=2), wav(seconds=2), wav()[:-2]):
            with self.assertRaises(ValueError):
                decode_wav(content, 1)

    def test_resampling_and_exact_24k_pcm(self):
        pcm, info = decode_wav(wav(value=1024), 1)
        np.testing.assert_array_equal(pcm, np.full(24000, 1024, dtype="<i2"))
        self.assertFalse(info["resampled"])
        pcm, info = decode_wav(wav(value=1024, rate=48000), 1)
        self.assertEqual(len(pcm), 24000)
        self.assertTrue(info["resampled"])
        self.assertAlmostEqual(float(np.median(pcm)), 1024, delta=1)


if __name__ == "__main__":
    unittest.main()
