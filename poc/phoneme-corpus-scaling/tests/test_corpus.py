"""Check split provenance, bounded alignment and immutable preparation writes."""

import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus import atomic, bounded_julius, fixed, official_segment, source_metadata


class CorpusTests(unittest.TestCase):
    def test_keep_held_out_split_and_original_extraction_provenance(self):
        row = {
            "split": "test",
            "speaker_id": "jtube_ABC",
            "sha256": "a" * 64,
            "path": "source.mp3",
            "local_path": "audio/source.mp3",
            "position": 7,
        }
        original = {
            "split": "pilot",
            "role": "held_out",
            "speaker_id": "jtube_abc",
            "cache_index": 0,
        }
        result = official_segment(original, row)
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["extraction_split"], "pilot")
        self.assertEqual(result["role"], "held_out")
        self.assertEqual(source_metadata(row)["role"], "held_out")
        self.assertEqual(original["split"], "pilot")
        with self.assertRaises(AssertionError):
            official_segment({**original, "speaker_id": "wrong"}, row)

    def test_alignment_limits_before_call_and_timeout_propagation(self):
        calls = []

        def fake(command, **kwargs):
            calls.append(kwargs)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        config = {"maximum_utterance_seconds": 30, "julius_timeout_seconds": 10}
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "input.wav"
            for frames, expected in [
                (31 * 16000, ValueError),
                (16000, subprocess.TimeoutExpired),
            ]:
                with wave.open(str(wav), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(b"\0\0" * frames)
                with self.assertRaises(expected):
                    bounded_julius(fake, config)(["/bin/julius"], input=str(wav) + "\n")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["timeout"], 10)

    def test_frozen_manifest_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/freeze.json"
            fixed(path, {"seed": 10})
            fixed(path, {"seed": 10})
            with self.assertRaises(AssertionError):
                fixed(path, {"seed": 11})
            self.assertEqual(json.loads(path.read_text()), {"seed": 10})
            atomic(path.parent / "progress.json", {"processed": 1})
            self.assertFalse(list(path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
