"""Recovery must only exclude a reproducible failure and remain restartable."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resume_pipeline as r


class RecoveryTests(unittest.TestCase):
    def test_unknown_failure_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            r.atomic(run / "pipeline-state.json", {"status": "failed", "error": "OOM"})
            self.assertFalse(r.recover(run))

    def test_empty_g2p_is_recorded_once_without_advancing_input(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(r, "ROOT", Path(directory)),
        ):
            run = Path(directory) / "run"
            source = Path(directory) / "source.mp3"
            source.write_bytes(b"fixture")
            digest = r.sha(source)
            row = {
                "sha256": digest,
                "local_path": "source.mp3",
                "path": "source.mp3",
                "split": "train",
                "speaker_id": "fixture",
            }
            r.atomic(run / "inputs.json", [row, {"sha256": "next-unprocessed"}])
            r.atomic(run / "extraction-progress.json", {"processed": 0})
            r.atomic(
                run / "pipeline-state.json", {"status": "failed", "error": r.EMPTY_G2P}
            )
            r.atomic(
                run / "asr" / (digest + ".json"),
                {
                    "input_sha256": digest,
                    "status": "transcribed",
                    "asr": {"text": "fixture"},
                },
            )
            with (
                patch.object(r.pyopenjtalk, "g2p", return_value="a"),
                self.assertRaises(AssertionError),
            ):
                r.recover(run)
            self.assertFalse((run / "extraction" / digest / "receipt.json").exists())
            with patch.object(r.pyopenjtalk, "g2p", return_value="") as g2p:
                self.assertTrue(r.recover(run))
                event = run / "recovery-events" / (digest + ".json")
                before = event.read_bytes()
                self.assertTrue(r.recover(run))
                self.assertEqual(g2p.call_count, 1)
                self.assertEqual(event.read_bytes(), before)
                self.assertFalse(r.read(event)["replacement_selected"])
            self.assertFalse((run / "extraction" / "next-unprocessed").exists())


if __name__ == "__main__":
    unittest.main()
