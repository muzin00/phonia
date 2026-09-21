from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_mfa_alignment import conda_command, prepare_corpus


class PrepareCorpusTest(unittest.TestCase):
    def test_writes_utf8_label_for_existing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            audio_path = temporary_path / "jvs001_sample.wav"
            audio_path.write_bytes(b"RIFF")
            corpus_path = temporary_path / "corpus"
            record = {
                "utterance_id": "sample_001",
                "speaker_id": "jvs001",
                "source_file": str(audio_path),
                "transcript": "日本語の発話。",
            }

            stems = prepare_corpus([record], corpus_path)

            self.assertEqual(stems, ["jvs001_sample"])
            self.assertEqual(
                (corpus_path / "jvs001_sample.lab").read_text(encoding="utf-8"),
                "日本語の発話。\n",
            )

    def test_rejects_filename_without_speaker_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            audio_path = temporary_path / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            record = {
                "utterance_id": "sample_001",
                "speaker_id": "jvs001",
                "source_file": str(audio_path),
                "transcript": "sample",
            }

            with self.assertRaisesRegex(ValueError, "filename must start"):
                prepare_corpus([record], temporary_path / "corpus")


class CondaCommandTest(unittest.TestCase):
    def test_builds_uncaptured_environment_command(self) -> None:
        command = conda_command(Path("/opt/conda"), "phonia-mfa", "mfa", "version")

        self.assertEqual(
            command,
            [
                "/opt/conda",
                "run",
                "--no-capture-output",
                "-n",
                "phonia-mfa",
                "mfa",
                "version",
            ],
        )


if __name__ == "__main__":
    unittest.main()
