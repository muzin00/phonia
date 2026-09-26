from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from scripts.inspect_jvs_corpus import (
    inspect_corpus,
    parse_transcripts,
    subset_issue_messages,
)


def write_wav(path: Path, frame_count: int = 2_400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * frame_count)


class ParseTranscriptsTest(unittest.TestCase):
    def test_parses_first_colon_and_reports_invalid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transcripts_utf8.txt"
            path.write_text(
                "sample_001:発話:一\ninvalid\nsample_002:発話二\n",
                encoding="utf-8",
            )

            transcripts, errors = parse_transcripts(path)

        self.assertEqual(transcripts, {"sample_001": "発話:一", "sample_002": "発話二"})
        self.assertEqual(errors, ["line 2: missing ':' separator"])


class InspectCorpusTest(unittest.TestCase):
    def test_reports_counts_formats_and_missing_design_elements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "jvs_ver1"
            subset = root / "jvs001" / "parallel100"
            write_wav(subset / "wav24kHz16bit" / "sample_001.wav")
            (subset / "transcripts_utf8.txt").write_text(
                "sample_001:テスト発話\n", encoding="utf-8"
            )

            inventory = inspect_corpus(root)

        self.assertEqual(inventory["speaker_count"], 1)
        self.assertEqual(len(inventory["missing_speakers"]), 99)
        parallel = inventory["subsets"]["parallel100"]
        self.assertEqual(parallel["wav_count"], 1)
        self.assertEqual(parallel["paired_wav_transcript_count"], 1)
        self.assertEqual(parallel["audio_formats"], {"24000Hz/1ch/16bit/NONE": 1})
        self.assertGreater(inventory["issue_count"], 0)

    def test_allows_unrecorded_nonparallel_transcript_candidates(self) -> None:
        subset = {
            "name": "nonpara30",
            "present": True,
            "expected_utterances": 30,
            "expected_transcripts": 50,
            "allow_unrecorded_transcripts": True,
            "wav_count": 30,
            "transcript_count": 50,
            "wav_without_transcript": [],
            "transcript_without_wav": [f"candidate_{index}" for index in range(20)],
            "transcript_errors": [],
            "audio_errors": [],
            "audio_formats": {"24000Hz/1ch/16bit/NONE": 30},
        }

        self.assertEqual(subset_issue_messages("jvs001", subset), [])


if __name__ == "__main__":
    unittest.main()
