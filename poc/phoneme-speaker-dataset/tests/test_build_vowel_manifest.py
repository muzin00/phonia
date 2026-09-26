from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from scripts.build_vowel_manifest import process_utterance
from scripts.run_julius_dataset import alignment_input_sha256, atomic_write_json, sha256


def write_test_wave(path: Path) -> None:
    samples = array(
        "h",
        [round(10_000 * math.sin(2 * math.pi * index / 20)) for index in range(24_000)],
    )
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(24_000)
        destination.writeframes(samples.tobytes())


def expected_record(source_path: Path) -> dict:
    return {
        "utterance_id": "jvs001_sample",
        "speaker_id": "jvs001",
        "source_file": str(source_path),
        "source_sha256": sha256(source_path),
        "source_subset": "parallel100",
        "transcript": "テスト",
        "split": "validation",
        "evaluation_role": "enrollment",
        "learning_curve_cohorts": [],
        "session_id": None,
        "g2p_engine": "pyopenjtalk",
        "g2p_version": "0.4.1",
        "g2p_dictionary": "dictionary",
        "raw_phonemes": ["m", "a", "s", "I"],
    }


def raw_alignment(expected: dict) -> dict:
    phones = ["silB", "m", "a", "s", "i", "silE"]
    return {
        "utterance_id": expected["utterance_id"],
        "source_sha256": expected["source_sha256"],
        "alignment_input_sha256": alignment_input_sha256(expected),
        "intervals": [
            {
                "phoneme": phone,
                "start_sec": index * 0.1,
                "end_sec": (index + 1) * 0.1,
                "score": -20.0 - index,
            }
            for index, phone in enumerate(phones)
        ],
    }


class ProcessUtteranceTest(unittest.TestCase):
    def test_builds_source_slice_records_and_preserves_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.wav"
            raw_path = root / "alignment.json"
            write_test_wave(source_path)
            expected = expected_record(source_path)
            atomic_write_json(raw_path, raw_alignment(expected))
            alignment = {
                "status": "success",
                "raw_alignment_file": str(raw_path),
            }

            segments, failure = process_utterance(
                expected, alignment, 0.03, -50.0, verify_source_checksum=True
            )

        self.assertIsNone(failure)
        self.assertEqual([item["normalized_phoneme"] for item in segments], ["a", "i"])
        self.assertEqual(segments[0]["split"], "validation")
        self.assertEqual(segments[0]["evaluation_role"], "enrollment")
        self.assertEqual(segments[0]["storage_mode"], "source_slice")
        self.assertEqual(segments[0]["start_frame"], 4_800)
        self.assertEqual(segments[0]["frame_count"], 2_400)
        self.assertFalse(segments[0]["is_devoiced"])
        self.assertTrue(segments[1]["is_devoiced"])

    def test_propagates_alignment_failure_without_reading_audio(self) -> None:
        expected = {
            "utterance_id": "jvs001_sample",
            "speaker_id": "jvs001",
            "source_file": "missing.wav",
            "split": "test",
            "evaluation_role": "verification",
        }
        alignment = {
            "status": "failure",
            "stage": "julius_alignment",
            "error_type": "RuntimeError",
            "reason": "alignment failed",
        }

        segments, failure = process_utterance(
            expected, alignment, 0.03, -50.0, verify_source_checksum=True
        )

        self.assertEqual(segments, [])
        self.assertEqual(failure["stage"], "julius_alignment")
        self.assertEqual(failure["reason"], "alignment failed")


if __name__ == "__main__":
    unittest.main()
