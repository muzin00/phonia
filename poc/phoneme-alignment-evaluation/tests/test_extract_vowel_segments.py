from __future__ import annotations

import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from scripts.extract_vowel_segments import (
    WaveSource,
    extract_record,
    read_wave,
    rms_dbfs,
    seconds_to_frame,
    slice_frames,
    validate_vowel_indices,
)


def make_wave_source(duration_sec: float = 1.0) -> WaveSource:
    sample_rate_hz = 1000
    samples = array(
        "h",
        [round(10_000 * math.sin(2 * math.pi * index / 20)) for index in range(1000)],
    )
    return WaveSource(
        channels=1,
        sample_width=2,
        sample_rate_hz=sample_rate_hz,
        frame_count=round(duration_sec * sample_rate_hz),
        compression_type="NONE",
        compression_name="not compressed",
        frames=samples.tobytes(),
    )


def make_record(start_sec: float = 0.2, end_sec: float = 0.4) -> dict[str, object]:
    return {
        "utterance_id": "sample_001",
        "vowel_interval_id": "sample_001-mfa-vowel-000",
        "vowel_index": 0,
        "normalized_phoneme": "a",
        "source_file": "sample.wav",
        "aligner": "mfa",
        "model_id": "japanese_mfa",
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


class FrameConversionTest(unittest.TestCase):
    def test_rounds_and_clamps_frame_index(self) -> None:
        self.assertEqual(seconds_to_frame(0.1236, 1000, 1000), 124)
        self.assertEqual(seconds_to_frame(-1, 1000, 1000), 0)
        self.assertEqual(seconds_to_frame(2, 1000, 1000), 1000)

    def test_slices_complete_pcm_frames(self) -> None:
        source = make_wave_source()

        frames = slice_frames(source, 100, 250)

        self.assertEqual(len(frames), 150 * source.bytes_per_frame)


class RmsTest(unittest.TestCase):
    def test_returns_none_for_silence(self) -> None:
        self.assertIsNone(rms_dbfs(bytes(200), 2))

    def test_rejects_unsupported_sample_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "16-bit"):
            rms_dbfs(bytes(100), 1)


class ExtractRecordTest(unittest.TestCase):
    def test_writes_exact_and_context_waves(self) -> None:
        source = make_wave_source()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)

            result = extract_record(
                make_record(), source, output_directory, 0.1, 0.03, -50.0
            )

            exact_path = (
                output_directory / "exact" / "sample_001-mfa-vowel-000_a.wav"
            )
            context_path = (
                output_directory / "context" / "sample_001-mfa-vowel-000_a.wav"
            )
            self.assertTrue(exact_path.is_file())
            self.assertTrue(context_path.is_file())
            self.assertEqual(result["exact_frame_count"], 200)
            self.assertEqual(result["context_frame_count"], 400)
            self.assertEqual(result["quality_flags"], [])
            with wave.open(str(exact_path), "rb") as exact:
                self.assertEqual(exact.getnframes(), 200)
                self.assertEqual(exact.getframerate(), 1000)

    def test_flags_short_silent_and_clipped_segment(self) -> None:
        source = WaveSource(
            channels=1,
            sample_width=2,
            sample_rate_hz=1000,
            frame_count=1000,
            compression_type="NONE",
            compression_name="not compressed",
            frames=bytes(2000),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = extract_record(
                make_record(0.0, 0.02),
                source,
                Path(temporary_directory),
                0.1,
                0.03,
                -50.0,
            )

        self.assertEqual(
            result["quality_flags"],
            ["short_exact", "silent_exact", "context_clipped_start"],
        )

    def test_rejects_interval_outside_audio(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(ValueError, "exceeds source audio"),
        ):
            extract_record(
                make_record(0.9, 1.1),
                make_wave_source(),
                Path(temporary_directory),
                0.1,
                0.03,
                -50.0,
            )


class InputValidationTest(unittest.TestCase):
    def test_rejects_non_contiguous_vowel_indices(self) -> None:
        records = [make_record(), {**make_record(), "vowel_index": 2}]

        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_vowel_indices(records)

    def test_reads_generated_wave(self) -> None:
        source = make_wave_source()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.wav"
            with wave.open(str(path), "wb") as destination:
                destination.setnchannels(source.channels)
                destination.setsampwidth(source.sample_width)
                destination.setframerate(source.sample_rate_hz)
                destination.writeframes(source.frames)

            loaded = read_wave(path)

        self.assertEqual(loaded.sample_rate_hz, source.sample_rate_hz)
        self.assertEqual(loaded.frame_count, source.frame_count)
        self.assertEqual(loaded.frames, source.frames)


if __name__ == "__main__":
    unittest.main()
