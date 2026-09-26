from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from scripts.build_phase3_manifest import filter_phase3_manifest


def write_test_wave(path: Path) -> None:
    samples = array(
        "h",
        [round(10_000 * math.sin(2 * math.pi * index / 20)) for index in range(2400)],
    )
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(24_000)
        destination.writeframes(samples.tobytes())


def segment(item_id: str, source: Path, flags: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "vowel_interval_id": item_id,
        "speaker_id": "jvs001",
        "source_file": str(source),
        "split": "train",
        "evaluation_role": "training",
        "learning_curve_cohorts": [1],
        "normalized_phoneme": "a",
        "start_frame": 0,
        "end_frame": 1200,
        "frame_count": 1200,
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "duration_sec": 0.05,
        "quality_flags": flags,
    }


class FilterPhase3ManifestTest(unittest.TestCase):
    def test_excludes_only_configured_flags_and_validates_audio(self) -> None:
        config = {
            "schema_version": 1,
            "policy_version": "test",
            "source_schema_version": 1,
            "exclude_quality_flags": ["near_silent"],
            "expected_counts": {"input": 3, "excluded": 1, "eligible": 2},
            "coverage": {
                "required_vowels": ["a"],
                "required_evaluation_roles_by_split": {"train": ["training"]},
            },
        }
        split_config = {
            "speaker_splits": {"train": ["jvs001"]},
            "learning_curve_sizes": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            write_test_wave(source)
            records = [
                segment("normal", source, []),
                segment("near-silent", source, ["near_silent"]),
                segment("other-flag", source, ["review_note"]),
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            summary = filter_phase3_manifest(
                input_path,
                output_path,
                config,
                split_config,
                verify_source_audio=True,
            )
            output = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual([record["vowel_interval_id"] for record in output], ["normal", "other-flag"])
        self.assertEqual(summary["input"]["segment_count"], 3)
        self.assertEqual(summary["eligible"]["segment_count"], 2)
        self.assertEqual(summary["excluded"]["segment_count"], 1)
        self.assertEqual(summary["exclusion_reason_counts"], {"near_silent": 1})
        self.assertEqual(summary["validated_source_file_count"], 1)
        self.assertEqual(summary["eligible"]["by_split_speaker_count"], {"train": 1})
        self.assertEqual(
            summary["eligible"]["by_learning_curve_cohort_speaker_count"],
            {"1": 1},
        )


if __name__ == "__main__":
    unittest.main()
