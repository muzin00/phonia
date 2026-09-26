from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sample_review_segments import select_records, stable_priority


def make_record(
    item_id: str,
    utterance_id: str,
    source_file: Path,
    *,
    vowel: str = "a",
    split: str = "train",
    duration: float = 0.06,
    near_silent: bool = False,
) -> dict[str, object]:
    return {
        "vowel_interval_id": item_id,
        "utterance_id": utterance_id,
        "speaker_id": "jvs001",
        "source_file": str(source_file),
        "split": split,
        "transcript": "テスト",
        "vowel_index": 0,
        "normalized_phoneme": vowel,
        "start_sec": 0.1,
        "end_sec": 0.1 + duration,
        "duration_sec": duration,
        "quality_flags": ["near_silent"] if near_silent else [],
    }


class SelectRecordsTest(unittest.TestCase):
    def test_selects_reproducibly_and_adds_mapping_examples(self) -> None:
        config = {
            "seed": 17,
            "target_count": 4,
            "base_per_stratum": 1,
            "phone_mapping_target_count": 1,
            "duration_bins_sec": {"short_max": 0.04, "middle_max": 0.09},
        }
        with tempfile.TemporaryDirectory() as directory:
            media_root = Path(directory)
            source = media_root / "jvs001" / "parallel100" / "sample.wav"
            duplicate_ids = ["normal-a", "normal-b"]
            mapped_id = max(
                duplicate_ids,
                key=lambda item_id: stable_priority(config["seed"], item_id),
            )
            records = [
                make_record(
                    item_id,
                    "mapped" if item_id == mapped_id else item_id,
                    source,
                )
                for item_id in duplicate_ids
            ]
            records.extend(
                (
                    make_record(
                        "near-silent", "near-silent", source, near_silent=True
                    ),
                    make_record("short", "short", source, duration=0.03),
                    make_record("long", "long", source, duration=0.12),
                )
            )

            selected = select_records(records, config, media_root, {"mapped": "v_to_b"})
            repeated = select_records(
                reversed(records), config, media_root, {"mapped": "v_to_b"}
            )

        self.assertEqual(
            [record["vowel_interval_id"] for record in selected],
            [record["vowel_interval_id"] for record in repeated],
        )
        self.assertEqual(len(selected), 4)
        mapped = next(record for record in selected if record["utterance_id"] == "mapped")
        self.assertEqual(mapped["sampling"]["phone_mapping"], "v_to_b")
        self.assertEqual(
            mapped["audio_url"], "/media/jvs001/parallel100/sample.wav"
        )
        self.assertEqual(mapped["expected_units"], 1)


if __name__ == "__main__":
    unittest.main()
