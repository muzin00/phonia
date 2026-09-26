from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_julius_dataset import (
    alignment_input_sha256,
    artifact_paths,
    atomic_write_json,
    parse_alignment_log,
    process_records,
    to_julius_phonemes,
)


def make_record(
    utterance_id: str,
    speaker_id: str,
    source_file: str = "data/VOICEACTRESS100_001.wav",
) -> dict:
    return {
        "utterance_id": utterance_id,
        "speaker_id": speaker_id,
        "source_file": source_file,
        "source_sha256": f"hash-{speaker_id}",
        "source_subset": "parallel100",
        "split": "train",
        "evaluation_role": "training",
        "learning_curve_cohorts": [10],
        "session_id": None,
        "g2p_engine": "pyopenjtalk",
        "g2p_version": "0.4.1",
        "g2p_dictionary": "dictionary",
        "raw_phonemes": ["m", "a"],
    }


def raw_alignment(record: dict) -> dict:
    return {
        "alignment_input_sha256": alignment_input_sha256(record),
        "intervals": [{"phoneme": "a", "start_sec": 0.1, "end_sec": 0.2}],
    }


class ArtifactPathsTest(unittest.TestCase):
    def test_same_source_stem_uses_distinct_utterance_paths(self) -> None:
        root = Path("output")
        first = make_record("jvs001_parallel100_voiceactress100_001", "jvs001")
        second = make_record("jvs002_parallel100_voiceactress100_001", "jvs002")

        first_path = artifact_paths(root, first)["raw"]
        second_path = artifact_paths(root, second)["raw"]

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.parent.name, "jvs001")
        self.assertEqual(second_path.parent.name, "jvs002")

    def test_rejects_path_traversal_identifier(self) -> None:
        record = make_record("../escape", "jvs001")

        with self.assertRaisesRegex(ValueError, "safe path components"):
            artifact_paths(Path("output"), record)


class PhoneMappingTest(unittest.TestCase):
    def test_maps_unsupported_openjtalk_phones_to_fixed_model(self) -> None:
        self.assertEqual(
            to_julius_phonemes(["v", "a", "ty", "u", "A", "pau", "cl"]),
            ["b", "a", "ch", "u", "a", "sp", "q"],
        )


class AlignmentLogTest(unittest.TestCase):
    def test_reports_search_failure_reason(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no alignment candidate"):
            parse_alignment_log("<search failed>")


class ProcessRecordsTest(unittest.TestCase):
    def test_records_failure_and_continues_with_next_utterance(self) -> None:
        records = [
            make_record("jvs001_good", "jvs001"),
            make_record("jvs002_bad", "jvs002"),
            make_record("jvs003_good", "jvs003"),
        ]
        called: list[str] = []

        def aligner(record: dict, raw_path: Path, _log_path: Path) -> dict:
            called.append(record["utterance_id"])
            if record["utterance_id"] == "jvs002_bad":
                raise RuntimeError("alignment failed")
            atomic_write_json(raw_path, raw_alignment(record))
            return {
                "duration_sec": 0.1,
                "phone_interval_count": 1,
                "raw_alignment_file": str(raw_path),
                "reused": False,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            results = process_records(
                records, Path(temporary_directory), aligner, resume=True
            )

        self.assertEqual(called, [item["utterance_id"] for item in records])
        self.assertEqual(
            [item["status"] for item in results],
            ["success", "failure", "success"],
        )
        self.assertEqual(results[1]["stage"], "julius_alignment")
        self.assertEqual(results[1]["error_type"], "RuntimeError")

    def test_reuses_only_compatible_success_checkpoint(self) -> None:
        record = make_record("jvs001_sample", "jvs001")
        calls = 0

        def aligner(_record: dict, _raw_path: Path, _log_path: Path) -> dict:
            nonlocal calls
            calls += 1
            raise AssertionError("compatible checkpoint should be reused")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            raw_path = artifact_paths(output, record)["raw"]
            atomic_write_json(raw_path, raw_alignment(record))

            results = process_records([record], output, aligner, resume=True)

            stored = json.loads(raw_path.read_text(encoding="utf-8"))

        self.assertEqual(calls, 0)
        self.assertTrue(results[0]["reused"])
        self.assertEqual(
            stored["alignment_input_sha256"], alignment_input_sha256(record)
        )


if __name__ == "__main__":
    unittest.main()
