from __future__ import annotations

import unittest

from scripts.generate_expected_phonemes import generate_records


def make_record(utterance_id: str = "jvs001_parallel100_sample_001") -> dict:
    return {
        "utterance_id": utterance_id,
        "speaker_id": "jvs001",
        "source_file": "data/sample.wav",
        "source_sha256": "source-hash",
        "source_subset": "parallel100",
        "transcript": "テスト",
        "split": "train",
        "evaluation_role": "training",
        "learning_curve_cohorts": [10, 25, 50, 70],
        "session_id": None,
        "duration_sec": 1.0,
    }


class GenerateRecordsTest(unittest.TestCase):
    def test_preserves_dataset_assignment_metadata(self) -> None:
        def g2p(_text: str, kana: bool = False) -> str:
            return "テスト" if kana else "t e s u t o"

        generated, failures = generate_records(
            [make_record()], "0.4.1", "dictionary", g2p
        )

        self.assertEqual(failures, [])
        self.assertEqual(generated[0]["raw_phonemes"], ["t", "e", "s", "u", "t", "o"])
        self.assertEqual(generated[0]["split"], "train")
        self.assertEqual(generated[0]["learning_curve_cohorts"], [10, 25, 50, 70])

    def test_records_failure_and_continues(self) -> None:
        records = [make_record("jvs001_good"), make_record("jvs001_bad")]

        def g2p(text: str, kana: bool = False) -> str:
            if not kana and text == "失敗":
                raise RuntimeError("conversion failed")
            return "テスト" if kana else "t e s u t o"

        records[1]["transcript"] = "失敗"
        generated, failures = generate_records(records, "0.4.1", "dictionary", g2p)

        self.assertEqual([item["utterance_id"] for item in generated], ["jvs001_good"])
        self.assertEqual(failures[0]["utterance_id"], "jvs001_bad")
        self.assertEqual(failures[0]["stage"], "g2p")
        self.assertEqual(failures[0]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
