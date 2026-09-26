from __future__ import annotations

import copy
import unittest

from scripts.build_dataset_manifest import (
    evaluation_role,
    learning_curve_cohorts,
    summarize_records,
    validate_config,
)


def make_config() -> dict:
    return {
        "schema_version": 1,
        "design_version": "test",
        "corpus": "JVS",
        "corpus_version": "jvs_ver1",
        "randomization": {
            "method": "python_random_shuffle",
            "seed": 9,
        },
        "included_subsets": ["parallel100", "nonpara30"],
        "speaker_splits": {
            "train": ["jvs001", "jvs002"],
            "validation": ["jvs003"],
            "test": ["jvs004"],
        },
        "learning_curve_sizes": [1, 2],
        "evaluation_roles": {
            "parallel100": {
                "enrollment": {
                    "utterance_number_start": 1,
                    "utterance_number_end": 50,
                },
                "verification": {
                    "utterance_number_start": 51,
                    "utterance_number_end": 100,
                },
            },
            "nonpara30": "cross_text_verification",
        },
        "minimum_evaluation_utterances_per_speaker": {
            "enrollment": 1,
            "verification": 1,
            "cross_text_verification": 1,
        },
    }


def make_record(
    utterance_id: str,
    speaker_id: str,
    split: str,
    role: str,
    checksum: str,
) -> dict:
    return {
        "utterance_id": utterance_id,
        "speaker_id": speaker_id,
        "source_file": f"data/{utterance_id}.wav",
        "source_sha256": checksum,
        "source_subset": "parallel100",
        "split": split,
        "evaluation_role": role,
        "duration_sec": 1.0,
    }


class ConfigValidationTest(unittest.TestCase):
    def test_accepts_disjoint_complete_speaker_splits(self) -> None:
        config = make_config()

        validate_config(config, ["jvs001", "jvs002", "jvs003", "jvs004"])

    def test_rejects_speaker_overlap(self) -> None:
        config = make_config()
        config["speaker_splits"]["test"] = ["jvs003"]

        with self.assertRaisesRegex(ValueError, "speaker overlap"):
            validate_config(config, ["jvs001", "jvs002", "jvs003", "jvs004"])

    def test_rejects_split_that_differs_from_seed(self) -> None:
        config = make_config()
        config["randomization"]["seed"] = 10

        with self.assertRaisesRegex(ValueError, "randomization settings"):
            validate_config(config, ["jvs001", "jvs002", "jvs003", "jvs004"])


class AssignmentTest(unittest.TestCase):
    def test_assigns_nested_learning_curve_cohorts(self) -> None:
        config = make_config()

        self.assertEqual(learning_curve_cohorts("jvs001", config), [1, 2])
        self.assertEqual(learning_curve_cohorts("jvs002", config), [2])
        self.assertEqual(learning_curve_cohorts("jvs003", config), [])

    def test_assigns_evaluation_roles_by_subset_and_number(self) -> None:
        config = make_config()

        self.assertEqual(
            evaluation_role("validation", "parallel100", "VOICEACTRESS100_050", config),
            "enrollment",
        )
        self.assertEqual(
            evaluation_role("test", "parallel100", "VOICEACTRESS100_051", config),
            "verification",
        )
        self.assertEqual(
            evaluation_role("test", "nonpara30", "BASIC5000_0001", config),
            "cross_text_verification",
        )


class ManifestValidationTest(unittest.TestCase):
    def make_valid_records(self) -> list[dict]:
        records = [
            make_record("train-1", "jvs001", "train", "training", "hash-1"),
            make_record("train-2", "jvs002", "train", "training", "hash-2"),
        ]
        for speaker, split in (("jvs003", "validation"), ("jvs004", "test")):
            records.extend(
                [
                    make_record(
                        f"{speaker}-enroll",
                        speaker,
                        split,
                        "enrollment",
                        f"{speaker}-1",
                    ),
                    make_record(
                        f"{speaker}-verify",
                        speaker,
                        split,
                        "verification",
                        f"{speaker}-2",
                    ),
                    make_record(
                        f"{speaker}-cross",
                        speaker,
                        split,
                        "cross_text_verification",
                        f"{speaker}-3",
                    ),
                ]
            )
        return records

    def test_passes_complete_unique_records(self) -> None:
        report = summarize_records(
            self.make_valid_records(),
            make_config(),
            sorted(
                [
                    "jvs030/VOICEACTRESS100_045",
                    "jvs074/VOICEACTRESS100_094",
                    "jvs089/VOICEACTRESS100_019",
                ]
            ),
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["error_count"], 0)

    def test_rejects_duplicate_audio_content(self) -> None:
        records = self.make_valid_records()
        records[1] = copy.deepcopy(records[1])
        records[1]["source_sha256"] = records[0]["source_sha256"]

        report = summarize_records(
            records,
            make_config(),
            sorted(
                [
                    "jvs030/VOICEACTRESS100_045",
                    "jvs074/VOICEACTRESS100_094",
                    "jvs089/VOICEACTRESS100_019",
                ]
            ),
        )

        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            any("duplicate source_sha256" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
