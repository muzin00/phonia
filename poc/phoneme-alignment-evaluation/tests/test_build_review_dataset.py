from __future__ import annotations

import unittest

from scripts.build_review_dataset import build_review_artifacts


def make_record(
    method: str,
    *,
    item_id: str = "sample-vowel-000",
    start_sec: float = 1.0,
    end_sec: float = 1.2,
) -> dict[str, object]:
    return {
        "vowel_interval_id": item_id,
        "utterance_id": "sample",
        "transcript": "サンプル",
        "source_file": "data/samples/sample.wav",
        "vowel_index": 0,
        "normalized_phoneme": "a",
        "expected_units": 1,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "is_long": False,
        "is_devoiced": False,
        "contains_devoiced": False,
        "aligner": method,
        "aligner_version": "1",
        "model_id": f"{method}-model",
        "model_version": "1",
    }


class BuildReviewArtifactsTest(unittest.TestCase):
    def test_separates_anonymous_candidate_from_method_mapping(self) -> None:
        dataset, mapping = build_review_artifacts(
            {"mfa": [make_record("mfa", start_sec=1.0, end_sec=1.2)]},
        )

        item = dataset["items"][0]
        self.assertEqual([candidate["id"] for candidate in item["candidates"]], ["A"])
        self.assertNotIn("aligner", str(item))
        self.assertEqual(
            {candidate["method"] for candidate in mapping["items"][0]["candidates"]},
            {"mfa"},
        )
        self.assertEqual(dataset["schemaVersion"], 2)
        self.assertEqual(dataset["protocol"]["version"], "5")
        self.assertEqual(
            [question["id"] for question in dataset["form"]["candidateQuestions"]],
            ["perceived_content"],
        )
        self.assertEqual(
            [choice["value"] for choice in dataset["form"]["reviewStatus"]["choices"]],
            ["accepted", "rejected", "uncertain"],
        )
        perceived_content = dataset["form"]["candidateQuestions"][0]
        self.assertEqual(
            [choice["value"] for choice in perceived_content["choices"]],
            [
                "target_vowel",
                "target_vowel_with_non_vowel",
                "other_vowel",
                "non_vowel_only",
                "near_silence",
                "uncertain",
            ],
        )
        self.assertEqual(
            dataset["form"]["reviewStatus"]["prompt"],
            "この区間を、表示された母音の学習データとして利用できますか？",
        )

    def test_rejects_multiple_candidate_methods(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exactly one candidate method"):
            build_review_artifacts(
                {"mfa": [make_record("mfa")], "julius": [make_record("julius")]}
            )


if __name__ == "__main__":
    unittest.main()
