from __future__ import annotations

import unittest

from scripts.build_review_dataset import (
    build_review_artifacts,
    candidate_order,
)


def make_record(
    method: str,
    *,
    item_id: str = "sample-vowel-000",
    start_sec: float = 1.0,
    end_sec: float = 1.2,
) -> dict[str, object]:
    return {
        "vowel_run_id": item_id,
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


class CandidateOrderTest(unittest.TestCase):
    def test_is_deterministic_for_seed_and_item(self) -> None:
        methods = ["mfa", "julius", "wav2vec2"]

        first = candidate_order(methods, "seed", "item-1")
        second = candidate_order(list(reversed(methods)), "seed", "item-1")

        self.assertEqual(first, second)


class BuildReviewArtifactsTest(unittest.TestCase):
    def test_separates_anonymous_candidates_from_method_mapping(self) -> None:
        dataset, mapping = build_review_artifacts(
            {
                "mfa": [make_record("mfa", start_sec=1.0, end_sec=1.2)],
                "julius": [make_record("julius", start_sec=0.9, end_sec=1.3)],
            },
            seed="seed",
        )

        item = dataset["items"][0]
        self.assertEqual(
            {candidate["id"] for candidate in item["candidates"]}, {"A", "B"}
        )
        self.assertNotIn("aligner", str(item))
        self.assertEqual(
            {candidate["method"] for candidate in mapping["items"][0]["candidates"]},
            {"mfa", "julius"},
        )

    def test_marks_a_method_without_an_interval_as_missing(self) -> None:
        dataset, _ = build_review_artifacts(
            {
                "mfa": [make_record("mfa")],
                "julius": [make_record("julius", item_id="other-vowel-000")],
            }
        )

        sample_item = next(
            item for item in dataset["items"] if item["id"] == "sample-vowel-000"
        )
        self.assertEqual(
            {candidate["status"] for candidate in sample_item["candidates"]},
            {"available", "missing"},
        )

    def test_rejects_candidates_for_different_targets(self) -> None:
        different = make_record("julius")
        different["normalized_phoneme"] = "i"

        with self.assertRaisesRegex(ValueError, "candidate target differs"):
            build_review_artifacts({"mfa": [make_record("mfa")], "julius": [different]})


if __name__ == "__main__":
    unittest.main()
