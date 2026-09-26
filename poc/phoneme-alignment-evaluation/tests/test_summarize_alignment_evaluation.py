from __future__ import annotations

import unittest

from scripts.summarize_alignment_evaluation import (
    latest_reviews,
    pair_atomic_records,
    pair_mfa_with_julius,
    review_summary,
)


class ReviewSummaryTest(unittest.TestCase):
    def test_aggregates_status_and_candidate_answers(self) -> None:
        records = [
            {
                "reviewStatus": "accepted",
                "candidateAnswers": [
                    {
                        "candidateId": "A",
                        "answers": {
                            "perceived_content": "target_vowel",
                        },
                    }
                ],
            },
            {
                "reviewStatus": "rejected",
                "candidateAnswers": [
                    {
                        "candidateId": "A",
                        "answers": {
                            "perceived_content": "other_vowel",
                        },
                    }
                ],
            },
        ]

        result = review_summary(records)

        self.assertEqual(result["reviewed_count"], 2)
        self.assertEqual(result["accepted_rate"], 0.5)
        self.assertEqual(result["status_counts"], {"accepted": 1, "rejected": 1})
        self.assertEqual(
            result["question_counts"]["perceived_content"],
            {"other_vowel": 1, "target_vowel": 1},
        )

    def test_latest_reviews_does_not_mix_protocol_batches(self) -> None:
        records = [
            {
                "datasetId": "review",
                "datasetVersion": "3",
                "protocol": {"id": "vowel-review", "version": "2"},
                "itemId": "item-1",
                "revision": 4,
            },
            {
                "datasetId": "review",
                "datasetVersion": "5",
                "protocol": {"id": "vowel-review", "version": "5"},
                "itemId": "item-2",
                "revision": 1,
            },
        ]

        self.assertEqual(latest_reviews(records), {"item-2": records[1]})


class PairingTest(unittest.TestCase):
    def test_expands_mfa_expected_units_against_atomic_julius_intervals(self) -> None:
        mfa = [
            {
                "utterance_id": "u1",
                "vowel_index": 0,
                "normalized_phoneme": "o",
                "expected_units": 2,
                "start_sec": 1.0,
                "end_sec": 1.2,
            }
        ]
        julius = [
            {
                "utterance_id": "u1",
                "vowel_index": 0,
                "normalized_phoneme": "o",
                "start_sec": 1.01,
                "end_sec": 1.1,
            },
            {
                "utterance_id": "u1",
                "vowel_index": 1,
                "normalized_phoneme": "o",
                "start_sec": 1.1,
                "end_sec": 1.19,
            },
        ]

        self.assertEqual(
            pair_mfa_with_julius(mfa, julius),
            [("o", 1.0, 1.2, 1.01, 1.19)],
        )

    def test_pairs_atomic_records_by_target(self) -> None:
        left = [
            {
                "utterance_id": "u1",
                "vowel_index": 0,
                "normalized_phoneme": "a",
                "start_sec": 0.1,
                "end_sec": 0.2,
            }
        ]
        right = [
            {
                "utterance_id": "u1",
                "vowel_index": 0,
                "normalized_phoneme": "a",
                "start_sec": 0.11,
                "end_sec": 0.21,
            }
        ]

        self.assertEqual(
            pair_atomic_records(left, right),
            [("a", 0.1, 0.2, 0.11, 0.21)],
        )


if __name__ == "__main__":
    unittest.main()
