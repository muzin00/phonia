from __future__ import annotations

import unittest

from scripts.summarize_review_results import build_summary


def sample(item_id: str, quality: str) -> dict[str, object]:
    return {
        "vowel_interval_id": item_id,
        "speaker_id": "jvs001",
        "normalized_phoneme": "a",
        "split": "train",
        "sampling": {
            "stratum": f"a/train/{quality}/middle",
            "quality_group": quality,
            "duration_bin": "middle",
            "phone_mapping": None,
        },
    }


def review(
    item_id: str, revision: int, status: str, content: str
) -> dict[str, object]:
    return {
        "itemId": item_id,
        "status": "completed",
        "revision": revision,
        "reviewStatus": status,
        "candidateAnswers": [
            {"answers": {"perceived_content": content}},
        ],
    }


class BuildSummaryTest(unittest.TestCase):
    def test_uses_latest_revision_and_reports_pending_coverage(self) -> None:
        summary = build_summary(
            [sample("one", "normal"), sample("two", "near_silent")],
            [
                review("one", 1, "rejected", "other_vowel"),
                review("one", 2, "accepted", "target_vowel"),
            ],
        )

        self.assertEqual(summary["coverage"]["reviewed"], 1)
        self.assertEqual(summary["coverage"]["pending"], 1)
        self.assertEqual(summary["overall"]["accepted"], 1)
        self.assertEqual(summary["overall"]["acceptance_rate"], 1.0)
        self.assertEqual(
            summary["coverage"]["unreviewed_strata"],
            ["a/train/near_silent/middle"],
        )


if __name__ == "__main__":
    unittest.main()
