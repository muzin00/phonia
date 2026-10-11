"""Guard training-only reproducible sampling and scarce phone allocation."""

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review import sample_segments, select_utterances


class ReviewSamplingTests(unittest.TestCase):
    def test_utterances_are_training_only_and_input_order_independent(self):
        rows = [
            {"split": split, "speaker_id": f"s{s}", "path": f"{split}/{s}/{i}"}
            for split in ["train", "validation", "test"]
            for s in range(3)
            for i in range(4)
        ]
        selected = select_utterances(rows, 12)
        self.assertEqual(selected, select_utterances(list(reversed(rows)), 12))
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(r["split"] == "train" for r in selected))

    def test_scarcity_redistributes_without_duplicates(self):
        rows = [
            {
                "source_split": "train",
                "quality_flags": [],
                "segment_id": f"{p}-{i}",
                "phoneme": p,
            }
            for p, count in [("a", 200), ("i", 100), ("ky", 1)]
            for i in range(count)
        ]
        selected = sample_segments(rows, 100, 13, 14)
        self.assertEqual(selected, sample_segments(list(reversed(rows)), 100, 13, 14))
        self.assertEqual(len({r["segment_id"] for r in selected}), 100)
        counts = Counter(r["phoneme"] for r in selected)
        self.assertEqual(counts["ky"], 1)
        self.assertLessEqual(abs(counts["a"] - counts["i"]), 1)
        self.assertNotEqual(selected, sample_segments(rows, 100, 15, 14))

    def test_rejects_held_out_or_failed_qc(self):
        for split, flags in [("test", []), ("train", ["low_rms"])]:
            with self.assertRaises(AssertionError):
                sample_segments(
                    [{"source_split": split, "quality_flags": flags}], 1, 13, 14
                )
        with self.assertRaises(ValueError):
            sample_segments([], 100, 13, 14)


if __name__ == "__main__":
    unittest.main()
