import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import count_math as cm
import study


class ScheduleTests(unittest.TestCase):
    def test_nested_balanced_prespecified_schedule(self):
        phones = list(study.fixed.PHONEMES)
        design = cm.schedule(phones, 20261010)
        self.assertEqual(len(design["conditions"]), 20)
        consonants = set(phones[5:])
        for count, expected in ((7, 2), (10, 5)):
            totals = Counter()
            sets = []
            for key in design["by_count"][str(count)]:
                group = set(design["conditions"][key])
                self.assertEqual(len(group), count)
                self.assertTrue(set(cm.VOWELS) <= group)
                totals.update(group & consonants)
                sets.append(frozenset(group))
            self.assertEqual(len(set(sets)), 9)
            self.assertEqual(totals, Counter({p: expected for p in consonants}))
        for i in range(9):
            self.assertTrue(
                set(design["conditions"][f"p7_o{i}"])
                < set(design["conditions"][f"p10_o{i}"])
            )

    def test_schedule_depends_only_on_inventory_and_seed(self):
        phones = list(study.fixed.PHONEMES)
        self.assertEqual(cm.schedule(phones, 1), cm.schedule(phones, 1))
        self.assertNotEqual(
            cm.schedule(phones, 1)["orders"], cm.schedule(phones, 2)["orders"]
        )
        with self.assertRaises(ValueError):
            cm.schedule(phones[:-1], 1)


class TimeTests(unittest.TestCase):
    def test_sparse_enrollment_is_capped_without_repetition_or_dropping_phone(self):
        config = study.read_json(study.CONFIG)
        rows = [
            {"segment_id": f"{p}-{i}", "vowel": p, "source_file": f"wave-{i}"}
            for p in config["phones"]
            for i in range(3 if p == "z" else 12)
        ]
        selected = study.enrollment_candidates(rows, config)
        counts = Counter(r["vowel"] for r in selected)
        self.assertEqual(counts["z"], 3)
        self.assertTrue(all(counts[p] == 10 for p in config["phones"] if p != "z"))
        self.assertEqual(len(selected), len({r["segment_id"] for r in selected}))

    def test_count_changes_keep_identical_real_time_without_duplicate_tokens(self):
        config = study.read_json(study.CONFIG)
        design = cm.schedule(config["phones"], config["selection_seed"])
        rows = [
            {"segment_id": f"{p}-{i}", "vowel": p, "start_frame": 0, "end_frame": 4800}
            for p in config["phones"]
            for i in range(2)
        ]
        with patch.object(study, "pcm_rms_dbfs", return_value=-20):
            match = study.matched_plans(rows, design["conditions"], 24000, config)
        self.assertEqual(match["budget_frames"], 24000)
        for key, selected in match["conditions"].items():
            self.assertEqual(
                sum(r["end_frame"] - r["start_frame"] for r in selected), 24000
            )
            self.assertEqual(
                {r["vowel"] for r in selected}, set(design["conditions"][key])
            )
            self.assertEqual(
                len(selected), len({r["original_segment_id"] for r in selected})
            )
            self.assertTrue(
                all(720 <= r["end_frame"] - r["start_frame"] <= 4800 for r in selected)
            )

    def test_incomplete_support_and_silent_crops_are_rejected(self):
        config = study.read_json(study.CONFIG)
        design = cm.schedule(config["phones"], config["selection_seed"])
        rows = [
            {"segment_id": p, "vowel": p, "start_frame": 0, "end_frame": 4800}
            for p in config["phones"]
        ]
        self.assertIsNone(
            study.matched_plans(rows[:-1], design["conditions"], 24000, config)
        )
        with patch.object(study, "pcm_rms_dbfs", return_value=-70):
            self.assertIsNone(
                study.matched_plans(rows, design["conditions"], 24000, config)
            )


class FusionTests(unittest.TestCase):
    def test_equal_phone_weight_despite_unequal_number_of_segments(self):
        rows = lambda names: [
            {"segment_id": name, "vowel": name.split("-")[0]} for name in names
        ]
        data = {
            "split": "test",
            "speakers": ["person"],
            "profiles": {
                "person": {"conditions": {"two": rows(["a-enroll", "i-enroll"])}}
            },
            "queries": [
                {
                    "query_id": "query",
                    "speaker_id": "person",
                    "role": "verification",
                    "budget_frames": 2400,
                    "conditions": {"two": rows(["a-q1", "a-q2", "a-q3", "i-q"])},
                }
            ],
        }
        vectors = {
            name: np.array(vector, dtype=float)
            for name, vector in {
                "a-enroll": [1, 0],
                "i-enroll": [0, 1],
                "a-q1": [1, 0],
                "a-q2": [1, 0],
                "a-q3": [1, 0],
                "i-q": [1, 0],
            }.items()
        }
        result = cm.fused_scores(data, vectors)[0]
        self.assertEqual(result["phone_scores"], {"a": 1, "i": 0})
        self.assertEqual(result["score"], 0.5)

    def test_count_mean_is_mean_of_integrated_eers_not_best_combination(self):
        by_count = {str(k): ["a", "b"] for k in (5, 7, 10, 14)}
        cells = {
            "a/verification": {"pooled_eer": 0.1},
            "b/verification": {"pooled_eer": 0.3},
        }
        result = cm.count_means(cells, by_count, "verification")
        self.assertAlmostEqual(result["7"]["mean"], 0.2)
        self.assertEqual(result["7"]["minimum"], 0.1)
        self.assertEqual(result["7"]["maximum"], 0.3)


if __name__ == "__main__":
    unittest.main()
