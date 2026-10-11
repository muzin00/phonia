"""Verify full-inventory retention and real sampling exposure parity."""

import importlib.util
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("all_phone_study", BASE / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
from common import VOWELS, np
from learner import schedule


class AllTrainingTests(unittest.TestCase):
    def test_full_inventory_keeps_rare_trainable_phones_and_records_unavailable(self):
        extras = ["N", "cl", "dy", "v", "ty"]
        report = {
            "candidates": extras,
            "phone_support": {p: {"trainable": p != "ty"} for p in [*VOWELS, *extras]},
        }
        phones, unavailable = study.all_trainable(report)
        self.assertEqual(set(phones), set(VOWELS) | {"N", "cl", "dy", "v"})
        self.assertEqual(unavailable, ["ty"])

    def test_scaled_budget_preserves_each_vowels_actual_sample_sequence(self):
        phones = [*VOWELS, *[f"c{i}" for i in range(31)]]
        speakers = {f"s{i}": i for i in range(12)}
        groups = {}
        for p_i, phone in enumerate(phones):
            for s_i, speaker in enumerate(speakers):
                first = 100 * p_i + 5 * s_i
                groups[speaker, phone] = [first, first + 1, first + 2]
        baseline = schedule(groups, list(VOWELS), 30, 23, speakers)
        all_phones = schedule(groups, phones, 216, 23, speakers)
        for before, after in zip(baseline, all_phones, strict=True):
            before, after = before.reshape(-1, 20), after.reshape(-1, 20)
            for i in range(5):
                np.testing.assert_array_equal(before[i::5], after[i::36])
        self.assertEqual(int(all_phones[2].sum()), 21600)


if __name__ == "__main__":
    unittest.main()
