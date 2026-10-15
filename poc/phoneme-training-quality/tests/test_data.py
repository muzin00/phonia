import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import subset_bank, subset_pool, validate_bank


class SubsetTests(unittest.TestCase):
    def fixture(self):
        phones = ["a", "i", "u", "e", "o"]
        owners = np.repeat([0, 1], 15)
        phonemes = np.tile(np.arange(5), 6)
        groups = [
            np.flatnonzero((owners == s) & (phonemes == p))
            for s in range(2)
            for p in range(5)
        ]
        arrays = {
            "indices": np.concatenate(groups),
            "offsets": np.arange(0, 31, 3),
            "group_speaker": np.repeat([0, 1], 5),
            "group_phone": np.tile(np.arange(5), 2),
            "segment_speaker": owners,
            "segment_phone": phonemes,
            "recording": np.repeat(np.arange(6), 5),
            "metadata": np.zeros((30, 2)),
        }
        info = {"phones": phones, "speakers": ["s1", "s2"], "segments": 30}
        bank = {
            "speakers": info["speakers"],
            "profiles": {
                s: {p: [j * 15 + i] for i, p in enumerate(phones)}
                for j, s in enumerate(info["speakers"])
            },
            "queries": [
                {
                    "query_id": f"{s}/{k}",
                    "speaker_id": s,
                    "groups": {p: [j * 15 + k * 5 + i] for i, p in enumerate(phones)},
                }
                for j, s in enumerate(info["speakers"])
                for k in (1, 2)
            ],
        }
        return info, arrays, bank

    def test_partial_query_removed_and_index_mapping_remains_correct(self):
        info, arrays, bank = self.fixture()
        keep = np.ones(30, bool)
        keep[5] = False
        stats, filtered, ids, mapping = subset_pool(info, arrays, keep)
        mapped, dropped = subset_bank(bank, mapping, info["phones"])
        self.assertEqual(len(mapped["queries"]), 3)
        self.assertEqual(dropped["queries_missing_vowels"], 1)
        self.assertEqual(mapping[5], -1)
        np.testing.assert_array_equal(mapping[ids], np.arange(29))
        validate_bank(mapped, stats, filtered)

    def test_empty_speaker_and_profile_are_removed_without_wrong_owner(self):
        info, arrays, bank = self.fixture()
        keep = np.arange(30) >= 15
        stats, filtered, _, mapping = subset_pool(info, arrays, keep)
        mapped, _ = subset_bank(bank, mapping, info["phones"])
        self.assertEqual(stats["speakers"], ["s2"])
        self.assertEqual(mapped["speakers"], ["s2"])
        self.assertTrue(np.all(filtered["segment_speaker"] == 0))
        validate_bank(mapped, stats, filtered)

    def test_missing_enrollment_vowel_removes_all_owner_queries(self):
        info, arrays, bank = self.fixture()
        keep = np.ones(30, bool)
        keep[0] = False
        stats, filtered, _, mapping = subset_pool(info, arrays, keep)
        mapped, dropped = subset_bank(bank, mapping, info["phones"])
        self.assertEqual(mapped["speakers"], ["s2"])
        self.assertEqual(dropped["queries_ineligible_profile"], 2)
        validate_bank(mapped, stats, filtered)

    def test_enrollment_query_recording_leak_is_detected(self):
        info, arrays, bank = self.fixture()
        arrays["recording"][5] = arrays["recording"][0]
        with self.assertRaisesRegex(ValueError, "recording overlap"):
            validate_bank(bank, info, arrays)

    def test_empty_selection_fails(self):
        info, arrays, _ = self.fixture()
        with self.assertRaises(ValueError):
            subset_pool(info, arrays, np.zeros(30, bool))


if __name__ == "__main__":
    unittest.main()
