from __future__ import annotations

import unittest

from scripts.generate_vowel_units import classify_vowel, extract_vowel_units


def make_record(raw_phonemes: list[str]) -> dict[str, object]:
    return {
        "utterance_id": "sample_001",
        "speaker_id": "speaker_001",
        "text": "sample",
        "reading_kana": "サンプル",
        "raw_phonemes": raw_phonemes,
        "review_status": "pending",
    }


class ClassifyVowelTest(unittest.TestCase):
    def test_classifies_voiced_devoiced_and_long_vowels(self) -> None:
        self.assertEqual(classify_vowel("o"), ("o", 1, False))
        self.assertEqual(classify_vowel("U"), ("u", 1, True))
        self.assertEqual(classify_vowel("o:"), ("o", 2, False))
        self.assertEqual(classify_vowel("oː"), ("o", 2, False))
        self.assertIsNone(classify_vowel("pau"))


class ExtractVowelUnitsTest(unittest.TestCase):
    def test_merges_directly_adjacent_identical_vowels(self) -> None:
        units = extract_vowel_units(make_record(["my", "o", "o", "o", "o", "t", "o"]))

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["normalized_phoneme"], "o")
        self.assertEqual(units[0]["raw_phonemes"], ["o", "o", "o", "o"])
        self.assertEqual(units[0]["source_phoneme_start"], 1)
        self.assertEqual(units[0]["source_phoneme_end"], 5)
        self.assertEqual(units[0]["expected_units"], 4)
        self.assertTrue(units[0]["is_long"])
        self.assertEqual(units[1]["expected_units"], 1)
        self.assertFalse(units[1]["is_long"])

    def test_keeps_different_adjacent_vowels_separate(self) -> None:
        units = extract_vowel_units(make_record(["o", "o", "i"]))

        self.assertEqual([unit["normalized_phoneme"] for unit in units], ["o", "i"])
        self.assertEqual([unit["expected_units"] for unit in units], [2, 1])

    def test_consonant_separates_identical_vowels(self) -> None:
        units = extract_vowel_units(make_record(["o", "n", "o"]))

        self.assertEqual(len(units), 2)
        self.assertEqual([unit["expected_units"] for unit in units], [1, 1])

    def test_preserves_devoicing_metadata(self) -> None:
        units = extract_vowel_units(make_record(["k", "U", "o:"]))

        self.assertTrue(units[0]["is_devoiced"])
        self.assertTrue(units[0]["contains_devoiced"])
        self.assertFalse(units[1]["is_devoiced"])
        self.assertEqual(units[1]["expected_units"], 2)

    def test_assigns_stable_zero_based_indices(self) -> None:
        units = extract_vowel_units(make_record(["a", "t", "i"]))

        self.assertEqual([unit["vowel_index"] for unit in units], [0, 1])
        self.assertEqual(
            [unit["vowel_unit_id"] for unit in units],
            ["sample_001-vowel-000", "sample_001-vowel-001"],
        )


if __name__ == "__main__":
    unittest.main()
