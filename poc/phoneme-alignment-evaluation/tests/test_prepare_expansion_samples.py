from __future__ import annotations

import unittest

from scripts.prepare_expansion_samples import select_stems


class SelectStemsTest(unittest.TestCase):
    def test_selects_sorted_paired_utterances_only(self) -> None:
        files = [
            {"name": "sample_002.wav", "type": "file"},
            {"name": "sample_001.lab", "type": "file"},
            {"name": "sample_003.wav", "type": "file"},
            {"name": "sample_002.lab", "type": "file"},
            {"name": "sample_001.wav", "type": "file"},
        ]

        self.assertEqual(select_stems(files, 2), ["sample_001", "sample_002"])

    def test_rejects_too_few_paired_utterances(self) -> None:
        files = [{"name": "sample_001.wav", "type": "file"}]

        with self.assertRaisesRegex(ValueError, "0 paired"):
            select_stems(files, 1)
