"""Check that duration conditions share recordings and cannot use outside audio."""

import copy
import json
import sys
import unittest
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison_inputs import describe_audio
from smoke import BASE
from utterance_windows import describe_query_window, recording_window


class QueryWindowTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "source_file": "source.wav",
            "source_sha256": "sha",
            "frame_count": 48000,
            "speaker_id": "speaker",
            "split": "validation",
            "evaluation_role": "verification",
        }
        self.query = {
            "query_id": "query",
            "source_file": "source.wav",
            "source_sha256": "sha",
            "speaker_id": "speaker",
            "split": "validation",
            "role": "verification",
            "segments": [
                {
                    "segment_id": vowel,
                    "vowel": vowel,
                    "source_file": "source.wav",
                    "source_sha256": "sha",
                    "start_frame": 12000 + index * 3000,
                    "end_frame": 12720 + index * 3000,
                }
                for index, vowel in enumerate("aiueo")
            ],
        }

    def test_duration_caps_are_nested_with_shared_center(self):
        for frames in (240001, 240000, 48001):
            windows = [recording_window(frames, cap) for cap in (1, 2, 3, 5, None)]
            for inner, outer in pairwise(windows):
                self.assertLessEqual(outer[0], inner[0])
                self.assertLessEqual(inner[1], outer[1])
            self.assertEqual(windows[-1], (0, frames))
        self.assertEqual(recording_window(240001, 1), (108000, 132000))

    def test_short_recording_is_not_padded_and_shares_cache_identity(self):
        full = describe_query_window(self.query, self.source, None)
        capped = describe_query_window(self.query, self.source, 5)
        self.assertEqual(capped["recording_frames"], [0, 48000])
        self.assertTrue(capped["source_shorter_than_cap"])
        self.assertEqual(capped["actual_recording_seconds"], 2)
        self.assertEqual(capped["window_id"], full["window_id"])
        self.assertNotEqual(capped["maximum_seconds"], full["maximum_seconds"])

    def test_context_is_clipped_to_recording_and_complete_vowels_retained(self):
        limited = describe_query_window(self.query, self.source, 1)
        self.assertTrue(limited["complete_five_vowels"])
        self.assertEqual(limited["counts"], {vowel: 1 for vowel in "aiueo"})
        first = limited["anchors"][0]
        self.assertEqual(first["exact_frames"], [12000, 12720])
        self.assertEqual(first["context_frames"], [12000, 13200])
        for anchor in limited["anchors"]:
            for key in ("exact_frames", "context_frames"):
                self.assertGreaterEqual(anchor[key][0], 12000)
                self.assertLessEqual(anchor[key][1], 36000)

    def test_partial_core_is_excluded_without_crop_rescue(self):
        query = copy.deepcopy(self.query)
        query["segments"][0]["start_frame"] = 11000
        query["segments"][0]["end_frame"] = 21000
        limited = describe_query_window(query, self.source, 1)
        self.assertFalse(limited["complete_five_vowels"])
        self.assertEqual(limited["counts"]["a"], 0)
        self.assertEqual(limited["excluded_anchor_ids"], ["a"])

    def test_full_window_matches_existing_vowel_geometry_and_order_independent(self):
        full = describe_query_window(self.query, self.source, None)
        old = describe_audio(self.query["segments"], {"source.wav": self.source})
        for method in ("vowel_exact", "vowel_context20"):
            self.assertEqual(
                full["used_audio"][method]["unique_frames"],
                old["unique_frames"][method],
            )
            self.assertEqual(
                full["used_audio"][method]["processed_frames"],
                old["processed_frames"][method],
            )
        query = copy.deepcopy(self.query)
        query["segments"].reverse()
        self.assertEqual(full, describe_query_window(query, self.source, None))

    def test_invalid_identity_intervals_and_cap_rejected(self):
        for frames, cap in ((0, 1), (48000, 0), (48000, 1.5), (48000, True)):
            with self.assertRaises(ValueError):
                recording_window(frames, cap)
        query = copy.deepcopy(self.query)
        query["source_sha256"] = "modified"
        with self.assertRaisesRegex(ValueError, "identity"):
            describe_query_window(query, self.source, 1)
        query = copy.deepcopy(self.query)
        query["segments"][0]["end_frame"] = 48001
        with self.assertRaisesRegex(ValueError, "interval"):
            describe_query_window(query, self.source, 1)

    def test_v2_protocol_uses_three_methods_and_seconds_of_query_recording(self):
        protocol = json.loads((BASE / "config/comparison-protocol.json").read_text())
        self.assertEqual(protocol["protocol_version"], "2.0.0")
        self.assertEqual(
            {method["id"] for method in protocol["methods"]},
            {"vowel_exact", "vowel_context20", "ecapa_whole"},
        )
        self.assertEqual(
            [
                condition["maximum_seconds"]
                for condition in protocol["query_windows"]["conditions"]
            ],
            [None, 1, 2, 3, 5],
        )
        self.assertEqual(
            protocol["query_windows"]["scope"], "query_only_enrollment_fixed"
        )
        self.assertIn(
            "query_window_condition_id",
            protocol["queries_and_trials"]["condition_score_id"],
        )
        self.assertIn("query_window_condition_id", protocol["thresholds"]["scope"])


if __name__ == "__main__":
    unittest.main()
