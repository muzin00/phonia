"""Check that duration controls preserve source-frame budgets and provenance."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison_inputs import (
    describe_audio,
    effective_window,
    matched_center_window,
    union_frames,
)
from smoke import BASE


class ComparisonGeometryTests(unittest.TestCase):
    def test_phase3_center_crop_and_source_boundary_clip(self):
        self.assertEqual(effective_window(100, 10101, 12000), (2100, 8100))
        self.assertEqual(effective_window(100, 1100, 10000, margin=480), (0, 1580))
        self.assertEqual(effective_window(9100, 9900, 10000, margin=480), (8620, 10000))

    def test_overlapping_context_counts_unique_frames_once(self):
        intervals = [(100, 300), (200, 400), (220, 250), (400, 500)]
        self.assertEqual(union_frames(intervals), 400)
        self.assertEqual(union_frames(list(reversed(intervals))), 400)
        self.assertEqual(union_frames([]), 0)

    def test_matched_window_uses_exact_budget_and_never_pads(self):
        for frames, budget in ((10000, 2000), (10001, 1999), (720, 720), (721, 1)):
            start, end = matched_center_window(frames, budget)
            self.assertEqual(end - start, budget)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, frames)
        self.assertIsNone(matched_center_window(10000, 0))
        with self.assertRaisesRegex(ValueError, "exceeds source"):
            matched_center_window(720, 721)

    def test_duration_is_matched_per_source_and_overlap_is_visible(self):
        sources = {
            "a.wav": {"frame_count": 10000, "source_sha256": "sha-a"},
            "b.wav": {"frame_count": 10000, "source_sha256": "sha-b"},
        }
        anchors = [
            {
                "segment_id": identity,
                "source_file": source,
                "source_sha256": sources[source]["source_sha256"],
                "start_frame": start,
                "end_frame": start + 1000,
            }
            for identity, source, start in (
                ("first", "a.wav", 1000),
                ("second", "a.wav", 2400),
                ("other-source", "b.wav", 1000),
            )
        ]
        result = describe_audio(anchors, sources)
        self.assertEqual(result["unique_frames"]["vowel_exact"], 3000)
        self.assertEqual(result["unique_frames"]["vowel_context20"], 5320)
        self.assertEqual(result["processed_frames"]["vowel_context20"], 5880)
        self.assertEqual(result["whole_source_frames"], 20000)
        for source in result["sources"].values():
            for method, (start, end) in source["ecapa_duration_windows"].items():
                self.assertEqual(end - start, source["budgets"][method])

    def test_long_anchor_can_lose_all_requested_context_to_crop(self):
        sources = {"a.wav": {"frame_count": 20000, "source_sha256": "sha"}}
        result = describe_audio(
            [
                {
                    "segment_id": "long",
                    "source_file": "a.wav",
                    "source_sha256": "sha",
                    "start_frame": 2000,
                    "end_frame": 10000,
                }
            ],
            sources,
        )
        self.assertEqual(result["context_changed_anchor_count"], 0)
        self.assertEqual(result["retained_context_frames_including_repeated_frames"], 0)
        self.assertEqual(result["unique_frames"]["vowel_exact"], 6000)
        self.assertEqual(result["unique_frames"]["vowel_context20"], 6000)
        self.assertEqual(result["context_cropped_anchor_count"], 1)

    def test_invalid_geometry_duplicate_anchor_and_source_identity(self):
        for start, end, frames in ((-1, 1000, 2000), (100, 100, 2000), (0, 2001, 2000)):
            with self.assertRaises(ValueError):
                effective_window(start, end, frames)
        source = {"a.wav": {"frame_count": 10000, "source_sha256": "sha"}}
        anchor = {
            "segment_id": "one",
            "source_file": "a.wav",
            "source_sha256": "sha",
            "start_frame": 1000,
            "end_frame": 2000,
        }
        with self.assertRaisesRegex(ValueError, "duplicate anchor"):
            describe_audio([anchor, anchor], source)
        with self.assertRaisesRegex(ValueError, "checksum identity"):
            describe_audio([{**anchor, "source_sha256": "changed"}], source)

    def test_archived_protocol_preserves_original_duration_pairs(self):
        protocol = json.loads(
            (BASE / "archive/v1/config/comparison-protocol.json").read_text()
        )
        methods = protocol["methods"]
        self.assertEqual(len({method["id"] for method in methods}), 5)
        self.assertEqual(
            sum(method["family"] == "source_matched" for method in methods), 3
        )
        self.assertEqual(
            {
                method["paired_with"]
                for method in methods
                if method["family"] == "duration_control"
            },
            {"vowel_exact", "vowel_context20"},
        )
        self.assertEqual(protocol["context"]["margin_frames_per_side"], 480)
        self.assertFalse(protocol["context"]["retrain"])


if __name__ == "__main__":
    unittest.main()
