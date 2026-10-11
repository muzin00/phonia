"""Enrollment expansion must retain sparse phones and reject query audio leaks."""

import importlib.util
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("all_enrollment_study", BASE / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
from prepare import diverse


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.segments = []
        for phone, count in (("a", 40), ("by", 3)):
            for i in range(count):
                self.segments.append(
                    {
                        "segment_id": f"{phone}-{i}",
                        "speaker_id": "s1",
                        "phoneme": phone,
                        "source_file": f"enroll-{i % 4}.wav",
                        "source_sha256": f"hash-{i % 4}",
                        "cache_index": len(self.segments),
                        "role": "enrollment",
                        "split": "test",
                        "quality_flags": [],
                    }
                )
        self.original = {
            "split": "test",
            "speakers": ["s1"],
            "universally_registered": ["a", "by"],
            "queries": [{"source_file": "query.wav"}],
            "profiles": {
                "s1": {
                    p: [
                        r["cache_index"]
                        for r in diverse(
                            [r for r in self.segments if r["phoneme"] == p], 10, 7
                        )
                    ]
                    for p in ("a", "by")
                }
            },
        }

    def test_real_nested_enrollment_keeps_sparse_phone_and_baseline(self):
        data = [
            study.expanded_inputs(self.original, self.segments, n, 7)
            for n in (10, 20, 30)
        ]
        self.assertEqual(data[0], self.original)
        self.assertEqual([len(d["profiles"]["s1"]["a"]) for d in data], [10, 20, 30])
        for d in data:
            self.assertEqual(len(d["profiles"]["s1"]["by"]), 3)
            self.assertEqual(
                len(set(d["profiles"]["s1"]["a"])), len(d["profiles"]["s1"]["a"])
            )
            self.assertEqual(d["queries"], self.original["queries"])
        self.assertEqual(
            data[2]["profiles"]["s1"]["a"][:20], data[1]["profiles"]["s1"]["a"]
        )

    def test_enrollment_rejects_query_audio_by_path_or_hash(self):
        query = {
            **self.segments[0],
            "role": "verification",
            "source_file": "query.wav",
            "source_sha256": "query-hash",
        }
        for key, value in (
            ("source_file", "query.wav"),
            ("source_sha256", "query-hash"),
        ):
            with self.subTest(key=key):
                leaked = [
                    {**r, key: value} if i == 0 else r
                    for i, r in enumerate(self.segments)
                ]
                with self.assertRaisesRegex(ValueError, "audio overlap"):
                    study.expanded_inputs(self.original, [*leaked, query], 20, 7)


if __name__ == "__main__":
    unittest.main()
