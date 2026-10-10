"""Check matched tokens, quiet-pair rejection and diagnostic support guards."""

import copy
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import diagnostics


def row(phone, source, first, last):
    return {
        "segment_id": phone,
        "vowel": phone,
        "source_file": str(source),
        "source_sha256": phone,
        "start_frame": first,
        "end_frame": last,
    }


def audio(path, samples):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class DiagnosticTest(unittest.TestCase):
    def test_matching_preserves_stop_end_and_equal_token_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio(root / "t.wav", [0] * 1100 + [5000] * 100)
            audio(root / "d.wav", [3000] * 720)
            rows = diagnostics.matched_tokens(
                row("t", root / "t.wav", 0, 1200),
                row("d", root / "d.wav", 0, 720),
                {"minimum_consonant_rms_dbfs": -50},
            )
            self.assertEqual(
                [(r["start_frame"], r["end_frame"]) for r in rows],
                [(480, 1200), (0, 720)],
            )

    def test_quiet_token_rejects_both_sides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio(root / "t.wav", [5000] * 1200)
            audio(root / "d.wav", [0] * 720)
            self.assertIsNone(
                diagnostics.matched_tokens(
                    row("t", root / "t.wav", 0, 1200),
                    row("d", root / "d.wav", 0, 720),
                    {"minimum_consonant_rms_dbfs": -50},
                )
            )

    def test_diagnostic_time_count_and_speaker_support_guards(self):
        config = {"diagnostics": {"pairs": {"td": ["t", "d"]}}}
        plans = {
            p: [
                {
                    "original_segment_id": p,
                    "vowel": p,
                    "source_sha256": p,
                    "start_frame": 0,
                    "end_frame": 720,
                }
            ]
            for p in ("t", "d")
        }
        data = {
            "speakers": ["s"],
            "studies": {
                "td": {
                    "profiles": {"s": {"conditions": plans, "budget_frames": 720}},
                    "queries": [
                        {
                            "query_id": role,
                            "speaker_id": "s",
                            "role": role,
                            "conditions": {
                                p: [{**r, "source_sha256": "query-" + p}]
                                for p, rows in plans.items()
                                for r in rows
                            },
                            "budget_frames": 720,
                        }
                        for role in diagnostics.study.ROLES
                    ],
                }
            },
        }
        self.assertEqual(diagnostics.validate_inputs(data, config), 6)
        for mutation in ("time", "count", "speaker", "leak"):
            broken = copy.deepcopy(data)
            item = broken["studies"]["td"]
            if mutation == "time":
                item["queries"][0]["conditions"]["t"][0]["end_frame"] += 1
            elif mutation == "count":
                item["queries"][0]["conditions"]["t"].append(
                    item["queries"][0]["conditions"]["t"][0]
                )
            elif mutation == "speaker":
                item["queries"].pop()
            else:
                item["queries"][0]["conditions"]["t"][0]["source_sha256"] = "t"
            with self.assertRaises(ValueError):
                diagnostics.validate_inputs(broken, config)


if __name__ == "__main__":
    unittest.main()
