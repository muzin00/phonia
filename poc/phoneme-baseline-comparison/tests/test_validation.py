"""Calibration leakage, ties, missing speakers and streaming regression cases."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import iter_rows, validate_trials, writer
from validation_metrics import calibrate_cell, evaluate_cell, rates


def fixture():
    rows = []
    for speaker in ("s1", "s2"):
        for index in range(2):
            for claimed in ("s1", "s2"):
                genuine = speaker == claimed
                rows.append(
                    {
                        "query_id": f"{speaker}-{index}",
                        "speaker_id": speaker,
                        "claimed_speaker_id": claimed,
                        "split": "validation",
                        "role": "verification",
                        "status": "scored",
                        "is_genuine": genuine,
                        "score": 0.8 if genuine else 0.5,
                    }
                )
    return rows


class CalibrationTests(unittest.TestCase):
    def test_tied_impostors_are_rejected_with_gte_acceptance(self):
        result = calibrate_cell(fixture(), ["s1", "s2"])
        point = result["operating_points"]["far_1pct"]
        self.assertGreater(point["threshold"], 0.5)
        self.assertEqual(point["false_accepts"], 0)
        self.assertEqual(point["false_rejects"], 0)
        self.assertEqual(result["far_resolution"], 0.25)

    def test_cross_text_and_test_are_forbidden_for_calibration(self):
        for field, value in (("role", "cross_text_verification"), ("split", "test")):
            rows = fixture()
            rows[-1][field] = value
            with self.assertRaisesRegex(ValueError, "validation/verification"):
                calibrate_cell(rows, ["s1", "s2"])

    def test_no_score_genuine_stays_in_all_input_denominator(self):
        rows = fixture()
        for row in rows:
            if row["query_id"] == "s1-0":
                row.update(status="no_score", score=None)
        result = rates(rows, 0.7)
        self.assertEqual(result["all_genuine"], 4)
        self.assertEqual(result["genuine"], 3)
        self.assertEqual(result["all_input_frr"], 0.25)
        self.assertEqual(result["frr"], 0.0)
        self.assertEqual(result["no_score_impostor"], 1)

    def test_zero_score_speaker_masks_conditional_metrics_without_dropping_all_input(
        self,
    ):
        threshold = calibrate_cell(fixture(), ["s1", "s2"])
        rows = fixture()
        for row in rows:
            if row["speaker_id"] == "s2":
                row.update(status="no_score", score=None)
        item, curve = evaluate_cell(rows, threshold, ["s1", "s2"])
        self.assertEqual(item["conditional_status"], "not_evaluable")
        self.assertEqual(item["zero_scored_speakers"], ["s2"])
        self.assertIsNone(item["pooled_eer"])
        self.assertEqual(curve["thresholds"], [])
        point = item["operating_points"]["far_1pct"]
        self.assertIsNone(point["far"])
        self.assertIsNone(point["frr"])
        self.assertEqual(point["all_input_frr"], 0.5)
        self.assertEqual(point["all_genuine"], 4)
        self.assertEqual(item["queries"], 4)

    def test_common_support_with_absent_speaker_is_not_pooled_as_full_population(self):
        rows = [r for r in fixture() if r["speaker_id"] == "s1"]
        result = calibrate_cell(rows, ["s1", "s2"])
        self.assertEqual(result["status"], "not_evaluable")
        self.assertIsNone(result["operating_points"])

    def test_cross_text_uses_fixed_threshold_without_retuning(self):
        threshold = calibrate_cell(fixture(), ["s1", "s2"])
        before = copy.deepcopy(threshold)
        cross = fixture()
        for row in cross:
            row.update(role="cross_text_verification", score=0.9)
        item, _ = evaluate_cell(cross, threshold, ["s1", "s2"])
        self.assertEqual(item["operating_points"]["far_1pct"]["far"], 1.0)
        self.assertEqual(threshold, before)

    def test_all_missing_reports_all_input_rejection_and_no_curve(self):
        threshold = calibrate_cell(fixture(), ["s1", "s2"])
        missing = [{**r, "status": "no_score", "score": None} for r in fixture()]
        item, curve = evaluate_cell(missing, threshold, ["s1", "s2"])
        self.assertEqual(item["operating_points"]["far_1pct"]["all_input_frr"], 1.0)
        self.assertEqual(item["operating_points"]["far_1pct"]["all_input_far"], 0.0)
        self.assertIsNone(curve["eer"])


class StorageAndIdentityTests(unittest.TestCase):
    def test_compressed_stream_is_reproducible_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = [
                Path(directory) / name for name in ("one.jsonl.gz", "two.jsonl.gz")
            ]
            for path in (first, second):
                with writer(path) as write:
                    for row in fixture():
                        write(row)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(list(iter_rows(first)), fixture())
            with self.assertRaises(FileExistsError), writer(first):
                pass

    def test_duplicate_mislabeled_missing_trials_are_rejected(self):
        queries = [
            {
                "query_id": "q",
                "speaker_id": "s1",
                "role": "verification",
                "split": "validation",
            }
        ]
        trials = [
            {
                "query_id": "q",
                "speaker_id": "s1",
                "claimed_speaker_id": s,
                "role": "verification",
                "split": "validation",
                "enrollment_count": c,
                "trial_id": f"{s}-{c}",
                "is_genuine": s == "s1",
            }
            for s in ("s1", "s2")
            for c in (1, 5, 10)
        ]
        validate_trials(queries, trials, ["s1", "s2"])
        for broken in (
            trials + [trials[0]],
            trials[:-1],
            [{**r, "is_genuine": False} for r in trials],
        ):
            with self.assertRaises(ValueError):
                validate_trials(queries, broken, ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
