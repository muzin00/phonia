"""FAR caps retain missing trials and ties; calibration never reads test rows."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnose as d


def records(impostors, genuine):
    return [
        {"is_genuine": flag, "score": value, "role": "verification"}
        for flag, values in ((False, impostors), (True, genuine))
        for value in values
    ]


class DiagnosticTests(unittest.TestCase):
    def test_ties_are_not_split_to_hit_the_far_cap(self):
        rows = records([0.9, 0.9, 0.8, None], [0.95, 0.85, None])
        point = d.empirical_cap_threshold(rows, 0.25)
        self.assertGreater(point, 0.9)
        metrics = d.s.operating(rows, point)
        self.assertEqual(metrics["far"], 0)
        self.assertEqual(metrics["frr"], 2 / 3)

    def test_missing_impostors_stay_in_denominator(self):
        rows = records([0.9, 0.8, None, None], [0.95, 0.85, None])
        point = d.empirical_cap_threshold(rows, 0.25)
        self.assertGreater(point, 0.8)
        self.assertLess(point, 0.9)
        metrics = d.s.operating(rows, point)
        self.assertEqual(metrics["far"], 0.25)
        self.assertEqual(metrics["scored_only_error_rates"]["far"], 0.5)
        self.assertEqual(metrics["frr"], 1 / 3)
        self.assertEqual(point, d.empirical_cap_threshold(list(reversed(rows)), 0.25))

    def test_empty_or_fully_permitted_impostors(self):
        with self.assertRaises(ValueError):
            d.empirical_cap_threshold(records([], [0.9]), 0.1)
        self.assertEqual(
            d.empirical_cap_threshold(records([None], [0.9]), 0), -math.inf
        )
        self.assertEqual(d.empirical_cap_threshold(records([0.7], [0.9]), 1), -math.inf)

    def test_calibration_only_loads_validation_and_freezes_all_points(self):
        rows = records([0.9, 0.8, 0.7], [0.95, 0.75])
        config = {
            "seeds": [1],
            "arms": ["arm"],
            "validation_scored_far_targets": [0.001, 0.002, 0.005, 0.01],
            "quantiles": [0.5, 0.99],
        }
        point, _ = d.s.far_target_threshold(*d.scored_arrays(rows), 0.01)
        original = {
            "seeds": {
                "1": {
                    "thresholds": {
                        c: {"arm": d.s.json_threshold(point)} for c in d.s.CORPORA
                    }
                }
            }
        }
        calls = []

        def load(config, seed, corpus, arm, split):
            self.assertEqual(split, "validation")
            calls.append((seed, corpus, arm, split))
            return rows

        with tempfile.TemporaryDirectory() as directory, patch.object(d, "load", load):
            run = Path(directory)
            (run / "design-freeze.json").write_text("{}")
            d.calibrate(config, run, original)
            frozen = d.s.read(run / "validation-freeze.json")
            self.assertFalse(frozen["test_used_to_choose_thresholds"])
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(frozen["thresholds"]["1"]["jvs"]["arm"]), 4)


if __name__ == "__main__":
    unittest.main()
