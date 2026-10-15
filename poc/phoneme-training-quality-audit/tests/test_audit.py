import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit import interval_measures


class MeasurementsTests(unittest.TestCase):
    def test_interval_measurements_match_direct_slices(self):
        pcm = np.random.default_rng(42).integers(-32768, 32768, 10000, dtype=np.int16)
        starts, ends = np.array([0, 7, 200, 5000]), np.array([720, 4000, 8000, 10000])
        values = interval_measures(pcm, starts, ends)
        for i, (s, e) in enumerate(zip(starts, ends)):
            x = pcm[s:e].astype(np.float64) / 32768
            self.assertAlmostEqual(
                values["measured_rms_dbfs"][i], 10 * np.log10(np.mean(x * x)), places=10
            )
            self.assertAlmostEqual(
                values["dc_removed_rms_dbfs"][i],
                10 * np.log10(np.mean((x - x.mean()) ** 2)),
                places=10,
            )
            self.assertAlmostEqual(
                values["near_clip_fraction"][i], np.mean(np.abs(x) >= 0.999), places=10
            )

    def test_dc_can_pass_raw_rms_gate_with_no_ac_signal(self):
        values = interval_measures(
            np.full(1000, 1024, np.int16), np.array([0]), np.array([1000])
        )
        self.assertGreater(values["measured_rms_dbfs"][0], -50)
        self.assertLess(values["dc_removed_rms_dbfs"][0], -100)

    def test_negative_pcm_rail_does_not_overflow(self):
        pcm = np.array([-32768, 32767, 0, 0], np.int16)
        values = interval_measures(pcm, np.array([0]), np.array([4]))
        self.assertEqual(values["rail_fraction"][0], 0.5)
        self.assertEqual(values["near_clip_fraction"][0], 0.5)
        self.assertEqual(values["zero_fraction"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
