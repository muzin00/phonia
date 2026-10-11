import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acoustics as a
import numpy as np


class AcousticTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(a.RATE) / a.RATE
        self.x = (np.sin(2 * np.pi * 1000 * t) * 0.1).astype(np.float32)

    def test_gain_changes_level_preserves_energy_contrast_and_band(self):
        before = a.quality(self.x)
        y, _ = a.transform(self.x, "gain-minus12", 0)
        after = a.quality(y)
        self.assertAlmostEqual(after["rms_dbfs"] - before["rms_dbfs"], -12, delta=0.01)
        self.assertAlmostEqual(
            after["energy_contrast_db"], before["energy_contrast_db"], delta=0.01
        )
        self.assertEqual(after["rolloff95_hz"], before["rolloff95_hz"])

    def test_noise_snr_and_nested_realizations(self):
        y10, d10 = a.transform(self.x, "white-snr10", 7)
        y20, d20 = a.transform(self.x, "white-snr20", 7)
        self.assertAlmostEqual(d10["effective_added_noise_snr_db"], 10, delta=0.01)
        self.assertAlmostEqual(d20["effective_added_noise_snr_db"], 20, delta=0.01)
        self.assertGreater(np.corrcoef(y10 - self.x, y20 - self.x)[0, 1], 0.999)
        np.testing.assert_array_equal(y10, a.transform(self.x, "white-snr10", 7)[0])

    def test_band_limit_preserves_overall_rms_rejects_high_frequency(self):
        t = np.arange(a.RATE) / a.RATE
        x = self.x + 0.1 * np.sin(2 * np.pi * 6000 * t)
        y, _ = a.transform(x, "band-300-3400", 0)
        self.assertLess(a.quality(y)["high_frequency_fraction"], 0.001)
        self.assertAlmostEqual(
            a.quality(y)["rms_dbfs"], a.quality(x)["rms_dbfs"], delta=0.01
        )

    def test_silence_finite_and_clipping_count(self):
        result = a.quality(np.zeros(a.RATE))
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertEqual(result["active_seconds"], 0)
        self.assertEqual(a.quality(np.ones(a.RATE))["near_clip_fraction"], 1)


if __name__ == "__main__":
    unittest.main()
