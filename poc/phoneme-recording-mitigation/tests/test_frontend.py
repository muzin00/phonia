import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as e


class FrontendTests(unittest.TestCase):
    def test_gain_restoration_and_no_clipping(self):
        config = e.s.read_json(e.CONFIG)
        x = (0.1 * e.np.sin(2 * e.np.pi * 1000 * e.np.arange(24000) / 24000)).astype(
            e.np.float32
        )
        y, _ = e.normalize(x, config)
        z, _ = e.normalize(x * 0.25, config)
        e.np.testing.assert_allclose(y, z, atol=1 / 32768)
        self.assertAlmostEqual(e.a.quality(y)["active_rms_dbfs"], -25, delta=0.01)
        peak = e.np.ones(24000) * 0.0001
        peak[0] = 1
        value, info = e.normalize(peak, config)
        self.assertLessEqual(float(abs(value).max()), 0.98004)
        self.assertTrue(info["limited"])
        __import__("json").dumps(info, allow_nan=False)

    def test_silence_is_not_amplified(self):
        result, info = e.normalize(e.np.zeros(24000), e.s.read_json(e.CONFIG))
        self.assertTrue(info["silence"])
        self.assertFalse(result.any())

    def test_window_rejects_partial_phone_even_if_center_crop_fits(self):
        segments = [
            {"raw_start_frame": 100, "raw_end_frame": 1000},
            {"raw_start_frame": 24000, "raw_end_frame": 25000},
        ]
        ids, first, last = e.window_indices({"groups": {"a": [0, 1]}}, segments, 1)
        self.assertEqual(ids, [0])
        self.assertEqual((first, last), (100, 24100))


if __name__ == "__main__":
    unittest.main()
