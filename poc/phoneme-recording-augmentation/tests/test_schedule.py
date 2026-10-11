import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as e


class ScheduleTests(unittest.TestCase):
    def test_positive_pairs_have_same_speaker_and_phone_with_distinct_real_segments(
        self,
    ):
        rows = [
            {"speaker_id": speaker, "phoneme": phone}
            for speaker in ["s1", "s2"]
            for phone in ["a", "i"]
            for _ in range(3)
        ]
        config = {
            "updates": 5,
            "conditions": ["clean", "gain", "noise", "band", "reverb", "noise10"],
        }
        design = {"training_speakers": ["s1", "s2"], "phones": ["a", "i"]}
        ids, classes, valid, views = e.schedule(config, design, rows, 42)
        self.assertTrue((views[:, ::2] == 0).all())
        self.assertTrue(((views[:, 1::2] >= 1) & (views[:, 1::2] <= 5)).all())
        for u in range(5):
            for i in range(0, 100, 2):
                self.assertEqual(valid[u, i], valid[u, i + 1])
                if valid[u, i]:
                    self.assertNotEqual(ids[u, i], ids[u, i + 1])
                    self.assertEqual(rows[ids[u, i]], rows[ids[u, i + 1]])
                    self.assertEqual(classes[u, i], classes[u, i + 1])
        repeated = e.schedule(config, design, rows, 42)
        for a, b in zip((ids, classes, valid, views), repeated, strict=True):
            e.np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
