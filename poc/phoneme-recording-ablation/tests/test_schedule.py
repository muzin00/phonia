import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as e


class ExposureTests(unittest.TestCase):
    def test_leave_one_out_preserves_all_other_exposures(self):
        views = e.np.array([[0, 1, 0, 2, 0, 3, 0, 4, 0, 5]])
        for k in range(1, 6):
            drop = e.arm_views(views, f"drop{k}")
            e.np.testing.assert_array_equal(drop[views != k], views[views != k])
            self.assertTrue((drop[views == k] == 0).all())
            single = e.arm_views(views, f"single{k}")
            self.assertTrue((single[:, ::2] == 0).all())
            self.assertTrue((single[:, 1::2] == k).all())
        e.np.testing.assert_array_equal(views, e.arm_views(views, "augmented"))
        self.assertTrue((e.arm_views(views, "clean") == 0).all())
        self.assertEqual(views.sum(), 15)

    def test_pairs_and_schedule_replay(self):
        rows = [
            {"speaker_id": sp, "phoneme": p}
            for sp in ["s1", "s2"]
            for p in ["a", "i"]
            for _ in range(3)
        ]
        config = {"updates": 5, "conditions": list(range(6))}
        design = {"training_speakers": ["s1", "s2"], "phones": ["a", "i"]}
        indices, labels, valid, views = e.schedule(config, design, rows, 42)
        for u in range(5):
            for i in range(0, 100, 2):
                self.assertEqual(valid[u, i], valid[u, i + 1])
                if valid[u, i]:
                    self.assertNotEqual(indices[u, i], indices[u, i + 1])
                    self.assertEqual(rows[indices[u, i]], rows[indices[u, i + 1]])
                    self.assertEqual(labels[u, i], labels[u, i + 1])
        for a, b in zip(
            (indices, labels, valid, views),
            e.schedule(config, design, rows, 42),
            strict=True,
        ):
            e.np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
