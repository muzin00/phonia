"""Check waveform corruption and that augmentation cannot reveal trial labels."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shared as s
from augmentation import augmented_batch, make_schedule, noisy_clip


class AugmentationTests(unittest.TestCase):
    def test_waveform_noise_has_requested_snr_and_is_reproducible(self):
        pcm = s.np.sin(s.np.arange(2400) * 0.03) * 0.05
        first, snr, clipped = noisy_clip(pcm, 1, 10)
        second, _, _ = noisy_clip(pcm, 1, 10)
        other, _, _ = noisy_clip(pcm, 2, 10)
        self.assertTrue(s.np.array_equal(first, second))
        self.assertFalse(s.np.array_equal(first, other))
        self.assertAlmostEqual(snr, 10, places=5)
        self.assertEqual(clipped, 0)

    def test_augmentation_is_shared_across_genuine_impostor_and_keeps_vowels(self):
        s.torch.set_num_threads(1)
        clean = {
            "enrollment": s.torch.randn(3, 8, 128),
            "emeta": s.torch.zeros(3, 8, 4),
            "emask": s.torch.ones(3, 8, dtype=s.torch.bool),
            "queries": s.torch.zeros(2, 8, 128),
            "qmeta": s.torch.zeros(2, 8, 4),
            "qmask": s.torch.ones(2, 8, dtype=s.torch.bool),
        }
        noisy = {"queries": s.torch.ones(2, 8, 128), "qmeta": s.torch.ones(2, 8, 4)}
        data = {
            "speakers": ["a", "b", "c"],
            "queries": [{"speaker_id": "a"}, {"speaker_id": "b"}],
        }
        pairs = {
            "queries": s.np.array([[0, 1]]),
            "negative_claims": s.np.array([[2, 2]]),
        }
        augmentation = {
            "modes": s.np.array([[1, 3]]),
            "keep": s.np.array([[[True] * 8, [True] * 5 + [False] * 3]]),
        }
        features, _, mask, labels = augmented_batch(
            clean, noisy, data, pairs, augmentation, 0
        )
        s.torch.testing.assert_close(features[:2, :, 128:256], features[2:, :, 128:256])
        self.assertTrue(bool((features[:, :, 128:256] == 1).all()))
        s.torch.testing.assert_close(mask[:2], mask[2:])
        self.assertTrue(bool(mask[:, :5].all()))
        self.assertFalse(bool(mask[1, 5:].any()))
        self.assertTrue(bool((clean["queries"] == 0).all()))
        s.torch.testing.assert_close(labels, s.torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_four_modes_keep_vowels_and_only_missing_modes_drop_consonants(self):
        settings = s.read_json(s.CONFIG)["augmentation"]
        schedule = make_schedule(
            {"updates": 1000, "queries_per_update": 8}, settings, 4, 36
        )
        self.assertTrue(schedule["keep"][..., :5].all())
        self.assertTrue(schedule["keep"][schedule["modes"] < 2].all())
        self.assertFalse(schedule["keep"][schedule["modes"] >= 2][..., 5:].all())
        fractions = s.np.bincount(schedule["modes"].ravel(), minlength=4) / 8000
        self.assertTrue(((fractions > 0.2) & (fractions < 0.3)).all())


if __name__ == "__main__":
    unittest.main()
