"""The weighted scorer must ignore padding and preserve phone-label symmetry."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from backend import Fusion, pair_features
from study import training_batch


class BackendTests(unittest.TestCase):
    def test_genuine_and_impostor_share_the_same_phone_mask(self):
        arrays = {
            "enrollment": torch.randn(2, 6, 128),
            "emeta": torch.randn(2, 6, 4),
            "emask": torch.tensor([[True] * 6, [True] * 5 + [False]]),
            "queries": torch.randn(1, 6, 128),
            "qmeta": torch.randn(1, 6, 4),
            "qmask": torch.ones(1, 6, dtype=torch.bool),
        }
        data = {"speakers": ["a", "b"], "queries": [{"speaker_id": "a"}]}
        schedule = {"queries": np.array([[0]]), "negative_claims": np.array([[1]])}
        _, _, mask, labels = training_batch(arrays, data, schedule, 0)
        torch.testing.assert_close(mask[0], mask[1])
        self.assertFalse(bool(mask[:, 5].any()))
        torch.testing.assert_close(labels, torch.tensor([1.0, 0.0]))

    def test_masked_values_and_phone_order_do_not_change_score(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        for kind in ("mlp", "transformer"):
            model = Fusion(kind, phone_count=8).eval()
            torch.nn.init.normal_(model.output.weight)
            features = torch.randn(3, 8, 520)
            components = torch.randn(3, 8)
            mask = torch.tensor([[True] * 5 + [False] * 3] * 3)
            score, weights = model(features, components, mask)
            changed = features.clone()
            changed[~mask] = 100000
            changed_components = components.clone()
            changed_components[~mask] = -100000
            altered, _ = model(changed, changed_components, mask)
            torch.testing.assert_close(score, altered)
            order = torch.tensor([7, 2, 0, 6, 5, 1, 4, 3])
            permuted, _ = model(
                features[:, order], components[:, order], mask[:, order], order
            )
            torch.testing.assert_close(score, permuted, rtol=1e-5, atol=1e-6)
            self.assertTrue((weights[~mask] == 0).all())
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))

    def test_initial_score_is_equal_mean_and_training_has_real_gradients(self):
        torch.set_num_threads(1)
        for kind in ("mlp", "transformer"):
            model = Fusion(kind, phone_count=6)
            enrollment, query = torch.randn(4, 6, 128), torch.randn(4, 6, 128)
            f, c = pair_features(
                enrollment, torch.randn(4, 6, 4), query, torch.randn(4, 6, 4)
            )
            mask = torch.tensor([[True] * 5 + [False]] * 4)
            score, _ = model(f, c, mask)
            torch.testing.assert_close(score, c[:, :5].mean(dim=1))
            loss = model.loss(f, c, mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))
            loss.backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
            with self.assertRaisesRegex(ValueError, "shared phone"):
                model(f, c, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
