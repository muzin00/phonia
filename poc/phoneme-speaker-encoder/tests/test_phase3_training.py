"""Model, loss and validation metric contracts for Phase 3."""

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_data.manifest import Segment
from phase3_data.sampling import BalancedSampler
from phase3_train.evaluation import _stratum_values, summarize_trials
from phase3_train.losses import AAMSoftmax, within_vowel_supcon
from phase3_train.metrics import eer_operating_threshold, far_target_threshold, roc_eer
from phase3_train.models import create_encoder
from phase3_train.training import Trainer, TrainSettings, seed_everything

NAMES = (
    "statistics_mlp",
    "tdnn",
    "framewise_cnn",
    "waveform_cnn_k80",
    "waveform_cnn_k240",
    "waveform_cnn_k240_context27",
)


class ModelTests(unittest.TestCase):
    def test_models_accept_short_input_and_ignore_padding(self):
        torch.manual_seed(3)
        for name in NAMES:
            with self.subTest(name=name):
                model = create_encoder(name).eval()
                short, long = (720, 1200) if model.input_kind == "waveform" else (2, 8)
                channels = 1 if model.input_kind == "waveform" else 64
                first = torch.randn(1, channels, short)
                batch = torch.full((2, channels, long + 3), 99.0)
                batch[0, :, :short] = first[0]
                batch[1, :, :long] = torch.randn(channels, long)
                alone_mask = torch.ones(1, short, dtype=torch.bool)
                batch_mask = torch.arange(long + 3)[None, :] < torch.tensor(
                    [[short], [long]]
                )
                with torch.no_grad():
                    alone, sequence, sequence_mask = model(
                        first, alone_mask, return_sequence=True
                    )
                    mixed, mixed_sequence, mixed_mask = model(
                        batch, batch_mask, return_sequence=True
                    )
                    batch[:, :, short:] = -37.0
                    changed = model(batch, batch_mask)
                self.assertEqual(tuple(alone.shape), (1, 128))
                torch.testing.assert_close(alone[0], mixed[0], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(mixed[0], changed[0], atol=0, rtol=0)
                torch.testing.assert_close(
                    sequence[0],
                    mixed_sequence[0, :, : sequence.shape[-1]],
                    atol=1e-6,
                    rtol=1e-5,
                )
                self.assertEqual(sequence_mask.sum().item(), mixed_mask[0].sum().item())
                self.assertAlmostEqual(float(alone.norm()), 1.0, places=5)

    def test_losses_and_microbatch_gradient(self):
        torch.manual_seed(4)
        model = create_encoder("statistics_mlp").eval()
        head = AAMSoftmax(10)
        replica, replica_head = deepcopy(model), deepcopy(head)
        features = torch.randn(100, 64, 3)
        mask = torch.ones(100, 3, dtype=torch.bool)
        labels = torch.tensor(
            [speaker for _ in range(5) for speaker in range(10) for _ in range(2)]
        )
        speakers = [f"s{index}" for index in labels.tolist()]
        embeddings = model(features, mask)
        full = head(embeddings, labels)
        full += (
            sum(
                within_vowel_supcon(
                    embeddings[start : start + 20], speakers[start : start + 20]
                )
                for start in range(0, 100, 20)
            )
            * 0.1
        )
        full.backward()
        grouped_loss = 0.0
        for start in range(0, 100, 20):
            group = replica(features[start : start + 20], mask[start : start + 20])
            loss = (
                replica_head(group, labels[start : start + 20])
                + 0.5 * within_vowel_supcon(group, speakers[start : start + 20])
            ) / 5
            grouped_loss += float(loss.detach())
            loss.backward()
        self.assertAlmostEqual(float(full.detach()), grouped_loss, places=5)
        for original, grouped in zip(
            (*model.parameters(), *head.parameters()),
            (*replica.parameters(), *replica_head.parameters()),
        ):
            torch.testing.assert_close(
                original.grad, grouped.grad, atol=1e-6, rtol=1e-4
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            within_vowel_supcon(torch.randn(3, 128), ["a", "b", "c"])


class MetricTests(unittest.TestCase):
    def test_eer_and_ties(self):
        self.assertEqual(
            roc_eer([0.9, 0.8, 0.2, 0.1], [True, True, False, False])["eer"], 0
        )
        self.assertEqual(roc_eer([0.5, 0.5], [True, False])["eer"], 0.5)
        self.assertIsNone(roc_eer([0.1, 0.2], [True, True])["eer"])
        self.assertAlmostEqual(
            roc_eer([0.9, 0.6, 0.6, 0.2], [True, True, False, False])["eer"], 0.25
        )

    def test_thresholds_do_not_split_ties(self):
        scores = np.array([0.9, 0.7, 0.7, 0.1])
        labels = np.array([True, False, False, True])
        threshold, rates = far_target_threshold(scores, labels, 0.5)
        self.assertEqual(rates["far"], 0)
        self.assertGreater(threshold, 0.7)
        threshold, rates = eer_operating_threshold(scores, labels)
        self.assertTrue(np.isfinite(threshold) or threshold == -np.inf)
        self.assertEqual(rates["genuine"], 2)
        with self.assertRaises(ValueError):
            roc_eer([float("nan")], [True])

    def test_validation_thresholds_are_calibrated_on_verification_only(self):
        groups = {}
        for role in ("verification", "cross_text_verification"):
            for count in (1, 5, 10):
                for vowel in "aiueo":
                    scores = (
                        [0.9, 0.8, 0.2, 0.1]
                        if role == "verification"
                        else [0.2, 0.1, 0.9, 0.8]
                    )
                    groups[(role, count, vowel)] = (scores, [True, True, False, False])
        metrics, thresholds = summarize_trials(groups)
        self.assertEqual(metrics["macro_eer"], 0)
        self.assertEqual(
            metrics["roles"]["cross_text_verification"]["10"]["macro_eer"], 1
        )
        self.assertEqual(
            metrics["roles"]["cross_text_verification"]["10"]["vowels"]["a"][
                "fixed_threshold_rates"
            ]["far_1pct"]["far"],
            1,
        )
        self.assertGreater(
            thresholds["per_count"]["10"]["a"]["far_1pct"]["threshold"], 0.2
        )

    def test_length_and_quality_strata(self):
        segment = Segment(
            "x",
            "s",
            "a",
            "validation",
            "verification",
            (),
            "unused.wav",
            0,
            7200,
            is_devoiced=None,
            is_long=True,
            quality_flags=("low_snr",),
        )
        strata = _stratum_values(segment)
        self.assertEqual(strata["duration"], ["100ms-plus"])
        self.assertEqual(strata["cropped_over_250ms"], ["yes"])
        self.assertEqual(strata["is_devoiced"], ["unknown"])
        self.assertEqual(strata["quality_flags"], ["low_snr"])


class _SyntheticDataset:
    kind = "log_mel"

    def __init__(self, segments, run_seed):
        self.segments, self.run_seed = segments, run_seed
        self.by_id = {segment.segment_id: segment for segment in segments}
        self.seen = []

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, request):
        self.seen.append((request.segment_id, request.logical_update % 3))
        segment = self.by_id[request.segment_id]
        speaker = int(segment.speaker_id[1:])
        vowel = "aiueo".index(segment.vowel)
        tensor = torch.full((64, 2 + speaker % 2), (speaker + vowel) / 20)
        return {
            "input": tensor,
            "length": tensor.shape[-1],
            "crop_start": request.logical_update % 3,
            "segment_id": segment.segment_id,
            "speaker_id": segment.speaker_id,
            "vowel": segment.vowel,
        }


class TrainerTests(unittest.TestCase):
    def test_aam_only_condition(self):
        segments = [
            Segment(
                f"s{speaker}-{vowel}-{take}",
                f"s{speaker}",
                vowel,
                "train",
                "training",
                (10, 70),
                "unused.wav",
                0,
                720,
            )
            for speaker in range(10)
            for vowel in "aiueo"
            for take in range(2)
        ]
        settings = TrainSettings("statistics_mlp", 10, True, False, 9, 2, 0, 1)
        seed_everything(9)
        trainer = Trainer(
            create_encoder("statistics_mlp"),
            AAMSoftmax(10),
            settings,
            _SyntheticDataset(segments, 9),
            BalancedSampler(segments, 9),
            manifest_sha256="manifest",
            feature_statistics_sha256="statistics",
        )
        record = trainer.train_until(1)[0]
        self.assertTrue(np.isfinite(record["loss"]))
        self.assertTrue(np.isfinite(record["gradient_norm"]))

    def test_checkpoint_resume_matches_uninterrupted_ten_updates(self):
        torch.set_num_threads(1)
        segments = [
            Segment(
                f"s{speaker}-{vowel}-{take}",
                f"s{speaker}",
                vowel,
                "train",
                "training",
                (10, 70),
                "unused.wav",
                0,
                720,
            )
            for speaker in range(10)
            for vowel in "aiueo"
            for take in range(2)
        ]
        settings = TrainSettings("statistics_mlp", 10, True, True, 7, 10, 2, 5)

        def build():
            seed_everything(7)
            dataset = _SyntheticDataset(segments, 7)
            return Trainer(
                create_encoder("statistics_mlp"),
                AAMSoftmax(10),
                settings,
                dataset,
                BalancedSampler(segments, 7),
                manifest_sha256="manifest",
                feature_statistics_sha256="statistics",
            )

        continuous = build()
        whole = continuous.train_until(10)
        with tempfile.TemporaryDirectory() as directory:
            staged = build()
            first = staged.train_until(5)
            checkpoint = Path(directory) / "last.pt"
            staged.save_checkpoint(checkpoint)
            resumed = build()
            resumed.load_checkpoint(checkpoint)
            second = resumed.train_until(10)
            incompatible = build()
            incompatible.manifest_sha256 = "changed"
            with self.assertRaisesRegex(ValueError, "manifest_sha256 mismatch"):
                incompatible.load_checkpoint(checkpoint)
        self.assertEqual(whole, first + second)
        for left, right in zip(
            continuous.model.parameters(), resumed.model.parameters()
        ):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        torch.testing.assert_close(
            continuous.head.weight, resumed.head.weight, atol=0, rtol=0
        )
        self.assertEqual(
            continuous.dataset.seen, staged.dataset.seen + resumed.dataset.seen
        )
        self.assertEqual(continuous.sampler.state_at(10), resumed.sampler.state_at(10))
        self.assertEqual(
            continuous.scheduler.state_dict(), resumed.scheduler.state_dict()
        )
        torch.testing.assert_close(
            continuous.optimizer.state_dict(),
            resumed.optimizer.state_dict(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
