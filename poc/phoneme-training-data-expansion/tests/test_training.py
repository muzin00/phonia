"""Leak prevention, pooled normalization and the expanded sampler contract."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import training
from phase3_data.manifest import Segment
from phase3_data.sampling import BalancedSampler


def settings(**changes):
    return training.ExpansionSettings(
        **{
            "encoder": "statistics_mlp",
            "cohort": 140,
            "rms_enabled": False,
            "supcon_enabled": True,
            "seed": 20260926,
            "maximum_updates": 30000,
            "warmup_updates": 1000,
            "validation_interval": 1000,
            **changes,
        }
    )


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_settings_retain_real_expanded_cohort_in_hash(self):
        expanded = settings()
        self.assertEqual(expanded.cohort, 140)
        baseline = training.TrainSettings(**{**training.asdict(expanded), "cohort": 70})
        self.assertNotEqual(expanded.sha256, baseline.sha256)
        for changed in (
            {"cohort": 70},
            {"overfit": True},
            {"warmup_updates": 30000},
            {"learning_rate": -1},
        ):
            with self.assertRaises(ValueError):
                settings(**changed)

    def test_pooled_moments_match_direct_combined_train_frames(self):
        generator = np.random.default_rng(123)
        first = generator.normal(size=(31, 64))
        second = 7 + 3 * generator.normal(size=(19, 64))

        def moments(frames):
            return {
                "frame_count": len(frames),
                "mean": frames.mean(axis=0).tolist(),
                "population_variance": frames.var(axis=0).tolist(),
            }

        merged = training.union_statistics(moments(first), moments(second))
        expected = moments(np.concatenate([first, second]))
        self.assertEqual(merged["frame_count"], 50)
        np.testing.assert_allclose(merged["mean"], expected["mean"], rtol=1e-14)
        np.testing.assert_allclose(
            merged["population_variance"], expected["population_variance"], rtol=1e-14
        )

    def test_invalid_population_moments_are_rejected(self):
        good = {"frame_count": 2, "mean": [0.0] * 64, "population_variance": [1.0] * 64}
        for bad in (
            {**good, "frame_count": 0},
            {**good, "mean": [float("nan")] * 64},
            {**good, "population_variance": [-1.0] * 64},
        ):
            with self.assertRaises(ValueError):
                training.union_statistics(good, bad)

    def test_validation_test_and_duplicate_ids_cannot_enter_training(self):
        row = {
            "split": "train",
            "evaluation_role": "training",
            "learning_curve_cohorts": [140],
            "normalized_phoneme": "a",
            "start_frame": 0,
            "end_frame": 960,
            "quality_flags": [],
            "vowel_interval_id": "id",
            "speaker_id": "jvs001",
            "source_file": "fake.wav",
            "source_sha256": "0" * 64,
            "sample_rate_hz": 24000,
            "channels": 1,
            "sample_width_bytes": 2,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inputs.jsonl"
            for changed in (
                {"split": "validation"},
                {"split": "test"},
                {"evaluation_role": "enrollment"},
                {"source_file": "../outside.wav"},
                {"quality_flags": ["near_silent"]},
            ):
                path.write_text(json.dumps({**row, **changed}) + "\n")
                with self.assertRaises(ValueError):
                    training.load_training(path)
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "invalid expanded"):
                training.load_training(path)

    def test_all_140_labels_participate_in_balanced_sampling_and_resume(self):
        segments = [
            Segment(
                segment_id=f"s{speaker}-{vowel}-{rank}",
                speaker_id=f"s{speaker}",
                vowel=vowel,
                split="train",
                role="training",
                cohorts=(140,),
                source_file="fake.wav",
                start_frame=0,
                end_frame=960,
            )
            for speaker in range(140)
            for vowel in "aiueo"
            for rank in range(2)
        ]
        original = BalancedSampler(segments, 20260926)
        batches = [original.batch(i) for i in range(28)]
        self.assertTrue(all(len(batch) == 100 for batch in batches))
        state = original.state_at(28)
        self.assertEqual(set(state["speaker_counts"].values()), {2})
        resumed = BalancedSampler(segments, 20260926)
        resumed.load_state_dict(state)
        self.assertEqual(original.batch(28), resumed.batch(28))

    def test_expanded_head_checkpoint_resume_reproduces_next_update(self):
        segments = [
            Segment(
                segment_id=f"s{speaker}-{vowel}-{rank}",
                speaker_id=f"s{speaker}",
                vowel=vowel,
                split="train",
                role="training",
                cohorts=(140,),
                source_file="fake.wav",
                start_frame=0,
                end_frame=960,
            )
            for speaker in range(140)
            for vowel in "aiueo"
            for rank in range(2)
        ]
        by_id = {s.segment_id: s for s in segments}

        def make():
            training.seed_everything(20260926)
            pipeline = training.InputPipeline(
                training.BASELINE["input"], rms_enabled=False
            )
            model = training.create_encoder("statistics_mlp")
            dataset = training.SegmentDataset(
                training.ROOT, segments, pipeline, run_seed=20260926, kind="log_mel"
            )
            return training.Trainer(
                model,
                training.AAMSoftmax(140),
                settings(),
                dataset,
                BalancedSampler(segments, 20260926),
                manifest_sha256="fixed",
                feature_statistics_sha256="fixed",
            )

        def batches(active, update):
            requests = active.sampler.batch(update)
            generator = torch.Generator().manual_seed(update)
            return [
                {
                    "input": torch.randn(20, 64, 8, generator=generator),
                    "mask": torch.ones(20, 8, dtype=torch.bool),
                    "speaker_ids": [
                        by_id[r.segment_id].speaker_id
                        for r in requests[offset : offset + 20]
                    ],
                    "vowels": [
                        by_id[r.segment_id].vowel
                        for r in requests[offset : offset + 20]
                    ],
                }
                for offset in range(0, 100, 20)
            ]

        first = make()
        first.train_update(batches(first, 0))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "last.pt"
            first.save_checkpoint(checkpoint)
            expected = first.train_update(batches(first, 1))
            resumed = make()
            resumed.load_checkpoint(checkpoint)
            actual = resumed.train_update(batches(resumed, 1))
        self.assertEqual(actual, expected)
        for left, right in zip(
            first.model.parameters(), resumed.model.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(first.head.weight, resumed.head.weight))


if __name__ == "__main__":
    unittest.main()
