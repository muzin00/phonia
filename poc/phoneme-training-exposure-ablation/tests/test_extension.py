"""Exposure matching and actual checkpoint continuation with early stopping off."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import train_extension as study


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_protocol_rejects_extra_changes_and_checkpoint_retuning(self):
        good = study.read_json(study.CONFIG)
        study.validate_protocol(good)
        for change in (
            {"target_update": 36000},
            {"change_feature_statistics": True},
            {"change_training_data": True},
            {"early_stopping": "enabled"},
            {"checkpoint_selection": "validation_best"},
        ):
            with self.assertRaises(ValueError):
                study.validate_protocol({**good, **change})

    def test_actual_exposure_matches_with_at_most_one_balanced_selection(self):
        baseline = {f"jvs{s:03d}": 2142 + (s % 2) for s in range(70)}
        current = {s: 2143 if n == 2142 else 2142 for s, n in baseline.items()}
        current.update({f"cv17_{s}": 2142 for s in range(70)})
        report = study.verify_exposure(current, baseline)
        self.assertEqual(report["maximum_jvs_difference_from_baseline"], 1)
        bad = {**current, "jvs000": 1071}
        with self.assertRaises(ValueError):
            study.verify_exposure(bad, baseline)
        bad = dict(current)
        bad.pop("cv17_0")
        with self.assertRaises(ValueError):
            study.verify_exposure(bad, baseline)

    @patch(
        "phase3_data.input.read_pcm_slice",
        side_effect=lambda *_: torch.linspace(-0.1, 0.1, 960),
    )
    def test_optimizer_scheduler_sampler_rng_and_updates_continue_exactly(self, _pcm):
        source = study.source
        segments = [
            source.Segment(
                segment_id=f"s{s}-{v}-{i}",
                speaker_id=f"s{s}",
                vowel=v,
                split="train",
                role="training",
                cohorts=(140,),
                source_file="unused.wav",
                start_frame=0,
                end_frame=960,
            )
            for s in range(140)
            for v in "aiueo"
            for i in range(2)
        ]

        def make():
            source.seed_everything(123)
            settings = source.ExpansionSettings(
                encoder="statistics_mlp",
                cohort=140,
                rms_enabled=False,
                supcon_enabled=True,
                seed=123,
                maximum_updates=10,
                warmup_updates=1,
                validation_interval=1,
            )
            pipeline = source.InputPipeline(source.BASELINE["input"], rms_enabled=False)
            model = source.create_encoder("statistics_mlp")
            dataset = source.SegmentDataset(
                source.ROOT,
                segments,
                pipeline,
                kind="log_mel",
                run_seed=123,
                mode="random",
            )
            return source.Trainer(
                model,
                source.AAMSoftmax(140),
                settings,
                dataset,
                source.BalancedSampler(segments, 123),
                manifest_sha256="fixed",
                feature_statistics_sha256="fixed",
            )

        original = make()
        original.train_until(1)
        original.early_stopping = {"reference_eer": 0.1, "patience": 8}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.pt"
            original.save_checkpoint(path)
            expected = []
            study.continue_fixed(
                original, 3, checkpoint_dir=None, on_update=expected.append
            )
            state = {k: v.clone() for k, v in original.model.state_dict().items()}
            head = original.head.weight.clone()
            scheduler = copy.deepcopy(original.scheduler.state_dict())
            sampler = copy.deepcopy(original.sampler.state_at(3))
            resumed = make()
            resumed.load_checkpoint(path)
            actual = []
            study.continue_fixed(
                resumed, 3, checkpoint_dir=None, on_update=actual.append
            )
            self.assertEqual(expected, actual)
            self.assertEqual(resumed.update, 3)
            self.assertEqual(resumed.scheduler.state_dict(), scheduler)
            self.assertEqual(resumed.sampler.state_at(3), sampler)
            self.assertTrue(torch.equal(head, resumed.head.weight))
            self.assertTrue(
                all(
                    torch.equal(state[k], v)
                    for k, v in resumed.model.state_dict().items()
                )
            )


if __name__ == "__main__":
    unittest.main()
