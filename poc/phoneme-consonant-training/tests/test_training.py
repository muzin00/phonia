"""Check seven-phone coverage, positive pairs and deterministic restart."""

import json
import sys
import tempfile
import unittest
import wave
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import torch
import train
from phase3_data.input import InputPipeline, SegmentDataset
from phase3_data.manifest import Segment
from phase3_train.losses import AAMSoftmax
from phase3_train.models import create_encoder
from phase3_train.training import Trainer, seed_everything
from sampling import PHONEMES, PhonemeSampler


def fixture(root):
    generator = torch.Generator().manual_seed(17)
    pcm = (torch.randn(6000, generator=generator) * 1500).to(torch.int16).numpy()
    with wave.open(str(root / "source.wav"), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm.tobytes())
    return [
        Segment(
            f"{speaker}-{phone}-{i}",
            f"s{speaker:02d}",
            phone,
            "train",
            "training",
            (140,),
            "source.wav",
            i * 960,
            (i + 1) * 960,
        )
        for speaker in range(10)
        for phone in PHONEMES
        for i in range(3)
    ]


def trainer(root, segments):
    seed_everything(20260926)
    settings = train.expanded.ExpansionSettings(
        encoder="statistics_mlp",
        cohort=140,
        rms_enabled=False,
        supcon_enabled=True,
        seed=20260926,
        maximum_updates=4,
        warmup_updates=1,
        validation_interval=2,
    )
    pipeline = InputPipeline(train.expanded.BASELINE["input"], rms_enabled=False)
    dataset = SegmentDataset(
        root, segments, pipeline, run_seed=settings.seed, mode="random"
    )
    return Trainer(
        create_encoder("statistics_mlp"),
        AAMSoftmax(10),
        settings,
        dataset,
        PhonemeSampler(segments, settings.seed),
        manifest_sha256="fixture",
        feature_statistics_sha256=None,
    )


class TrainingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sampler_coverage_and_positive_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            segments = fixture(Path(directory))
            lookup = {s.segment_id: s for s in segments}
            sampler = PhonemeSampler(segments, 23)
            counts = Counter()
            for update in range(7):
                groups = sampler.microbatches(update)
                self.assertEqual(len(groups), 5)
                for group in groups:
                    rows = [lookup[r.segment_id] for r in group]
                    self.assertEqual(len(rows), 20)
                    self.assertEqual(len({s.segment_id for s in rows}), 20)
                    self.assertEqual(len({s.vowel for s in rows}), 1)
                    self.assertEqual(
                        set(Counter(s.speaker_id for s in rows).values()), {2}
                    )
                    counts[rows[0].vowel] += 1
            self.assertEqual(counts, Counter({p: 5 for p in PHONEMES}))
            resumed = PhonemeSampler(segments, 23)
            resumed.load_state_dict(sampler.state_at(7))
            self.assertEqual(sampler.batch(7), resumed.batch(7))

    def test_missing_consonant_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            segments = [s for s in fixture(Path(directory)) if s.vowel != "n"]
            with self.assertRaises(ValueError):
                PhonemeSampler(segments, 23)

    def test_checkpoint_restart_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = fixture(root)
            continuous = trainer(root, segments)
            continuous.train_until(4)
            expected = {k: v.clone() for k, v in continuous.model.state_dict().items()}
            interrupted = trainer(root, segments)
            interrupted.train_until(2)
            interrupted.save_checkpoint(root / "checkpoint.pt")
            resumed = trainer(root, segments)
            resumed.load_checkpoint(root / "checkpoint.pt")
            resumed.train_until(4)
            for key, value in resumed.model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            self.assertEqual(
                resumed.sampler.state_at(4), continuous.sampler.state_at(4)
            )

    def test_manifest_rejects_nontraining_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.jsonl"
            row = {
                "split": "test",
                "evaluation_role": "training",
                "normalized_phoneme": "m",
                "learning_curve_cohorts": [140],
                "start_frame": 0,
                "end_frame": 720,
                "quality_flags": [],
                "vowel_interval_id": "duplicate",
                "source_file": "test.wav",
            }
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                train.load_training(path)
            row["split"] = "train"
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                train.load_training(path)


if __name__ == "__main__":
    unittest.main()
