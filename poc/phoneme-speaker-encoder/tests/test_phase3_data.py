"""Input, sampler and artifact contracts for design version 2.0.0."""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3_data import (
    BalancedSampler,
    InputPipeline,
    SampleRequest,
    Segment,
    compute_feature_statistics,
    load_feature_statistics,
    make_enrollment,
    make_trials,
)
from phase3_data.artifacts import write_json
from phase3_data.input import SegmentDataset, collate_segments, crop_start
from phase3_data.manifest import VOWELS, ids_checksum
from phase3_data.sampling import VowelMicrobatchSampler

BASELINE = Path(__file__).resolve().parents[1] / "config/baseline-log-mel.json"
CONFIG = json.loads(BASELINE.read_text(encoding="utf-8"))["input"]


def segment(
    identifier,
    speaker="jvs001",
    vowel="a",
    *,
    split="train",
    role="training",
    source=None,
    length=720,
    cohorts=(10, 25, 50, 70),
):
    return Segment(
        identifier,
        speaker,
        vowel,
        split,
        role,
        cohorts,
        source or f"audio/{identifier}.wav",
        0,
        length,
    )


def write_wav(root, item, values):
    path = root / item.source_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(np.asarray(values, dtype="<i2").tobytes())


def sampler_segments():
    return [
        segment(f"s{speaker:02d}-{vowel}-{index}", f"s{speaker:02d}", vowel)
        for speaker in range(25)
        for vowel in VOWELS
        for index in range(3)
    ]


class InputTests(unittest.TestCase):
    def test_frame_counts_and_crop(self):
        pipeline = InputPipeline(CONFIG, rms_enabled=False)
        for size, frames in ((720, 2), (6000, 46), (6240, 46)):
            signal = torch.linspace(-0.25, 0.25, size)
            item = pipeline.prepare(signal, "interval", mode="center")
            self.assertEqual(item["length"], frames)
            self.assertEqual(item["crop_start"], max(0, (size - 6000) // 2))
        expected = crop_start(6240, "interval", "random", 20260926, 17)
        self.assertEqual(expected, crop_start(6240, "interval", "random", 20260926, 17))
        self.assertTrue(0 <= expected <= 240)
        self.assertNotEqual(
            expected, crop_start(6240, "interval", "random", 20260926, 18)
        )

    def test_batch_padding_and_finiteness(self):
        pipeline = InputPipeline(CONFIG, rms_enabled=True)
        items = [
            pipeline.prepare(
                torch.arange(size, dtype=torch.float32) / 30000,
                str(size),
                mode="center",
            )
            for size in (720, 1680, 6000)
        ]
        alone = collate_segments(items[:1])
        together = collate_segments(items)
        changed = collate_segments(items, padding_value=99.0)
        self.assertTrue(
            torch.equal(alone["input"][0, :, :2], together["input"][0, :, :2])
        )
        self.assertTrue(
            torch.equal(
                together["input"][
                    together["mask"][:, None, :].expand_as(together["input"])
                ],
                changed["input"][
                    changed["mask"][:, None, :].expand_as(changed["input"])
                ],
            )
        )
        self.assertEqual(together["mask"].sum(dim=1).tolist(), [2, 10, 46])
        with self.assertRaises(ValueError):
            pipeline.prepare(torch.zeros(719), "short", mode="center")
        with self.assertRaises(ValueError):
            pipeline.prepare(torch.full((720,), float("nan")), "nan", mode="center")
        with self.assertRaises(ValueError):
            collate_segments([])

    def test_pcm_and_worker_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [segment("short"), segment("long", length=6480)]
            for item in items:
                write_wav(root, item, np.arange(item.length, dtype=np.int16) - 3000)
            pipeline = InputPipeline(CONFIG, rms_enabled=False)
            dataset = SegmentDataset(root, items, pipeline, run_seed=13, mode="random")
            request = [SampleRequest("long", 5)]
            one = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_sampler=[request],
                        collate_fn=collate_segments,
                        num_workers=0,
                    )
                )
            )
            many = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_sampler=[request],
                        collate_fn=collate_segments,
                        num_workers=2,
                        prefetch_factor=2,
                    )
                )
            )
            self.assertTrue(torch.equal(one["input"], many["input"]))
            self.assertEqual(one["crop_starts"], many["crop_starts"])
            self.assertEqual(
                one["crop_starts"], [crop_start(6480, "long", "random", 13, 5)]
            )
            self.assertTrue(
                torch.isfinite(
                    pipeline.prepare(torch.zeros(720), "silent", mode="center")["input"]
                ).all()
            )
            bad = segment("bad", length=720)
            path = root / bad.source_file
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(24000)
                stream.writeframes(b"\0" * 720 * 4)
            with self.assertRaises(ValueError):
                SegmentDataset(root, [bad], pipeline)[0]


class SamplingTests(unittest.TestCase):
    def test_balance_order_and_resume(self):
        items = sampler_segments()
        sampler = BalancedSampler(items, 20260926)
        first = sampler.batch(0)
        self.assertEqual(len(first), 100)
        self.assertEqual(len({r.segment_id for r in first}), 100)
        self.assertEqual(
            [r.segment_id.split("-")[1] for r in first[::20]], list(VOWELS)
        )
        for _ in range(13):
            sampler.batch(sampler.next_update)
        self.assertLessEqual(
            max(sampler.counts.values()) - min(sampler.counts.values()), 1
        )
        state = sampler.state_at(9)
        resumed = BalancedSampler(items, 20260926)
        resumed.load_state_dict(state)
        original = BalancedSampler(items, 20260926)
        for update in range(9):
            original.batch(update)
        self.assertEqual(resumed.batch(9), original.batch(9))
        self.assertEqual(
            len(
                list(
                    VowelMicrobatchSampler(
                        BalancedSampler(items, 7), start_update=2, updates=3
                    )
                )
            ),
            15,
        )
        with self.assertRaises(ValueError):
            BalancedSampler(
                items + [segment("leak", split="validation", role="enrollment")], 1
            )
        state["speaker_counts"]["s00"] += 1
        with self.assertRaises(ValueError):
            resumed.load_state_dict(state)

    def test_dataloader_prefetch_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = sampler_segments()[:150]
            shared = "audio/shared.wav"
            items = [
                Segment(
                    s.segment_id,
                    s.speaker_id,
                    s.vowel,
                    s.split,
                    s.role,
                    s.cohorts,
                    shared,
                    0,
                    6240,
                )
                for s in items
            ]
            write_wav(root, items[0], np.arange(6240, dtype=np.int16) - 3000)
            pipeline = InputPipeline(CONFIG, rms_enabled=False)
            dataset = SegmentDataset(
                root, items, pipeline, run_seed=20260926, mode="random"
            )

            def collect(start, updates, workers):
                sampler = BalancedSampler(items, 20260926)
                loader = DataLoader(
                    dataset,
                    batch_sampler=VowelMicrobatchSampler(
                        sampler, start_update=start, updates=updates
                    ),
                    collate_fn=collate_segments,
                    num_workers=workers,
                    **({"prefetch_factor": 2} if workers else {}),
                )
                return [
                    (batch["segment_ids"], batch["crop_starts"]) for batch in loader
                ]

            full = collect(0, 2, 0)
            parallel = collect(0, 2, 2)
            resumed = collect(1, 1, 2)
            self.assertEqual(full, parallel)
            self.assertEqual(full[5:], resumed)


class ArtifactTests(unittest.TestCase):
    def test_train_statistics_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = [
                segment(f"s{speaker}", f"s{speaker}", length=720)
                for speaker in range(10)
            ]
            for item in items:
                write_wav(
                    root, item, np.tile(np.array([-400, 400], dtype=np.int16), 360)
                )
            pipeline = InputPipeline(CONFIG, rms_enabled=False)
            stats = compute_feature_statistics(
                root,
                items,
                pipeline,
                cohort=10,
                manifest_sha256="abc",
                git_commit="deadbeef",
                calculation_code_sha256="code",
            )
            self.assertEqual(stats["frame_count"], 20)
            self.assertEqual(stats["segment_count"], 10)
            self.assertTrue(np.isfinite(stats["mean"]).all())
            path = root / "stats.json"
            write_json(path, stats)
            expected = {
                "manifest_sha256": "abc",
                "cohort": 10,
                "speaker_ids_sha256": ids_checksum(s.speaker_id for s in items),
                "segment_ids_sha256": ids_checksum(s.segment_id for s in items),
                "preprocessing_sha256": pipeline.preprocessing_sha256,
            }
            loaded = load_feature_statistics(path, **expected)
            self.assertEqual(loaded, stats)
            self.assertTrue(
                torch.isfinite(
                    InputPipeline(CONFIG, rms_enabled=False, statistics=loaded).log_mel(
                        torch.zeros(720)
                    )
                ).all()
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_feature_statistics(
                    path, **{**expected, "manifest_sha256": "wrong"}
                )
            modified = dict(stats)
            modified["mean"] = list(stats["mean"])
            modified["mean"][0] += 1
            write_json(path, modified)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_feature_statistics(path, **expected)
            with self.assertRaises(ValueError):
                compute_feature_statistics(
                    root,
                    items
                    + [
                        segment(
                            "leak",
                            "outsider",
                            split="validation",
                            role="enrollment",
                            cohorts=(),
                        )
                    ],
                    pipeline,
                    cohort=10,
                    manifest_sha256="abc",
                    git_commit="deadbeef",
                    calculation_code_sha256="code",
                )
            with self.assertRaises(ValueError):
                compute_feature_statistics(
                    root,
                    items + [segment("outside", "outsider")],
                    pipeline,
                    cohort=10,
                    manifest_sha256="abc",
                    git_commit="deadbeef",
                    calculation_code_sha256="code",
                )

    def test_enrollment_diversity_and_trials(self):
        enrolled = []
        queries = []
        for speaker in range(15):
            sid = f"s{speaker:02d}"
            for vowel in VOWELS:
                for index in range(12):
                    enrolled.append(
                        segment(
                            f"e-{sid}-{vowel}-{index}",
                            sid,
                            vowel,
                            split="validation",
                            role="enrollment",
                            source=f"{sid}/{vowel}/source-{index // 2}",
                            cohorts=(),
                        )
                    )
                queries.append(
                    segment(
                        f"q-{sid}-{vowel}",
                        sid,
                        vowel,
                        split="validation",
                        role="verification",
                        source=f"{sid}/{vowel}/query",
                        cohorts=(),
                    )
                )
        chosen = make_enrollment(enrolled, split="validation")
        self.assertEqual(len(chosen), 750)
        self.assertEqual(
            chosen, make_enrollment(reversed(enrolled), split="validation")
        )
        group = [r for r in chosen if r["speaker_id"] == "s00" and r["vowel"] == "a"]
        self.assertEqual(len({r["source_file"] for r in group[:6]}), 6)
        trials = list(make_trials(queries, chosen, split="validation"))
        self.assertEqual(len(trials), len(queries) * 15 * 3)
        self.assertEqual(sum(r["is_genuine"] for r in trials), len(queries) * 3)
        self.assertEqual(len({r["trial_id"] for r in trials}), len(trials))


if __name__ == "__main__":
    unittest.main()
