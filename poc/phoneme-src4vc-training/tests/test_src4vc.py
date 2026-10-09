"""Protect actual corpus selection, held-out speaker separation and exposure."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_src4vc as preparation
import train_src4vc as training


class CorpusTests(unittest.TestCase):
    def test_raw_spoken_inventory_is_balanced_and_reserved_speakers_excluded(self):
        config = preparation.read_json(preparation.CONFIG)
        names = set()
        for i in range(1, 101):
            parent = f"SRC4VC_ver1/SRC4VC{i:03d}"
            names.add(parent + "/speaker_metadata.yml")
            for subset, count in (("RECITATION", 10), ("CALLS", 5), ("STUDIES", 5)):
                for number in range(count):
                    names.add(f"{parent}/wav/{subset}_{number:03d}.wav")
                    names.add(f"{parent}/txt/{subset}_{number:03d}.txt")
            for emotion in (
                "anger",
                "disgust",
                "fear",
                "happiness",
                "sadness",
                "surprise",
            ):
                for number in range(5):
                    names.add(f"{parent}/wav/JVNV{emotion}_{number:03d}.wav")
                    names.add(f"{parent}/txt/JVNV{emotion}_{number:03d}.txt")
            names.add(parent + "/wav/SONGS_001.wav")
            names.add(parent + "/wav-R/RECITATION_001.wav")
        splits, selected = preparation.select_inventory(names, config)
        self.assertEqual(len(selected), 3500)
        self.assertEqual(
            {k: len(v) for k, v in splits.items()},
            {"train": 70, "validation_reserved": 15, "test_reserved": 15},
        )
        self.assertEqual(len(set().union(*map(set, splits.values()))), 100)
        self.assertEqual(
            {r["source_speaker_id"] for r in selected}, set(splits["train"])
        )
        self.assertTrue(
            all(
                preparation.source.alignment.SAFE_ID.fullmatch(r["utterance_id"])
                for r in selected
            )
        )
        self.assertTrue(
            all(
                preparation.source.alignment.SAFE_ID.fullmatch(r["speaker_id"])
                for r in selected
            )
        )
        self.assertTrue(
            all(
                "/wav/" in r["raw_member"] and "SONGS" not in r["raw_member"]
                for r in selected
            )
        )
        self.assertEqual(
            (splits, selected),
            preparation.select_inventory(
                dict.fromkeys(sorted(names, reverse=True)), config
            ),
        )
        missing = selected[0]["transcript_member"]
        with self.assertRaises(ValueError):
            preparation.select_inventory(names - {missing}, config)

    def test_training_loader_rejects_reserved_and_duplicate_segments(self):
        jvs = training.read_json(
            training.ROOT / "poc/phoneme-speaker-dataset/config/dataset-split.json"
        )["speaker_splits"]
        additional = [f"SRC4VC{i:03d}" for i in range(1, 71)]
        speakers = sorted(jvs["train"]) + ["ext_src4vc" + s[6:] for s in additional]
        rows = [
            {
                "split": "train",
                "evaluation_role": "training",
                "learning_curve_cohorts": [140],
                "normalized_phoneme": "a",
                "start_frame": 0,
                "end_frame": 960,
                "quality_flags": [],
                "vowel_interval_id": s + "-a",
                "speaker_id": s,
                "source_file": "artifacts/fixture.wav",
                "source_sha256": "0" * 64,
                "sample_rate_hz": 24000,
                "channels": 1,
                "sample_width_bytes": 2,
            }
            for s in speakers
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "segments.jsonl"
            (path.parent / "speaker-splits.json").write_text(
                json.dumps({"splits": {"train": additional}})
            )

            def load(values):
                path.write_text("".join(json.dumps(r) + "\n" for r in values))
                return training.load_training(path)

            self.assertEqual(len(load(rows)), 140)
            for replacement in (
                {"speaker_id": jvs["test"][0]},
                {"speaker_id": "ext_src4vc071"},
                {"split": "validation"},
                {"quality_flags": ["near_silent"]},
                {"source_file": "../../outside.wav"},
            ):
                bad = copy.deepcopy(rows)
                bad[0].update(replacement)
                with self.assertRaises(ValueError):
                    load(bad)
            with self.assertRaises(ValueError):
                load(rows + [rows[0]])

    def test_fixed_budget_cannot_be_reselected_from_validation(self):
        config = training.read_json(training.CONFIG)
        training.validate_protocol(config)
        for change in (
            {"checkpoint_selection": "validation_best"},
            {"maximum_updates": 36000},
            {"early_stopping": "enabled"},
            {"test_used": True},
        ):
            with self.assertRaises(ValueError):
                training.validate_protocol({**config, **change})

    def test_exposure_is_checked_for_each_jvs_speaker(self):
        baseline = {f"jvs{s:03d}": 2142 + s % 2 for s in range(70)}
        current = {**baseline, **{f"ext_src4vc{s:03d}": 2142 for s in range(70)}}
        self.assertEqual(
            training.verify_exposure(current, baseline)[
                "maximum_jvs_difference_from_baseline"
            ],
            0,
        )
        with self.assertRaises(ValueError):
            training.verify_exposure({**current, "jvs000": 1071}, baseline)


if __name__ == "__main__":
    unittest.main()
