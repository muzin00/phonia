"""Protect validation-only selection, source identity, and model snapshots."""

import copy
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke import (
    CONFIG,
    check_model_snapshot,
    read_pcm,
    select_validation,
    sha256_file,
    write_json,
)


class SmokeInputTests(unittest.TestCase):
    def setUp(self):
        self.selection = json.loads(CONFIG.read_text())["selection"]
        self.selection["speaker_count"] = 1
        self.splits = {
            "speaker_splits": {
                "train": ["train1"],
                "validation": ["valid1", "valid2"],
                "test": ["test1"],
            }
        }
        self.rows = [
            {
                "utterance_id": f"{speaker}_{role}_{index}",
                "speaker_id": speaker,
                "split": "validation",
                "evaluation_role": role,
                "frame_count": 48000 if index == 1 else 24000,
                "source_file": f"{speaker}/{role}/{index}.wav",
            }
            for speaker in ("valid1", "valid2")
            for role in self.selection["roles"]
            for index in (1, 2)
        ]

    def test_selection_is_order_independent_and_includes_short_queries(self):
        selected = select_validation(self.rows, self.splits, self.selection)
        self.assertEqual(
            selected,
            select_validation(list(reversed(self.rows)), self.splits, self.selection),
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            {row["utterance_id"] for row in selected},
            {
                "valid1_enrollment_1",
                "valid1_verification_1",
                "valid1_verification_2",
                "valid1_cross_text_verification_1",
                "valid1_cross_text_verification_2",
            },
        )

    def test_reject_test_selection_or_mislabeled_speaker(self):
        for field, value in (("split", "test"), ("speaker_id", "test1")):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, "validation assignment"):
                    select_validation(rows, self.splits, self.selection)
        self.selection["split"] = "test"
        with self.assertRaisesRegex(ValueError, "must use validation"):
            select_validation(self.rows, self.splits, self.selection)

    def test_reject_overlapping_splits_and_missing_role(self):
        splits = copy.deepcopy(self.splits)
        splits["speaker_splits"]["test"].append("valid1")
        with self.assertRaisesRegex(ValueError, "split overlap"):
            select_validation(self.rows, splits, self.selection)
        rows = [row for row in self.rows if row["evaluation_role"] != "enrollment"]
        with self.assertRaisesRegex(ValueError, "missing validation role"):
            select_validation(rows, self.splits, self.selection)

    def test_reject_duplicate_sources(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["source_file"] = rows[0]["source_file"]
        with self.assertRaisesRegex(ValueError, "duplicate validation utterance"):
            select_validation(rows, self.splits, self.selection)

    def test_test_wav_is_rejected_before_any_file_access(self):
        with self.assertRaisesRegex(ValueError, "only validation WAVs"):
            read_pcm({"split": "test"})

    def test_source_integrity_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.wav"
            pcm = b"\x01\x00" * 240
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            row = {
                "split": "validation",
                "source_file": "sample.wav",
                "source_sha256": sha256_file(source),
                "frame_count": 240,
            }
            self.assertEqual(read_pcm(row, root=root), pcm)
            row["frame_count"] = 241
            with self.assertRaisesRegex(ValueError, "PCM metadata"):
                read_pcm(row, root=root)
            row["frame_count"] = 240
            row["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_pcm(row, root=root)

    def test_source_cannot_escape_audio_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for source in ("../outside.wav", "/outside.wav"):
                with (
                    self.subTest(source=source),
                    self.assertRaisesRegex(ValueError, "outside audio root"),
                ):
                    read_pcm(
                        {"split": "validation", "source_file": source},
                        root=Path(directory),
                    )

    def test_snapshot_tampering_and_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights").write_bytes(b"frozen weights")
            model = {"id": "test/model", "revision": "fixed", "files": ["weights"]}
            model["file_sha256"] = {"weights": sha256_file(root / "weights")}
            manifest = {
                "model_id": model["id"],
                "revision": model["revision"],
                "files": {"weights": {"sha256": sha256_file(root / "weights")}},
            }
            write_json(root / "model-manifest.json", manifest)
            self.assertEqual(check_model_snapshot(root, {"model": model}), manifest)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                check_model_snapshot(root, {"model": {**model, "revision": "new"}})
            (root / "weights").write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "file checksum mismatch"):
                check_model_snapshot(root, {"model": model})
            manifest["files"]["weights"]["sha256"] = sha256_file(root / "weights")
            (root / "model-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "file checksum mismatch"):
                check_model_snapshot(root, {"model": model})


if __name__ == "__main__":
    unittest.main()
