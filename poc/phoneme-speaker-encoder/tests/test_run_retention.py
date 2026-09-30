"""Bounded validation artifacts and safe interrupted-run history."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))

from phase3_data import sha256_file
from phase3_data.artifacts import write_json
from run_phase3 import _compact_validation_outputs, _trim_history_to_checkpoint


class ValidationRetentionTests(unittest.TestCase):
    def _validation(self, root: Path, update: int) -> Path:
        directory = root / f"validation/update-{update:06d}"
        score = directory / "scores/validation.jsonl"
        score.parent.mkdir(parents=True)
        score.write_text('{"cosine_score":0.5}\n', encoding="utf-8")
        write_json(
            directory / "metrics/validation.json",
            {
                "split": "validation",
                "partial": False,
                "macro_eer": update / 10000,
                "score_sha256": sha256_file(score),
                "roles": {"verification": {"roc_det_curve": {"far": [0, 1]}}},
            },
        )
        return directory

    def test_only_selected_validation_keeps_full_scores_and_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = self._validation(root, 1000)
            best = self._validation(root, 2000)
            _compact_validation_outputs(root, keep_update=2000)
            self.assertFalse((old / "scores/validation.jsonl").exists())
            compact = json.loads(
                (old / "metrics/validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(compact["macro_eer"], 0.1)
            self.assertNotIn("roc_det_curve", compact["roles"]["verification"])
            self.assertTrue((best / "scores/validation.jsonl").exists())
            _compact_validation_outputs(root, keep_update=2000)

    def test_score_checksum_mismatch_prevents_compaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self._validation(root, 1000)
            (directory / "scores/validation.jsonl").write_text("changed\n")
            with self.assertRaises(ValueError):
                _compact_validation_outputs(root, keep_update=None)
            self.assertTrue((directory / "scores/validation.jsonl").exists())


class ResumeHistoryTests(unittest.TestCase):
    def test_trims_only_updates_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"update": update}) + "\n" for update in range(1, 5)
                ),
                encoding="utf-8",
            )
            _trim_history_to_checkpoint(path, 2)
            self.assertEqual(
                [json.loads(line)["update"] for line in path.read_text().splitlines()],
                [1, 2],
            )

    def test_rejects_history_shorter_than_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.jsonl"
            path.write_text('{"update":1}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                _trim_history_to_checkpoint(path, 2)


if __name__ == "__main__":
    unittest.main()
