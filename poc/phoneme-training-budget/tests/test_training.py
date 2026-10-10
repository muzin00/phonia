"""Protect schedule fairness, fixed endpoints and trajectory randomness."""

import copy
import random
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget_training as study
import numpy as np
import torch
from phase3_train.training import WarmupCosine


class TrainingTests(unittest.TestCase):
    def test_protocol_rejects_early_stopping_test_selection_and_budget_changes(self):
        good = study.read_json(study.CONFIG)
        study.validate_protocol(good)
        for change in (
            {"early_stopping": True},
            {"maximum_updates": 30000},
            {"snapshot_updates": [30000, 60000]},
            {"test_checkpoint_selection": "test_best"},
            {"initialization": "resume_old_30k"},
            {"test_used_for_training": True},
        ):
            with self.assertRaises(ValueError):
                study.validate_protocol({**good, **change})

    def test_source_settings_change_only_the_schedule_budget(self):
        config = study.read_json(study.CONFIG)
        references = []
        for corpus in config["corpora"]:
            source = study.ROOT / config["source_runs"][corpus]
            freeze = study.read_json(source / "training-freeze.json")
            before = copy.deepcopy(freeze)
            result = study.derived_config(config, freeze)
            self.assertEqual(freeze, before)
            self.assertEqual(result, {**freeze["config"], "maximum_updates": 60000})
            reference = study.read_json(source / "expanded/run.json")["settings"]
            current = asdict(
                study.cv.ExpansionSettings(**{**reference, "maximum_updates": 60000})
            )
            study.validate_settings(current, reference)
            for key, value in (
                ("seed", 1),
                ("learning_rate", 1e-3),
                ("rms_enabled", True),
            ):
                with self.assertRaises(ValueError):
                    study.validate_settings({**current, key: value}, reference)
            references.append(current)
        self.assertEqual(references[0], references[1])

    def test_sixty_thousand_schedule_still_learns_at_thirty_thousand(self):
        reference = study.read_json(
            study.ROOT
            / study.read_json(study.CONFIG)["source_runs"]["cv"]
            / "expanded/run.json"
        )["settings"]
        settings = study.cv.ExpansionSettings(**{**reference, "maximum_updates": 60000})
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=settings.learning_rate)
        scheduler = WarmupCosine(optimizer, settings)
        values = []
        for update in (30000, 45000, 60000):
            scheduler.load_state_dict({"completed_steps": update})
            values.append(optimizer.param_groups[0]["lr"])
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(values[2], settings.minimum_learning_rate)
        self.assertGreater(values[0], 10 * values[2])

    def test_export_probe_preserves_all_random_streams_even_when_it_fails(self):
        study.cv.seed_everything(20260926)
        expected = (random.random(), np.random.random(), torch.rand(3))
        study.cv.seed_everything(20260926)
        with self.assertRaises(RuntimeError), study.preserve_rng():
            random.random()
            np.random.random()
            torch.rand(100)
            raise RuntimeError("export failed")
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_conflicting_frozen_sources_cannot_silently_replace_checksums(self):
        files = {"input": "one"}
        study.merge_files(files, {"other": "two", "input": "one"})
        with self.assertRaises(ValueError):
            study.merge_files(files, {"input": "changed"})


if __name__ == "__main__":
    unittest.main()
