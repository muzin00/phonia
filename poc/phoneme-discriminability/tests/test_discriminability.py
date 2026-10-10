"""Guard matched phone inputs, held-out calibration and speaker-balanced diagnostics."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import study
from diagnostics import embedding_scatter, primary_results, score_separation
from inputs import fixed_slice, validate_plans
from support import PHONES, core


def fixture(config):
    def item(query):
        count = 1 if query else 10
        conditions = {}
        for index, phone in enumerate(PHONES):
            conditions[phone] = []
            for token in range(count):
                first = (index * 10 + token) * 3600
                conditions[phone].append(
                    {
                        "segment_id": f"{'q' if query else 'e'}-{phone}-{token}",
                        "original_segment_id": f"{'q' if query else 'e'}-{phone}-{token}",
                        "vowel": phone,
                        "source_file": "q.wav" if query else "e.wav",
                        "source_sha256": "query" if query else "enrollment",
                        "original_start_frame": first,
                        "original_end_frame": first + 2400,
                        "start_frame": first + 600,
                        "end_frame": first + 1800,
                        "selected_rms_dbfs": -20,
                    }
                )
        common = {"conditions": conditions, "budget_frames": count * 1200}
        if query:
            common.update(
                query_id="q",
                speaker_id="s1",
                role="verification",
                split="test",
                common8=True,
                source_file="q.wav",
            )
        else:
            common["candidate_source_files"] = ["e.wav"]
        return common

    return {
        "profiles": {"s1": item(False)},
        "queries": [item(True)],
        "speakers": ["s1", "s2"],
    }


class DiscriminabilityTest(unittest.TestCase):
    def setUp(self):
        self.config = study.read_json(study.CONFIG)

    def test_fixed_crop_quality_checks_selected_pcm(self):
        samples = np.ones(3000) * 0.2
        self.assertEqual(fixed_slice(samples, 0, 3000, 1200, -50)[:2], (900, 2100))
        samples[900:2100] = 0
        self.assertIsNone(fixed_slice(samples, 0, 3000, 1200, -50))
        self.assertIsNone(fixed_slice(samples, 0, 1000, 1200, -50))
        with self.assertRaises(ValueError):
            fixed_slice(samples, -1, 3000, 1200, -50)

    def test_counts_durations_missing_phone_and_leaks(self):
        valid = fixture(self.config)
        self.assertEqual(
            validate_plans(valid, self.config), {"plans": 16, "slices": 88}
        )
        for change in ("duration", "count", "missing", "leak", "quiet", "crop"):
            broken = copy.deepcopy(valid)
            query = broken["queries"][0]
            row = query["conditions"]["s"][0]
            if change == "duration":
                row["end_frame"] += 1
            elif change == "count":
                query["conditions"]["s"].append(dict(row))
            elif change == "missing":
                del query["conditions"]["s"]
            elif change == "leak":
                row["source_sha256"] = "enrollment"
            elif change == "quiet":
                row["selected_rms_dbfs"] = -60
            else:
                row["start_frame"] += 1
                row["end_frame"] += 1
            with self.assertRaises(ValueError):
                validate_plans(broken, self.config)

    def test_complete_eight_phone_matrix_and_labels(self):
        inputs = fixture(self.config)
        query = inputs["queries"][0]
        rows = [
            {
                **{
                    k: query[k]
                    for k in ("query_id", "speaker_id", "role", "split", "common8")
                },
                "condition": phone,
                "claimed_speaker_id": speaker,
                "is_genuine": speaker == query["speaker_id"],
                "status": "scored",
                "score": 0.5,
                "phone_scores": {phone: 0.5},
                "used_frames": 1200,
            }
            for phone in PHONES
            for speaker in inputs["speakers"]
        ]
        study.validate_scores(inputs, rows, self.config)
        for change in ("missing", "duplicate", "label", "frames", "phone"):
            broken = copy.deepcopy(rows)
            if change == "missing":
                broken.pop()
            elif change == "duplicate":
                broken.append(broken[0])
            elif change == "label":
                broken[0]["is_genuine"] = False
            elif change == "frames":
                broken[0]["used_frames"] = 1201
            else:
                broken[0]["phone_scores"] = {"s": 0.5}
            with self.assertRaises(ValueError):
                study.validate_scores(inputs, broken, self.config)

    def test_test_requires_frozen_validation_and_rejects_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            study.write_json(
                run / "design-freeze.json",
                {
                    "config": self.config,
                    "files": {},
                    "runtime": core.old.previous.runtime(),
                },
            )
            study.write_json(run / "test-inputs.json", fixture(self.config))
            with self.assertRaises(FileNotFoundError):
                study.frozen(self.config, run, "test")
            threshold = run / "threshold.json"
            threshold.write_text("fixed")
            study.write_json(
                run / "evaluation-freeze.json",
                {
                    "status": "thresholds_frozen_before_test_inference",
                    "files": {"threshold.json": study.sha256_file(threshold)},
                },
            )
            study.frozen(self.config, run, "test")
            threshold.write_text("changed")
            with self.assertRaises(ValueError):
                study.frozen(self.config, run, "test")

    def test_calibration_accepts_validation_normal_text_only(self):
        for split, role in (
            ("test", "verification"),
            ("validation", "cross_text_verification"),
        ):
            with self.assertRaises(ValueError):
                core.old.calibrate_cell([{"split": split, "role": role}], ["s"])

    def test_speaker_scatter_does_not_overweight_frequent_speaker(self):
        vectors = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        balanced = embedding_scatter(vectors, ["a", "a", "b", "b"], ["a", "b"])
        repeated = embedding_scatter(
            np.concatenate([np.tile(vectors[:2], (10, 1)), vectors[2:]]),
            ["a"] * 20 + ["b"] * 2,
            ["a", "b"],
        )
        self.assertAlmostEqual(
            balanced["between_within_ratio"], repeated["between_within_ratio"]
        )

    def test_score_separation_rewards_stable_distinct_scores(self):
        rows = [
            {"score": v, "is_genuine": i < 2}
            for i, v in enumerate((0.8, 0.9, 0.1, 0.2))
        ]
        result = score_separation(rows)
        self.assertGreater(result["dprime"], 10)
        self.assertEqual(sum(result["genuine"]["histogram_counts"]), 2)

    def test_primary_requires_both_contrasts_and_excludes_zero_endpoint(self):
        cells = {
            f"{p}/verification": {"pooled_eer": value}
            for p, value in (("m", 0.1), ("n", 0.2), ("s", 0.4))
        }
        replicas = {
            key: {"pooled_eer": np.full(100, value["pooled_eer"])}
            for key, value in cells.items()
        }
        result = primary_results(self.config, cells, replicas, core.old.interval)
        self.assertTrue(result["both_m_and_n_better_than_s"])
        replicas["n/verification"]["pooled_eer"][:] = 0.4
        result = primary_results(self.config, cells, replicas, core.old.interval)
        self.assertFalse(result["both_m_and_n_better_than_s"])
        self.assertFalse(result["contrasts"]["n_minus_s"]["supports_lower_eer_than_s"])


if __name__ == "__main__":
    unittest.main()
