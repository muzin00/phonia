"""Verify class expansion, failed-query accounting, and cached training export."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("corpus_scaling_study", BASE / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class StudyTests(unittest.TestCase):
    def test_nested_speakers_and_equal_count_replacement(self):
        existing = [f"jvs{i:03}" for i in range(70)] + [
            f"cv17_{i:03}" for i in range(70)
        ]
        plans = study.planned_speakers(existing, [f"S{i:03}" for i in range(140)], 15)
        self.assertEqual([len(plans[a]) for a in study.ARMS], [140, 140, 210, 280])
        self.assertEqual(sum(s.startswith("jtube_") for s in plans["mixed140"]), 35)
        self.assertLess(set(plans["expanded210"]), set(plans["expanded280"]))
        self.assertEqual(
            plans,
            study.planned_speakers(
                list(reversed(existing)), [f"S{i:03}" for i in range(140)], 15
            ),
        )

    def test_failed_queries_stay_in_frr_denominator(self):
        records = [
            {"is_genuine": True, "score": 0.9},
            {"is_genuine": True, "score": None},
            {"is_genuine": False, "score": 0.8},
            {"is_genuine": False, "score": None},
        ]
        result = study.operating(records, 0.7)
        self.assertEqual(result["frr"], 0.5)
        self.assertEqual(result["far"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["scored_only_error_rates"]["frr"], 0)

    def test_missing_vowel_is_not_silently_relaxed(self):
        phones = list(study.VOWELS)
        data = {
            "speakers": ["a", "b"],
            "universally_registered": phones,
            "profiles": {s: {p: [i] for i, p in enumerate(phones)} for s in ("a", "b")},
            "queries": [
                {
                    "query_id": "q",
                    "speaker_id": "a",
                    "role": "verification",
                    "groups": {
                        p: [i] if p != "u" else [] for i, p in enumerate(phones)
                    },
                }
            ],
        }
        result = study.score(data, study.np.eye(5, dtype=study.np.float32), phones)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r["score"] is None for r in result))

    def test_paired_bootstrap_zero_when_every_arm_is_identical(self):
        records = [
            {
                "query_id": "q" + s,
                "speaker_id": s,
                "claimed_speaker_id": c,
                "is_genuine": s == c,
                "score": 0.9 if s == c else 0.1,
            }
            for s in ("a", "b")
            for c in ("a", "b")
        ]
        result = study.paired_intervals(
            {arm: records for arm in study.ARMS},
            {arm: 0.5 for arm in study.ARMS},
            100,
            17,
        )
        self.assertTrue(
            all(
                bounds == [0.0, 0.0]
                for item in result.values()
                for bounds in item.values()
            )
        )

    def test_jtube_excluded_query_slot_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            inputs = [
                {
                    "split": "test",
                    "speaker_id": "jtube_ABC",
                    "sha256": digest,
                    "position": position,
                }
                for digest, position in [("a" * 64, 0), ("b" * 64, 20)]
            ]
            segments = [
                {
                    "speaker_id": "jtube_abc",
                    "phoneme": phone,
                    "cache_index": i,
                    "segment_id": str(i),
                    "source_position": source["position"],
                    "original_mp3_sha256": source["sha256"],
                    "source_sha256": source["sha256"],
                }
                for i, (source, phone) in enumerate(
                    (r, p) for r in inputs for p in study.VOWELS
                )
            ]
            (path / "inputs.json").write_text(json.dumps(inputs))
            (path / "test-segments.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in segments)
            )
            data = study.jtube_inputs(
                {
                    "enrollment_clips": 20,
                    "enrollment_seed": 16,
                    "enrollment_interval_limit": 30,
                },
                path,
                "test",
                list(study.VOWELS),
                {"b" * 64},
            )
            self.assertEqual(len(data["queries"]), 1)
            self.assertFalse(any(data["queries"][0]["groups"].values()))
            self.assertTrue(all(data["profiles"]["jtube_abc"][p] for p in study.VOWELS))

    def test_train_export_uses_actual_class_count_and_can_resume(self):
        study.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            speakers = [f"s{i}" for i in range(3)]
            rows = [
                {"speaker_id": speaker, "phoneme": phone, "cache_index": i}
                for i, (speaker, phone) in enumerate(
                    (s, p) for s in speakers for p in study.VOWELS for _ in range(2)
                )
            ]
            (run / "preparation-report.json").write_text(
                json.dumps(
                    {"training_speakers": speakers, "train_feature_count": len(rows)}
                )
            )
            (run / "train-segments.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            study.np.random.default_rng(1).normal(size=(len(rows), 128)).astype(
                "<f4"
            ).tofile(run / "train-features.f32")
            (run / "design-freeze.json").write_text("{}\n")
            config = {
                "encoder": "statistics_mlp",
                "training_seed": 20260926,
                "maximum_updates": 2,
                "warmup_updates": 1,
                "checkpoint_interval": 1,
                "feature_statistics_sha256": "test-only",
            }
            pool = study.TrainingPool(config, run)
            trial = run / "trial"
            study.train_trial(config, run, trial, list(study.VOWELS), pool)
            self.assertEqual(
                json.loads((trial / "training-freeze.json").read_text())["settings"][
                    "cohort"
                ],
                3,
            )
            before = (trial / "encoder.pt").read_bytes()
            study.load_model(trial)
            study.train_trial(config, run, trial, list(study.VOWELS), pool)
            self.assertEqual(before, (trial / "encoder.pt").read_bytes())


if __name__ == "__main__":
    unittest.main()
