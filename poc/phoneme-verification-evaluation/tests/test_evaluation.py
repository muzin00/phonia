"""Known-score, synthetic-WAV and leakage checks before test inference."""

import copy
import json
import math
import sys
import tempfile
import unittest
import wave
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(BASE),
    str(BASE.parent / "phoneme-speaker-encoder"),
    str(BASE.parent / "phoneme-user-registration"),
    str(BASE.parent / "phoneme-verification"),
    str(BASE.parent / "phoneme-speaker-encoder/scripts"),
]

from phase3_data.input import InputPipeline
from phase3_data.manifest import VOWELS, Segment, json_sha256, sha256_file
from phase3_train.metrics import roc_eer
from phase3_train.models import create_encoder
from registration import FrozenEncoder
from registration.encoder import PHASE3_CONFIG
from verification import VerificationResult
from weighted_bootstrap import weighted_eers

from evaluation import pipeline
from evaluation.inference import CachedEncoder, score_split
from evaluation.metrics import (
    KINDS,
    bootstrap_draws,
    bootstrap_intervals,
    calibrate,
    evaluate,
    operating_rates,
    validate_scores,
)
from evaluation.storage import load_document, read_rows, save_document, save_rows
from evaluation.trials import build_trials, checked_source, prepare_split


def fixture_scores(*, split="validation", role="verification"):
    speakers = ["A", "B", "C"]
    queries = [
        {
            "query_id": f"q-{speaker}-{i}",
            "speaker_id": speaker,
            "split": split,
            "role": role,
        }
        for speaker in speakers
        for i in range(2)
    ]
    trials = list(build_trials(queries, speakers, [1, 5, 10], "1.0.0"))
    rows = []
    for trial in trials:
        missing = trial["query_id"] == "q-A-1"
        score = (0.75 if trial["is_genuine"] else 0.25) + 0.05 * int(
            trial["query_id"][-1]
        )
        rows.append(
            {
                **trial,
                "status": "no_score" if missing else "scored",
                "reason": "missing_vowels" if missing else None,
                "scores": None if missing else {kind: score for kind in KINDS},
            }
        )
    return speakers, trials, rows


class MetricTests(unittest.TestCase):
    def test_complete_matrix_and_labels(self):
        _, trials, rows = fixture_scores()
        validate_scores(rows, trials)
        changed = copy.deepcopy(rows)
        changed[0]["is_genuine"] = not changed[0]["is_genuine"]
        with self.assertRaisesRegex(ValueError, "identity/label"):
            validate_scores(changed, trials)
        with self.assertRaisesRegex(ValueError, "count"):
            validate_scores(rows[:-1], trials)
        duplicate = [rows[0], *rows[1:-1], rows[0]]
        with self.assertRaises(ValueError):
            validate_scores(duplicate, trials)

    def test_no_score_denominators_and_acceptance_equality(self):
        _, _, rows = fixture_scores()
        selected = [r for r in rows if r["enrollment_count"] == 10]
        rates = operating_rates(selected, "fused", 0.75)
        self.assertEqual((rates["genuine"], rates["impostor"]), (5, 10))
        self.assertEqual(
            (rates["no_score_genuine"], rates["no_score_impostor"]), (1, 2)
        )
        self.assertEqual(rates["frr"], 0)
        self.assertEqual(rates["all_input_frr"], 1 / 6)
        rates = operating_rates(selected, "fused", 0.25)
        self.assertEqual(rates["far"], 1)
        self.assertEqual(rates["all_input_far"], 10 / 12)

    def test_calibration_ties_and_floor(self):
        speakers, _, rows = fixture_scores()
        thresholds = calibrate(rows, [1, 5, 10], speakers)
        t = thresholds["conditions"]["n10/fused"]["far_1pct"]
        self.assertEqual(t["threshold"], float(np.nextafter(0.3, np.inf)))
        self.assertEqual(t["false_accepts"], 0)
        self.assertEqual(t["frr"], 0)
        self.assertEqual(len(thresholds["conditions"]), 18)

    def test_calibration_rejects_test_and_cross_text(self):
        for split, role in (
            ("test", "verification"),
            ("validation", "cross_text_verification"),
        ):
            speakers, _, rows = fixture_scores(split=split, role=role)
            with self.assertRaisesRegex(ValueError, "validation/verification"):
                calibrate(rows, [1, 5, 10], speakers)

    def test_fixed_validation_thresholds_are_not_retuned(self):
        speakers, _, rows = fixture_scores()
        thresholds = calibrate(rows, [1, 5, 10], speakers)
        _, _, test = fixture_scores(split="test")
        for row in test:
            if row["status"] == "scored":
                row["scores"] = dict.fromkeys(KINDS, 0.9)
        before = json_sha256(thresholds)
        metrics, _ = evaluate(test, thresholds, [1, 5, 10], ["verification"], speakers)
        self.assertEqual(
            metrics["conditions"]["verification/n10/fused"]["operating_points"][
                "far_1pct"
            ]["far"],
            1,
        )
        self.assertEqual(before, json_sha256(thresholds))
        invalid = {**thresholds, "split": "test"}
        with self.assertRaisesRegex(ValueError, "validation/verification"):
            evaluate(test, invalid, [1, 5, 10], ["verification"], speakers)

    def test_zero_scored_speaker_is_not_removed(self):
        speakers, _, rows = fixture_scores()
        for row in rows:
            if row["speaker_id"] == "A":
                row.update(status="no_score", reason="missing_vowels", scores=None)
        with self.assertRaisesRegex(ValueError, "not evaluable"):
            calibrate(rows, [1, 5, 10], speakers)

    def test_invalid_scores_are_rejected(self):
        for bad in (math.nan, math.inf, -1.001, True):
            _, trials, rows = fixture_scores()
            rows[0]["scores"]["a"] = bad
            with self.assertRaises(ValueError):
                validate_scores(rows, trials)

    def test_bootstrap_pair_weights_match_expanded_trials_with_ties(self):
        scores = np.array([0.5, 0.5, 0.9, 0.4, 0.5, 0.1, 0.5, 0.4])
        q = np.array([0, 1, 2, 0, 1, 2, 1, 2])
        c = np.array([0, 1, 2, 1, 0, 0, 2, 1])
        counts = np.array([[1, 1, 1], [2, 1, 0], [0, 1, 2]])
        actual = weighted_eers(scores, q, c, counts)
        expected = []
        for draw in counts:
            weights = np.where(q == c, draw[q], draw[q] * draw[c])
            expected.append(
                roc_eer(np.repeat(scores, weights), np.repeat(q == c, weights))["eer"]
            )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_shared_bootstrap_no_score_weighted_denominators(self):
        speakers, _, rows = fixture_scores()
        selected = [r for r in rows if r["enrollment_count"] == 10]
        points = {"primary": {"threshold": 0.75}}
        draws = {"speakers": speakers, "multiplicities": [[2, 1, 0], [0, 1, 2]]}
        result = bootstrap_intervals(selected, "fused", points, draws)
        # Genuine total=6; missing A contributes n_A (2 or 0), never n_A squared.
        expected = np.percentile([2 / 6, 0], [2.5, 97.5], method="linear")
        np.testing.assert_allclose(
            result["operating_points"]["primary"]["all_input_frr"], expected
        )
        self.assertEqual(result["pooled_eer"], [0.0, 0.0])

    def test_draws_are_reproducible_and_shared(self):
        a = bootstrap_draws(["A", "B", "C"], replicates=20)
        self.assertEqual(a, bootstrap_draws(["A", "B", "C"], replicates=20))
        self.assertTrue(
            all(sum(c) == 3 and sum(x > 0 for x in c) >= 2 for c in a["multiplicities"])
        )


class SourceAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.original = []
        self.segments = []
        for index, speaker in enumerate(("A", "B")):
            for role, count in (("enrollment", 10), ("verification", 1)):
                length = 960 * 5 * count
                path = self.root / f"{speaker}-{role}.wav"
                pcm = 5000 * np.sin(
                    np.arange(length)
                    * (300 + index * 170 + (0 if role == "enrollment" else 100))
                    * 2
                    * np.pi
                    / 24000
                )
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(pcm.astype("<i2").tobytes())
                checksum = sha256_file(path)
                row = {
                    "utterance_id": path.stem,
                    "source_file": path.name,
                    "source_sha256": checksum,
                    "frame_count": length,
                    "speaker_id": speaker,
                    "split": "validation",
                    "evaluation_role": role,
                    "session_id": None,
                }
                self.original.append(row)
                for v, vowel in enumerate(VOWELS):
                    for rank in range(count):
                        offset = (v * count + rank) * 960
                        self.segments.append(
                            Segment(
                                f"{path.stem}-{vowel}-{rank}",
                                speaker,
                                vowel,
                                "validation",
                                role,
                                (),
                                path.name,
                                offset,
                                offset + 960,
                                checksum,
                            )
                        )
        (self.root / "original.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.original)
        )
        (self.root / "split.json").write_text(
            json.dumps(
                {
                    "speaker_splits": {
                        "train": ["train"],
                        "validation": ["A", "B"],
                        "test": ["X", "Y"],
                    }
                }
            )
        )
        self.protocol = {
            "inputs": {
                "speaker_split": {"path": "split.json"},
                "utterance_manifest": {"path": "original.jsonl"},
                "segment_manifest": {"path": "segments.jsonl"},
            },
            "splits": {"expected_speakers_per_split": 2, "roles": ["verification"]},
            "enrollment": {"seed": 20260930, "counts_per_vowel": [1, 5, 10]},
            "protocol_version": "1.0.0",
        }

    def prepared(self):
        with (
            patch("evaluation.trials.ROOT", self.root),
            patch("evaluation.trials.iter_segments", return_value=iter(self.segments)),
        ):
            return prepare_split(self.protocol, "validation", audio_root=self.root)

    def encoder(self):
        config = json.loads((PHASE3_CONFIG / "baseline-log-mel.json").read_text())[
            "input"
        ]
        raw = InputPipeline(config, rms_enabled=False)
        statistics = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "manifest_sha256": "1" * 64,
            "cohort": 70,
            "rms_enabled": False,
            "preprocessing_sha256": raw.preprocessing_sha256,
            "mean": [0.0] * 64,
            "population_variance": [1.0] * 64,
        }
        statistics["sha256"] = json_sha256(statistics)
        pipeline = InputPipeline(config, rms_enabled=False, statistics=statistics)
        torch.manual_seed(29)
        identity = {
            "design_version": "2.0.0",
            "encoder": "statistics_mlp",
            "seed": 20260926,
            **dict.fromkeys(
                (
                    "checkpoint_sha256",
                    "encoder_config_sha256",
                    "feature_statistics_sha256",
                    "implementation_sha256",
                ),
                "2" * 64,
            ),
            "preprocessing_sha256": pipeline.preprocessing_sha256,
        }
        return FrozenEncoder(create_encoder("statistics_mlp"), pipeline, identity)

    def test_nested_enrollment_and_one_wav_queries(self):
        prepared = self.prepared()
        for speaker in ("A", "B"):
            inputs = [r for r in prepared["enrollment"] if r["user_id"] == speaker]
            for small, large in pairwise(inputs):
                self.assertTrue(
                    {r["segment_id"] for r in small["segments"]}
                    < {r["segment_id"] for r in large["segments"]}
                )
        for q in prepared["queries"]:
            self.assertEqual(
                {r["source_file"] for r in q["segments"]}, {q["source_file"]}
            )

    def test_checksum_frame_and_root_errors_are_fatal(self):
        row = self.original[0]
        for changed in (
            {**row, "source_sha256": "0" * 64},
            {**row, "frame_count": 1},
            {**row, "source_file": "../outside.wav"},
        ):
            with self.assertRaises(ValueError):
                checked_source(changed, self.root)

    def test_copied_wav_cannot_cross_enrollment_and_query(self):
        source = self.root / "A-enrollment.wav"
        copied = self.root / "A-verification.wav"
        copied.write_bytes(source.read_bytes())
        self.original[1].update(
            source_sha256=sha256_file(source),
            frame_count=self.original[0]["frame_count"],
        )
        (self.root / "original.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.original)
        )
        with self.assertRaisesRegex(ValueError, "content leakage"):
            self.prepared()

    def test_score_api_roundtrip_cache_equivalence_and_no_score(self):
        prepared = self.prepared()
        prepared["queries"][1]["segments"] = [
            r for r in prepared["queries"][1]["segments"] if r["vowel"] != "o"
        ]
        trials = list(
            build_trials(prepared["queries"], ["A", "B"], [1, 5, 10], "1.0.0")
        )
        output = self.root / "result"
        invariants = score_split(
            prepared, trials, self.encoder(), output, audio_root=self.root
        )
        rows = list(read_rows(output / "scores/validation.jsonl"))
        validate_scores(rows, trials)
        self.assertEqual(sum(r["status"] == "scored" for r in rows), 6)
        self.assertEqual(sum(r["status"] == "no_score" for r in rows), 6)
        self.assertGreater(invariants["cache_hits"], 0)
        self.assertTrue(invariants["parameters_unchanged"])
        for row in read_rows(output / "phase5-results/validation.jsonl.gz"):
            document = row["result"]
            checksum = document.pop("sha256")
            self.assertEqual(checksum, VerificationResult(document).checksum)

    def test_cache_key_includes_pcm_and_returns_copies(self):
        cached = CachedEncoder(self.encoder())
        cached.select(
            [
                {
                    "segment_id": "one",
                    "source_sha256": "3" * 64,
                    "start_frame": 0,
                    "end_frame": 960,
                }
            ]
        )
        samples = torch.sin(torch.arange(960) * 0.1)
        original = cached.embed(samples, "one")
        changed = cached.embed(samples * 0.5, "one")
        self.assertEqual(cached.misses, 2)
        original[:] = 0
        self.assertAlmostEqual(np.linalg.norm(cached.embed(samples, "one")), 1)
        self.assertTrue(np.isfinite(changed).all())

    def test_immutable_documents_and_rows(self):
        path = self.root / "document.json"
        save_document(path, {"value": 1})
        self.assertEqual(load_document(path), {"value": 1})
        with self.assertRaises(FileExistsError):
            save_document(path, {"value": 2})
        document = json.loads(path.read_text())
        document["value"] = 3
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_document(path)
        for suffix in ("jsonl", "jsonl.gz"):
            path = self.root / f"rows.{suffix}"
            save_rows(path, [{"value": 1}])
            self.assertEqual(list(read_rows(path)), [{"value": 1}])
            with self.assertRaises(FileExistsError):
                save_rows(path, [{"value": 2}])

    def test_frozen_plan_rejects_changed_artifact(self):
        path = self.root / "pinned-input.jsonl"
        save_rows(path, [{"value": 1}])
        plan = {
            "implementation_files": {},
            "runtime": {},
            "files": {path.name: sha256_file(path)},
        }
        save_document(self.root / "evaluation-plan.json", plan)
        with (
            patch(
                "evaluation.pipeline.require_prepared",
                return_value=({"implementation_files": {}}, {}),
            ),
            patch("evaluation.pipeline.runtime", return_value={}),
        ):
            pipeline.require_plan(self.root)
            path.write_text('{"value":2}\n')
            with self.assertRaisesRegex(ValueError, "frozen file changed"):
                pipeline.require_plan(self.root)

    def test_test_inference_requires_frozen_plan(self):
        with (
            patch(
                "evaluation.pipeline.require_prepared",
                return_value=({"implementation_files": {}}, {}),
            ),
            patch("evaluation.pipeline.FrozenEncoder.from_bundle") as load,
        ):
            with self.assertRaises(FileNotFoundError):
                pipeline.test(self.root)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
