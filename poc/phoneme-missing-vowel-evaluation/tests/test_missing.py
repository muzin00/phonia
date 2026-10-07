"""Partial fusion and failure-sensitive validation/threshold contracts."""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import missing_study as study
import numpy as np
import torch
from partial_scoring import fuse_available, prepare_available, verify_available
from verification import VerificationInput, verify

sys.path.insert(0, str(study.ROOT / "poc/phoneme-verification/tests"))
import test_verification as fixtures


def vector(index=0):
    result = np.zeros(128)
    result[index] = 1
    return result


class FusionTests(unittest.TestCase):
    def test_present_vowels_have_equal_weight_despite_segment_count(self):
        references = {v: vector() for v in study.VOWELS}
        grouped = {
            "a": [vector()] * 10,
            "i": [vector(1)],
            "u": [vector(1)],
            "e": [vector(1)],
            "o": [],
        }
        values = fuse_available(references, grouped, 4)
        self.assertEqual(values["fused"], 0.25)
        self.assertIsNone(values["o"])
        self.assertEqual(values, fuse_available(references, grouped, 3))
        self.assertIsNone(fuse_available(references, grouped, 5))

    def test_five_vowels_match_original_fusion(self):
        references = {v: vector() for v in study.VOWELS}
        grouped = {v: [vector(i % 2)] for i, v in enumerate(study.VOWELS)}
        expected = study.vowel_scores(references, grouped)
        for minimum in (3, 4, 5):
            self.assertEqual(fuse_available(references, grouped, minimum), expected)

    def test_minimum_support_and_invalid_vectors(self):
        references = {v: vector() for v in study.VOWELS}
        for count in range(6):
            groups = {
                v: [vector()] if i < count else [] for i, v in enumerate(study.VOWELS)
            }
            for minimum in (3, 4, 5):
                self.assertEqual(
                    fuse_available(references, groups, minimum) is not None,
                    count >= minimum,
                )
        complete = {v: [vector()] for v in study.VOWELS}
        for bad in (np.zeros(128), np.ones(128), np.full(128, np.nan), np.ones(1)):
            with self.assertRaises(ValueError):
                fuse_available(references, {**complete, "a": [bad]}, 3)
        for minimum in (0, 2, 6, True, 3.0):
            with self.assertRaises(ValueError):
                fuse_available(references, complete, minimum)


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.VerificationTests.setUp(self)

    def partial(self, count=4):
        return VerificationInput(
            "partial",
            tuple(s for s in self.query.segments if s.vowel in study.VOWELS[:count]),
        )

    def run_partial(self, query=None, minimum=4):
        return verify_available(
            self.profile,
            self.partial() if query is None else query,
            self.encoder,
            audio_root=self.root,
            minimum_vowels=minimum,
        )

    def test_complete_query_matches_public_verification(self):
        original = verify(self.profile, self.query, self.encoder, audio_root=self.root)
        result = self.run_partial(self.query)
        self.assertEqual(result["fused"], original.score)
        self.assertEqual(
            result,
            {
                "fused": original.score,
                **{v: original.data["vowels"][v]["score"] for v in study.VOWELS},
            },
        )

    def test_real_pcm_partial_support_and_order_invariance(self):
        result = self.run_partial()
        self.assertIsNone(result["o"])
        self.assertEqual(
            result["fused"],
            float(np.mean([result[v] for v in study.VOWELS[:4]], dtype=np.float64)),
        )
        reversed_query = VerificationInput(
            "reverse", tuple(reversed(self.partial().segments))
        )
        self.assertEqual(result, self.run_partial(reversed_query))
        self.assertIsNone(self.run_partial(minimum=5))
        self.assertIsNone(self.run_partial(self.partial(2), minimum=3))
        self.assertIsNotNone(self.run_partial(self.partial(3), minimum=3))

    def test_duplicate_segments_and_enrollment_overlap_are_still_rejected(self):
        for query in (
            VerificationInput(
                "duplicate", self.partial().segments + (self.partial().segments[0],)
            ),
            VerificationInput(
                "overlap", (self.enrollment[0], *self.partial().segments)
            ),
        ):
            with self.assertRaises(ValueError):
                self.run_partial(query)

    def test_wav_content_overlap_and_checksum_mismatch_are_rejected(self):
        segment = self.partial().segments[0]
        (self.root / "query/copy.wav").write_bytes(
            (self.root / self.enrollment[0].source_file).read_bytes()
        )
        for changed in (
            replace(
                segment,
                source_file="query/copy.wav",
                end_frame=self.enrollment[0].end_frame,
            ),
            replace(segment, source_sha256="0" * 64),
        ):
            query = VerificationInput(
                "bad-source", (changed, *self.partial().segments[1:])
            )
            with self.assertRaises(ValueError):
                self.run_partial(query)

    def test_quality_exclusions_do_not_become_available_vowels(self):
        fixtures.write_wav(self.root / "query/silent.wav", np.zeros(960))
        silent = replace(
            self.partial().segments[0], source_file="query/silent.wav", end_frame=960
        )
        query = VerificationInput(
            "silent",
            tuple([silent] + [s for s in self.partial().segments if s.vowel != "a"]),
        )
        prepared = prepare_available(
            query, self.profile, self.encoder, audio_root=self.root
        )
        self.assertEqual(prepared.counts["a"], 0)
        self.assertEqual(prepared.records[0]["status"], "excluded")
        self.assertIsNone(self.run_partial(query, minimum=4))
        self.assertIsNotNone(self.run_partial(query, minimum=3))

    def test_encoder_and_source_changes_are_rejected(self):
        original = copy.deepcopy(self.encoder.identity)
        self.encoder.identity["seed"] += 1
        with self.assertRaises(ValueError):
            self.run_partial()
        self.encoder.identity = original
        prepared = prepare_available(
            self.partial(), self.profile, self.encoder, audio_root=self.root
        )
        (self.root / self.partial().segments[0].source_file).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            prepared.require_unchanged_sources()


class EvaluationTests(unittest.TestCase):
    def test_calibration_rejects_test_and_cross_text(self):
        for split, role in (
            ("test", "verification"),
            ("validation", "cross_text_verification"),
        ):
            with self.assertRaises(ValueError):
                study.calibrate_cell([{"split": split, "role": role}], ["s1", "s2"])

    def test_new_test_requires_frozen_thresholds_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "design-freeze.json").write_text(
                json.dumps(
                    {
                        "status": "design_frozen_before_new_inference",
                        "runtime": study.previous.runtime(),
                        "files": {},
                    }
                )
            )
            (run / "test-inputs.json").write_text(json.dumps({"split": "test"}))
            with self.assertRaises(FileNotFoundError):
                study.frozen_inputs(run, "test")
            threshold = run / "thresholds.json"
            threshold.write_text("fixed")
            (run / "evaluation-freeze.json").write_text(
                json.dumps(
                    {
                        "status": "thresholds_frozen_before_new_test_inference",
                        "files": {threshold.name: study.sha256_file(threshold)},
                    }
                )
            )
            self.assertEqual(study.frozen_inputs(run, "test")["split"], "test")
            threshold.write_text("changed")
            with self.assertRaises(ValueError):
                study.frozen_inputs(run, "test")

    def test_missing_rescue_and_complete_regression_are_counted_separately(self):
        groups, thresholds = {}, {}
        for role in study.ROLES:
            for method in study.METHODS:
                scores = (
                    (None, 0.8, 0.4, None)
                    if method == "strict5"
                    else (0.8, 0.4, 0.8, 0.8)
                )
                groups[method, role] = [
                    {
                        "query_id": str(i),
                        "claimed_speaker_id": "s",
                        "is_genuine": i != 3,
                        "score": score,
                    }
                    for i, score in enumerate(scores)
                ]
                thresholds[method] = {
                    "operating_points": {
                        point: {"threshold": 0.5} for point, _ in study.POINTS
                    }
                }
        result = study.transitions(groups, thresholds)[
            "available4/verification/far_1pct"
        ]
        self.assertEqual(result["rescued_missing_genuine"], 1)
        self.assertEqual(result["rescued_scored_genuine"], 1)
        self.assertEqual(result["newly_rejected_scored_genuine"], 1)
        self.assertEqual(result["added_false_accepts_missing"], 1)
        self.assertEqual(result["candidate_all_false_rejects"], 1)

    def test_raw_score_audit_counts_no_score_as_rejection(self):
        rows = [
            {
                "query_id": str(i),
                "speaker_id": "s1",
                "claimed_speaker_id": "s1" if genuine else "s2",
                "is_genuine": genuine,
                "status": "no_score" if score is None else "scored",
                "score": score,
            }
            for i, (genuine, score) in enumerate(
                ((True, 0.7), (True, None), (False, 0.7), (False, None))
            )
        ]
        threshold = {
            "operating_points": {point: {"threshold": 0.5} for point, _ in study.POINTS}
        }
        cell = {
            "operating_points": {
                point: study.rates(rows, 0.5) for point, _ in study.POINTS
            }
        }
        self.assertEqual(cell["operating_points"]["far_1pct"]["all_input_frr"], 0.5)
        self.assertEqual(
            study.audit_rates(
                {("strict5", "verification"): rows},
                {"strict5/verification": cell},
                {"strict5": threshold},
            ),
            3,
        )
        cell["operating_points"]["far_1pct"]["all_input_frr"] = 0
        with self.assertRaises(ValueError):
            study.audit_rates(
                {("strict5", "verification"): rows},
                {"strict5/verification": cell},
                {"strict5": threshold},
            )


if __name__ == "__main__":
    unittest.main()
