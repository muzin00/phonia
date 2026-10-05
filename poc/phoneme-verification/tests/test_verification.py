"""Verification contracts with synthetic PCM and analytically known scores."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(BASE),
    str(BASE.parent / "phoneme-user-registration"),
    str(BASE.parent / "phoneme-speaker-encoder"),
]

from phase3_data.input import InputPipeline, read_pcm_slice
from phase3_data.manifest import VOWELS, json_sha256
from phase3_train.models import create_encoder
from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    UserProfile,
    register_user,
    save_profile,
)
from registration.encoder import PHASE3_CONFIG

from verification import (
    IncompleteVerification,
    VerificationInput,
    VerificationResult,
    load_result,
    load_verification_input,
    save_result,
    verify,
)
from verification.jvs import prepare_jvs_input
from verification.scoring import write_document

spec = importlib.util.spec_from_file_location(
    "verify_user_cli", BASE / "scripts/verify_user.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def write_wav(path, samples, *, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config = json.loads((PHASE3_CONFIG / "baseline-log-mel.json").read_text())[
            "input"
        ]
        raw = InputPipeline(config, rms_enabled=False)
        self.statistics = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "manifest_sha256": "1" * 64,
            "cohort": 70,
            "rms_enabled": False,
            "preprocessing_sha256": raw.preprocessing_sha256,
            "mean": [0.0] * 64,
            "population_variance": [1.0] * 64,
        }
        self.statistics["sha256"] = json_sha256(self.statistics)
        pipeline = InputPipeline(config, rms_enabled=False, statistics=self.statistics)
        torch.manual_seed(19)
        model = create_encoder("statistics_mlp")
        identity = {
            "design_version": "2.0.0",
            "encoder": "statistics_mlp",
            "seed": 20260926,
            **{
                key: "2" * 64
                for key in (
                    "checkpoint_sha256",
                    "encoder_config_sha256",
                    "feature_statistics_sha256",
                    "implementation_sha256",
                )
            },
            "preprocessing_sha256": pipeline.preprocessing_sha256,
        }
        self.encoder = FrozenEncoder(model, pipeline, identity)
        enrollment, queries = [], []
        for index, vowel in enumerate(VOWELS):
            length = 840 + index * 120
            samples = 6000 * np.sin(
                np.arange(length) * (300 + index * 90) * 2 * np.pi / 24000
            )
            write_wav(self.root / f"enrollment/{vowel}.wav", samples)
            enrollment.append(
                EnrollmentSegment(
                    f"enroll-{vowel}", vowel, f"enrollment/{vowel}.wav", 0, length
                )
            )
            chunks, offset = [], 0
            for number, length in enumerate((960, 1200)):
                chunks.append(
                    5000
                    * np.sin(
                        np.arange(length)
                        * (500 + index * 100 + number * 20)
                        * 2
                        * np.pi
                        / 24000
                    )
                )
                queries.append(
                    EnrollmentSegment(
                        f"query-{vowel}-{number}",
                        vowel,
                        f"query/{vowel}.wav",
                        offset,
                        offset + length,
                    )
                )
                offset += length
            write_wav(self.root / f"query/{vowel}.wav", np.concatenate(chunks))
        self.enrollment = enrollment
        self.profile = register_user(
            "user-A", enrollment, self.encoder, audio_root=self.root, minimum_segments=1
        )
        self.query = VerificationInput("query-001", tuple(queries))

    def compare(self, query=None, profile=None):
        return verify(
            self.profile if profile is None else profile,
            self.query if query is None else query,
            self.encoder,
            audio_root=self.root,
        )

    def test_known_cosines_and_equal_vowel_weight_despite_unequal_counts(self):
        reference = np.zeros(128)
        reference[0] = 1
        for item in self.profile.data["vowels"].values():
            item["vector"] = reference.tolist()
        query = VerificationInput(
            "analytic",
            tuple(
                s
                for s in self.query.segments
                if s.vowel == "a" or s.segment_id.endswith("0")
            ),
        )
        values = {
            "query-a-0": 1.0,
            "query-a-1": -1.0,
            "query-i-0": 0.5,
            "query-u-0": -0.5,
            "query-e-0": 1.0,
            "query-o-0": 0.0,
        }

        def embedding(pcm, segment_id):
            vector = np.zeros(128)
            vector[0] = values[segment_id]
            vector[1] = np.sqrt(1 - values[segment_id] ** 2)
            return vector

        with patch.object(self.encoder, "embed", side_effect=embedding):
            result = self.compare(query)
        self.assertEqual(result.data["vowels"]["a"]["score"], 0.0)
        self.assertEqual(result.score, 0.2)
        self.assertNotAlmostEqual(result.score, np.mean(list(values.values())))
        self.assertEqual(
            {row["segment_id"]: row["score"] for row in result.data["segments"]}, values
        )

    def test_single_segment_matches_phase3_cosine_calculation(self):
        query = VerificationInput(
            "one-each",
            tuple(s for s in self.query.segments if s.segment_id.endswith("0")),
        )
        result = self.compare(query)
        for segment in query.segments:
            vector = self.encoder.embed(
                read_pcm_slice(self.root, segment), segment.segment_id
            )
            expected = float(np.dot(self.profile.vector(segment.vowel), vector))
            self.assertAlmostEqual(
                result.data["vowels"][segment.vowel]["score"], expected, places=14
            )

    def test_reordered_inputs_roundtrip_and_inference_preserve_state(self):
        before_model = {
            name: tensor.clone()
            for name, tensor in self.encoder.model.state_dict().items()
        }
        before_profile, before_statistics = (
            copy.deepcopy(self.profile.data),
            copy.deepcopy(self.statistics),
        )
        result = self.compare()
        repeated = self.compare(
            VerificationInput(self.query.query_id, tuple(reversed(self.query.segments)))
        )
        first, second = (
            self.root / "results/first.json",
            self.root / "results/second.json",
        )
        save_result(result, first)
        save_result(repeated, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(load_result(first, encoder=self.encoder).data, result.data)
        self.assertEqual(self.profile.data, before_profile)
        self.assertEqual(self.statistics, before_statistics)
        self.assertFalse(self.encoder.model.training)
        for name, value in self.encoder.model.state_dict().items():
            self.assertTrue(torch.equal(value, before_model[name]))
        self.assertTrue(
            all(
                not p.requires_grad and p.grad is None
                for p in self.encoder.model.parameters()
            )
        )

    def test_missing_vowel_or_empty_input_fails_before_inference(self):
        for segments, missing in (
            (tuple(s for s in self.query.segments if s.vowel != "u"), ["u"]),
            ((), list(VOWELS)),
        ):
            with (
                self.subTest(missing=missing),
                patch.object(self.encoder, "embed") as embed,
            ):
                with self.assertRaises(IncompleteVerification) as raised:
                    self.compare(VerificationInput("missing", segments))
                self.assertEqual(raised.exception.missing_vowels, missing)
                embed.assert_not_called()

    def test_quality_exclusions_and_shortage_after_exclusion(self):
        extras = []
        for identifier, samples in (
            ("short", np.ones(240) * 1000),
            ("silent", np.zeros(720)),
            ("quiet", np.tile([10, -10], 360)),
        ):
            write_wav(self.root / f"query/{identifier}.wav", samples)
            extras.append(
                EnrollmentSegment(
                    identifier, "a", f"query/{identifier}.wav", 0, len(samples)
                )
            )
        result = self.compare(
            VerificationInput("quality", self.query.segments + tuple(extras))
        )
        self.assertEqual(result.data["vowels"]["a"]["query_segment_count"], 2)
        self.assertEqual(
            {
                row["reason"]
                for row in result.data["segments"]
                if row["status"] == "excluded"
            },
            {"too_short", "silent", "near_silent"},
        )
        baseline = self.compare()
        self.assertEqual(result.score, baseline.score)
        missing = tuple(s for s in self.query.segments if s.vowel != "a") + tuple(
            extras
        )
        with self.assertRaises(IncompleteVerification) as raised:
            self.compare(VerificationInput("quality-missing", missing))
        self.assertEqual(raised.exception.counts["a"], 0)

    def test_enrollment_sources_ids_and_renamed_copies_are_rejected(self):
        copied = self.root / "query/copied-enrollment.wav"
        copied.write_bytes((self.root / "enrollment/a.wav").read_bytes())
        candidates = (
            replace(
                self.query.segments[0], source_file="enrollment/a.wav", end_frame=840
            ),
            replace(
                self.query.segments[0],
                source_file="query/copied-enrollment.wav",
                end_frame=840,
            ),
            replace(self.query.segments[0], segment_id=self.enrollment[0].segment_id),
        )
        for candidate in candidates:
            with (
                self.subTest(candidate=candidate),
                patch.object(self.encoder, "embed") as embed,
            ):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self.compare(
                        VerificationInput(
                            "overlap", (candidate, *self.query.segments[1:])
                        )
                    )
                embed.assert_not_called()

    def test_duplicate_ids_intervals_and_copied_query_audio_are_rejected(self):
        copied = self.root / "query/copy.wav"
        copied.write_bytes((self.root / "query/a.wav").read_bytes())
        candidates = (
            self.query.segments[0],
            replace(self.query.segments[0], segment_id="duplicate-range"),
            replace(
                self.query.segments[0],
                segment_id="duplicate-copy",
                source_file="query/copy.wav",
            ),
        )
        for extra in candidates:
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "duplicate"),
            ):
                self.compare(
                    VerificationInput("duplicate", self.query.segments + (extra,))
                )

    def test_invalid_checksum_range_format_and_paths_fail_before_inference(self):
        write_wav(self.root / "query/wrong-rate.wav", np.ones(1200) * 1000, rate=16000)
        (self.root / "query/broken.wav").write_bytes(b"not a WAV")
        candidates = (
            (replace(self.query.segments[0], source_sha256="0" * 64), "checksum"),
            (replace(self.query.segments[0], end_frame=100000), "exceeds"),
            (replace(self.query.segments[0], source_file="../outside.wav"), "outside"),
            (
                replace(
                    self.query.segments[0], source_file=str(self.root / "query/a.wav")
                ),
                "outside",
            ),
            (
                replace(self.query.segments[0], source_file="query/wrong-rate.wav"),
                "24 kHz",
            ),
            (
                replace(self.query.segments[0], source_file="query/broken.wav"),
                "invalid source WAV",
            ),
        )
        for candidate, message in candidates:
            with (
                self.subTest(message=message),
                patch.object(self.encoder, "embed") as embed,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    self.compare(
                        VerificationInput("bad", (candidate, *self.query.segments[1:]))
                    )
                embed.assert_not_called()

    def test_incompatible_profile_fails_before_source_access(self):
        data = copy.deepcopy(self.profile.data)
        data["encoder"]["preprocessing_sha256"] = "3" * 64
        with patch("verification.inputs.read_pcm_slice") as reader:
            with self.assertRaisesRegex(ValueError, "incompatible"):
                self.compare(profile=UserProfile(data))
            reader.assert_not_called()

    def test_invalid_embedding_values_shape_and_norm_fail(self):
        for vector in (
            np.zeros(128),
            np.ones(127),
            np.ones(128) * np.nan,
            np.ones(128) * np.inf,
        ):
            with (
                self.subTest(shape=vector.shape),
                patch.object(self.encoder, "embed", return_value=vector),
                self.assertRaisesRegex(ValueError, "invalid unit embedding"),
            ):
                self.compare()

    def test_source_or_profile_changes_during_inference_are_detected(self):
        original_embed = self.encoder.embed

        def changed_source(pcm, segment_id):
            result = original_embed(pcm, segment_id)
            with (self.root / "query/a.wav").open("ab") as stream:
                stream.write(b"changed")
            return result

        with (
            patch.object(self.encoder, "embed", side_effect=changed_source),
            self.assertRaisesRegex(ValueError, "source changed"),
        ):
            self.compare()

        def changed_profile(pcm, segment_id):
            self.profile.data["user_id"] = "modified-user"
            return original_embed(pcm, segment_id)

        with (
            patch.object(self.encoder, "embed", side_effect=changed_profile),
            self.assertRaisesRegex(ValueError, "profile changed"),
        ):
            self.compare()

    def test_result_corruption_and_recomputed_invalid_results_are_rejected(self):
        result = self.compare()
        path = self.root / "result.json"
        save_result(result, path)
        document = json.loads(path.read_text())
        document["user_id"] = "tampered"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_result(path)
        changes = (
            lambda d: d.update(schema_version=True),
            lambda d: d["vowels"].pop("u"),
            lambda d: d["vowels"]["a"].update(query_segment_count=3),
            lambda d: d["vowels"]["a"].update(score=-1.0),
            lambda d: d.update(verification_score=-1.0),
            lambda d: d.update(verification_score=float("nan")),
            lambda d: d["segments"][0].update(score=True),
            lambda d: d["segments"][0].update(source_sha256=None),
            lambda d: d["policy"].update(fusion="pooled_mean"),
            lambda d: d["segments"].append(copy.deepcopy(d["segments"][0])),
        )
        for change in changes:
            data = copy.deepcopy(result.data)
            change(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                VerificationResult(data)
        path.unlink()
        save_result(result, path)
        other = FrozenEncoder(
            self.encoder.model,
            self.encoder.pipeline,
            {**self.encoder.identity, "checkpoint_sha256": "3" * 64},
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            load_result(path, encoder=other)

    def test_atomic_save_refuses_overwrite_and_removes_temporary_files(self):
        path = self.root / "results/score.json"
        result = self.compare()
        save_result(result, path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_result(result, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_input_schema_query_identity_and_segment_validation(self):
        path = self.root / "input.json"
        data = {
            "schema_version": 1,
            "query_id": self.query.query_id,
            "segments": [asdict(s) for s in self.query.segments],
        }
        write_document(path, data)
        self.assertEqual(load_verification_input(path), self.query)
        for change in (
            lambda d: d.update(schema_version=True),
            lambda d: d.update(query_id=" "),
            lambda d: d.update(segments={}),
            lambda d: d["segments"][0].update(vowel="x"),
            lambda d: d["segments"][0].update(start_frame=True),
            lambda d: d["segments"].append(None),
        ):
            invalid = copy.deepcopy(data)
            change(invalid)
            path.write_text(json.dumps(invalid))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_verification_input(path)

    def test_cli_verification_inspection_missing_input_and_no_overwrite(self):
        profile_path, input_path, result_path = (
            self.root / "profile.json",
            self.root / "query.json",
            self.root / "result.json",
        )
        save_profile(self.profile, profile_path)
        write_document(
            input_path,
            {
                "schema_version": 1,
                "query_id": self.query.query_id,
                "segments": [asdict(s) for s in self.query.segments],
            },
        )

        def run(arguments):
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch.object(sys, "argv", ["verify_user.py", *arguments]),
                patch.object(
                    cli.FrozenEncoder, "from_bundle", return_value=self.encoder
                ),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                code = cli.main()
            return code, output.getvalue(), errors.getvalue()

        args = [
            "verify",
            "--profile",
            str(profile_path),
            "--input",
            str(input_path),
            "--audio-root",
            str(self.root),
            "--output",
            str(result_path),
        ]
        code, output, errors = run(args)
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(
            json.loads(output)["verification_score"], load_result(result_path).score
        )
        self.assertEqual(
            run(["inspect", "--result", str(result_path), "--bundle", "unused"])[0], 0
        )
        before = result_path.read_bytes()
        self.assertEqual(run(args)[0], 2)
        self.assertEqual(result_path.read_bytes(), before)
        result_path.unlink()
        data = json.loads(input_path.read_text())
        data["segments"] = [row for row in data["segments"] if row["vowel"] != "u"]
        input_path.write_text(json.dumps(data))
        code, output, errors = run(args)
        self.assertEqual((code, output), (2, ""))
        self.assertEqual(json.loads(errors)["missing_vowels"], ["u"])
        self.assertFalse(result_path.exists())

    def test_jvs_helper_selects_validation_queries_only_without_score_selection(self):
        rows = []
        for split, role in (
            ("train", "training"),
            ("test", "verification"),
            ("validation", "enrollment"),
            ("validation", "verification"),
            ("validation", "cross_text_verification"),
        ):
            for vowel in VOWELS:
                for number in range(3):
                    rows.append(
                        {
                            "schema_version": 1,
                            "design_version": "1.0.0",
                            "sample_rate_hz": 24000,
                            "channels": 1,
                            "sample_width_bytes": 2,
                            "storage_mode": "source_slice",
                            "frame_count": 1200,
                            "vowel_interval_id": f"{split}-{role}-{vowel}-{number}",
                            "speaker_id": "fixture",
                            "normalized_phoneme": vowel,
                            "split": split,
                            "evaluation_role": role,
                            "learning_curve_cohorts": [70] if split == "train" else [],
                            "source_file": f"{split}/{role}/{vowel}-{number}.wav",
                            "start_frame": 0,
                            "end_frame": 1200,
                            "source_sha256": "4" * 64,
                        }
                    )
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        for role in ("verification", "cross_text_verification"):
            first = prepare_jvs_input(manifest, "fixture", role=role, count=2)
            self.assertEqual(
                first, prepare_jvs_input(manifest, "fixture", role=role, count=2)
            )
            self.assertEqual(len(first["segments"]), 10)
            self.assertTrue(
                all(
                    row["segment_id"].startswith(f"validation-{role}-")
                    for row in first["segments"]
                )
            )
        with self.assertRaises(ValueError):
            prepare_jvs_input(manifest, "fixture", role="enrollment")
        with self.assertRaises(ValueError):
            prepare_jvs_input(manifest, "fixture", count=4)


if __name__ == "__main__":
    unittest.main()
