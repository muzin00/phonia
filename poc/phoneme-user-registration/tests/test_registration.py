"""Enrollment contracts using synthetic audio, without JVS or training artifacts."""

import copy
import json
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE.parent / "phoneme-speaker-encoder")]

from phase3_data.input import InputPipeline, read_pcm_slice
from phase3_data.manifest import VOWELS, json_sha256, sha256_file
from phase3_train.evaluation import _profiles
from phase3_train.models import create_encoder

from registration import (
    EnrollmentSegment,
    FrozenEncoder,
    IncompleteEnrollment,
    UserProfile,
    load_profile,
    load_registration_input,
    register_user,
    save_profile,
)
from registration.encoder import PHASE3_CONFIG, POLICY


def write_wav(path, samples, *, rate=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        config = json.loads((PHASE3_CONFIG / "baseline-log-mel.json").read_text())[
            "input"
        ]
        raw_pipeline = InputPipeline(config, rms_enabled=False)
        self.statistics = {
            "schema_version": 1,
            "design_version": "2.0.0",
            "manifest_sha256": "1" * 64,
            "cohort": 70,
            "rms_enabled": False,
            "preprocessing_sha256": raw_pipeline.preprocessing_sha256,
            "mean": [0.0] * 64,
            "population_variance": [1.0] * 64,
        }
        self.statistics["sha256"] = json_sha256(self.statistics)
        self.pipeline = InputPipeline(
            config, rms_enabled=False, statistics=self.statistics
        )
        torch.manual_seed(19)
        self.model = create_encoder("statistics_mlp")
        self.identity = {
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
            "preprocessing_sha256": self.pipeline.preprocessing_sha256,
        }
        self.encoder = FrozenEncoder(self.model, self.pipeline, self.identity)
        self.segments = []
        for v, vowel in enumerate(VOWELS):
            chunks, offset = [], 0
            for n in range(10):
                length = 720 + n * 120
                samples = 6000 * np.sin(
                    np.arange(length) * (240 + v * 90 + n * 7) * 2 * np.pi / 24000
                )
                chunks.append(samples)
                self.segments.append(
                    EnrollmentSegment(
                        f"{vowel}-{n:02d}",
                        vowel,
                        f"audio/{vowel}.wav",
                        offset,
                        offset + length,
                    )
                )
                offset += length
            write_wav(self.root / f"audio/{vowel}.wav", np.concatenate(chunks))

    def register(self, segments=None, **kwargs):
        return register_user(
            "new-user",
            self.segments if segments is None else segments,
            self.encoder,
            audio_root=self.root,
            **kwargs,
        )

    def test_repeat_and_input_order_produce_identical_profile_bytes(self):
        first = self.register()
        second = self.register(list(reversed(self.segments)))
        first_path, second_path = self.root / "first.json", self.root / "second.json"
        save_profile(first, first_path)
        save_profile(second, second_path)
        self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        loaded = load_profile(first_path, encoder=self.encoder)
        self.assertEqual(loaded.user_id, "new-user")
        for vowel in VOWELS:
            np.testing.assert_array_equal(loaded.vector(vowel), first.vector(vowel))

    def test_registration_matches_phase3_profile_aggregation(self):
        profile = self.register()
        embeddings = {
            s.segment_id: self.encoder.embed(read_pcm_slice(self.root, s), s.segment_id)
            for s in self.segments
        }
        rows = [
            {
                "speaker_id": f"speaker-{speaker}",
                "vowel": s.vowel,
                "rank": int(s.segment_id.split("-")[1]) + 1,
                "segment_id": s.segment_id,
            }
            for speaker in range(15)
            for s in self.segments
        ]
        reference = _profiles(rows, embeddings)
        for vowel in VOWELS:
            np.testing.assert_allclose(
                profile.vector(vowel),
                reference[("speaker-0", vowel, 10)],
                atol=1e-15,
                rtol=0,
            )

    def test_registration_preserves_model_and_statistics(self):
        before = {
            name: value.clone() for name, value in self.model.state_dict().items()
        }
        statistics = copy.deepcopy(self.statistics)
        self.register()
        self.assertEqual(statistics, self.statistics)
        self.assertFalse(self.model.training)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(
            all(not p.requires_grad and p.grad is None for p in self.model.parameters())
        )

    def test_missing_and_insufficient_vowels_fail_before_inference(self):
        for segments, count in (
            ([s for s in self.segments if s.vowel != "u"], 0),
            (self.segments[:-1], 9),
        ):
            with (
                self.subTest(count=count),
                patch.object(self.encoder, "embed") as embed,
            ):
                with self.assertRaises(IncompleteEnrollment) as raised:
                    self.register(segments)
                self.assertIn(count, raised.exception.counts.values())
                embed.assert_not_called()

    def test_short_silent_and_near_silent_are_reported_not_weighted(self):
        extras = []
        for identifier, samples in (
            ("short", np.ones(240) * 1000),
            ("silent", np.zeros(720)),
            ("quiet", np.tile([10, -10], 360)),
        ):
            write_wav(self.root / f"audio/{identifier}.wav", samples)
            extras.append(
                EnrollmentSegment(
                    identifier, "a", f"audio/{identifier}.wav", 0, len(samples)
                )
            )
        profile = self.register(self.segments + extras)
        self.assertEqual(profile.data["vowels"]["a"]["segment_count"], 10)
        self.assertEqual(
            {
                r["reason"]
                for r in profile.data["segments"]
                if r["status"] == "excluded"
            },
            {"too_short", "silent", "near_silent"},
        )
        insufficient = [s for s in self.segments if s.segment_id != "a-00"] + extras
        with self.assertRaises(IncompleteEnrollment) as raised:
            self.register(insufficient)
        self.assertEqual(raised.exception.counts["a"], 9)

    def test_excess_segments_are_all_used_and_lower_minimum_is_explicit(self):
        source = self.root / "audio/extra.wav"
        write_wav(source, np.tile([1000, -1000], 600))
        extra = EnrollmentSegment("a-extra", "a", "audio/extra.wav", 0, 1200)
        self.assertEqual(
            self.register(self.segments + [extra]).data["vowels"]["a"]["segment_count"],
            11,
        )
        one_each = [s for s in self.segments if s.segment_id.endswith("00")]
        profile = self.register(one_each, minimum_segments=1)
        self.assertEqual(profile.data["minimum_segments_per_vowel"], 1)
        self.assertTrue(
            all(item["segment_count"] == 1 for item in profile.data["vowels"].values())
        )

    def test_duplicate_ids_and_duplicate_audio_cannot_inflate_count(self):
        for extra in (
            self.segments[0],
            replace(self.segments[0], segment_id="different-id"),
        ):
            with (
                self.subTest(extra=extra.segment_id),
                self.assertRaisesRegex(ValueError, "duplicate"),
            ):
                self.register(self.segments + [extra])

    def test_wrong_checksum_format_range_and_escape_fail(self):
        cases = (
            (replace(self.segments[0], source_sha256="0" * 64), "checksum"),
            (replace(self.segments[0], end_frame=1000000), "exceeds"),
            (replace(self.segments[0], source_file="../outside.wav"), "outside"),
            (
                replace(self.segments[0], source_file=str(self.root / "audio/a.wav")),
                "outside",
            ),
        )
        for segment, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.register([segment, *self.segments[1:]])
        write_wav(self.root / "wrong-rate.wav", np.ones(720) * 1000, rate=16000)
        with self.assertRaisesRegex(ValueError, "24 kHz"):
            self.register(
                [
                    replace(self.segments[0], source_file="wrong-rate.wav"),
                    *self.segments[1:],
                ]
            )

    def test_profile_corruption_schema_counts_vectors_and_compatibility_fail(self):
        profile = self.register()
        path = self.root / "profile.json"
        save_profile(profile, path)
        document = json.loads(path.read_text())
        document["user_id"] = "tampered"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_profile(path)
        changes = (
            lambda d: d.update(schema_version=2),
            lambda d: d["vowels"].pop("u"),
            lambda d: d["vowels"]["a"].update(segment_count=11),
            lambda d: d["vowels"]["a"].update(vector=[0.0] * 128),
            lambda d: d["vowels"]["a"].update(vector=[float("nan")] * 128),
        )
        for change in changes:
            data = copy.deepcopy(profile.data)
            change(data)
            with self.assertRaises(ValueError):
                UserProfile(data)
        other = FrozenEncoder(
            self.model, self.pipeline, {**self.identity, "checkpoint_sha256": "3" * 64}
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            profile.require_compatible(other)

    def test_save_does_not_overwrite_and_leaves_no_partial_files(self):
        profile = self.register()
        path = self.root / "profiles/user.json"
        save_profile(profile, path)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_profile(profile, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_input_schema_and_segment_validation(self):
        path = self.root / "input.json"
        from dataclasses import asdict

        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "user_id": "new-user",
                    "segments": [asdict(s) for s in self.segments],
                }
            )
        )
        self.assertEqual(load_registration_input(path), ("new-user", self.segments))
        for kwargs in (
            {"vowel": "x"},
            {"start_frame": -1},
            {"start_frame": True},
            {"end_frame": 0},
            {"segment_id": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.segments[0], **kwargs)

    def test_bundle_load_needs_no_training_data_and_checks_pinned_files(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        stats_path = bundle / "feature-statistics.json"
        stats_path.write_text(json.dumps(self.statistics))
        settings = {
            "encoder": "statistics_mlp",
            "seed": 20260926,
            "cohort": 70,
            "rms_enabled": False,
            "supcon_enabled": True,
            "overfit": False,
        }
        config_hash = json_sha256(
            {
                name: sha256_file(PHASE3_CONFIG / name)
                for name in ("log-mel-encoders.json", "waveform-encoders.json")
            }
        )
        run = {
            "settings": settings,
            "configuration_sha256": json_sha256(settings),
            "encoder_config_sha256": config_hash,
            "manifest": "/unavailable/training.jsonl",
        }
        (bundle / "run.json").write_text(json.dumps(run))
        torch.save(
            {
                "schema_version": 1,
                "design_version": "2.0.0",
                "run_seed": 20260926,
                "config_sha256": json_sha256(settings),
                "manifest_sha256": "1" * 64,
                "feature_statistics_sha256": sha256_file(stats_path),
                "model": self.model.state_dict(),
            },
            bundle / "best.pt",
        )
        hashes = {name: sha256_file(bundle / name) for name in POLICY["bundle_sha256"]}
        with patch.dict(POLICY, {"bundle_sha256": hashes}):
            loaded = FrozenEncoder.from_bundle(bundle)
            pcm = read_pcm_slice(self.root, self.segments[0])
            np.testing.assert_array_equal(
                loaded.embed(pcm, "sample"), self.encoder.embed(pcm, "sample")
            )
            (bundle / "best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum.*best.pt"):
                FrozenEncoder.from_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
