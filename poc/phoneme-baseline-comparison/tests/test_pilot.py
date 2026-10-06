"""Regression cases for the aggregation and cache mistakes a pilot must catch."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import (
    VOWELS,
    EmbeddingCache,
    digest,
    mean_profile,
    score_row,
    select_queries,
    vowel_scores,
)


class AggregationTests(unittest.TestCase):
    def test_query_mean_is_not_normalized_before_cosine(self):
        references = {v: np.array([1.0, 0.0]) for v in VOWELS}
        queries = {v: [np.array([1.0, 0.0]), np.array([0.0, 1.0])] for v in VOWELS}
        self.assertEqual(vowel_scores(references, queries)["fused"], 0.5)

    def test_vowels_have_equal_weight_despite_segment_counts(self):
        profile = {v: np.array([1.0, 0.0]) for v in VOWELS}
        query = {v: [np.array([0.0, 1.0])] for v in VOWELS}
        query["a"] = [np.array([1.0, 0.0])] * 10
        self.assertEqual(vowel_scores(profile, query)["fused"], 0.2)

    def test_missing_vowel_never_becomes_numeric_zero(self):
        query = {v: [np.array([1.0, 0.0])] for v in VOWELS}
        query["o"] = []
        self.assertIsNone(vowel_scores({}, query))

    def test_enrollment_mean_is_unit_and_equal_weighted(self):
        result = mean_profile([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        np.testing.assert_allclose(result, [2**-0.5, 2**-0.5], atol=1e-15)
        with self.assertRaises(ValueError):
            mean_profile([np.array([2.0, 0.0])])
        with self.assertRaises(ValueError):
            mean_profile([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])

    def test_nonfinite_and_zero_embeddings_stop_inference(self):
        for vector in ([0.0, 0.0], [float("nan"), 1.0]):
            with self.assertRaises(ValueError):
                mean_profile([np.array(vector)])

    def test_condition_ids_differ_even_when_audio_is_identical(self):
        inputs = {"protocol": {"protocol_version": "2.0.0"}}
        window = {
            "condition_id": "full",
            "window_id": "same",
            "counts": {v: 1 for v in VOWELS},
        }
        first = score_row(
            inputs,
            "ecapa_whole",
            window,
            {"trial_id": "trial"},
            {"vector": [1.0]},
            {"fused": 1.0},
        )
        second = score_row(
            inputs,
            "ecapa_whole",
            {**window, "condition_id": "max_5s"},
            {"trial_id": "trial"},
            {"vector": [1.0]},
            {"fused": 1.0},
        )
        self.assertNotEqual(first["score_id"], second["score_id"])
        self.assertEqual(first["profile_sha256"], second["profile_sha256"])
        self.assertEqual(first["window_id"], second["window_id"])


class CacheTests(unittest.TestCase):
    def audio(self):
        class Audio:
            def __init__(self):
                self.sources = {"one.wav": {"source_sha256": "a" * 64}}

            def slice(self, filename, first, last):
                return np.arange(first, last, dtype=np.float32)

        return Audio()

    def test_cache_uses_actual_bounds_and_returns_copies(self):
        calls = []

        def forward(pcm, sid):
            calls.append(pcm.tolist())
            return np.array([pcm[0] + 1, pcm[-1] + 1])

        cache = EmbeddingCache(self.audio(), {"checkpoint": "one"}, forward, 3)
        first = cache.embed("one.wav", 0, 3, "anchor")
        first[:] = 0
        self.assertGreater(np.linalg.norm(cache.embed("one.wav", 0, 3, "other-cap")), 0)
        second = cache.embed("one.wav", 1, 3, "same-anchor")
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(cache.vectors), 2)
        self.assertFalse(np.array_equal(second, next(iter(cache.vectors.values()))))

    def test_cache_rejects_nonreproducible_forward(self):
        counter = 0

        def forward(pcm, sid):
            nonlocal counter
            counter += 1
            return np.array([1.0, counter])

        with self.assertRaisesRegex(ValueError, "repeated embedding"):
            EmbeddingCache(self.audio(), {}, forward, 3).embed("one.wav", 0, 3, "a")

    def test_already_unit_vectors_preserve_exact_native_bits(self):
        vector = np.array([0.7, (1 - 0.7**2) ** 0.5])
        cache = EmbeddingCache(
            self.audio(), {}, lambda pcm, sid: vector, 3, already_unit=True
        )
        np.testing.assert_array_equal(cache.embed("one.wav", 0, 3, "a"), vector)


class SelectionTests(unittest.TestCase):
    def fixture(self):
        queries, sources = [], {}
        for speaker in ("s1", "s2", "s3"):
            for role in ("verification", "cross_text_verification"):
                name = f"{speaker}-{role}"
                source = {
                    "source_file": name,
                    "source_sha256": digest(name),
                    "speaker_id": speaker,
                    "split": "validation",
                    "evaluation_role": role,
                    "frame_count": 24000,
                }
                segments = [
                    {
                        "segment_id": name + v,
                        "vowel": v,
                        "source_file": name,
                        "source_sha256": source["source_sha256"],
                        "start_frame": i * 1000,
                        "end_frame": i * 1000 + 720,
                    }
                    for i, v in enumerate(VOWELS)
                ]
                queries.append(
                    {
                        "query_id": digest(name),
                        "utterance_id": name,
                        "source_file": name,
                        "source_sha256": source["source_sha256"],
                        "speaker_id": speaker,
                        "split": "validation",
                        "role": role,
                        "segments": segments,
                    }
                )
                sources[name] = source
        return queries, sources

    def test_selection_is_order_independent_and_retains_reasons(self):
        queries, sources = self.fixture()
        first = select_queries(queries, sources, ["s3", "s1", "s2"])
        second = select_queries(list(reversed(queries)), sources, ["s1", "s2", "s3"])
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 6)
        self.assertTrue(any(len(reasons) > 1 for reasons in first[1].values()))

    def test_test_split_and_missing_role_are_rejected(self):
        queries, sources = self.fixture()
        with self.assertRaises(ValueError):
            select_queries(
                [{**q, "split": "test"} for q in queries], sources, ["s1", "s2", "s3"]
            )
        with self.assertRaises(ValueError):
            select_queries(queries[:-1], sources, ["s1", "s2", "s3"])


if __name__ == "__main__":
    unittest.main()
