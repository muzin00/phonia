import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import score, score_inputs
from inputs import backend, corpus, make_schedule, s, select_sources, training_batch


class InputTests(unittest.TestCase):
    def test_source_split_is_disjoint_and_excludes_single_recording(self):
        catalog = {"jvs001": {"a", "b", "c"}, "jtube_a": {"d"}}
        enroll, query, excluded = select_sources(catalog, 20, 17)
        self.assertEqual(len(enroll), 1)
        self.assertEqual(len(query), 2)
        self.assertFalse(enroll & query)
        self.assertEqual(enroll | query, catalog["jvs001"])
        self.assertEqual(excluded, {"jtube_a": "fewer_than_two_distinct_wavs"})
        self.assertEqual((enroll, query, excluded), select_sources(catalog, 20, 17))

    def test_schedule_has_same_corpus_negatives_and_shared_mask(self):
        speakers = ["jvs001", "jvs002", "cv17_a", "cv17_b", "jtube_a", "jtube_b"]
        data = {"speakers": speakers, "queries": [{"speaker_id": v} for v in speakers]}
        schedule = make_schedule({"updates": 3, "queries_per_update": 12}, data, 7)
        for q, n in zip(schedule["queries"].flat, schedule["negative_claims"].flat):
            self.assertNotEqual(q, n)
            self.assertEqual(corpus(speakers[q]), corpus(speakers[n]))
        arrays = {
            "enrollment": s.torch.randn(6, 6, 128),
            "queries": s.torch.randn(6, 6, 128),
            "emeta": s.torch.zeros(6, 6, 4),
            "qmeta": s.torch.zeros(6, 6, 4),
            "emask": s.torch.ones(6, 6, dtype=s.torch.bool),
            "qmask": s.torch.ones(6, 6, dtype=s.torch.bool),
        }
        arrays["emask"][::2, 5] = False
        _, _, mask, labels = training_batch(arrays, data, schedule, 0)
        s.torch.testing.assert_close(mask[:12], mask[12:])
        self.assertTrue(mask[:, :5].all())
        self.assertEqual(labels.tolist(), [1.0] * 12 + [0.0] * 12)

    def test_uniform_fusion_preserves_baseline_scores_and_missing_vowel_gate(self):
        s.torch.set_num_threads(1)
        rng = s.np.random.default_rng(19)
        phones = ["a", "i", "u", "e", "o", "k"]
        arrays = {
            "enrollment": rng.normal(size=(2, 6, 128)).astype("float32"),
            "queries": rng.normal(size=(2, 6, 128)).astype("float32"),
            "emeta": s.np.zeros((2, 6, 4), "float32"),
            "qmeta": s.np.zeros((2, 6, 4), "float32"),
            "emask": s.np.ones((2, 6), bool),
            "qmask": s.np.ones((2, 6), bool),
        }
        arrays["qmask"][1, 0] = False
        data = {
            "speakers": ["s0", "s1"],
            "queries": [{"query_id": "q0"}, {"query_id": "q1"}],
            "universally_registered": phones,
        }
        baseline = []
        for q in range(2):
            for c in range(2):
                value = (
                    float(
                        s.np.mean(
                            [
                                float(
                                    arrays["queries"][q, p] @ arrays["enrollment"][c, p]
                                )
                                for p in range(6)
                            ]
                        )
                    )
                    if q == 0
                    else None
                )
                baseline.append(
                    {"query_id": f"q{q}", "claimed_speaker_id": f"s{c}", "score": value}
                )
        components, masks = score_inputs(data, arrays, phones, baseline)
        tensors = {k: s.torch.from_numpy(v) for k, v in arrays.items()}
        tensors.update(components=components, masks=masks)
        for kind in ("mlp", "transformer"):
            output = score(backend.Fusion(kind, phone_count=6), tensors, baseline)
            for expected, actual in zip(baseline, output):
                if expected["score"] is None:
                    self.assertIsNone(actual["score"])
                else:
                    self.assertAlmostEqual(
                        expected["score"], actual["score"], places=12
                    )
            self.assertIsNone(baseline[2]["score"])


if __name__ == "__main__":
    unittest.main()
