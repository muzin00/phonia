"""Verify selection, sparse sampling and numerical equivalence before training."""

import copy
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from common import (
    CONFIG,
    VOWELS,
    canonical,
    check_cache_equivalence,
    create_encoder,
    expanded,
    np,
    pipeline,
    read_json,
    torch,
)
from evaluate import scores_for
from learner import grouped_aam, grouped_supcon, schedule
from phase3_train.losses import AAMSoftmax, within_vowel_supcon
from selection import baseline_state, draw_order, next_candidate, record_attempt


class SearchTests(unittest.TestCase):
    def test_missing_extra_phone_does_not_change_query_support_or_claim_phone_sets(
        self,
    ):
        vectors = np.eye(6, dtype=np.float64)
        profiles = {
            s: {p: [i] for i, p in enumerate((*VOWELS, "m"))} for s in ("s0", "s1")
        }
        queries = [
            {
                "query_id": "q",
                "speaker_id": "s0",
                "role": "verification",
                "scorable": True,
                "groups": {p: [i] for i, p in enumerate(VOWELS)} | {"m": []},
            }
        ]
        data = {
            "speakers": ["s0", "s1"],
            "profiles": profiles,
            "queries": queries,
            "split": "validation",
            "universally_registered": [*VOWELS, "m"],
        }
        scored = scores_for(data, vectors, [*VOWELS, "m"])
        self.assertEqual([r["status"] for r in scored], ["scored", "scored"])
        self.assertEqual(scored[0]["used_phones"], scored[1]["used_phones"])
        self.assertEqual(set(scored[0]["used_phones"]), set(VOWELS))

    def test_full_box_preserves_nasal_geminate_palatalized_and_rare_labels(self):
        raw = ["a", "I", "U", "N", "cl", "by", "py", "r", "w", "dy", "ty", "v", "pau"]
        phones = {canonical(p) for p in raw} - {None} - set(VOWELS)
        self.assertEqual(phones, {"N", "cl", "by", "py", "r", "w", "dy", "ty", "v"})
        self.assertEqual(set(draw_order(list(phones), 12)), phones)

    def test_rejected_phone_is_retried_after_later_acceptance(self):
        state = baseline_state(["n", "m"], 0.03)
        self.assertEqual(next_candidate(state), "n")
        record_attempt(state, "n", "n-only", 0.04)
        self.assertEqual(next_candidate(state), "m")
        record_attempt(state, "m", "m-only", 0.02)
        self.assertEqual(next_candidate(state), "n")
        entry = record_attempt(state, "n", "mn", 0.01)
        self.assertTrue(entry["accepted"])
        self.assertEqual(set(state["accepted"]), {*VOWELS, "m", "n"})
        self.assertIsNone(next_candidate(state))

    def test_tie_and_worse_candidates_never_replace_best(self):
        state = baseline_state(["s", "m"], 0.03)
        record_attempt(state, "s", "tie", 0.03)
        record_attempt(state, "m", "worse", 0.04)
        self.assertEqual(state["best_trial"], "trial-000-baseline")
        self.assertEqual(state["best_eer"], 0.03)
        self.assertEqual(state["accepted"], list(VOWELS))

    def test_resume_pending_does_not_draw_another_phone(self):
        state = baseline_state(["s", "m"], 0.03)
        phone = next_candidate(state)
        state["pending"] = {"phoneme": phone}
        resumed = copy.deepcopy(state)
        self.assertEqual(resumed["pending"]["phoneme"], "s")
        self.assertEqual(resumed["position"], 1)

    def test_unavailable_stays_in_box_without_fake_eer(self):
        state = baseline_state(["ty"], 0.03)
        next_candidate(state)
        entry = record_attempt(state, "ty", None, None, "no_training_data")
        self.assertEqual(entry["status"], "unavailable")
        self.assertIsNone(entry["validation_eer"])
        self.assertEqual(state["order"], ["ty"])
        self.assertIsNone(next_candidate(state))

    def test_sparse_phone_uses_real_pairs_and_masked_slots(self):
        groups = {}
        speakers = {f"s{i}": i for i in range(12)}
        i = 0
        for phone in (*VOWELS, "dy"):
            for speaker in list(speakers)[: 1 if phone == "dy" else 12]:
                groups[speaker, phone] = [i, i + 1, i + 2]
                i += 3
        planned, labels, valid = schedule(groups, [*VOWELS, "dy"], 12, 23, speakers)
        np.testing.assert_array_equal(
            planned, schedule(groups, [*VOWELS, "dy"], 12, 23, speakers)[0]
        )
        for ids, classes, mask in zip(
            planned.reshape(-1, 20),
            labels.reshape(-1, 20),
            valid.reshape(-1, 20),
            strict=True,
        ):
            pairs = ids[mask].reshape(-1, 2)
            self.assertTrue(np.all(pairs[:, 0] != pairs[:, 1]))
            self.assertEqual(len(set(classes[mask][::2])), len(pairs))
            self.assertTrue(np.all(ids[~mask] == -1))
        self.assertEqual(int((valid.sum(axis=1) < 100).sum()), 10)

    def test_batched_loss_matches_original_with_sparse_groups_and_gradients(self):
        expanded.seed_everything(23)
        raw = torch.randn(100, 128, requires_grad=True)
        embeddings = torch.nn.functional.normalize(raw, dim=1)
        labels = torch.arange(20).repeat_interleave(2)[:20].repeat(5)
        valid = torch.ones(100, dtype=torch.bool)
        valid[42:60] = False
        head = AAMSoftmax(20)
        expected = []
        for i in range(5):
            use = valid[i * 20 : (i + 1) * 20]
            e, l = (
                embeddings[i * 20 : (i + 1) * 20][use],
                labels[i * 20 : (i + 1) * 20][use],
            )
            expected.append(head(e, l) + 0.5 * within_vowel_supcon(e, l))
        reference = torch.stack(expected).mean()
        cached = grouped_aam(head, embeddings, labels, valid) + 0.5 * grouped_supcon(
            embeddings, valid
        )
        self.assertTrue(torch.allclose(reference, cached, atol=2e-6, rtol=1e-6))
        a = torch.autograd.grad(reference, raw, retain_graph=True)[0]
        b = torch.autograd.grad(cached, raw)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.isfinite(b).all())
        self.assertTrue(torch.equal(b[42:60], torch.zeros_like(b[42:60])))

    def test_cache_matches_original_waveform_encoder(self):
        config = read_json(CONFIG)
        pipe = pipeline(config)
        expanded.seed_everything(23)
        probes = [
            (
                str(i),
                np.random.default_rng(i).normal(0, 0.03, length).astype(np.float32),
            )
            for i, length in enumerate((720, 1200, 3120, 6000))
        ]
        error = check_cache_equivalence(create_encoder("statistics_mlp"), pipe, probes)
        self.assertLessEqual(error, 2e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
