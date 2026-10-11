"""The intervention changes corpus exposure, never pair validity or budget."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as experiment
import sampler


class SamplerTests(unittest.TestCase):
    def test_common_phone_quotas_match_the_prespecified_ratios(self):
        capacities = dict.fromkeys(sampler.CORPORA, 20)
        equal = sampler.quotas(capacities, list(sampler.CORPORA), 30)
        self.assertEqual(equal.sum(axis=0).tolist(), [100, 100, 100])
        weighted = sampler.quotas(
            capacities, ["jvs", "common_voice", "jtube", "common_voice"], 30
        )
        self.assertEqual(weighted.sum(axis=0).tolist(), [75, 150, 75])
        self.assertTrue((equal.sum(axis=1) == 10).all())
        self.assertTrue((weighted.sum(axis=1) == 10).all())

    def test_rare_phone_redistribution_does_not_duplicate_speakers(self):
        counts = sampler.quotas(
            {"jvs": 1, "common_voice": 2, "jtube": 0}, list(sampler.CORPORA), 20
        )
        self.assertEqual(counts.tolist(), [[1, 2, 0]] * 20)
        with self.assertRaises(ValueError):
            sampler.quotas(dict.fromkeys(sampler.CORPORA, 0), list(sampler.CORPORA), 1)

    def test_schedule_keeps_original_phone_budget_and_distinct_positive_pairs(self):
        s = experiment.s
        speakers = [
            f"{prefix}{i}" for prefix in ("jvs_", "cv17_", "jtube_") for i in range(12)
        ]
        groups, owner = {}, {}
        phones = ["a", "i", "u", "e", "o", "rare"]
        index = 0
        for speaker in speakers:
            for phone in phones:
                if phone == "rare" and speaker != "cv17_0":
                    continue
                ids = list(range(index, index + 3))
                groups[speaker, phone] = ids
                owner.update({i: (speaker, phone) for i in ids})
                index += 3
        classes = {speaker: i for i, speaker in enumerate(speakers)}
        kwargs = {"cycle": list(sampler.CORPORA), "rank": s.rank}
        planned, labels, valid = sampler.schedule(
            groups, phones, 24, 123, classes, **kwargs
        )
        replay = sampler.schedule(groups, phones, 24, 123, classes, **kwargs)
        self.assertTrue(
            all(
                s.np.array_equal(a, b)
                for a, b in zip((planned, labels, valid), replay, strict=True)
            )
        )
        original = experiment.learner.schedule(groups, phones, 24, 123, classes)
        self.assertTrue(s.np.array_equal(valid.sum(axis=1), original[2].sum(axis=1)))
        for pos, (row, row_labels, mask) in enumerate(
            zip(
                planned.reshape(-1, 20),
                labels.reshape(-1, 20),
                valid.reshape(-1, 20),
                strict=True,
            )
        ):
            selected = row[mask]
            selected_labels = row_labels[mask]
            selected_speakers = []
            for i in range(0, len(selected), 2):
                first, second = int(selected[i]), int(selected[i + 1])
                self.assertNotEqual(first, second)
                self.assertEqual(owner[first], owner[second])
                self.assertEqual(owner[first][1], phones[pos % len(phones)])
                self.assertEqual(classes[owner[first][0]], selected_labels[i])
                self.assertEqual(selected_labels[i], selected_labels[i + 1])
                selected_speakers.append(owner[first][0])
            self.assertEqual(len(selected_speakers), len(set(selected_speakers)))
        measured = sampler.exposure(labels, valid, phones, speakers)
        self.assertEqual(measured["valid_examples"], int(valid.sum()))
        self.assertEqual(
            measured["phone_corpus_examples"]["rare"],
            {"jvs": 0, "common_voice": 40, "jtube": 0},
        )


if __name__ == "__main__":
    unittest.main()
