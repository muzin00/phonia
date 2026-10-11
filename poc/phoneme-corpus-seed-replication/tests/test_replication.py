"""Check paired-seed accounting and unchanged pools across seed runs."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replicate as r


class ReplicationTests(unittest.TestCase):
    def test_deltas_pair_each_seed_with_its_own_baseline(self):
        results = []
        for base in (0.1, 0.2, 0.3, 0.4, 0.5):
            conditions = {}
            for corpus in r.s.CORPORA:
                conditions[corpus] = {
                    arm: {
                        "verification": dict.fromkeys(
                            ("eer", "far", "frr", "coverage"),
                            base - (0.01 if arm != "baseline140" else 0),
                        )
                    }
                    for arm in r.s.ARMS
                }
            results.append({"conditions": conditions})
        result = r.summarize(results)
        for corpus in r.s.CORPORA:
            metric = result[corpus]["expanded280"]["eer"]
            self.assertEqual(metric["decreased_seeds"], 5)
            self.assertAlmostEqual(metric["paired_delta_mean"], -0.01)
            self.assertTrue(
                all(abs(v + 0.01) < 1e-10 for v in metric["paired_delta_values"])
            )
            self.assertGreater(metric["sd"], 0)

    def test_seed_list_is_fixed_and_unique(self):
        with self.assertRaises(AssertionError):
            r.seeds({"reference_seed": 1, "additional_seeds": [1, 2, 3, 4]})
        self.assertEqual(
            r.seeds({"reference_seed": 1, "additional_seeds": [2, 3, 4, 5]}),
            [1, 2, 3, 4, 5],
        )

    def test_prepare_changes_seed_but_preserves_all_pool_bytes(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(r.s, "ROOT", Path(directory)),
        ):
            root = Path(directory)
            reference, run = root / "reference", root / "replication"
            config = {"reference_run": "reference", "interpretation": "test"}
            frozen = {
                "files": {},
                "reference_design": {
                    "config": {
                        "encoder": "statistics_mlp",
                        "training_seed": 1,
                        "maximum_updates": 216000,
                    },
                    "runtime": {},
                    "speaker_plans": {},
                    "phonemes": [],
                },
            }
            r.s.fixed(run / "design-freeze.json", frozen)
            for name in ("prepared-inputs-freeze.json", "corpus-overlap-audit.json"):
                r.s.fixed(reference / name, {})
            for arm in r.s.ARMS:
                for name in r.POOL_FILES:
                    path = reference / arm / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((arm + name).encode())
            initial_hashes = []
            for seed in (2, 3):
                current, destination, initial = r.prepare_seed(
                    config, run, frozen, seed
                )
                initial_hashes.append(initial)
                self.assertEqual(current["training_seed"], seed)
                self.assertEqual(current["maximum_updates"], 216000)
                self.assertTrue(
                    all(
                        r.s.read(destination / "design-freeze.json")[
                            "previously_observed_test"
                        ].values()
                    )
                )
                for arm in r.s.ARMS:
                    for name in r.POOL_FILES:
                        self.assertEqual(
                            (reference / arm / name).read_bytes(),
                            (destination / arm / name).read_bytes(),
                        )
            self.assertNotEqual(*initial_hashes)


if __name__ == "__main__":
    unittest.main()
