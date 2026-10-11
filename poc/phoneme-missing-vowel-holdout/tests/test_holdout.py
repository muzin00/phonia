import importlib.util
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("holdout_study_test", BASE / "study.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)


class HoldoutTests(unittest.TestCase):
    def test_missing_query_stays_in_denominators(self):
        rows = []
        for query in range(3):
            for claim in range(3):
                rows.append(
                    {
                        "status": "scored" if query < 2 else "no_score",
                        "score": (0.9 if query == claim else 0.1)
                        if query < 2
                        else None,
                        "is_genuine": query == claim,
                    }
                )
        result = study.metrics(rows, {"far_1pct": 0.5})
        self.assertEqual(result["all_queries"], 3)
        self.assertEqual(result["scored_queries"], 2)
        self.assertEqual(result["operating_points"]["far_1pct"]["all_input_frr"], 1 / 3)
        self.assertEqual(result["eer"], 0)

    def test_all36_encoder_and_extraction_configuration(self):
        config = study.s.read_json(study.CONFIG)
        path = study.ROOT / config["encoder_run"] / "trials" / config["encoder_trial"]
        study.s.checked(path / "encoder.pt", config["encoder_sha256"])
        _, bundle = study.s.original.load_model(path)
        self.assertEqual(len(bundle["phonemes"]), 36)
        self.assertTrue(set(study.s.VOWELS) <= set(bundle["phonemes"]))

    def test_speaker_and_clip_ranking_deterministic(self):
        self.assertEqual(study.digest(1, "speaker"), study.digest(1, "speaker"))
        self.assertNotEqual(study.digest(1, "speaker"), study.digest(2, "speaker"))


if __name__ == "__main__":
    unittest.main()
