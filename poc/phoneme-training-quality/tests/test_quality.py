import json
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from quality import REASONS, decide, measure_pcm, reason_names, source_quality


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads((BASE / "config.json").read_text())["policy"]

    def measures(self, count=1):
        return {
            "dc_removed_rms_dbfs": np.full(count, -20.0),
            "near_clip_fraction": np.zeros(count),
            "zero_fraction": np.zeros(count),
            "alignment_score": np.full(count, -100.0),
            "raw_start": np.zeros(count),
            "raw_end": np.full(count, 2400),
        }

    def test_dc_signal_fails_after_centering_even_when_raw_rms_passes(self):
        measured = measure_pcm(np.full(720, 1024, np.int16))
        self.assertGreater(measured["measured_rms_dbfs"], -50)
        values = self.measures()
        values["dc_removed_rms_dbfs"][:] = measured["dc_removed_rms_dbfs"]
        bits, status = decide(values, [0], self.policy, 24000)
        self.assertEqual(status.tolist(), [2])
        self.assertEqual(reason_names(bits[0]), ["low_rms_after_dc"])

    def test_strict_boundaries_and_uncalibrated_alignment_score(self):
        m = self.measures(4)
        m["dc_removed_rms_dbfs"][:] = [-50, -50.01, -20, -20]
        m["zero_fraction"][:] = [0.5, 0, 0.50001, 0]
        m["raw_end"][:] = [24000, 2400, 2400, 24001]
        bits, status = decide(m, np.zeros(4), self.policy, 24000)
        self.assertEqual(status.tolist(), [0, 2, 1, 1])
        self.assertEqual(bits[0], 0)  # low raw alignment score alone is not a gate

    def test_rejection_keeps_all_held_reasons(self):
        m = self.measures()
        m["dc_removed_rms_dbfs"][:] = -60
        m["near_clip_fraction"][:] = 0.01
        bits, status = decide(m, [REASONS["asr_low_logprob"]], self.policy, 24000)
        self.assertEqual(status[0], 2)
        self.assertEqual(
            set(reason_names(bits[0])),
            {"low_rms_after_dc", "near_clipping", "asr_low_logprob"},
        )

    def test_missing_diagnostics_not_equivalent_to_clean_asr(self):
        source = {
            "source_file": "x",
            "source_sha256": "hash",
            "speaker_id": "s",
            "corpus": "jtube",
        }
        row = source_quality(source, None, self.policy)
        self.assertEqual(row["reason_bits"], REASONS["asr_diagnostics_missing"])
        source["corpus"] = "jvs"
        self.assertEqual(
            source_quality(source, None, self.policy)["asr_status"], "not_required"
        )

    def test_repetition_held_despite_high_asr_logprob(self):
        source = {
            "source_file": "x",
            "source_sha256": "hash",
            "speaker_id": "s",
            "corpus": "jtube",
        }
        asr = {
            "source_file": "x",
            "speaker_id": "s",
            "text": "4" + "〜" * 222,
            "minimum_avg_logprob": -0.1,
            "maximum_compression_ratio": 39,
            "maximum_no_speech_prob": 0,
        }
        row = source_quality(source, asr, self.policy)
        self.assertEqual(row["asr"]["maximum_same_character_run"], 222)
        self.assertEqual(
            set(reason_names(row["reason_bits"])),
            {"asr_high_compression", "asr_character_repetition"},
        )

    def test_nonfinite_measurements_fail_instead_of_passing(self):
        m = self.measures()
        m["dc_removed_rms_dbfs"][:] = np.nan
        with self.assertRaises(ValueError):
            decide(m, [0], self.policy, 24000)

    def test_pcm_rail_and_mean_removal_match_float32_training(self):
        import torch

        pcm = np.array([32767, -32768, 1100, 1102, 1101], np.int16)
        m = measure_pcm(pcm)
        x = torch.from_numpy(pcm.astype(np.float32) / 32768)
        x -= x.mean()
        self.assertAlmostEqual(
            m["dc_removed_rms_dbfs"],
            float(10 * torch.log10(x.square().mean())),
            places=5,
        )
        self.assertEqual(m["rail_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
