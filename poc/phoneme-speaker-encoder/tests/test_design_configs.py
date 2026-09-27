"""Static design consistency checks; these do not implement or train encoders."""

import itertools
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).resolve().parents[1] / "config"


def read_config(name):
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def convolution_parameters(block):
    # All designed convolutions have no bias and affine channel LayerNorm.
    return (
        block["input_channels"]
        * block["output_channels"]
        * block["kernel_size"]
        // block.get("groups", 1)
        + 2 * block["output_channels"]
    )


def projection_parameters(inputs, hidden, outputs):
    return inputs * hidden + hidden + hidden * outputs + outputs


class DesignConfigTests(unittest.TestCase):
    def setUp(self):
        self.search = read_config("search-space.json")
        self.protocol = read_config("experiment-protocol.json")
        self.baseline = read_config("baseline-log-mel.json")
        self.mel = read_config("log-mel-encoders.json")
        self.wave = read_config("waveform-encoders.json")

    def test_versions_and_references(self):
        for config in (self.search, self.protocol, self.baseline, self.mel, self.wave):
            self.assertEqual(config["design_version"], "2.0.0")
        for family in self.search["families"]:
            self.assertTrue((ROOT / family["encoder_config"]).is_file())
        for config in (self.search, self.baseline):
            self.assertTrue((ROOT / config["protocol_config"]).is_file())

    def test_search_and_run_budget(self):
        configurations = []
        for family in self.search["families"]:
            axes = family["axes"]
            tuples = list(
                itertools.product(
                    axes["encoder"], axes["rms_normalization"], axes["loss"]
                )
            )
            self.assertEqual(len(tuples), family["combination_count"])
            configurations.extend((family["input"], *item) for item in tuples)
        main_count = len(configurations)
        self.assertEqual(main_count, self.search["main_combination_count"])
        for item in self.search["limited_comparisons"]:
            control = (
                item["input"],
                item["control_encoder"],
                item["rms_normalization"],
                item["loss"],
            )
            self.assertIn(control, configurations)
            self.assertTrue(item["selection_eligible"])
            configurations.append(
                (
                    item["input"],
                    item["encoder"],
                    item["rms_normalization"],
                    item["loss"],
                )
            )
        self.assertEqual(len(set(configurations)), len(configurations))
        self.assertEqual(len(configurations), self.search["total_combination_count"])
        available = {
            "log_mel": {item["id"] for item in self.mel["candidates"]},
            "waveform": {item["id"] for item in self.wave["candidates"]},
        }
        for input_type, encoder, _, _ in configurations:
            self.assertIn(encoder, available[input_type])
        encoders = {item[1] for item in configurations}
        seeds = len(self.search["seeds"])
        additional = self.search["learning_curve"]["additional_cohorts"]
        expected = {
            "overfit_checks_by_encoder_implementation": len(encoders),
            "sanity_runs_10_speakers": len(configurations),
            "main_full_runs_70_speakers": main_count * seeds,
            "limited_full_runs_70_speakers": len(self.search["limited_comparisons"])
            * seeds,
            "learning_curve_additional_runs": len(additional) * seeds,
        }
        expected["total_planned_runs"] = sum(expected.values())
        self.assertEqual(expected, self.search["runs"])
        self.assertEqual(expected["total_planned_runs"], 87)

    def test_mel_parameters_and_context(self):
        for candidate in self.mel["candidates"]:
            if "blocks" not in candidate:
                params = projection_parameters(128, 256, 128)
            else:
                blocks = candidate["blocks"]
                params = sum(convolution_parameters(b) for b in blocks)
                params += projection_parameters(
                    2 * blocks[-1]["output_channels"], 256, 128
                )
                receptive_field = 1 + sum(
                    (b["kernel_size"] - 1) * b["dilation"] for b in blocks
                )
                expected_field = 13 if candidate["id"] == "tdnn" else 1
                self.assertEqual(receptive_field, expected_field)
            self.assertEqual(params, candidate["expected_parameter_count"])
        models = {m["id"]: m for m in self.mel["candidates"]}
        ratio = (
            models["framewise_cnn"]["expected_parameter_count"]
            / models["tdnn"]["expected_parameter_count"]
        )
        self.assertLess(abs(ratio - 1), 0.002)

    def test_waveform_parameters_lengths_and_context(self):
        for candidate in self.wave["candidates"]:
            blocks = [candidate["first_block"]]
            blocks += self.wave["common_model"]["blocks_after_first"]
            blocks += candidate.get("context_blocks", [])
            params = sum(convolution_parameters(b) for b in blocks)
            params += projection_parameters(512, 256, 128)
            self.assertEqual(params, candidate["expected_parameter_count"])
            receptive_field, jump = 1, 1
            for block in blocks:
                receptive_field += (
                    (block["kernel_size"] - 1) * block.get("dilation", 1) * jump
                )
                jump *= block["stride"]
            self.assertEqual(receptive_field, candidate["receptive_field_samples"])
            for size, key in (
                (720, "minimum_input_output_frames"),
                (6000, "maximum_input_output_frames"),
            ):
                for block in blocks:
                    size = (
                        size
                        + 2 * block["padding"]
                        - block.get("dilation", 1) * (block["kernel_size"] - 1)
                        - 1
                    ) // block["stride"] + 1
                    self.assertGreater(size, 1)
                self.assertEqual(size, candidate[key])

    def test_logical_batch_and_overfit_exception(self):
        batch = self.protocol["logical_batch"]
        self.assertEqual(
            batch["microbatch_size"],
            batch["speakers"] * batch["segments_per_speaker_vowel"],
        )
        self.assertEqual(batch["size"], batch["microbatch_size"] * len(batch["vowels"]))
        self.assertTrue(
            math.isclose(batch["microbatch_loss_weight"] * len(batch["vowels"]), 1)
        )
        self.assertEqual(batch["optimizer_steps_per_logical_batch"], 1)
        self.assertEqual(batch["size"], self.baseline["sampler"]["batch_size"])
        smoke = self.protocol["overfit"]
        self.assertEqual(
            smoke["batch_size"],
            smoke["speakers"]
            * len(smoke["vowels"])
            * smoke["segments_per_speaker_vowel"],
        )
        self.assertEqual(
            smoke["batch_size"], smoke["microbatch_size"] * len(smoke["vowels"])
        )
        self.assertEqual(smoke["microbatch_loss_weight"] * len(smoke["vowels"]), 1)
        self.assertEqual(smoke["batch_size"], 32)

    def test_selection_learning_curve_and_test_freeze(self):
        selection = self.search["selection"]
        self.assertEqual(selection["test_seeds"], self.search["seeds"])
        self.assertEqual(selection["required_seed_count"], len(self.search["seeds"]))
        self.assertIn(selection["single_distribution_seed"], selection["test_seeds"])
        self.assertTrue(selection["bootstrap_for_reporting_only"])
        self.assertFalse(selection["ensemble"])
        curve = self.search["learning_curve"]
        self.assertEqual(curve["cohorts"], curve["additional_cohorts"] + [70])
        self.assertTrue(curve["reuse_70_speaker_runs"])
        self.assertFalse(curve["used_for_reselection"])
        self.assertEqual(self.search["diagnostics"]["training_runs"], 0)
        self.assertEqual(self.search["diagnostics"]["split"], "validation")

    def test_shared_reference_values(self):
        tdnn = next(m for m in self.mel["candidates"] if m["id"] == "tdnn")
        for key in (
            "blocks",
            "normalization",
            "normalization_epsilon",
            "dropout",
            "projection_hidden_dimension",
        ):
            self.assertEqual(self.baseline["model"][key], tdnn[key])
        supcon = self.baseline["loss"]["supervised_contrastive_within_vowel"]
        for key in ("weight", "temperature"):
            self.assertEqual(supcon[key], self.protocol["supcon"][key])
        numeric = self.protocol["numeric"]
        for model in (self.mel, self.wave["common_model"], self.baseline["model"]):
            self.assertEqual(
                model["output_normalization_epsilon"],
                numeric["l2_normalization_epsilon"],
            )
            self.assertEqual(
                model["pooling_population_variance_floor"],
                numeric["pooling_population_variance_floor"],
            )
        normalization = self.baseline["input"]["feature_normalization"]
        self.assertEqual(
            normalization["standard_deviation_floor"],
            numeric["feature_standard_deviation_floor"],
        )
        self.assertEqual(
            self.baseline["evaluation"]["threshold_scope"],
            self.protocol["metrics"]["primary_threshold_scope"],
        )


if __name__ == "__main__":
    unittest.main()
