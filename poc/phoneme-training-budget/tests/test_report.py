"""Exercise the real metric/CI schema and reject a corrupted published table."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget_evaluation as study
import budget_report as report
import numpy as np


class ReportTests(unittest.TestCase):
    def test_report_round_trip_uses_real_interval_schema_and_detects_html_corruption(
        self,
    ):
        config = study.read_json(study.training.CONFIG)
        prior = study.ROOT / config["source_evaluation_run"]
        reference_validation = study.read_json(prior / "validation-metrics.json")[
            "conditions"
        ]
        reference_test = study.read_json(prior / "test-metrics.json")["conditions"]
        reference_thresholds = study.read_json(prior / "validation-thresholds.json")[
            "conditions"
        ]
        cells, validation, thresholds, replicas = {}, {}, {}, {}
        for condition in study.CONDITIONS:
            original = (
                study.BASELINE
                if condition == study.BASELINE
                else ("cv70_u30000" if condition.startswith("cv_") else "src70_u30000")
            )
            thresholds[condition] = copy.deepcopy(reference_thresholds[original])
            for role in study.ROLES:
                key = f"{condition}/{role}"
                cells[key] = copy.deepcopy(reference_test[f"{original}/{role}"])
                validation[key] = copy.deepcopy(
                    reference_validation[f"{original}/{role}"]
                )
                replicas[key] = {
                    name: np.full(10, study.old.metric_value(cells[key], name))
                    for name in study.NAMES
                }
        differences, _ = study.paired(cells, replicas, study.read_json(study.CONFIG))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            run = base / "evaluation"
            run.mkdir()
            (base / "training-results.json").write_text(json.dumps({"fixture": True}))
            for name, value in (
                ("validation-thresholds.json", {"conditions": thresholds}),
                ("validation-metrics.json", {"conditions": validation}),
                (
                    "test-metrics.json",
                    {"conditions": cells, "paired_differences": differences},
                ),
            ):
                (run / name).write_text(json.dumps(value))
            with patch.object(report, "BASE", base):
                report.render(run)
                self.assertEqual(
                    report.audit_report(run),
                    {"html_tables": 5, "condition_rows": 84, "paired_rows": 112},
                )
                page = run / "report/evaluation-results.html"
                body = page.read_text()
                self.assertIn("2.657%", body)
                page.write_text(body.replace("2.657%", "99.999%", 1))
                with self.assertRaises(ValueError):
                    report.audit_report(run)


if __name__ == "__main__":
    unittest.main()
