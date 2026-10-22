"""Synthetic report contract tests; no private reports or model execution."""
from __future__ import annotations

import contextlib
import csv
from datetime import date, datetime, time, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nyx_runtime import cli, reporting as report_io
from tests import report_helpers as report_import


def synthetic_rows(first_day="2026-03-28"):
    first = date.fromisoformat(first_day)
    origin = datetime.combine(first - timedelta(days=1), time(8), report_io.LOCAL_TZ)
    rows = []
    for horizon in (1, 2):
        day = (first + timedelta(days=horizon - 1)).isoformat()
        for zi, zone in enumerate(report_io.ZONES):
            for hi, start in enumerate(report_io.physical_day_hours(day)):
                price = -12.3456789012345 + zi * 20 + hi / 7
                row = dict.fromkeys(report_io.CSV_COLUMNS, "")
                row.update(delivery_day=day, zone=zone,
                    delivery_start_utc=start.isoformat(),
                    delivery_start_local=start.astimezone(report_io.LOCAL_TZ).isoformat(),
                    forecast_origin_utc=origin.astimezone(timezone.utc).isoformat(),
                    horizon_day=horizon, prediction_eur_mwh=price,
                    lower_bound_eur_mwh=price - 3.14159265358979,
                    upper_bound_eur_mwh=price + 6.23456789012345,
                    storm_eur_mwh=None if hi == 3 else price + .123456789012345,
                    observed_eur_mwh=None if horizon == 2 else price - .23456789012345,
                    interval_method=report_io.INTERVAL_METHOD,
                    nominal_interval_level=.8, interval_calibrated=False)
                rows.append(row)
    return rows


class ProductionReportTests(unittest.TestCase):
    def test_civil_day_grids_keep_spring_and_autumn_dst_hours(self):
        self.assertEqual(len(report_io.physical_day_hours("2026-03-29")), 23)
        autumn = report_io.physical_day_hours("2026-10-25")
        self.assertEqual(len(autumn), 25)
        labels = [p.astimezone(report_io.LOCAL_TZ).isoformat() for p in autumn]
        self.assertTrue(any("T02:00:00+02:00" in p for p in labels))
        self.assertTrue(any("T02:00:00+01:00" in p for p in labels))

    def test_csv_html_import_roundtrip_preserves_exact_values_and_missing_observations(self):
        rows = synthetic_rows()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "predictions.csv"
            report_io.write_predictions(csv_path, rows)
            # A tiny synthetic asset exercises embedding without a browser or CDN.
            asset = "/* plotly.js v0.test Licensed under the MIT license */"
            with patch.object(report_io, "_find_plotly", return_value=asset):
                rendered = report_io.render_reports(csv_path, root / "rendered")
            self.assertEqual(rendered["row_count"], 4 * 47)
            self.assertEqual(len(rendered["html_files"]), 3)
            imported = report_import.import_reports(rendered["html_files"], root / "imported")
            self.assertFalse(imported["new_prediction_computed"])
            self.assertEqual(imported["overlapping_rows_verified"], 4 * 47)
            actual = report_io.read_predictions(Path(imported["predictions_csv"]))
            expected = sorted(rows, key=report_io.row_key)
            self.assertEqual(len(actual), len(expected))
            for observed, original in zip(actual, expected):
                for column in report_io.VALUE_COLUMNS:
                    self.assertEqual(observed[column], original[column])
                self.assertEqual(observed["forecast_origin_utc"], original["forecast_origin_utc"])
            self.assertEqual(imported["missing_observed_by_delivery_day"]["2026-03-29"], 4 * 23)
            combined = next(Path(p) for p in rendered["html_files"] if "J1_J2" in p)
            html = combined.read_text()
            self.assertIn("coverage is not calibrated for the final forecast", html)
            self.assertIn(asset, html)

    def test_duplicate_physical_hour_is_rejected(self):
        rows = synthetic_rows()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(report_io.ReportFormatError, "dupliqu"):
            report_io.validate_rows(rows)

    def test_missing_country_or_hour_is_rejected(self):
        for rows in ([r for r in synthetic_rows() if r["zone"] != "BE"], synthetic_rows()[1:]):
            with self.subTest(rows=len(rows)):
                with self.assertRaisesRegex(report_io.ReportFormatError, "incomplète"):
                    report_io.validate_rows(rows)

    def test_common_origin_and_correct_horizon_are_required(self):
        rows = synthetic_rows()
        rows[-1]["forecast_origin_utc"] = "2026-03-28T07:00:00+00:00"
        rows[-1]["horizon_day"] = 1
        with self.assertRaisesRegex(report_io.ReportFormatError, "cutoff commun"):
            report_io.validate_rows(rows)
        rows = synthetic_rows()
        rows[-1]["horizon_day"] = 1
        with self.assertRaisesRegex(report_io.ReportFormatError, "horizon_day incohérent"):
            report_io.validate_rows(rows)

    def test_calibrated_label_and_missing_forecast_are_rejected(self):
        for column, value in (("interval_calibrated", True), ("prediction_eur_mwh", None)):
            rows = synthetic_rows()
            rows[0][column] = value
            with self.subTest(column=column), self.assertRaises(report_io.ReportFormatError):
                report_io.validate_rows(rows)

    def test_existing_html_destination_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "predictions.csv"
            report_io.write_predictions(source, synthetic_rows())
            with patch.object(report_io, "_find_plotly", return_value="synthetic asset"):
                first = report_io.render_reports(source, root / "rendered")
                saved = {p: Path(p).read_bytes() for p in first["html_files"]}
                with self.assertRaises(FileExistsError):
                    report_io.render_reports(source, root / "rendered")
                self.assertEqual(saved, {p: Path(p).read_bytes() for p in saved})

    def test_cli_failed_render_preserves_last_success_pointer_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            latest = root / "latest.json"
            latest.write_text('{"run_id":"prior-success"}\n')
            source = root / "broken.csv"
            source.write_text("zone,prediction_eur_mwh\nFR,2\n")
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["reports", "--csv", str(source), "--output-root", str(root)])
            self.assertEqual(code, 2)
            self.assertEqual(latest.read_text(), '{"run_id":"prior-success"}\n')
            statuses = list(root.glob("*/status.json"))
            self.assertEqual(len(statuses), 1)
            self.assertEqual(json.loads(statuses[0].read_text())["state"], "FAILED")

    def test_cli_reports_renders_three_html_and_tracks_the_original_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "predictions.csv", root / "reports-output"
            report_io.write_predictions(source, synthetic_rows())
            with patch.object(report_io, "_find_plotly", return_value="synthetic asset"), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["reports", "--csv", str(source), "--output-root", str(output)])
            self.assertEqual(code, 0)
            pointer = json.loads((output / "latest.json").read_text())
            self.assertEqual(pointer["action"], "reports")
            self.assertEqual((output / pointer["predictions_csv"]).resolve(), source)
            status = json.loads((output / pointer["status_file"]).read_text())
            self.assertEqual(status["state"], "COMPLETE")
            self.assertFalse(status["new_predictions_computed"])
            html_files = status["result"]["html_files"]
            self.assertEqual(len(html_files), 3)
            self.assertTrue(all(Path(path).is_file() for path in html_files))

    def test_json_duplicate_keys_and_javascript_input_are_rejected_without_execution(self):
        for raw in ('{"data":[],"data":[]}', '{"x":NaN}', 'alert("no execution")'):
            with self.subTest(raw=raw), self.assertRaises(report_io.ReportFormatError):
                report_import._strict_json(raw)


if __name__ == "__main__":
    unittest.main()
