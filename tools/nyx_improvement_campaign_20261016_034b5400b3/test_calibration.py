"""Scientific guard tests for the isolated NYX adaptive study."""
from __future__ import annotations

from datetime import timedelta
import unittest
import tempfile
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from calibration import (COMPONENTS, convex_weights, evaluate, frozen_current,
                         generate_predictions, interval_radius, main, validate_input)


def synthetic(days: int = 76, horizon: int = 1, country: str = "FR", start: str = "2026-07-01") -> pd.DataFrame:
    rows = []
    random = np.random.default_rng(481)
    for number in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=number)
        first = day.tz_localize("Europe/Paris").tz_convert("UTC")
        stop = (day + pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
        origin = ((day - pd.Timedelta(days=horizon)) + pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
        available = (day + pd.Timedelta(hours=14)).tz_localize("Europe/Paris").tz_convert("UTC")
        for index in pd.date_range(first, stop, freq="h", inclusive="left"):
            hour = index.tz_convert("Europe/Paris").hour
            base = 35. + .1 * number + 12. * np.sin(hour * np.pi / 12.)
            fr = base + 8. * np.sin(number * .13) + 3. * np.cos(hour * .3)
            residual = base + 15. * np.sin(number * .07 + hour * .1)
            absolute = base - 3. * np.cos(number * .08) + 7. * np.cos(hour * .4)
            target = .2 * base + .5 * fr + .2 * residual + .1 * absolute + random.normal(0., .4)
            rows.append({"country": country, "delivery_utc": index.isoformat(), "forecast_origin_utc": origin.isoformat(),
                "horizon_day": horizon, "target_price": target, "target_available_utc": available.isoformat(),
                "reference_price": base, "expert_fr_residual": fr, "expert_cwe_residual": residual,
                "expert_cwe_absolute": absolute, "storm_price": target + 4.})
    frame = pd.DataFrame(rows)
    frame["nyx_current"] = frozen_current(frame)
    return frame


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = validate_input(synthetic())
        cls.predictions, cls.audit = generate_predictions(cls.frame, (7, 14), minimum_days=3)

    def test_future_labels_and_storm_do_not_change_prior_predictions(self):
        cutoff = self.frame["forecast_origin_utc"].drop_duplicates().sort_values().iloc[45]
        altered = self.frame.copy()
        altered.loc[altered["target_available_utc"] > cutoff, "target_price"] += 250.
        altered["storm_price"] = -10000.
        predictions, _ = generate_predictions(altered, (7, 14), minimum_days=3)
        before = self.frame["forecast_origin_utc"] <= cutoff
        columns = [name for name in predictions if name.startswith(("prediction_", "lower_", "upper_"))]
        np.testing.assert_allclose(predictions.loc[before, columns], self.predictions.loc[before, columns], equal_nan=True, atol=1e-10)

    def test_holdout_labels_do_not_select_hyperparameters(self):
        _, initial = evaluate(self.predictions, (7, 14), holdout_days=21, warmup_days=14)
        cutoff = pd.Timestamp(initial["selections"][0]["selection_cutoff_utc"])
        altered = self.frame.copy()
        altered.loc[altered["target_available_utc"] > cutoff, "target_price"] -= 100.
        predictions, _ = generate_predictions(altered, (7, 14), minimum_days=3)
        _, second = evaluate(predictions, (7, 14), holdout_days=21, warmup_days=14)
        self.assertEqual(initial, second)

    def test_unavailable_history_is_not_fitted(self):
        late = self.frame.copy()
        late["target_available_utc"] = self.frame["target_available_utc"].max() + pd.Timedelta(days=30)
        _, audit = generate_predictions(late.iloc[:24 * 8].copy(), (7,), minimum_days=3)
        self.assertTrue(all(row["windows"]["7"]["fit_rows"] == 0 for row in audit))
        self.assertTrue(all(not row["windows"]["7"]["ready"] for row in audit))

    def test_horizon_streams_are_separate(self):
        j2 = validate_input(synthetic(days=12, horizon=2))
        j1 = validate_input(synthetic(days=12))
        mixed = pd.concat([j1, j2], ignore_index=True)
        baseline, _ = generate_predictions(mixed, (7,), minimum_days=3)
        altered = mixed.copy()
        altered.loc[altered.horizon_day == 2, "target_price"] += 100.
        changed, _ = generate_predictions(altered, (7,), minimum_days=3)
        mask = mixed.horizon_day == 1
        np.testing.assert_allclose(baseline.loc[mask, "prediction_adaptive_bias_7"], changed.loc[mask, "prediction_adaptive_bias_7"])

    def test_fit_audit_availability_and_convexity(self):
        for row in self.audit:
            for window in row["windows"].values():
                weights = np.asarray(window["weights"])
                self.assertTrue(np.all(weights >= 0.))
                self.assertAlmostEqual(float(weights.sum()), 1.)
                if window["latest_used_target_availability_utc"]:
                    self.assertLessEqual(pd.Timestamp(window["latest_used_target_availability_utc"]), pd.Timestamp(row["forecast_origin_utc"]))

    def test_physical_dst_grid_and_horizon(self):
        autumn = validate_input(synthetic(days=3, start="2026-10-24"))
        self.assertEqual(sorted(autumn.groupby("delivery_day").size()), [24, 24, 25])
        spring = validate_input(synthetic(days=3, start="2026-03-28"))
        self.assertEqual(sorted(spring.groupby("delivery_day").size()), [23, 24, 24])
        broken = synthetic(days=3)
        with self.assertRaisesRegex(ValueError, "incomplete physical-hour"):
            validate_input(broken.iloc[1:])
        broken = synthetic(days=3)
        broken["horizon_day"] = 2
        with self.assertRaisesRegex(ValueError, "Horizon"):
            validate_input(broken)

    def test_refuses_retrofitted_current_composition(self):
        broken = synthetic(days=3)
        broken.loc[0, "nyx_current"] += 1.
        with self.assertRaisesRegex(ValueError, "frozen production"):
            validate_input(broken)

    def test_missing_365_days_not_qualified_and_ties_not_wins(self):
        tied = self.predictions.copy()
        tied["storm_price"] = tied["prediction_current"]
        metrics, _ = evaluate(tied, (7, 14), holdout_days=21, warmup_days=14)
        annual = metrics[metrics.period == "365"]
        self.assertFalse(annual.complete_requested_days.any())
        current = metrics[(metrics.algorithm == "current") & metrics.matched_storm_only]
        self.assertTrue((current.win_rate == 0.).all())
        self.assertTrue((current.tie_rate == 1.).all())

    def test_interval_radius_is_outward_order_statistic(self):
        self.assertEqual(interval_radius(np.arange(1, 11), .2, minimum=3), 9.)
        self.assertTrue(np.isnan(interval_radius(np.arange(10), .2)))

    def test_generic_reference_pilot_cli_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = synthetic(days=97)
            frame["nyx_baseline"] = frame["expert_fr_residual"]
            frame["test2_price"] = frame["expert_cwe_residual"]
            source = directory / "oof.csv"
            frame.to_csv(source, index=False)
            manifest = source.with_suffix(".manifest.json")
            manifest.write_text(json.dumps({"scope": "reference_component_pilot", "provenance_status": "conditional_research",
                "oof_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}), encoding="utf-8")
            arguments = ["--input", str(source), "--output", str(directory / "results"),
                "--component-columns", "reference_price,nyx_baseline,test2_price", "--control-column", "nyx_baseline",
                "--scope", "reference_component_pilot", "--warmup-days", "45", "--holdout-days", "30"]
            self.assertEqual(main(arguments), 0)
            study = json.loads((directory / "results" / "study.json").read_text(encoding="utf-8"))
            self.assertEqual(study["scope"], "reference_component_pilot")
            self.assertFalse(study["production_modified"])
            self.assertTrue((directory / "results" / "report.html").is_file())
            with self.assertRaisesRegex(ValueError, "already exist"):
                main(arguments)
            manifest.write_text(json.dumps({"scope": "reference_component_pilot", "oof_csv_sha256": "0" * 64}), encoding="utf-8")
            arguments[arguments.index(str(directory / "results"))] = str(directory / "different_results")
            with self.assertRaisesRegex(ValueError, "exact oof_csv_sha256"):
                main(arguments)

    def test_missing_storm_hours_are_not_complete_period(self):
        missing = self.predictions.copy()
        missing.loc[missing.delivery_hour == 10, "storm_price"] = np.nan
        metrics, _ = evaluate(missing, (7, 14), holdout_days=21, warmup_days=14)
        self.assertFalse(metrics.loc[metrics.matched_storm_only, "complete_requested_days"].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
