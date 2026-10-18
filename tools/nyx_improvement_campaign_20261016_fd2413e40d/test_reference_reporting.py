"""Verify the reporting-only Storm path used by the Windows experiment."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import calibration
import reference_pilot
from replay import ZONES, seed_record, sha
import test_reference_pilot as reference_fixture


class ReferenceReportingTests(unittest.TestCase):
    def setUp(self):
        # Reuse fixture construction without inheriting or duplicating its tests.
        self.fixture = reference_fixture.ReferencePilotTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.app = self.fixture.root / "app"
        config = self.app / "config/experiment_console.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"state_root": "custom_state"}))
        self.report = self.app / "custom_state/executions/forecast/artifacts/reports"
        self.sources = self.report / "report_only_sources"
        self.snapshot = self.sources / "reporting_sources"
        self.snapshot.mkdir(parents=True)
        self.report_audit = self.report / "report_audit.json"
        self.report_audit.write_text(json.dumps({"status": "REPORTS_COMPLETE",
            "completed_run": str(self.fixture.run)}))
        hashes, countries = {}, {}
        for zone in ZONES:
            baseline = self.fixture.history[zone]
            # Distinct values prove that refreshed reporting data is actually used.
            actual = (baseline.actual + 7.).rename("actual")
            storm = (actual + 2.).rename("storm_dashboard_official__q50")
            for kind, values in (("observed", actual), ("storm", storm)):
                path = self.sources / f"reporting_sources/{zone}/{kind}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                values.to_frame().to_parquet(path)
                hashes[path.relative_to(self.sources).as_posix()] = sha(path)
            series = f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
            countries[zone] = {"observed": {"status": "complete", "source": {
                "origin": "EPEX", "policy": "epex_only_v1", "zone": zone,
                "source_mixing": False, "used_for_prediction": False}},
                "storm": {"status": "complete", "used_for_prediction": False,
                    "source": {"zone": zone, "series": series,
                        "used_for_prediction": False}, "comparator": {
                        "role": "evaluation_only_dashboard_comparator",
                        "column": "storm_dashboard_official__q50", "series": series,
                        "used_for_prediction": False}}}
        self.audit = self.snapshot / "audit.json"
        self.document = {"protocol": "nyx_saved_horizon_reporting_sources_v1",
            "extracted_at_utc": "2026-04-01T13:00:00Z",
            "used_for_prediction": False, "used_for_training": False,
            "countries": countries, "artifact_sha256": hashes}
        self.save_audit()

    def save_audit(self):
        self.audit.write_text(json.dumps(self.document), encoding="utf-8")

    def test_app_state_root_discovers_real_matching_storm_and_binds_hashes(self):
        csv = reference_pilot.export(seed_record(self.fixture.repo),
            self.fixture.root / "pilot_with_storm", self.app)
        raw = pd.read_csv(csv)
        self.assertEqual(raw.country.nunique(), 4)
        self.assertTrue(raw.storm_price.notna().all())
        np.testing.assert_allclose(raw.storm_price - raw.target_price, 2., atol=1e-12)
        manifest = json.loads(csv.with_suffix(".manifest.json").read_text())
        self.assertEqual(manifest["benchmark"]["benchmark"], "Storm officiel dashboard")
        consumed = manifest["benchmark"]["consumed_artifact_sha256"]
        self.assertEqual(consumed[str(self.audit)], sha(self.audit))
        self.assertEqual(consumed[str(self.report_audit)], sha(self.report_audit))
        self.assertEqual(consumed[str(self.snapshot / "FR/storm.parquet")],
            sha(self.snapshot / "FR/storm.parquet"))
        frame = calibration.validate_input(raw,
            ("reference_price", "nyx_baseline", "test2_price"),
            "nyx_baseline", "reference_component_pilot")
        predictions, _ = calibration.generate_predictions(frame,
            (30, 60, 90), 30,
            component_columns=("reference_price", "nyx_baseline", "test2_price"))
        metrics, _ = calibration.evaluate(predictions, (30, 60, 90), 30, 45)
        matched = metrics[(metrics.period == "holdout") & metrics.matched_storm_only]
        self.assertTrue((matched.hours > 0).all())
        self.assertTrue(matched.storm_rmse.notna().all())
        self.assertTrue(matched.win_rate.notna().all())

    def test_tampered_reporting_parquet_rejected(self):
        path = self.snapshot / "FR/storm.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            reference_pilot.export(seed_record(self.fixture.repo),
                self.fixture.root / "tampered", self.app)

    def test_wrong_storm_country_rejected(self):
        self.document["countries"]["FR"]["storm"]["source"]["zone"] = "DE"
        self.save_audit()
        with self.assertRaises(ValueError):
            reference_pilot.export(seed_record(self.fixture.repo),
                self.fixture.root / "wrong_country", self.app)

    def test_unofficial_storm_column_rejected(self):
        self.document["countries"]["FR"]["storm"]["comparator"]["column"] = "storm_strict_08__q50"
        self.save_audit()
        with self.assertRaises(ValueError):
            reference_pilot.export(seed_record(self.fixture.repo),
                self.fixture.root / "wrong_comparator", self.app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
