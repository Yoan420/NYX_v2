"""Boundary tests for the retained CPU refit wrapper, with a fake producer."""
from __future__ import annotations

import contextlib
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nyx_runtime import cli, engine, forecast as live, preflight, runtime, snapshot


FIRST_DAY = "2026-03-28"
CUTOFF = "2026-03-27T07:00:00+00:00"


def forecasts():
    index = pd.date_range(FIRST_DAY, "2026-03-30", freq="h", inclusive="left",
                          tz="Europe/Paris").tz_convert("UTC")
    output = {}
    for zi, zone in enumerate(engine.ZONES):
        point = np.arange(len(index)) / 7 + zi
        output[zone] = pd.DataFrame({
            "price_eur_mwh": point, "p_negative": .1, "p_negative_raw": .2,
            "reference_eur_mwh": point + .5,
            "nyx__q10": point - 2, "nyx__q50": point, "nyx__q90": point + 2,
            "forecast_origin_utc": CUTOFF,
            "horizon_day": [(p.date() - date.fromisoformat(FIRST_DAY)).days + 1
                            for p in index.tz_convert("Europe/Paris")],
        }, index=index)
    return output


def fake_model_audit():
    return {"daily_refit_completed": True, "future_labels_used": False,
            "refit_all_trainable_nyx_models": True,
            "seed_inference_used_for_final_forecast": False,
            "final_model_refit_attempts": 7,
            "reference": {"current_fit_counts": {"hgb": 12, "test2_pairs": 2}}}


def fake_producer(repo, bundle, seed, first_day, cutoff, extension, context=None):
    """Produce a synthetic sealed result instead of fitting any estimator."""
    directory = extension / "daily_refit"
    directory.mkdir(parents=True)
    artifact = directory / "synthetic_model.bin"
    artifact.write_bytes(b"synthetic fixture, not an estimator")
    receipt = {"protocol": "nyx_daily_model_refit_v1", "status": "MODELS_COMPLETE",
               "future_labels_used": False, "first_delivery_day": first_day,
               "forecast_days": 2, "shared_cutoff_utc": cutoff.isoformat(),
               "new_forecast_refit_required": True,
               "artifact_sha256": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    frames = forecasts()
    return {"forecasts": frames, "baseline": frames, "model_audit": fake_model_audit()}


class ProductionLiveTests(unittest.TestCase):
    def test_two_day_canonical_output_keeps_native_quantiles_origin_and_dst_grid(self):
        frames = forecasts()
        original = {zone: frame.copy(deep=True) for zone, frame in frames.items()}
        result = live.canonical_forecasts(frames, FIRST_DAY, CUTOFF)
        self.assertEqual(len(result), 4 * 47)
        self.assertEqual(set(result.forecast_origin_utc), {CUTOFF})
        self.assertTrue(result.observed_eur_mwh.isna().all())
        self.assertFalse(result.interval_calibrated.any())
        self.assertEqual(result.groupby("horizon_day").size().to_dict(), {1: 4 * 24, 2: 4 * 23})
        for zone in engine.ZONES:
            pd.testing.assert_frame_equal(frames[zone], original[zone])

    def test_future_actual_labels_are_rejected(self):
        frames = forecasts()
        frames["FR"]["actual"] = 2.0
        with self.assertRaisesRegex(ValueError, "labels observés interdits"):
            live.canonical_forecasts(frames, FIRST_DAY, CUTOFF)

    def test_origins_cannot_drift_for_j2(self):
        frames = forecasts()
        frames["NL"].loc[frames["NL"].horizon_day.eq(2), "forecast_origin_utc"] = "2026-03-28T07:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "origine différente"):
            live.canonical_forecasts(frames, FIRST_DAY, CUTOFF)

    def test_missing_hour_or_crossed_quantiles_are_rejected(self):
        frames = forecasts()
        frames["BE"] = frames["BE"].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "grille physique"):
            live.canonical_forecasts(frames, FIRST_DAY, CUTOFF)
        frames = forecasts()
        frames["DE"].iloc[0, frames["DE"].columns.get_loc("nyx__q10")] = 10
        with self.assertRaisesRegex(ValueError, "quantiles natifs croisés"):
            live.canonical_forecasts(frames, FIRST_DAY, CUTOFF)

    def test_request_requires_exact_shared_08h_paris_cutoff(self):
        with self.assertRaisesRegex(ValueError, "exactement"):
            engine.validate_request(FIRST_DAY, "2026-03-27T08:00:00+00:00")
        with self.assertRaisesRegex(ValueError, "fuseau explicite"):
            engine.validate_request(FIRST_DAY, "2026-03-27T08:00:00")

    def test_refit_wrapper_publishes_only_validated_new_forecasts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("runtime", "bundle", "seed"):
                (root / name).mkdir()
            config = {"repo_root": str(root / "runtime"), "seed_bundle": str(root / "bundle"),
                      "seed_output": str(root / "seed"), "collect_reporting_sources": False}
            module = SimpleNamespace(produce_refit_horizon=fake_producer)
            original_import = live.importlib.import_module
            def import_module(name, *args, **kwargs):
                return module if name == "chronos2_hourly.nyx_daily_refit" else original_import(name, *args, **kwargs)
            with patch.object(live, "inspect_live", return_value={"ready": True}), \
                 patch.object(live, "activate_runtime"), \
                 patch.object(live.importlib, "import_module", side_effect=import_module):
                result = live.run_live(config, FIRST_DAY, CUTOFF, root / "new-run")
            self.assertEqual(result["status"], "CSV_COMPLETE")
            self.assertEqual(result["rows"], 4 * 47)
            self.assertFalse(result["qualified_for_production"])
            self.assertTrue(Path(result["predictions_csv"]).is_file())
            self.assertEqual(set(result["forecasts_csv"]), set(engine.ZONES))
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(audit["reporting_audit"]["status"], "not_requested")

    def test_incomplete_refit_attestation_blocks_csv_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("runtime", "bundle", "seed"):
                (root / name).mkdir()
            config = {"repo_root": str(root / "runtime"), "seed_bundle": str(root / "bundle"),
                      "seed_output": str(root / "seed"), "collect_reporting_sources": False}
            def incomplete(*args, **kwargs):
                result = fake_producer(*args, **kwargs)
                result["model_audit"]["refit_all_trainable_nyx_models"] = False
                return result
            module = SimpleNamespace(produce_refit_horizon=incomplete)
            original_import = live.importlib.import_module
            def import_module(name, *args, **kwargs):
                return module if name == "chronos2_hourly.nyx_daily_refit" else original_import(name, *args, **kwargs)
            with patch.object(live, "inspect_live", return_value={"ready": True}), \
                 patch.object(live, "activate_runtime"), \
                 patch.object(live.importlib, "import_module", side_effect=import_module), \
                 self.assertRaisesRegex(ValueError, "réentraînement quotidien complet"):
                live.run_live(config, FIRST_DAY, CUTOFF, root / "new-run")
            self.assertFalse((root / "new-run/predictions.csv").exists())

    def test_keyboard_interrupt_preserves_success_pointer_and_records_interrupted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, outputs = root / "inputs", root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "snapshot.json").write_text("{}")
            config = root / "config.json"
            config.write_text(json.dumps({"input_root": "inputs", "output_root": "outputs"}))
            latest = outputs / "latest.json"
            latest.write_text('{"run_id":"previous-success"}')
            with patch.object(preflight, "inspect_live", return_value={"ready": True}), \
                 patch.object(live, "run_live", side_effect=KeyboardInterrupt), \
                 patch.object(runtime, "runtime_lock", return_value=contextlib.nullcontext()), \
                 patch.object(runtime, "RUNTIME", root / "runtime"), \
                 patch.object(runtime, "verify_runtime"), \
                 patch.object(runtime, "copy_state"), \
                 patch.object(snapshot, "validate_snapshot", return_value={"protocol": snapshot.PROTOCOL}), \
                 patch.dict("os.environ"), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["forecast", "--config", str(config),
                                 "--first-day", FIRST_DAY, "--as-of", CUTOFF])
            self.assertEqual(code, 130)
            self.assertEqual(latest.read_text(), '{"run_id":"previous-success"}')
            statuses = list(outputs.glob("*/status.json"))
            self.assertEqual(len(statuses), 1)
            self.assertEqual(json.loads(statuses[0].read_text())["state"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
