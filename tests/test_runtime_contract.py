"""Compatibility boundaries for the scientific sources retained by this project."""
from pathlib import Path
from types import ModuleType
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from nyx_runtime import engine, preflight, runtime


class RuntimeContractTests(unittest.TestCase):
    def test_retained_sources_match_their_recorded_sha256(self):
        runtime.verify_runtime()

    def test_daily_hgb_schemas_load_from_the_shipped_default_path(self):
        from chronos2_hourly.nyx_historical_hgb_reference import archived_columns

        for zone in ("FR", "DE", "BE", "NL"):
            base = archived_columns(zone, "hist_residual_400")
            absolute = archived_columns(zone, "hist_absolute_400")
            augmented = archived_columns(zone, "augmented_hist_residual_400")
            with self.subTest(zone=zone):
                self.assertEqual(len(base), 292)
                self.assertEqual(absolute, base)
                self.assertEqual(len(augmented), 334)
                self.assertEqual(augmented[:292], base)
                self.assertEqual(len(set(augmented)), 334)

    def test_missing_or_changed_hgb_schema_blocks_before_training(self):
        document = json.loads((runtime.PRODUCTION / "SOURCE_MANIFEST.json").read_text(encoding="utf8"))
        schema = "runtime/config/nyx_historical_hgb_feature_schemas.json"
        self.assertIn(schema, [item["destination"] for item in document["source_files"]])
        with tempfile.TemporaryDirectory() as folder:
            production = Path(folder) / "production"
            production.mkdir()
            shutil.copyfile(runtime.PRODUCTION / "SOURCE_MANIFEST.json", production / "SOURCE_MANIFEST.json")
            for item in document["source_files"]:
                copied = production / item["destination"]
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(runtime.PRODUCTION / item["destination"], copied)
            target = production / schema
            for state in ("missing", "changed"):
                with self.subTest(state=state):
                    if state == "missing":
                        target.unlink()
                    else:
                        target.write_text("{}", encoding="utf8")
                    with patch.object(runtime, "PRODUCTION", production):
                        with self.assertRaisesRegex(ValueError, "nyx_historical_hgb_feature_schemas"):
                            runtime.verify_runtime()
                    if state == "missing":
                        inspection = preflight.inspect_live(
                            {"repo_root": str(production / "runtime")},
                            "2026-10-24", "2026-10-23T06:00:00Z")
                        check = inspection["checks"]["runtime_files"]
                        self.assertFalse(check["passed"])
                        self.assertIn("nyx_historical_hgb_feature_schemas", str(check))

    def test_all_historical_namespaces_reject_an_unrelated_installation(self):
        for name in ("chronos2_hourly", "chronos2_modular", "experiment_console", "marginal_cost_expert"):
            foreign = ModuleType(name)
            foreign.__file__ = str(Path("/unrelated-installation") / name / "__init__.py")
            with self.subTest(namespace=name), patch.dict(sys.modules, {name: foreign}):
                with self.assertRaisesRegex(RuntimeError, "runtime"):
                    engine.activate_runtime(runtime.RUNTIME)


if __name__ == "__main__":
    unittest.main()
