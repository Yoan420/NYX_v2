"""User configuration remains portable and command-line overrides are explicit."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nyx_runtime import cli, pipeline
from nyx_runtime.settings import load_settings


class RuntimeSettingsTests(unittest.TestCase):
    def _config(self, root, document):
        directory = root / "configuration"
        directory.mkdir()
        path = directory / "nyx.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_relative_data_paths_follow_configuration_even_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self._config(root, {"input_root": "../inputs", "output_root": "results"})
            other = root / "unrelated-work-directory"
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(other)
                settings = load_settings(config)
            finally:
                os.chdir(previous)
            self.assertEqual(settings.input_root, root / "inputs")
            self.assertEqual(settings.output_root, config.parent / "results")

    def test_explicit_overrides_win_over_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self._config(root, {"input_root": "inputs", "output_root": "outputs",
                "collect_reporting_sources": False})
            settings = load_settings(config, input_root=root / "selected-input",
                output_root=root / "selected-output", collect_reporting_sources=True)
            self.assertEqual(settings.input_root, root / "selected-input")
            self.assertEqual(settings.output_root, root / "selected-output")
            self.assertTrue(settings.collect_reporting_sources)


    def test_configuration_cannot_silently_change_scientific_parameters_or_coerce_types(self):
        documents = ({"learning_rate": .1}, {"collect_reporting_sources": "false"},
                     {"input_root": 12}, {"output_s3_uri": ["s3://bucket/results"]})
        for document in documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as folder:
                config = self._config(Path(folder), document)
                with self.assertRaises(ValueError):
                    load_settings(config)

    def test_public_cli_applies_configuration_and_explicit_reporting_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self._config(root, {"input_root": "inputs", "output_root": "outputs",
                "collect_reporting_sources": True})
            selected = root / "selected-input"
            with patch.dict(os.environ, {}, clear=True), patch.object(pipeline, "execute", return_value=0) as execute:
                code = cli.main(["preflight", "--config", str(config), "--input-root", str(selected),
                                 "--skip-reporting-sources", "--first-day", "2026-03-28",
                                 "--as-of", "2026-03-27T07:00:00Z"])
            self.assertEqual(code, 0)
            command, settings, plan = execute.call_args.args
            self.assertEqual(command, "preflight")
            self.assertEqual(settings.input_root, selected)
            self.assertEqual(settings.output_root, config.parent / "outputs")
            self.assertFalse(settings.collect_reporting_sources)
            self.assertEqual(plan["forecast_origin_utc"], "2026-03-27T07:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
