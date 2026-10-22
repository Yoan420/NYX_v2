"""Compatibility boundaries for the scientific sources retained by this project."""
from pathlib import Path
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

from nyx_runtime import engine, runtime


class RuntimeContractTests(unittest.TestCase):
    def test_retained_sources_match_their_recorded_sha256(self):
        runtime.verify_runtime()

    def test_all_historical_namespaces_reject_an_unrelated_installation(self):
        for name in ("chronos2_hourly", "chronos2_modular", "experiment_console", "marginal_cost_expert"):
            foreign = ModuleType(name)
            foreign.__file__ = str(Path("/unrelated-installation") / name / "__init__.py")
            with self.subTest(namespace=name), patch.dict(sys.modules, {name: foreign}):
                with self.assertRaisesRegex(RuntimeError, "runtime"):
                    engine.activate_runtime(runtime.RUNTIME)


if __name__ == "__main__":
    unittest.main()
