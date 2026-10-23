"""A coworker's daily run must not inherit another operator or cutoff."""
import os
from unittest.mock import patch

from nyx_runtime import preflight


def test_preflight_requires_operator_without_exposing_identity():
    with patch.dict(os.environ, {}, clear=True):
        blocked = preflight.inspect_live({}, "2026-10-22", "2026-10-21T06:00:00Z")
    assert not blocked["checks"]["saturn_author"]["passed"]
    assert not blocked["ready"]
    with patch.dict(os.environ, {"SATURN_AUTHOR": "synthetic-operator"}, clear=True):
        configured = preflight.inspect_live({}, "2026-10-22", "2026-10-21T06:00:00Z")
    assert configured["checks"]["saturn_author"]["passed"]
    assert "synthetic-operator" not in str(configured)


def test_preflight_rejects_inherited_anytime_cutoff():
    with patch.dict(os.environ, {"SATURN_AUTHOR": "synthetic", "NYX_ANNUAL_OPERATIONAL_CONTEXT": "{}"}):
        blocked = preflight.inspect_live({}, "2026-10-22", "2026-10-21T06:00:00Z")
    assert not blocked["checks"]["fixed_cutoff_context"]["passed"]
    assert not blocked["ready"]
