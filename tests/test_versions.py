"""The forecast environment must match the historical scientific recipe."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nyx_runtime import preflight


@pytest.mark.parametrize("observed", ["2.11.0", "2.11.0+cpu", "2.11.0+cu128", "2.11", "2.11.0.0", "0!2.11.0"])
def test_matching_public_release_accepts_valid_local_build(observed):
    with patch.object(preflight.importlib.metadata, "version", return_value=observed):
        result = preflight.inspect_versions({"torch": "2.11.0"})
    assert result["passed"]
    assert result["installed_versions"]["torch"] == observed


@pytest.mark.parametrize("observed", [
    "2.11.1", "2.10.0", "2.11.0rc1", "2.11.0.dev1", "2.11.0.post1",
    "1!2.11.0", "2.11.0+", "unidentified",
])
def test_different_public_version_or_invalid_metadata_is_blocked(observed):
    with patch.object(preflight.importlib.metadata, "version", return_value=observed):
        result = preflight.inspect_versions({"torch": "2.11.0"})
    assert not result["passed"]
    assert result["mismatches"] == [{"distribution": "torch", "expected": "2.11.0", "observed": observed}]


def test_missing_distribution_is_reported_even_without_importing_package():
    with patch.object(preflight.importlib.metadata, "version",
                      side_effect=preflight.importlib.metadata.PackageNotFoundError("catboost")):
        result = preflight.inspect_versions({"catboost": "1.2.10"})
    assert not result["passed"]
    assert result["missing_distributions"] == ["catboost"]
    assert result["installed_versions"]["catboost"] is None
    assert result["metadata_only"]


def test_read_exact_pins_and_private_client_contract(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("# recipe\nTorch==2.11.0\nscikit_learn==1.9.1 # frozen\n", encoding="utf-8")
    assert preflight.pinned_requirements(path) == {
        "torch": "2.11.0", "scikit-learn": "1.9.1", "tshistory-lite": "0.5",
    }


@pytest.mark.parametrize("text", ["torch>=2.11", "-r another.txt", "", "torch==2.11.0\nTORCH==2.10.0"])
def test_unpinned_or_ambiguous_manifest_is_rejected(tmp_path, text):
    path = tmp_path / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="requirements.txt"):
        preflight.pinned_requirements(path)


def test_python_contract_accepts_only_311():
    with patch.object(preflight.sys, "version_info", (3, 11, 15)):
        assert preflight.check_python_version()["observed"] == "3.11.15"
    for version in ((3, 10, 19), (3, 12, 8), (3, 14, 0)):
        with patch.object(preflight.sys, "version_info", version):
            with pytest.raises(ValueError, match="attendu 3.11"):
                preflight.check_python_version()


def test_preflight_version_blockers_include_observed_expected_and_private_remedy():
    def metadata(name):
        if name == "tshistory-lite":
            raise preflight.importlib.metadata.PackageNotFoundError(name)
        return "1.2.9"

    with patch.object(preflight, "pinned_requirements",
                      return_value={"catboost": "1.2.10", "tshistory-lite": "0.5"}), \
         patch.object(preflight.importlib.metadata, "version", side_effect=metadata), \
         patch.object(preflight, "validate_request",
                      return_value=("2026-10-22", SimpleNamespace(isoformat=lambda: "2026-10-21T06:00:00+00:00"))), \
         patch.object(preflight.sys, "version_info", (3, 11, 15)):
        report = preflight.inspect_live({}, "2026-10-22", "2026-10-21T06:00:00Z")
    assert not report["ready"]
    assert not report["checks"]["dependencies"]["passed"]
    assert any("catboost observé 1.2.9 ; attendu 1.2.10" in reason for reason in report["blockers"])
    assert any("tshistory-lite observé absent ; attendu 0.5" in reason and "wheel Saturn approuvé" in reason
               for reason in report["blockers"])
