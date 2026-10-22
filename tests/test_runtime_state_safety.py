"""State restoration must not delete through aliases or overlapping roots."""
from pathlib import Path
from types import SimpleNamespace
import stat
from unittest.mock import patch

import pytest

from nyx_runtime import runtime


def _state(root: Path, relative: str, text="state"):
    path = root / relative / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("source_relative,destination_relative", [
    ("state", "state"), ("state", "state/child"), ("state/child", "state"),
])
def test_overlapping_roots_fail_before_any_recursive_delete(tmp_path, source_relative, destination_relative):
    source, target = tmp_path / source_relative, tmp_path / destination_relative
    with patch.object(runtime.shutil, "rmtree") as delete, patch.object(runtime.shutil, "copytree") as copy:
        with pytest.raises(ValueError, match="non imbriquées"):
            runtime.copy_state(source, target)
    delete.assert_not_called()
    copy.assert_not_called()


def test_resolved_target_must_stay_inside_intended_root(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "target"
    escaped = destination / runtime.STATE_PATHS[-1]
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == escaped:
            return tmp_path / "outside"
        return original_resolve(path, *args, **kwargs)

    with patch.object(Path, "resolve", resolve), patch.object(runtime.shutil, "rmtree") as delete:
        with pytest.raises(ValueError, match="hors de la racine"):
            runtime.copy_state(source, destination)
    delete.assert_not_called()


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("tag", [0xA0000003, 0xA000000C])
def test_junction_or_symlink_ancestor_is_rejected_even_when_resolve_stays_inside(
        tmp_path, side, tag):
    source, destination = tmp_path / "source", tmp_path / "destination"
    selected = source if side == "source" else destination
    redirected = selected / "data"
    # Simulates a junction to another directory *within* the same intended root:
    # containment alone would permit it, but deleting through aliases must not.
    original_lstat = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == redirected:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=tag)
        return original_lstat(path, *args, **kwargs)

    with patch.object(Path, "lstat", lstat), patch.object(runtime.shutil, "rmtree") as delete:
        with pytest.raises(ValueError, match="lien/jonction"):
            runtime.copy_state(source, destination)
    delete.assert_not_called()


def test_nested_source_link_is_checked_before_any_destination_removal(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _state(destination, runtime.STATE_PATHS[0], "keep until entire source verified")
    nested = _state(source, runtime.STATE_PATHS[-1], "untrusted nested link")
    original_lstat = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == nested:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_reparse_tag=0)
        return original_lstat(path, *args, **kwargs)

    with patch.object(Path, "lstat", lstat), patch.object(runtime.shutil, "rmtree") as delete:
        with pytest.raises(ValueError, match="lien/jonction"):
            runtime.copy_state(source, destination)
    delete.assert_not_called()
    assert (destination / runtime.STATE_PATHS[0] / "receipt.json").read_text() == "keep until entire source verified"


def test_source_root_alias_is_rejected_before_resolving_it(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    original = runtime._redirected
    with patch.object(runtime, "_redirected", side_effect=lambda path: path == source or original(path)), \
         patch.object(runtime.shutil, "rmtree") as delete:
        with pytest.raises(ValueError, match="Racine ou parent"):
            runtime.copy_state(source, destination)
    delete.assert_not_called()


def test_normal_restore_copies_only_selected_state_and_removes_stale_state(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    _state(source, runtime.STATE_PATHS[0], "new verified state")
    _state(destination, runtime.STATE_PATHS[0], "old state")
    _state(destination, runtime.STATE_PATHS[1], "stale state absent from source")
    unrelated = destination / "keep.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    runtime.copy_state(source, destination)
    assert (destination / runtime.STATE_PATHS[0] / "receipt.json").read_text() == "new verified state"
    assert not (destination / runtime.STATE_PATHS[1]).exists()
    assert unrelated.read_text() == "unrelated"


def test_runtime_lock_refuses_redirected_runs_ancestor_before_open(tmp_path, monkeypatch):
    core = tmp_path / "runtime"
    monkeypatch.setattr(runtime, "RUNTIME", core)
    original = runtime._redirected
    with patch.object(runtime, "_redirected", side_effect=lambda path: path == core / "runs" or original(path)):
        with pytest.raises(ValueError, match="lien/jonction"):
            with runtime.runtime_lock():
                pytest.fail("redirected lock acquired")
    assert not core.exists()


def test_runtime_lock_cleanup_preserves_regular_lock_behavior(tmp_path, monkeypatch):
    core = tmp_path / "runtime"
    monkeypatch.setattr(runtime, "RUNTIME", core)
    with runtime.runtime_lock():
        assert (core / "runs/.nyx_cloud.lock").is_file()
        with pytest.raises(ValueError, match="déjà verrouillé"):
            with runtime.runtime_lock():
                pytest.fail("concurrent lock acquired")
    assert not (core / "runs/.nyx_cloud.lock").exists()
