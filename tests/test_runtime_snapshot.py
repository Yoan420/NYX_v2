from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from nyx_runtime.snapshot import (
    CHRONOS_REVISION, MODEL_DIRECTORY, create_snapshot, validate_snapshot,
)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle, self.output = self.root / "bundle", self.root / "output"
        self.runtime, self.hub = self.root / "runtime", self.root / "hub"
        self.weights = self.hub / MODEL_DIRECTORY / "snapshots" / CHRONOS_REVISION
        for directory in (self.bundle, self.output, self.runtime, self.weights):
            directory.mkdir(parents=True)
        self.receipt = b'{"source":"untouched Windows C:\\\\NYX","state":"COMPLETE"}\r\n'
        (self.bundle / "receipt.json").write_bytes(self.receipt)
        (self.output / "model.cbm").write_bytes(b"saved estimator bytes")
        (self.weights / "config.json").write_text("{}")
        (self.weights / "model.safetensors").write_bytes(b"model weights")
        self.destination = self.root / "snapshot"

    def export(self, **overrides):
        options = dict(seed_bundle=self.bundle, seed_output=self.output,
                       runtime_root=self.runtime, hf_hub_cache=self.hub,
                       destination=self.destination)
        options.update(overrides)
        return create_snapshot(**options)

    def test_exports_exact_bytes_and_selected_states_only(self):
        state = self.runtime / "runs/nyx_daily_cpu_states/seed/20261022"
        state.mkdir(parents=True)
        (state / "receipt.json").write_bytes(b'{"retained":"unchanged"}')
        capture = self.runtime / "data/pit/nyx_annual_hydro_captures/day"
        capture.mkdir(parents=True)
        (capture / "capture.json").write_bytes(b"source capture")
        (self.runtime / "credentials.json").write_bytes(b"must not be exported")
        document = self.export()
        self.assertFalse(document["receipts_rewritten"])
        self.assertEqual((self.destination / "seed_bundle/receipt.json").read_bytes(), self.receipt)
        self.assertEqual(document["files"]["seed_bundle/receipt.json"], hashlib.sha256(self.receipt).hexdigest())
        self.assertTrue((self.destination / "state/runs/nyx_daily_cpu_states/seed/20261022/receipt.json").is_file())
        self.assertTrue((self.destination / "state/data/pit/nyx_annual_hydro_captures/day/capture.json").is_file())
        self.assertFalse(any("credentials" in name for name in document["files"]))
        self.assertEqual(validate_snapshot(self.destination), document)

    def test_materializes_huggingface_file_links_to_model_blobs(self):
        weights = self.weights / "model.safetensors"
        weights.unlink()
        blob = self.hub / MODEL_DIRECTORY / "blobs/digest"
        blob.parent.mkdir()
        blob.write_bytes(b"shared model blob")
        weights.symlink_to(blob)
        self.export()
        copied = self.destination / "huggingface/hub" / MODEL_DIRECTORY / "snapshots" / CHRONOS_REVISION / weights.name
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), blob.read_bytes())
        self.assertTrue(weights.is_symlink())

    def test_rejects_seed_links_without_creating_destination(self):
        (self.bundle / "linked.json").symlink_to(self.output / "model.cbm")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.export()
        self.assertFalse(self.destination.exists())

    def test_rejects_huggingface_links_outside_model_directory(self):
        (self.weights / "escaped.json").symlink_to(self.bundle / "receipt.json")
        with self.assertRaisesRegex(ValueError, "escaping symbolic link"):
            self.export()
        self.assertFalse(self.destination.exists())

    def test_rejects_directory_links_and_source_overlaps(self):
        (self.weights / "directory_link").symlink_to(self.bundle, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.export()
        (self.weights / "directory_link").unlink()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.export(destination=self.runtime / "export")
        self.assertFalse((self.runtime / "export").exists())

    def test_existing_destination_is_never_removed(self):
        self.destination.mkdir()
        protected = self.destination / "existing.txt"
        protected.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(protected.read_text(), "keep")

    def test_rejects_changed_and_stale_snapshot_artifacts(self):
        self.export()
        artifact = self.destination / "seed_output/model.cbm"
        artifact.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_snapshot(self.destination)
        artifact.write_bytes(b"saved estimator bytes")
        (self.destination / "stale.txt").write_text("old S3 object")
        with self.assertRaisesRegex(ValueError, "new, empty S3 prefix"):
            validate_snapshot(self.destination)

    def test_rejects_traversal_in_manifest(self):
        self.export()
        path = self.destination / "snapshot.json"
        document = json.loads(path.read_text())
        document["files"]["../outside"] = "0" * 64
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "portable snapshot"):
            validate_snapshot(self.destination)

    def test_cleanup_after_source_changes_during_export(self):
        from nyx_runtime import snapshot
        copy = snapshot.shutil.copyfile

        def change_source(source, destination):
            result = copy(source, destination)
            if Path(source) == self.output / "model.cbm":
                Path(source).write_bytes(b"concurrent source modification")
            return result

        with patch.object(snapshot.shutil, "copyfile", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.export()
        self.assertFalse(self.destination.exists())
        self.assertTrue((self.bundle / "receipt.json").is_file())

    def test_rejects_retained_lock_sentinel(self):
        state = self.runtime / "runs/nyx_daily_cpu_states/seed"
        state.mkdir(parents=True)
        (state / ".daily_state.lock").write_text('{"hostname":"other-host"}')
        with self.assertRaisesRegex(ValueError, "stop the NYX run"):
            self.export()


if __name__ == "__main__":
    unittest.main()
