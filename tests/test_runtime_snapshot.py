from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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

    def seed_inventory(self, files):
        """Write a real-shaped original receipt, retaining its CRLF bytes."""
        receipt = json.dumps({"input_sha256": {
            name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()
        }}, indent=2).replace("\n", "\r\n").encode("utf-8") + b"\r\n"
        (self.output / "receipt.json").write_bytes(receipt)
        return receipt

    def make_symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows does not grant symbolic-link creation privilege")
            raise

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
        self.make_symlink(weights, blob)
        self.export()
        copied = self.destination / "huggingface/hub" / MODEL_DIRECTORY / "snapshots" / CHRONOS_REVISION / weights.name
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), blob.read_bytes())
        self.assertTrue(weights.is_symlink())

    def test_rejects_seed_links_without_creating_destination(self):
        self.make_symlink(self.bundle / "linked.json", self.output / "model.cbm")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            self.export()
        self.assertFalse(self.destination.exists())

    def test_rejects_huggingface_links_outside_model_directory(self):
        self.make_symlink(self.weights / "escaped.json", self.bundle / "receipt.json")
        with self.assertRaisesRegex(ValueError, "escaping symbolic link"):
            self.export()
        self.assertFalse(self.destination.exists())

    def test_rejects_directory_links(self):
        self.make_symlink(self.weights / "directory_link", self.bundle, directory=True)
        with self.assertRaises(ValueError):
            self.export()

    def test_rejects_source_overlaps(self):
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
        (self.destination / "stale.txt").write_text("stale snapshot artifact")
        with self.assertRaisesRegex(ValueError, "new, empty directory"):
            validate_snapshot(self.destination)

    def test_lfs_pointer_blocks_runtime_preparation_with_download_instruction(self):
        from nyx_runtime.pipeline import _prepare_runtime
        from nyx_runtime.settings import RunSettings

        document = self.export()
        name = f"huggingface/hub/{MODEL_DIRECTORY}/snapshots/{CHRONOS_REVISION}/model.safetensors"
        artifact = self.destination / name
        settings = RunSettings(self.destination, self.root / "results")
        for newline in (b"\n", b"\r\n"):
            pointer = newline.join((b"version https://git-lfs.github.com/spec/v1",
                                    b"oid sha256:" + document["files"][name].encode("ascii"),
                                    b"size 13", b""))
            artifact.write_bytes(pointer)
            with self.subTest(newline=newline), \
                    patch("nyx_runtime.pipeline.runtime.verify_runtime") as verify, \
                    patch("nyx_runtime.pipeline.runtime.copy_state") as copy:
                with self.assertRaisesRegex(ValueError, r"Git LFS.*model\.safetensors.*git lfs pull"):
                    _prepare_runtime(settings)
                verify.assert_not_called()
                copy.assert_not_called()
                self.assertEqual(artifact.read_bytes(), pointer)
                self.assertFalse(settings.output_root.exists())

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

    def test_restores_only_receipt_proven_json_in_destination(self):
        originals = {
            "audits/weather.json": b'{\r\n  "source": "weather",\r\n  "rows": 48\r\n}\r\n',
            "captures/receipt.json": b'{\r\n  "complete": true\r\n}\r\n',
            "unchanged.json": b'{"existing": "CRLF"}\r\n',
            "features.parquet": b"unchanged binary\x00\xff\n",
        }
        for name, original in originals.items():
            path = self.bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original.replace(b"\r\n", b"\n") if "/" in name else original)
        receipt = self.seed_inventory(originals)
        source_before = {path.relative_to(self.bundle): path.read_bytes()
                         for path in self.bundle.rglob("*") if path.is_file()}
        document = self.export(restore_git_newlines=True)
        self.assertEqual(validate_snapshot(self.destination), document)
        self.assertEqual((self.output / "receipt.json").read_bytes(), receipt)
        self.assertEqual((self.destination / "seed_output/receipt.json").read_bytes(), receipt)
        self.assertEqual({path.relative_to(self.bundle): path.read_bytes()
                          for path in self.bundle.rglob("*") if path.is_file()}, source_before)
        expected_restorations = []
        for name, original in originals.items():
            self.assertEqual((self.destination / "seed_bundle" / name).read_bytes(), original)
            if "/" in name:
                expected_restorations.append({
                    "path": "seed_bundle/" + name,
                    "source_sha256": hashlib.sha256(source_before[Path(name)]).hexdigest(),
                    "restored_sha256": hashlib.sha256(original).hexdigest(),
                })
        self.assertEqual(document["git_newline_restorations"], expected_restorations)
        self.assertFalse(document["receipts_rewritten"])
        self.assertEqual(document["receipts_rewritten_scope"], ["seed_output/receipt.json", "state/"])
        for entry in expected_restorations:
            self.assertEqual(document["files"][entry["path"]], entry["restored_sha256"])

    def test_git_newline_recovery_is_opt_in(self):
        original = b'{\r\n  "value": 1\r\n}\r\n'
        raw = original.replace(b"\r\n", b"\n")
        (self.bundle / "audit.json").write_bytes(raw)
        self.seed_inventory({"audit.json": original})
        document = self.export()
        self.assertEqual((self.destination / "seed_bundle/audit.json").read_bytes(), raw)
        self.assertEqual(document["git_newline_restorations"], [])

    def test_recovery_checks_every_required_input_before_creating_destination(self):
        original = b'{\r\n  "value": 1\r\n}\r\n'
        (self.bundle / "audit.json").write_bytes(original.replace(b"\r\n", b"\n"))
        for name, value, pattern in (("damaged.json", b'{"value":2}\n', "cannot be restored"),
                                     ("missing.json", None, "missing")):
            with self.subTest(name=name):
                if value is not None:
                    (self.bundle / name).write_bytes(value)
                receipt = self.seed_inventory({"audit.json": original, name: original})
                with self.assertRaisesRegex(ValueError, pattern):
                    self.export(restore_git_newlines=True)
                self.assertFalse(self.destination.exists())
                self.assertEqual((self.output / "receipt.json").read_bytes(), receipt)
                self.assertEqual((self.bundle / "audit.json").read_bytes(), original.replace(b"\r\n", b"\n"))

    def test_recovery_rejects_non_json_or_invalid_json_even_if_newlines_match(self):
        for name, raw in (("model.cbm", b'{"value":1}\n'),
                          ("invalid.json", b"\xff\n"),
                          ("invalid.json", b"not JSON\n"),
                          ("invalid.json", b'{"value":NaN}\n')):
            with self.subTest(name=name, raw=raw):
                (self.bundle / name).write_bytes(raw)
                self.seed_inventory({name: raw.replace(b"\n", b"\r\n")})
                with self.assertRaisesRegex(ValueError, "JSON"):
                    self.export(restore_git_newlines=True)
                self.assertFalse(self.destination.exists())

    def test_recovery_rejects_unsafe_receipt_paths_and_hashes(self):
        for name in ("../outside.json", "/outside.json", "C:/outside.json", "x\\outside.json",
                     "x//outside.json", "./outside.json", "x/../outside.json", "x:stream.json",
                     "NUL.json", "x./outside.json", "x /outside.json", "x\x00.json"):
            with self.subTest(name=name):
                self.seed_inventory({name: b"{}\r\n"})
                with self.assertRaisesRegex(ValueError, "portable seed receipt"):
                    self.export(restore_git_newlines=True)
                self.assertFalse(self.destination.exists())
        for digest in ("0" * 63, "G" * 64, "A" * 64, None):
            with self.subTest(digest=digest):
                (self.output / "receipt.json").write_text(json.dumps({"input_sha256": {"audit.json": digest}}))
                with self.assertRaisesRegex(ValueError, "portable seed receipt"):
                    self.export(restore_git_newlines=True)
                self.assertFalse(self.destination.exists())

    def test_recovery_requires_receipt_inventory_and_rejects_duplicate_paths(self):
        for receipt in (None, {}, {"input_sha256": {}}, {"input_sha256": []},
                        {"input_sha256": {"audit.json": "0" * 64, "AUDIT.json": "0" * 64}}):
            with self.subTest(receipt=receipt):
                if receipt is not None:
                    (self.output / "receipt.json").write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    self.export(restore_git_newlines=True)
                self.assertFalse(self.destination.exists())
        (self.output / "receipt.json").write_text('{"input_sha256":{},"input_sha256":{}}')
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            self.export(restore_git_newlines=True)

    def test_recovery_still_rejects_source_changes_during_export(self):
        from nyx_runtime import snapshot
        original = b'{"value":1}\r\n'
        source = self.bundle / "audit.json"
        source.write_bytes(original.replace(b"\r\n", b"\n"))
        self.seed_inventory({"audit.json": original})
        copy = snapshot.shutil.copyfile

        def change_source(current, destination):
            result = copy(current, destination)
            if Path(current) == source:
                source.write_bytes(b'{"value":2}\n')
            return result

        with patch.object(snapshot.shutil, "copyfile", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.export(restore_git_newlines=True)
        self.assertFalse(self.destination.exists())

    def test_export_cli_forwards_explicit_recovery_flag(self):
        from nyx_runtime.cli import main
        args = ["export-input", "--seed-bundle", str(self.bundle), "--seed-output", str(self.output),
                "--runtime-root", str(self.runtime), "--hf-hub-cache", str(self.hub),
                "--destination", str(self.destination)]
        with patch("nyx_runtime.snapshot.create_snapshot", return_value={}) as export, patch("builtins.print"):
            self.assertEqual(main(args), 0)
            self.assertFalse(export.call_args.kwargs["restore_git_newlines"])
            self.assertEqual(main(args + ["--restore-git-newlines"]), 0)
            self.assertTrue(export.call_args.kwargs["restore_git_newlines"])

    def test_rejects_directory_junctions_at_root_or_inside_bundle(self):
        nested = self.bundle / "redirected"
        nested.mkdir()
        original = Path.lstat
        for redirected in (self.bundle, nested):
            def reparse_stat(path):
                info = original(path)
                return (SimpleNamespace(st_mode=info.st_mode, st_reparse_tag=0xA0000003)
                        if path == redirected else info)

            with self.subTest(path=redirected), patch.object(Path, "lstat", new=reparse_stat):
                with self.assertRaisesRegex(ValueError, "junction"):
                    self.export()
                self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
