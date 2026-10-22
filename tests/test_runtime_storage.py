"""Offline S3 protocol tests: no AWS account, credentials, or network needed."""
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from nyx_runtime.storage import (
    S3Store, StorageError, download_prefix, parse_s3_uri, upload_directory,
)


class S3Error(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.events = []
        self.fail_upload = None

    def put_object(self, Bucket, Key, Body, **kwargs):
        pair = (Bucket, Key)
        if kwargs.get("IfNoneMatch") == "*" and pair in self.objects:
            raise S3Error("PreconditionFailed")
        self.objects[pair] = Body.encode() if isinstance(Body, str) else Body
        self.events.append(("put", Key))
        return {"ETag": '"etag"'}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise S3Error("NoSuchKey")
        return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.events.append(("delete", Key))
        self.objects.pop((Bucket, Key), None)

    def upload_file(self, Filename, Bucket, Key):
        if self.fail_upload and self.fail_upload in Key:
            raise OSError("simulated interrupted upload")
        self.objects[(Bucket, Key)] = Path(Filename).read_bytes()
        self.events.append(("upload", Key))

    def download_file(self, Bucket, Key, Filename):
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise S3Error("NoSuchKey") from error
        Path(Filename).write_bytes(data)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                # One object per page proves callers do not silently lose pages.
                for bucket, key in sorted(client.objects):
                    if bucket == Bucket and key.startswith(Prefix):
                        yield {"Contents": [{"Key": key}]}

        return Paginator()


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.s3 = FakeS3()
        self.store = S3Store("s3://forecast-bucket/results", self.s3)

    def tearDown(self):
        self.directory.cleanup()

    def source(self, name="source"):
        source = self.root / name
        (source / "nyx_daily_cpu_states/seed/day").mkdir(parents=True)
        (source / "nyx_daily_cpu_states/seed/day/values.parquet").write_bytes(b"state values")
        (source / "nyx_daily_cpu_states/seed/day/receipt.json").write_text('{"state":"COMPLETE"}')
        (source / "nyx_daily_cpu_states/seed/.daily_state.lock").write_text("transient lock")
        return source

    def test_uri_and_traversal_rejection(self):
        self.assertEqual(parse_s3_uri("s3://forecast-bucket/input/current/"),
                         ("forecast-bucket", "input/current"))
        for uri in ("s3:///input", "https://bucket/input", "s3://bucket/input/../secret",
                    "s3://bucket/input/%2e%2e/secret", "s3://bucket/C:/secret",
                    "s3://bucket/input\\secret", "s3://bucket/input?token=secret"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_s3_uri(uri)

    def test_downloads_all_pages_and_rejects_remote_traversal(self):
        self.s3.objects[("forecast-bucket", "input/a.txt")] = b"one"
        self.s3.objects[("forecast-bucket", "input/nested/b.txt")] = b"two"
        target = self.root / "download"
        self.assertEqual(download_prefix("s3://forecast-bucket/input", target, self.s3),
                         ["a.txt", "nested/b.txt"])
        self.assertEqual((target / "nested/b.txt").read_bytes(), b"two")
        self.s3.objects[("forecast-bucket", "input/../escape")] = b"unsafe"
        with self.assertRaises(ValueError):
            download_prefix("s3://forecast-bucket/input", target, self.s3)
        self.assertFalse((self.root / "escape").exists())

    def test_symlinks_are_rejected_in_both_directions(self):
        outside = self.root / "outside"
        outside.mkdir()
        download = self.root / "download"
        download.mkdir()
        (download / "nested").symlink_to(outside, target_is_directory=True)
        self.s3.objects[("forecast-bucket", "input/nested/secret")] = b"unsafe"
        with self.assertRaises(StorageError):
            download_prefix("s3://forecast-bucket/input", download, self.s3)
        self.assertFalse((outside / "secret").exists())
        with self.assertRaises(StorageError):
            upload_directory(download, "s3://forecast-bucket/output", self.s3)

    def test_lock_contention_and_owner_release(self):
        with self.store.lock():
            another = S3Store("s3://forecast-bucket/results", self.s3)
            with self.assertRaisesRegex(StorageError, "Another forecast"):
                with another.lock():
                    self.fail("contender entered lock")
        self.assertNotIn(("forecast-bucket", "results/locks/forecast.json"), self.s3.objects)

    def test_changed_lock_owner_is_never_deleted(self):
        with self.assertRaisesRegex(StorageError, "ownership changed"):
            with self.store.lock():
                self.s3.objects[("forecast-bucket", "results/locks/forecast.json")] = b'{"token":"replacement"}'
        self.assertIn(("forecast-bucket", "results/locks/forecast.json"), self.s3.objects)

    def test_state_roundtrip_is_verified_and_lock_files_are_excluded(self):
        source = self.source()
        with self.store.lock():
            published = self.store.publish_state(source, "run1")
        self.assertEqual(len(published["files"]), 2)
        destination = self.root / "restored"
        self.assertEqual(self.store.restore_state(destination), published)
        self.assertEqual((destination / "nyx_daily_cpu_states/seed/day/values.parquet").read_bytes(), b"state values")
        self.assertFalse((destination / "nyx_daily_cpu_states/seed/.daily_state.lock").exists())
        puts = [key for kind, key in self.s3.events if kind == "put"]
        self.assertEqual(puts[-1], "results/state/latest.json")

    def test_partial_state_upload_preserves_last_complete_pointer(self):
        source = self.source()
        with self.store.lock():
            self.store.publish_state(source, "complete")
        pointer = self.s3.objects[("forecast-bucket", "results/state/latest.json")]
        self.s3.fail_upload = "versions/interrupted/"
        with self.store.lock():
            with self.assertRaises(OSError):
                self.store.publish_state(source, "interrupted")
        self.assertEqual(self.s3.objects[("forecast-bucket", "results/state/latest.json")], pointer)
        self.assertNotIn(("forecast-bucket", "results/state/versions/interrupted/_MANIFEST.json"), self.s3.objects)

    def test_corrupt_state_is_rejected_before_overwriting_local_file(self):
        source = self.source()
        with self.store.lock():
            self.store.publish_state(source, "run1")
        name = "nyx_daily_cpu_states/seed/day/values.parquet"
        self.s3.objects[("forecast-bucket", "results/state/versions/run1/" + name)] = b"corrupted"
        destination = self.root / "restored"
        target = destination / name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"locally valid")
        with self.assertRaisesRegex(StorageError, "checksum mismatch"):
            self.store.restore_state(destination)
        self.assertEqual(target.read_bytes(), b"locally valid")

    def test_success_marker_is_last_and_partial_run_has_none(self):
        source = self.root / "run"
        source.mkdir()
        (source / "report.html").write_text("report")
        (source / "receipt.json").write_text("{}")
        with self.store.lock():
            self.store.publish_run(source, "run1")
            self.assertEqual(self.s3.events[-1], ("put", "results/runs/run1/_SUCCESS.json"))
            self.s3.fail_upload = "runs/partial/"
            with self.assertRaises(OSError):
                self.store.publish_run(source, "partial")
        self.assertNotIn(("forecast-bucket", "results/runs/partial/_SUCCESS.json"), self.s3.objects)

    def test_version_id_cannot_be_reused(self):
        source = self.source()
        with self.store.lock():
            self.store.publish_state(source, "run1")
            with self.assertRaisesRegex(StorageError, "already exists"):
                self.store.publish_state(source, "run1")

    def test_publication_without_lock_is_rejected(self):
        with self.assertRaisesRegex(StorageError, "lock is required"):
            self.store.publish_state(self.source(), "run1")


if __name__ == "__main__":
    unittest.main()
