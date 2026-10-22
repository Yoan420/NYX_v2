"""S3 transport for immutable forecast runs and verified continuation state.

Credentials follow boto3's normal provider chain.  Merely importing this module
does not import boto3 or contact AWS.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit
from uuid import uuid4


class StorageError(RuntimeError):
    """A snapshot or publication cannot be safely completed."""


def _safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or value.startswith("/") or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Unsafe relative S3 path: {value!r}")
    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return bucket and slash-free prefix, rejecting ambiguous local paths."""
    parts = urlsplit(uri)
    if (parts.scheme != "s3" or not parts.netloc or parts.query or parts.fragment
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]{1,253}", parts.netloc)):
        raise ValueError("Expected s3://bucket/prefix without query or credentials")
    path = unquote(parts.path)
    prefix = path[1:] if path.startswith("/") else path
    prefix = prefix[:-1] if prefix.endswith("/") else prefix
    if prefix:
        _safe_relative(prefix)
    return parts.netloc, prefix


def _client(client=None):
    if client is None:
        import boto3
        return boto3.client("s3")
    return client


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _missing(error: Exception) -> bool:
    response = getattr(error, "response", {})
    return str(response.get("Error", {}).get("Code", "")) in {"NoSuchKey", "404", "NotFound"}


def _collision(error: Exception) -> bool:
    response = getattr(error, "response", {})
    return str(response.get("Error", {}).get("Code", "")) in {
        "PreconditionFailed", "ConditionalRequestConflict", "412", "409"}


def _join(prefix: str, suffix: str) -> str:
    return f"{prefix}/{suffix}" if prefix else suffix


def _safe_destination(root: Path, relative: str) -> Path:
    _safe_relative(relative)
    target = root.joinpath(*PurePosixPath(relative).parts)
    for candidate in (target, *target.parents):
        if candidate == root.parent:
            break
        if candidate.is_symlink():
            raise StorageError(f"Refusing a symlink in download destination: {candidate}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise StorageError("S3 destination escapes the requested directory")
    return target


def download_prefix(uri: str, destination: Path, client=None) -> list[str]:
    """Download a prefix atomically per file; never follow destination links."""
    bucket, prefix = parse_s3_uri(uri)
    client = _client(client)
    root = Path(destination).absolute()
    if root.is_symlink():
        raise StorageError("Download destination cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    match = prefix + "/" if prefix else ""
    downloaded = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=match):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not isinstance(key, str) or not key.startswith(match):
                raise StorageError("S3 returned an object outside the requested prefix")
            relative = key[len(match):]
            if not relative:  # The prefix's own directory marker.
                continue
            if relative.endswith("/"):
                _safe_relative(relative[:-1])
                continue
            target = _safe_destination(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not target.is_file():
                raise StorageError(f"Download target is not a regular file: {target}")
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.download")
            try:
                client.download_file(bucket, key, str(temporary))
                # Recheck after network I/O, before touching the destination.
                _safe_destination(root, relative)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            downloaded.append(relative)
    return downloaded


def _inventory(root: Path, exclude: Iterable[str] = ()) -> dict[str, str]:
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise StorageError("Upload source must be a regular directory")
    excluded = tuple(str(value).strip("/") for value in exclude)
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(relative == item or relative.startswith(item + "/")
               or ("/" not in item and path.name == item) for item in excluded):
            continue
        if path.is_symlink():
            raise StorageError(f"Upload source contains a symlink: {relative}")
        if path.is_file():
            _safe_relative(relative)
            inventory[relative] = _sha(path)
        elif not path.is_dir():
            raise StorageError(f"Upload source contains a special file: {relative}")
    return inventory


def upload_directory(path: Path, uri: str, client=None,
                     exclude: Iterable[str] = ()) -> dict[str, str]:
    """Upload regular files and return checksums; callers publish markers last."""
    bucket, prefix = parse_s3_uri(uri)
    client = _client(client)
    root = Path(path).absolute()
    inventory = _inventory(root, exclude)
    for relative, expected in inventory.items():
        source = _safe_destination(root, relative)
        if _sha(source) != expected:
            raise StorageError(f"Upload source changed before publication: {relative}")
        client.upload_file(str(source), bucket, _join(prefix, relative))
        if _sha(source) != expected:
            raise StorageError(f"Upload source changed during publication: {relative}")
    return inventory


class S3Store:
    """Serialize forecast publishers and promote complete state snapshots only."""

    def __init__(self, output_uri: str, client=None):
        self.bucket, self.prefix = parse_s3_uri(output_uri)
        if not self.prefix:
            raise ValueError("The output S3 URI must contain a dedicated prefix")
        self.client = _client(client)
        self._lock_token = None

    def _key(self, suffix: str) -> str:
        return _join(self.prefix, suffix)

    def _read(self, suffix: str):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(suffix))
        except Exception as error:
            if _missing(error):
                return None
            raise
        return json.loads(response["Body"].read())

    def _put(self, suffix: str, document: dict, *, exclusive=False):
        args = {"Bucket": self.bucket, "Key": self._key(suffix),
                "Body": _json_bytes(document), "ContentType": "application/json"}
        if exclusive:
            args["IfNoneMatch"] = "*"
        return self.client.put_object(**args)

    @contextmanager
    def lock(self):
        """Acquire a non-expiring S3 lock. Stale locks require operator review."""
        if self._lock_token is not None:
            raise StorageError("Forecast lock is already held by this process")
        token = uuid4().hex
        document = {"protocol": "nyx_forecast_lock_v1", "token": token,
                    "created_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            self._put("locks/forecast.json", document, exclusive=True)
        except Exception as error:
            if _collision(error):
                raise StorageError("Another forecast owns the S3 lock; check its job before removing a stale lock") from error
            raise
        self._lock_token = token
        try:
            yield self
        finally:
            try:
                current = self._read("locks/forecast.json")
                if current is None or current.get("token") != token:
                    raise StorageError("Forecast lock ownership changed; refusing to delete it")
                self.client.delete_object(Bucket=self.bucket, Key=self._key("locks/forecast.json"))
            finally:
                self._lock_token = None

    def _require_lock(self):
        if self._lock_token is None:
            raise StorageError("A forecast lock is required before publication")
        current = self._read("locks/forecast.json")
        if current is None or current.get("token") != self._lock_token:
            raise StorageError("Forecast lock is no longer owned by this process")

    @staticmethod
    def _run_id(run_id: str) -> str:
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,159}", run_id):
            raise ValueError("Unsafe run identifier")
        return run_id

    def _reserve(self, suffix: str):
        """Reserve a version once, including when a previous upload was partial."""
        try:
            self._put(suffix + "/_RESERVED.json", {"token": self._lock_token}, exclusive=True)
        except Exception as error:
            if _collision(error):
                raise StorageError("S3 run/version already exists; choose a new run identifier") from error
            raise

    def publish_run(self, source: Path, run_id: str, success: bool = True) -> dict:
        self._require_lock()
        run_id = self._run_id(run_id)
        suffix = f"runs/{run_id}"
        inventory = _inventory(Path(source))
        if any(name in inventory for name in ("_SUCCESS.json", "_FAILED.json", "_RESERVED.json")):
            raise StorageError("Local output contains a reserved S3 publication filename")
        self._reserve(suffix)
        files = upload_directory(source, f"s3://{self.bucket}/{self._key(suffix)}", self.client)
        document = {"protocol": "nyx_run_publication_v1", "run_id": run_id,
                    "success": bool(success), "files": files}
        self._require_lock()
        self._put(suffix + ("/_SUCCESS.json" if success else "/_FAILED.json"), document, exclusive=True)
        return document

    def publish_state(self, source: Path, run_id: str) -> dict:
        self._require_lock()
        run_id = self._run_id(run_id)
        suffix = f"state/versions/{run_id}"
        inventory = _inventory(Path(source), exclude=(".daily_state.lock",))
        if any(name in inventory for name in ("_MANIFEST.json", "_RESERVED.json")):
            raise StorageError("State source contains a reserved publication filename")
        self._reserve(suffix)
        files = upload_directory(source, f"s3://{self.bucket}/{self._key(suffix)}", self.client,
                                 exclude=(".daily_state.lock",))
        document = {"protocol": "nyx_state_snapshot_v1", "run_id": run_id,
                    "version_prefix": suffix, "files": files}
        self._require_lock()
        self._put(suffix + "/_MANIFEST.json", document, exclusive=True)
        # Readers see the previous complete version until this final write.
        self._put("state/latest.json", document)
        return document

    def restore_state(self, destination: Path) -> dict | None:
        document = self._read("state/latest.json")
        if document is None:
            return None
        if not isinstance(document, dict) or document.get("protocol") != "nyx_state_snapshot_v1":
            raise StorageError("Unknown state snapshot protocol")
        run_id = self._run_id(document.get("run_id"))
        suffix = f"state/versions/{run_id}"
        if document.get("version_prefix") != suffix:
            raise StorageError("State snapshot points outside its version prefix")
        manifest = self._read(suffix + "/_MANIFEST.json")
        if manifest != document:
            raise StorageError("State pointer does not match the sealed version manifest")
        files = document.get("files")
        if not isinstance(files, dict):
            raise StorageError("State snapshot has no checksum inventory")
        for name, digest in files.items():
            _safe_relative(name)
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise StorageError("State snapshot contains an invalid checksum")
        root = Path(destination).absolute()
        if root.is_symlink():
            raise StorageError("State destination cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        # Download each inventoried file, ignoring unsealed objects in the version.
        # Verify bytes before promotion so a corrupt remote file never overwrites
        # a locally valid state file.
        for name, digest in files.items():
            target = _safe_destination(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.state")
            try:
                self.client.download_file(self.bucket, self._key(suffix + "/" + name), str(temporary))
                if _sha(temporary) != digest:
                    raise StorageError(f"State snapshot checksum mismatch: {name}")
                _safe_destination(root, name)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return document
