"""Export only the immutable inputs and retained histories used by NYX.

No estimator or provider client is imported. Receipts are copied byte for byte;
export does not establish numerical or operating-system compatibility.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
from typing import Any


PROTOCOL = "nyx_aws_input_snapshot_v1"
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
MODEL_DIRECTORY = "models--amazon--chronos-2"
STATE_DIRECTORIES = (
    "runs/nyx_daily_cpu_states",
    "runs/nyx_daily_refit_states",
    "data/pit/nyx_annual_hydro_captures",
    "data/pit/nyx_annual_exchange_captures",
    "data/pit/nyx_annual_jao_initial_live",
)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _directory(value: Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"{label}: symbolic directory links are not supported: {path}")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{label}: directory required: {path}")
    return path


def _scan(root: Path, *, model_root: Path | None = None,
          reject_locks: bool = False) -> dict[str, tuple[Path, str]]:
    """Inventory regular files, allowing only bounded HF file links."""
    result: dict[str, tuple[Path, str]] = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                actual = path.resolve(strict=True)
                if (model_root is None or not actual.is_relative_to(model_root)
                        or not actual.is_file()):
                    raise ValueError(f"Unsupported or escaping symbolic link: {path}")
                if any(item.is_symlink() for item in (actual, *actual.parents)
                       if item.is_relative_to(model_root)):
                    raise ValueError(f"Redirected Hugging Face artifact: {path}")
            elif path.is_dir():
                visit(path)
                continue
            else:
                actual = path
            if not stat.S_ISREG(actual.stat().st_mode):
                raise ValueError(f"Regular file required: {path}")
            if reject_locks and path.name.endswith(".lock"):
                raise ValueError(f"State lock present; stop the NYX run before export: {path}")
            result[path.relative_to(root).as_posix()] = (actual, _sha(actual))

    visit(root)
    return result


def _identity(inventory: dict[str, tuple[Path, str]]) -> dict[str, tuple[str, str]]:
    return {name: (str(path), digest) for name, (path, digest) in inventory.items()}


def create_snapshot(*, seed_bundle: Path, seed_output: Path, runtime_root: Path,
                    hf_hub_cache: Path, destination: Path) -> dict[str, Any]:
    """Create a new, checked snapshot without altering any source receipt."""
    bundle = _directory(seed_bundle, "seed_bundle")
    output = _directory(seed_output, "seed_output")
    runtime = _directory(runtime_root, "runtime_root")
    hub = _directory(hf_hub_cache, "hf_hub_cache")
    model = _directory(hub / MODEL_DIRECTORY, "Chronos model cache")
    weights = _directory(model / "snapshots" / CHRONOS_REVISION, "Pinned Chronos snapshot")
    target = Path(destination).expanduser().absolute()
    if any(item.is_symlink() for item in (target, *target.parents)):
        raise ValueError("Snapshot destination cannot use symbolic directory links")
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"Snapshot destination already exists: {target}")
    for source in (bundle, output, runtime, hub):
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"Snapshot destination overlaps a source directory: {source}")

    sources: list[tuple[Path, str, Path | None, bool]] = [
        (bundle, "seed_bundle", None, False),
        (output, "seed_output", None, False),
        (weights, f"huggingface/hub/{MODEL_DIRECTORY}/snapshots/{CHRONOS_REVISION}", model, False),
    ]
    for relative in STATE_DIRECTORIES:
        source = runtime / relative
        if source.exists() or source.is_symlink():
            sources.append((_directory(source, relative), "state/" + relative, None, True))
    inventories = [_scan(source, model_root=allowed, reject_locks=locks)
                   for source, _, allowed, locks in sources]
    model_files = inventories[2]
    if "config.json" not in model_files or not any(
            name.endswith((".safetensors", ".bin")) for name in model_files):
        raise ValueError("Pinned Chronos snapshot must contain config.json and model weights")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    try:
        hashes: dict[str, str] = {}
        for (_, relative, _, _), inventory in zip(sources, inventories):
            (target / relative).mkdir(parents=True, exist_ok=True)
            for name, (source, digest) in inventory.items():
                portable = relative + "/" + name
                copied = target / portable
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, copied)
                if _sha(copied) != digest:
                    raise ValueError(f"Snapshot copy checksum differs: {portable}")
                hashes[portable] = digest
        for (source, _, allowed, locks), before in zip(sources, inventories):
            after = _scan(source, model_root=allowed, reject_locks=locks)
            if _identity(after) != _identity(before):
                raise ValueError(f"Source changed during snapshot export: {source}")
        document: dict[str, Any] = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "chronos_revision": CHRONOS_REVISION,
            "files": dict(sorted(hashes.items())),
            "state_directories": [relative.removeprefix("state/")
                                  for _, relative, _, _ in sources if relative.startswith("state/")],
            "receipts_rewritten": False,
            "notes": [
                "Receipts and retained state are copied unchanged; export does not establish runtime compatibility.",
                "Run the live preflight with the target platform's exact dependencies before forecasting.",
                "The snapshot inventory excludes snapshot.json itself.",
            ],
        }
        manifest = runtime.parent / "SOURCE_MANIFEST.json"
        if manifest.is_file() and not manifest.is_symlink():
            document["source_manifest_sha256"] = _sha(manifest)
        with (target / "snapshot.json").open("x", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        validate_snapshot(target)
        return document
    except BaseException:
        # target was created exclusively by this operation; sources are untouched.
        shutil.rmtree(target)
        raise


def validate_snapshot(root: Path) -> dict[str, Any]:
    """Verify the exact portable inventory of an immutable input snapshot."""
    directory = _directory(root, "snapshot")
    metadata = directory / "snapshot.json"
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("Snapshot metadata is missing or redirected")
    with metadata.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if (not isinstance(document, dict) or document.get("protocol") != PROTOCOL
            or document.get("chronos_revision") != CHRONOS_REVISION
            or document.get("receipts_rewritten") is not False):
        raise ValueError("Unsupported snapshot identity")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Snapshot file inventory is missing")
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name == "snapshot.json" or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Invalid portable snapshot artifact identity")
    inventory = _scan(directory)
    observed = {name: digest for name, (_, digest) in inventory.items() if name != "snapshot.json"}
    if observed.keys() != files.keys():
        raise ValueError("Snapshot file inventory differs; export to a new, empty S3 prefix")
    for name, expected in files.items():
        if observed[name] != expected:
            raise ValueError(f"Snapshot artifact checksum differs: {name}")
    required = ("seed_bundle/", "seed_output/",
                f"huggingface/hub/{MODEL_DIRECTORY}/snapshots/{CHRONOS_REVISION}/")
    if any(not any(name.startswith(prefix) for name in files) for prefix in required):
        raise ValueError("Snapshot lacks a required seed or pinned model directory")
    return document


__all__ = ["create_snapshot", "validate_snapshot"]
