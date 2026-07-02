"""Local file inventory and integrity verification."""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from ecg_anomaly_detection.config import DatasetConfig

BUFFER_SIZE = 1024 * 1024


class InventoryError(ValueError):
    """Raised when a local dataset does not satisfy its recorded inventory."""


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Size and content digest for one required file."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class InventoryManifest:
    """Machine-readable snapshot of required local dataset files."""

    schema_version: int
    dataset_slug: str
    dataset_version: str
    created_at_utc: str
    files: tuple[FileDigest, ...]

    def to_json(self) -> str:
        """Serialize with stable formatting for review and hashing."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, content: str) -> InventoryManifest:
        """Parse and validate a serialized inventory manifest."""
        try:
            document = json.loads(content)
            files = tuple(FileDigest(**item) for item in document["files"])
            manifest = cls(
                schema_version=document["schema_version"],
                dataset_slug=document["dataset_slug"],
                dataset_version=document["dataset_version"],
                created_at_utc=document["created_at_utc"],
                files=files,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise InventoryError(f"invalid inventory manifest: {error}") from error

        if manifest.schema_version != 1:
            raise InventoryError("inventory manifest must use schema_version 1")
        if len({item.path for item in manifest.files}) != len(manifest.files):
            raise InventoryError("inventory manifest contains duplicate file paths")
        if any(
            item.size_bytes < 0
            or len(item.sha256) != 64
            or any(character not in string.hexdigits for character in item.sha256)
            for item in manifest.files
        ):
            raise InventoryError("inventory manifest contains an invalid size or SHA-256 digest")
        return manifest


def create_inventory(
    config: DatasetConfig,
    data_dir: Path,
    *,
    clock: Callable[[], datetime] | None = None,
) -> InventoryManifest:
    """Hash every required file and return a local integrity baseline."""
    missing: list[str] = []
    invalid: list[str] = []
    digests: list[FileDigest] = []

    for relative_path in config.expected_files:
        file_path = data_dir / relative_path
        if not file_path.exists():
            missing.append(relative_path)
            continue
        if file_path.is_symlink() or not file_path.is_file():
            invalid.append(relative_path)
            continue
        size_bytes, sha256 = _measure_and_hash(file_path)
        digests.append(FileDigest(path=relative_path, size_bytes=size_bytes, sha256=sha256))

    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if invalid:
            details.append(f"not regular files: {', '.join(invalid)}")
        raise InventoryError("; ".join(details))

    now = (clock or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None:
        raise InventoryError("inventory clock must return a timezone-aware datetime")
    return InventoryManifest(
        schema_version=1,
        dataset_slug=config.slug,
        dataset_version=config.version,
        created_at_utc=now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        files=tuple(digests),
    )


def verify_inventory(config: DatasetConfig, data_dir: Path, manifest: InventoryManifest) -> None:
    """Verify required file names, sizes, and SHA-256 digests against a baseline."""
    if (manifest.dataset_slug, manifest.dataset_version) != (config.slug, config.version):
        raise InventoryError("manifest dataset identity does not match the configured dataset")

    expected_paths = set(config.expected_files)
    manifest_paths = {item.path for item in manifest.files}
    if manifest_paths != expected_paths:
        missing = sorted(expected_paths - manifest_paths)
        extra = sorted(manifest_paths - expected_paths)
        raise InventoryError(f"manifest file set mismatch; missing={missing}, extra={extra}")

    current = create_inventory(config, data_dir)
    baseline_by_path = {item.path: item for item in manifest.files}
    changed = [item.path for item in current.files if item != baseline_by_path[item.path]]
    if changed:
        raise InventoryError(f"size or SHA-256 mismatch: {', '.join(changed)}")


def write_manifest(manifest: InventoryManifest, output_path: Path) -> None:
    """Write a manifest without creating missing parent directories implicitly."""
    if not output_path.parent.is_dir():
        raise InventoryError(f"manifest parent directory does not exist: {output_path.parent}")
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def read_manifest(path: Path) -> InventoryManifest:
    """Read an inventory manifest from disk."""
    try:
        return InventoryManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InventoryError(f"could not read inventory manifest {path}: {error}") from error


def _measure_and_hash(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as data_file:
        while chunk := data_file.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()
