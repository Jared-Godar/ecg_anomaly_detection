"""Local file inventory and integrity verification."""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ecg_anomaly_detection.config import DatasetConfig

# Chunk size for streaming file reads during digest computation. 1 MiB balances syscall
# overhead against peak memory for source files that can be tens of megabytes.
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
        # Manifests are untrusted JSON read back from disk; collapse the ways parsing
        # or dict/attribute access can fail into one InventoryError so callers only
        # need to catch this module's own exception type.
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

        # This module has only ever written schema_version 1; any other value means
        # the manifest was produced by an incompatible tool or corrupted.
        if manifest.schema_version != 1:
            raise InventoryError("inventory manifest must use schema_version 1")
        # A duplicated path would mean two entries claim digest evidence for the same
        # file, making the path-keyed lookups in verify_inventory ambiguous.
        if len({item.path for item in manifest.files}) != len(manifest.files):
            raise InventoryError("inventory manifest contains duplicate file paths")
        # size_bytes must be nonnegative and sha256 must be exactly 64 hex characters
        # (the fixed length of a SHA-256 digest in hex); either violation means the
        # manifest was hand-edited or corrupted rather than machine-written by this module.
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
    expected_paths = set(config.expected_files)
    unexpected = sorted(path.name for path in data_dir.iterdir() if path.name not in expected_paths)
    # Any entry not in the config's declared set is unaccounted-for and untrusted.
    if unexpected:
        raise InventoryError(f"unexpected source files or directories: {', '.join(unexpected)}")
    expectations = config.expected_source_files_by_path

    # Check every expected file in the config's declared order, collecting every
    # missing/invalid entry together rather than failing on the first one, so a
    # single run reports the complete problem set instead of requiring repeated retries.
    for relative_path in config.expected_files:
        file_path = data_dir / relative_path
        # A missing file is recorded but doesn't stop the loop, so every missing file
        # is collected before raising below.
        if not file_path.exists():
            missing.append(relative_path)
            continue
        # Reject a symlink as well as a non-regular file: resolving through a link
        # could hash a file other than the one this inventory entry claims to describe.
        if file_path.is_symlink() or not file_path.is_file():
            invalid.append(relative_path)
            continue
        size_bytes, sha256 = _measure_and_hash(file_path)
        expected = expectations.get(relative_path)
        # expected_source_files metadata is optional per-file (see config.py); only
        # cross-check against it when this specific file has committed digest evidence.
        if expected is not None and size_bytes != expected.size_bytes:
            raise InventoryError(
                f"expected size mismatch for {relative_path}: "
                f"expected {expected.size_bytes} bytes, got {size_bytes}"
            )
        # Same optional cross-check as above, for the digest instead of the size.
        if expected is not None and sha256 != expected.sha256:
            raise InventoryError(
                f"expected SHA-256 mismatch for {relative_path}: "
                f"expected {expected.sha256}, got {sha256}"
            )
        digests.append(FileDigest(path=relative_path, size_bytes=size_bytes, sha256=sha256))

    # Report every missing and invalid file together in one error, rather than
    # raising on the first category found.
    if missing or invalid:
        details = []
        # Include the missing-files clause only if there were any.
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        # Include the invalid-files clause only if there were any.
        if invalid:
            details.append(f"not regular files: {', '.join(invalid)}")
        raise InventoryError("; ".join(details))

    now = (clock or (lambda: datetime.now(UTC)))()
    # created_at_utc is serialized with an explicit "Z" suffix below; a naive datetime
    # would make that suffix a lie about the actual timezone the timestamp represents.
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
    # A manifest recorded for a different dataset/version would otherwise be
    # silently accepted as a valid baseline for this config.
    if (manifest.dataset_slug, manifest.dataset_version) != (config.slug, config.version):
        raise InventoryError("manifest dataset identity does not match the configured dataset")

    expected_paths = set(config.expected_files)
    manifest_paths = {item.path for item in manifest.files}
    # The manifest's file set and the config's currently expected file set must match
    # exactly; a mismatch means the config changed since the manifest was created.
    if manifest_paths != expected_paths:
        missing = sorted(expected_paths - manifest_paths)
        extra = sorted(manifest_paths - expected_paths)
        raise InventoryError(f"manifest file set mismatch; missing={missing}, extra={extra}")

    current = create_inventory(config, data_dir)
    baseline_by_path = {item.path: item for item in manifest.files}
    changed = [item.path for item in current.files if item != baseline_by_path[item.path]]
    # A file whose freshly computed digest no longer matches the recorded baseline was
    # modified (or replaced) since the manifest was created.
    if changed:
        raise InventoryError(f"size or SHA-256 mismatch: {', '.join(changed)}")


def write_manifest(manifest: InventoryManifest, output_path: Path) -> None:
    """Write a manifest without creating missing parent directories implicitly."""
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from write_text.
    if not output_path.parent.is_dir():
        raise InventoryError(f"manifest parent directory does not exist: {output_path.parent}")
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def read_manifest(path: Path) -> InventoryManifest:
    """Read an inventory manifest from disk."""
    # Translate a missing or unreadable file into InventoryError so callers only need
    # to catch one exception type for every inventory-related failure.
    try:
        return InventoryManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InventoryError(f"could not read inventory manifest {path}: {error}") from error


def _measure_and_hash(path: Path) -> tuple[int, str]:
    """Measure one file and calculate its SHA-256 digest in a single streaming pass.

    Args:
        path: The local file to measure and hash.

    Returns:
        The file's size in bytes and its SHA-256 digest.
    """

    digest = hashlib.sha256()
    size_bytes = 0
    # Read in fixed-size chunks rather than the whole file at once, since source files
    # can be tens of megabytes.
    with path.open("rb") as data_file:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := data_file.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return size_bytes, digest.hexdigest()
