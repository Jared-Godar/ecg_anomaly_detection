"""Repeatable, fail-safe retrieval of versioned public dataset files."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import string
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ecg_anomaly_detection.config import DatasetConfig, ExpectedSourceFile

# Centralize BUFFER_SIZE so every caller shares the same documented invariant.
BUFFER_SIZE = 1024 * 1024
# Centralize DEFAULT_TIMEOUT_SECONDS so every caller shares the same documented invariant.
DEFAULT_TIMEOUT_SECONDS = 60.0
# Centralize DEFAULT_MAX_FILE_SIZE_BYTES so every caller shares the same documented invariant.
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class AcquisitionError(ValueError):
    """Raised when acquisition cannot preserve its source and integrity contracts."""


@dataclass(frozen=True, slots=True)
class AcquiredFile:
    """Authoritative URL and local digest for one required source file."""

    path: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """Stable baseline written before downloaded files are committed locally."""

    schema_version: int
    dataset_slug: str
    dataset_version: str
    source_url: str
    download_url: str
    created_at_utc: str
    files: tuple[AcquiredFile, ...]

    def to_json(self) -> str:
        """Serialize with deterministic formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, content: str) -> AcquisitionManifest:
        """Parse and validate a serialized acquisition baseline."""
        # Attempt this boundary operation here so (KeyError, TypeError, json.JSONDecodeError) can be
        # translated or cleaned up under the repository contract.
        try:
            document = json.loads(content)
            files = tuple(AcquiredFile(**item) for item in document["files"])
            manifest = cls(
                schema_version=document["schema_version"],
                dataset_slug=document["dataset_slug"],
                dataset_version=document["dataset_version"],
                source_url=document["source_url"],
                download_url=document["download_url"],
                created_at_utc=document["created_at_utc"],
                files=files,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise AcquisitionError(f"invalid acquisition manifest: {error}") from error
        # Evaluate `manifest.schema_version != 1` explicitly so invalid or alternate states follow
        # the documented contract.
        if manifest.schema_version != 1:
            raise AcquisitionError("acquisition manifest must use schema_version 1")
        # Evaluate `not all((isinstance(value, str) and value for value in (manifest.dataset_slug,
        # manifest.dataset_version, manifest.sou...` explicitly so invalid or alternate states
        # follow the documented contract.
        if not all(
            isinstance(value, str) and value
            for value in (
                manifest.dataset_slug,
                manifest.dataset_version,
                manifest.source_url,
                manifest.download_url,
                manifest.created_at_utc,
            )
        ):
            raise AcquisitionError("acquisition manifest identity fields must be non-empty strings")
        # Evaluate `len({item.path for item in manifest.files}) != len(manifest.files)` explicitly
        # so invalid or alternate states follow the documented contract.
        if len({item.path for item in manifest.files}) != len(manifest.files):
            raise AcquisitionError("acquisition manifest contains duplicate file paths")
        # Evaluate `any((not _valid_acquired_file(item) for item in manifest.files))` explicitly so
        # invalid or alternate states follow the documented contract.
        if any(not _valid_acquired_file(item) for item in manifest.files):
            raise AcquisitionError("acquisition manifest contains invalid file evidence")
        return manifest


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Digest returned by a transport after writing a staged file."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Acquisition baseline plus idempotency counts for this invocation."""

    manifest: AcquisitionManifest
    downloaded_file_count: int
    reused_file_count: int


# Construct Fetcher once so the module exposes one stable shared definition.
Fetcher = Callable[[str, Path, float, int], TransferResult]


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Fail before following an HTTP redirect to preserve the configured source boundary."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        """Reject redirect request for the documented repository workflow.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            request: Validated request object crossing the external boundary.
            file_pointer: The file pointer value supplied by the caller or surrounding test fixture.
            code: The code value supplied by the caller or surrounding test fixture.
            message: The message value supplied by the caller or surrounding test fixture.
            headers: The headers value supplied by the caller or surrounding test fixture.
            new_url: The new url value supplied by the caller or surrounding test fixture.
        """

        del request, file_pointer, code, message, headers
        raise AcquisitionError(f"retrieval redirect rejected: {new_url}")


def acquire_dataset(
    config: DatasetConfig,
    repository_root: Path,
    data_dir: Path,
    manifest_path: Path,
    *,
    fetcher: Fetcher | None = None,
    clock: Callable[[], datetime] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> AcquisitionResult:
    """Retrieve required files or verify them against an existing acquisition baseline."""
    expectations = _expected_source_files(config)
    # Evaluate `timeout_seconds <= 0` explicitly so invalid or alternate states follow the
    # documented contract.
    if timeout_seconds <= 0:
        raise AcquisitionError("download timeout must be positive")
    # Evaluate `max_file_size_bytes <= 0` explicitly so invalid or alternate states follow the
    # documented contract.
    if max_file_size_bytes <= 0:
        raise AcquisitionError("maximum file size must be positive")
    root = repository_root.resolve()
    # Evaluate `not (root / 'pyproject.toml').is_file()` explicitly so invalid or alternate states
    # follow the documented contract.
    if not (root / "pyproject.toml").is_file():
        raise AcquisitionError(f"repository root does not contain pyproject.toml: {root}")
    destination = _canonical_data_directory(config, root, data_dir)
    output = _canonical_manifest_path(root, manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    _reject_unexpected_source_files(config, destination)
    transport = fetcher or _fetch_https_file

    # Evaluate `output.exists()` explicitly so invalid or alternate states follow the documented
    # contract.
    if output.exists():
        # Evaluate `output.is_symlink() or not output.is_file()` explicitly so invalid or alternate
        # states follow the documented contract.
        if output.is_symlink() or not output.is_file():
            raise AcquisitionError(f"acquisition manifest must be a regular file: {output}")
        manifest = _read_manifest(output)
        _validate_manifest_for_config(config, manifest)
        return _resume_acquisition(
            config,
            destination,
            manifest,
            transport,
            timeout_seconds,
            max_file_size_bytes,
        )

    existing = [name for name in config.expected_files if (destination / name).exists()]
    # Evaluate `existing` explicitly so invalid or alternate states follow the documented contract.
    if existing:
        raise AcquisitionError(
            "required files already exist without an acquisition manifest: " + ", ".join(existing)
        )
    now = (clock or (lambda: datetime.now(UTC)))()
    # Evaluate `now.tzinfo is None` explicitly so invalid or alternate states follow the documented
    # contract.
    if now.tzinfo is None:
        raise AcquisitionError("acquisition clock must return a timezone-aware datetime")

    # Scope `tempfile.TemporaryDirectory(prefix='.acquire-', dir=destination)` here so resource
    # cleanup occurs on both success and failure paths.
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
        staging = Path(staging_name)
        acquired: list[AcquiredFile] = []
        # Iterate over `config.expected_files` one item at a time so ordering, validation, and
        # failure attribution remain explicit.
        for relative_path in config.expected_files:
            url = _file_url(config, relative_path)
            staged_path = staging / relative_path
            transfer = transport(url, staged_path, timeout_seconds, max_file_size_bytes)
            _validate_transfer(transfer, relative_path)
            _validate_expected_transfer(transfer, expectations[relative_path])
            acquired.append(
                AcquiredFile(
                    path=relative_path,
                    url=url,
                    size_bytes=transfer.size_bytes,
                    sha256=transfer.sha256,
                )
            )
        manifest = AcquisitionManifest(
            schema_version=1,
            dataset_slug=config.slug,
            dataset_version=config.version,
            source_url=config.source_url,
            download_url=config.download_url,
            created_at_utc=now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            files=tuple(acquired),
        )
        _write_manifest_once(manifest, output)
        # Iterate over `manifest.files` one item at a time so ordering, validation, and failure
        # attribution remain explicit.
        for item in manifest.files:
            _install_without_overwrite(staging / item.path, destination / item.path)
    return AcquisitionResult(
        manifest=manifest,
        downloaded_file_count=len(manifest.files),
        reused_file_count=0,
    )


def _resume_acquisition(
    config: DatasetConfig,
    destination: Path,
    manifest: AcquisitionManifest,
    fetcher: Fetcher,
    timeout_seconds: float,
    max_file_size_bytes: int,
) -> AcquisitionResult:
    """Resume acquisition by verifying existing files and retrieving only missing files.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        destination: The destination value supplied by the caller or surrounding test fixture.
        manifest: The manifest value supplied by the caller or surrounding test fixture.
        fetcher: The fetcher value supplied by the caller or surrounding test fixture.
        timeout_seconds: The timeout seconds value supplied by the caller or surrounding test fixture.
        max_file_size_bytes: The max file size bytes value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    expectations = _expected_source_files(config)
    downloaded = 0
    reused = 0
    # Iterate over `manifest.files` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for item in manifest.files:
        destination_path = destination / item.path
        # Evaluate `destination_path.exists()` explicitly so invalid or alternate states follow the
        # documented contract.
        if destination_path.exists():
            # Evaluate `destination_path.is_symlink() or not destination_path.is_file()` explicitly
            # so invalid or alternate states follow the documented contract.
            if destination_path.is_symlink() or not destination_path.is_file():
                raise AcquisitionError(f"acquired path must be a regular file: {destination_path}")
            current = _hash_file(destination_path)
            _validate_expected_transfer(current, expectations[item.path], existing=True)
            # Evaluate `current != TransferResult(item.size_bytes, item.sha256)` explicitly so
            # invalid or alternate states follow the documented contract.
            if current != TransferResult(item.size_bytes, item.sha256):
                raise AcquisitionError(
                    f"existing file differs from acquisition manifest: {item.path}"
                )
            reused += 1
            continue
        # Scope `tempfile.TemporaryDirectory(prefix='.acquire-', dir=destination)` here so resource
        # cleanup occurs on both success and failure paths.
        with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
            staged_path = Path(staging_name) / item.path
            transfer = fetcher(item.url, staged_path, timeout_seconds, max_file_size_bytes)
            _validate_transfer(transfer, item.path)
            _validate_expected_transfer(transfer, expectations[item.path])
            # Evaluate `transfer != TransferResult(item.size_bytes, item.sha256)` explicitly so
            # invalid or alternate states follow the documented contract.
            if transfer != TransferResult(item.size_bytes, item.sha256):
                raise AcquisitionError(
                    f"retrieved file differs from acquisition manifest: {item.path}"
                )
            _install_without_overwrite(staged_path, destination_path)
        downloaded += 1
    return AcquisitionResult(
        manifest=manifest,
        downloaded_file_count=downloaded,
        reused_file_count=reused,
    )


def _fetch_https_file(
    url: str,
    output_path: Path,
    timeout_seconds: float,
    max_file_size_bytes: int,
) -> TransferResult:
    """Fetch one HTTPS resource into a staged file under strict size and digest controls.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        url: The url value supplied by the caller or surrounding test fixture.
        output_path: The output path value supplied by the caller or surrounding test fixture.
        timeout_seconds: The timeout seconds value supplied by the caller or surrounding test fixture.
        max_file_size_bytes: The max file size bytes value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    _validate_requested_url(url)
    # url's scheme is already validated by _validate_requested_url above.
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "ecg-anomaly-detection-portfolio/0.1",
        },
        method="GET",
    )
    digest = hashlib.sha256()
    size_bytes = 0
    # Attempt this boundary operation here so (OSError, TimeoutError, urllib.error.URLError),
    # AcquisitionError can be translated or cleaned up under the repository contract.
    try:
        # Scope `_open_https_request(request, timeout_seconds)` here so resource cleanup occurs on
        # both success and failure paths.
        with _open_https_request(request, timeout_seconds) as response:
            # Evaluate `response.getcode() != 200` explicitly so invalid or alternate states follow
            # the documented contract.
            if response.getcode() != 200:
                raise AcquisitionError(f"unexpected HTTP status for {url}: {response.getcode()}")
            _validate_final_url(url, response.geturl())
            content_length = _content_length(response.headers.get("Content-Length"), url)
            # Evaluate `content_length is not None and content_length > max_file_size_bytes`
            # explicitly so invalid or alternate states follow the documented contract.
            if content_length is not None and content_length > max_file_size_bytes:
                raise AcquisitionError(f"remote file exceeds size limit: {url}")
            # Scope `output_path.open('xb')` here so resource cleanup occurs on both success and
            # failure paths.
            with output_path.open("xb") as output:
                # Continue while `(chunk := response.read(BUFFER_SIZE))` so the loop's termination
                # rule remains visible to reviewers.
                while chunk := response.read(BUFFER_SIZE):
                    size_bytes += len(chunk)
                    # Evaluate `size_bytes > max_file_size_bytes` explicitly so invalid or alternate
                    # states follow the documented contract.
                    if size_bytes > max_file_size_bytes:
                        raise AcquisitionError(f"remote file exceeds size limit: {url}")
                    output.write(chunk)
                    digest.update(chunk)
            # Evaluate `content_length is not None and content_length != size_bytes` explicitly so
            # invalid or alternate states follow the documented contract.
            if content_length is not None and content_length != size_bytes:
                raise AcquisitionError(f"Content-Length mismatch for {url}")
    except AcquisitionError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise AcquisitionError(f"could not retrieve {url}: {error}") from error
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _open_https_request(request: urllib.request.Request, timeout_seconds: float) -> Any:
    """Open one HTTPS request with redirect rejection and a bounded timeout.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        request: Validated request object crossing the external boundary.
        timeout_seconds: The timeout seconds value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout_seconds)


def _canonical_data_directory(config: DatasetConfig, root: Path, data_dir: Path) -> Path:
    """Resolve and validate canonical data directory for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        root: Repository root used to enforce path and trust boundaries.
        data_dir: The data dir value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    raw_root_path = root / "data" / "raw"
    # Evaluate `raw_root_path.is_symlink() or not raw_root_path.is_dir()` explicitly so invalid or
    # alternate states follow the documented contract.
    if raw_root_path.is_symlink() or not raw_root_path.is_dir():
        raise AcquisitionError(
            f"raw data root must be an existing regular directory: {raw_root_path}"
        )
    raw_root = raw_root_path.resolve()
    expected = raw_root / config.slug / config.version
    candidate = data_dir if data_dir.is_absolute() else root / data_dir
    # Evaluate `candidate.is_symlink() or candidate.resolve() != expected` explicitly so invalid or
    # alternate states follow the documented contract.
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise AcquisitionError(
            f"data directory must be data/raw/{config.slug}/{config.version} within the repository"
        )
    return expected


def _canonical_manifest_path(root: Path, manifest_path: Path) -> Path:
    """Resolve and validate canonical manifest path for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        manifest_path: The manifest path value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    candidate = manifest_path if manifest_path.is_absolute() else root / manifest_path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise AcquisitionError("acquisition manifest must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise AcquisitionError("acquisition manifest must stay within repository root") from error
    # Evaluate `not relative.parts or relative.parts[0] != 'artifacts'` explicitly so invalid or
    # alternate states follow the documented contract.
    if not relative.parts or relative.parts[0] != "artifacts":
        raise AcquisitionError("acquisition manifest must be written under artifacts/")
    # Evaluate `resolved.suffix != '.json'` explicitly so invalid or alternate states follow the
    # documented contract.
    if resolved.suffix != ".json":
        raise AcquisitionError("acquisition manifest must use the .json extension")
    # Evaluate `not resolved.parent.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.parent.is_dir():
        raise AcquisitionError(f"acquisition manifest parent does not exist: {resolved.parent}")
    return resolved


def _reject_unexpected_source_files(config: DatasetConfig, destination: Path) -> None:
    """Reject source-directory files that are absent from the configured closed inventory.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        destination: The destination value supplied by the caller or surrounding test fixture.
    """

    expected = set(config.expected_files)
    unexpected = sorted(path.name for path in destination.iterdir() if path.name not in expected)
    # Evaluate `unexpected` explicitly so invalid or alternate states follow the documented
    # contract.
    if unexpected:
        raise AcquisitionError(
            "unexpected source file or directory in configured data directory: "
            + ", ".join(unexpected)
        )


def _validate_manifest_for_config(config: DatasetConfig, manifest: AcquisitionManifest) -> None:
    """Validate manifest for config according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
        manifest: The manifest value supplied by the caller or surrounding test fixture.
    """

    # Evaluate `(manifest.dataset_slug, manifest.dataset_version, manifest.source_url,
    # manifest.download_url) != (config.slug, config...` explicitly so invalid or alternate states
    # follow the documented contract.
    if (
        manifest.dataset_slug,
        manifest.dataset_version,
        manifest.source_url,
        manifest.download_url,
    ) != (config.slug, config.version, config.source_url, config.download_url):
        raise AcquisitionError("acquisition manifest identity does not match dataset configuration")
    expected = config.expected_files
    # Evaluate `tuple((item.path for item in manifest.files)) != expected` explicitly so invalid or
    # alternate states follow the documented contract.
    if tuple(item.path for item in manifest.files) != expected:
        raise AcquisitionError(
            "acquisition manifest file order does not match dataset configuration"
        )
    # Evaluate `any((item.url != _file_url(config, item.path) for item in manifest.files))`
    # explicitly so invalid or alternate states follow the documented contract.
    if any(item.url != _file_url(config, item.path) for item in manifest.files):
        raise AcquisitionError("acquisition manifest file URL does not match dataset configuration")
    expectations = _expected_source_files(config)
    # Iterate over `manifest.files` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for item in manifest.files:
        _validate_expected_transfer(
            TransferResult(item.size_bytes, item.sha256), expectations[item.path]
        )


def _expected_source_files(config: DatasetConfig) -> dict[str, ExpectedSourceFile]:
    """Build expected source files for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.

    Returns:
        The value produced by the documented operation.
    """

    expected_paths = set(config.expected_files)
    by_path = config.expected_source_files_by_path
    missing = sorted(expected_paths - set(by_path))
    unexpected = sorted(set(by_path) - expected_paths)
    # Evaluate `missing or unexpected` explicitly so invalid or alternate states follow the
    # documented contract.
    if missing or unexpected:
        raise AcquisitionError(
            "committed expected source metadata is incomplete; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return by_path


def _validate_expected_transfer(
    actual: TransferResult, expected: ExpectedSourceFile, *, existing: bool = False
) -> None:
    """Validate expected transfer according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        actual: The actual value supplied by the caller or surrounding test fixture.
        expected: The expected value supplied by the caller or surrounding test fixture.
        existing: The existing value supplied by the caller or surrounding test fixture.
    """

    prefix = "existing source file" if existing else "retrieved source file"
    # Evaluate `actual.size_bytes != expected.size_bytes` explicitly so invalid or alternate states
    # follow the documented contract.
    if actual.size_bytes != expected.size_bytes:
        raise AcquisitionError(
            f"{prefix} size mismatch for {expected.path}: "
            f"expected {expected.size_bytes} bytes, got {actual.size_bytes}"
        )
    # Evaluate `actual.sha256.lower() != expected.sha256` explicitly so invalid or alternate states
    # follow the documented contract.
    if actual.sha256.lower() != expected.sha256:
        raise AcquisitionError(
            f"{prefix} SHA-256 mismatch for {expected.path}: "
            f"expected {expected.sha256}, got {actual.sha256.lower()}"
        )


def _read_manifest(path: Path) -> AcquisitionManifest:
    """Read manifest according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    # Attempt this boundary operation here so OSError can be translated or cleaned up under the
    # repository contract.
    try:
        return AcquisitionManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AcquisitionError(f"could not read acquisition manifest {path}: {error}") from error


def _write_manifest_once(manifest: AcquisitionManifest, output_path: Path) -> None:
    """Write manifest once according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        manifest: The manifest value supplied by the caller or surrounding test fixture.
        output_path: The output path value supplied by the caller or surrounding test fixture.
    """

    temporary_path: Path | None = None
    # Attempt this boundary operation here so FileExistsError, OSError can be translated or cleaned
    # up under the repository contract.
    try:
        # Scope `tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
        # prefix=f'.{output_path.name}.', suffix='.tmp', dir=output_pat...` here so resource cleanup
        # occurs on both success and failure paths.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(manifest.to_json())
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise AcquisitionError(f"acquisition manifest already exists: {output_path}") from error
    except OSError as error:
        raise AcquisitionError(
            f"could not write acquisition manifest {output_path}: {error}"
        ) from error
    finally:
        # Evaluate `temporary_path is not None` explicitly so invalid or alternate states follow the
        # documented contract.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _install_without_overwrite(source: Path, destination: Path) -> None:
    """Install a staged file atomically without overwriting an existing destination.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        source: Source object or input text consumed by the operation.
        destination: The destination value supplied by the caller or surrounding test fixture.
    """

    # Attempt this boundary operation here so FileExistsError, OSError can be translated or cleaned
    # up under the repository contract.
    try:
        os.link(source, destination)
        return
    except FileExistsError as error:
        raise AcquisitionError(
            f"refusing to overwrite acquired file: {destination.name}"
        ) from error
    except OSError as error:
        # Evaluate `error.errno != errno.EXDEV` explicitly so invalid or alternate states follow the
        # documented contract.
        if error.errno != errno.EXDEV:
            raise AcquisitionError(
                f"could not install acquired file {destination.name}: {error}"
            ) from error

    # source and destination are on different filesystems, so a hard link is
    # not possible. Copy into a temporary file alongside destination (always
    # the same filesystem as destination) and hard-link from there, so the
    # atomic no-overwrite guarantee above still holds instead of being lost to
    # a plain copy-and-replace.
    temporary_path: Path | None = None
    # Attempt this boundary operation here so FileExistsError, OSError can be translated or cleaned
    # up under the repository contract.
    try:
        # Scope `tempfile.NamedTemporaryFile(prefix=f'.{destination.name}.', suffix='.tmp',
        # dir=destination.parent, delete=False)` here so resource cleanup occurs on both success and
        # failure paths.
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            # Scope `source.open('rb')` here so resource cleanup occurs on both success and failure
            # paths.
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary, BUFFER_SIZE)
            temporary.flush()
            os.fsync(temporary.fileno())
        shutil.copystat(source, temporary_path)
        os.link(temporary_path, destination)
    except FileExistsError as error:
        raise AcquisitionError(
            f"refusing to overwrite acquired file: {destination.name}"
        ) from error
    except OSError as error:
        raise AcquisitionError(
            f"could not install acquired file {destination.name}: {error}"
        ) from error
    finally:
        # Evaluate `temporary_path is not None` explicitly so invalid or alternate states follow the
        # documented contract.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _hash_file(path: Path) -> TransferResult:
    """Calculate stable size and SHA-256 evidence for one local file.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    digest = hashlib.sha256()
    size_bytes = 0
    # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
    with path.open("rb") as source:
        # Continue while `(chunk := source.read(BUFFER_SIZE))` so the loop's termination rule
        # remains visible to reviewers.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _file_url(config: DatasetConfig, relative_path: str) -> str:
    """Construct file url for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        relative_path: The relative path value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    return config.download_url + quote(relative_path, safe="")


def _validate_transfer(transfer: TransferResult, relative_path: str) -> None:
    """Validate transfer according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        transfer: The transfer value supplied by the caller or surrounding test fixture.
        relative_path: The relative path value supplied by the caller or surrounding test fixture.
    """

    # Evaluate `not isinstance(transfer.size_bytes, int) or isinstance(transfer.size_bytes, bool) or
    # transfer.size_bytes <= 0` explicitly so invalid or alternate states follow the documented
    # contract.
    if (
        not isinstance(transfer.size_bytes, int)
        or isinstance(transfer.size_bytes, bool)
        or transfer.size_bytes <= 0
    ):
        raise AcquisitionError(f"retrieved file is empty: {relative_path}")
    # Evaluate `not isinstance(transfer.sha256, str) or len(transfer.sha256) != 64 or any((character
    # not in string.hexdigits for char...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(transfer.sha256, str)
        or len(transfer.sha256) != 64
        or any(character not in string.hexdigits for character in transfer.sha256)
    ):
        raise AcquisitionError(f"retrieved file has invalid SHA-256 evidence: {relative_path}")


def _valid_acquired_file(item: AcquiredFile) -> bool:
    """Return whether valid acquired file under the documented validation contract.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        item: The item value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(item.path, str) or not item.path or '/' in item.path or ('\\' in
    # item.path) or (not isinstance(item.ur...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(item.path, str)
        or not item.path
        or "/" in item.path
        or "\\" in item.path
        or not isinstance(item.url, str)
        or not isinstance(item.size_bytes, int)
        or isinstance(item.size_bytes, bool)
        or item.size_bytes <= 0
        or not isinstance(item.sha256, str)
        or len(item.sha256) != 64
        or any(character not in string.hexdigits for character in item.sha256)
    ):
        return False
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        parsed = urlsplit(item.url)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _validate_requested_url(url: str) -> None:
    """Validate requested url according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        url: The url value supplied by the caller or surrounding test fixture.
    """

    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as error:
        raise AcquisitionError(f"retrieval URL must be an HTTPS file URL: {url}") from error
    # Evaluate `parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None or
    # (parsed.password is not None) or pa...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AcquisitionError(f"retrieval URL must be an HTTPS file URL: {url}")


def _validate_final_url(requested_url: str, final_url: str) -> None:
    """Validate final url according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        requested_url: The requested url value supplied by the caller or surrounding test fixture.
        final_url: The final url value supplied by the caller or surrounding test fixture.
    """

    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        requested = urlsplit(requested_url)
        final = urlsplit(final_url)
    except ValueError as error:
        raise AcquisitionError(f"retrieval returned an invalid final URL: {final_url}") from error
    # Evaluate `final.scheme != 'https' or final.netloc != requested.netloc or final.path !=
    # requested.path or (final.username is not...` explicitly so invalid or alternate states follow
    # the documented contract.
    if (
        final.scheme != "https"
        or final.netloc != requested.netloc
        or final.path != requested.path
        or final.username is not None
        or final.password is not None
        or final.query
        or final.fragment
    ):
        raise AcquisitionError(f"retrieval redirected outside the configured file URL: {final_url}")


def _content_length(value: str | None, url: str) -> int | None:
    """Read content length for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        url: The url value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `value is None` explicitly so invalid or alternate states follow the documented
    # contract.
    if value is None:
        return None
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        length = int(value)
    except ValueError as error:
        raise AcquisitionError(f"invalid Content-Length for {url}") from error
    # Evaluate `length < 0` explicitly so invalid or alternate states follow the documented
    # contract.
    if length < 0:
        raise AcquisitionError(f"invalid Content-Length for {url}")
    return length
