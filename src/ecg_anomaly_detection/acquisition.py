"""Repeatable, fail-safe retrieval of versioned public dataset files."""

from __future__ import annotations

import hashlib
import json
import os
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

BUFFER_SIZE = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0
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
        if manifest.schema_version != 1:
            raise AcquisitionError("acquisition manifest must use schema_version 1")
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
        if len({item.path for item in manifest.files}) != len(manifest.files):
            raise AcquisitionError("acquisition manifest contains duplicate file paths")
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
    if timeout_seconds <= 0:
        raise AcquisitionError("download timeout must be positive")
    if max_file_size_bytes <= 0:
        raise AcquisitionError("maximum file size must be positive")
    root = repository_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise AcquisitionError(f"repository root does not contain pyproject.toml: {root}")
    destination = _canonical_data_directory(config, root, data_dir)
    output = _canonical_manifest_path(root, manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    _reject_unexpected_source_files(config, destination)
    transport = fetcher or _fetch_https_file

    if output.exists():
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
    if existing:
        raise AcquisitionError(
            "required files already exist without an acquisition manifest: " + ", ".join(existing)
        )
    now = (clock or (lambda: datetime.now(UTC)))()
    if now.tzinfo is None:
        raise AcquisitionError("acquisition clock must return a timezone-aware datetime")

    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
        staging = Path(staging_name)
        acquired: list[AcquiredFile] = []
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
    expectations = _expected_source_files(config)
    downloaded = 0
    reused = 0
    for item in manifest.files:
        destination_path = destination / item.path
        if destination_path.exists():
            if destination_path.is_symlink() or not destination_path.is_file():
                raise AcquisitionError(f"acquired path must be a regular file: {destination_path}")
            current = _hash_file(destination_path)
            _validate_expected_transfer(current, expectations[item.path], existing=True)
            if current != TransferResult(item.size_bytes, item.sha256):
                raise AcquisitionError(
                    f"existing file differs from acquisition manifest: {item.path}"
                )
            reused += 1
            continue
        with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
            staged_path = Path(staging_name) / item.path
            transfer = fetcher(item.url, staged_path, timeout_seconds, max_file_size_bytes)
            _validate_transfer(transfer, item.path)
            _validate_expected_transfer(transfer, expectations[item.path])
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
    try:
        with _open_https_request(request, timeout_seconds) as response:
            if response.getcode() != 200:
                raise AcquisitionError(f"unexpected HTTP status for {url}: {response.getcode()}")
            _validate_final_url(url, response.geturl())
            content_length = _content_length(response.headers.get("Content-Length"), url)
            if content_length is not None and content_length > max_file_size_bytes:
                raise AcquisitionError(f"remote file exceeds size limit: {url}")
            with output_path.open("xb") as output:
                while chunk := response.read(BUFFER_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > max_file_size_bytes:
                        raise AcquisitionError(f"remote file exceeds size limit: {url}")
                    output.write(chunk)
                    digest.update(chunk)
            if content_length is not None and content_length != size_bytes:
                raise AcquisitionError(f"Content-Length mismatch for {url}")
    except AcquisitionError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        raise AcquisitionError(f"could not retrieve {url}: {error}") from error
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _open_https_request(request: urllib.request.Request, timeout_seconds: float) -> Any:
    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout_seconds)


def _canonical_data_directory(config: DatasetConfig, root: Path, data_dir: Path) -> Path:
    raw_root_path = root / "data" / "raw"
    if raw_root_path.is_symlink() or not raw_root_path.is_dir():
        raise AcquisitionError(
            f"raw data root must be an existing regular directory: {raw_root_path}"
        )
    raw_root = raw_root_path.resolve()
    expected = raw_root / config.slug / config.version
    candidate = data_dir if data_dir.is_absolute() else root / data_dir
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise AcquisitionError(
            f"data directory must be data/raw/{config.slug}/{config.version} within the repository"
        )
    return expected


def _canonical_manifest_path(root: Path, manifest_path: Path) -> Path:
    candidate = manifest_path if manifest_path.is_absolute() else root / manifest_path
    if candidate.is_symlink():
        raise AcquisitionError("acquisition manifest must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise AcquisitionError("acquisition manifest must stay within repository root") from error
    if not relative.parts or relative.parts[0] != "artifacts":
        raise AcquisitionError("acquisition manifest must be written under artifacts/")
    if resolved.suffix != ".json":
        raise AcquisitionError("acquisition manifest must use the .json extension")
    if not resolved.parent.is_dir():
        raise AcquisitionError(f"acquisition manifest parent does not exist: {resolved.parent}")
    return resolved


def _reject_unexpected_source_files(config: DatasetConfig, destination: Path) -> None:
    expected = set(config.expected_files)
    unexpected = sorted(path.name for path in destination.iterdir() if path.name not in expected)
    if unexpected:
        raise AcquisitionError(
            "unexpected source file or directory in configured data directory: "
            + ", ".join(unexpected)
        )


def _validate_manifest_for_config(config: DatasetConfig, manifest: AcquisitionManifest) -> None:
    if (
        manifest.dataset_slug,
        manifest.dataset_version,
        manifest.source_url,
        manifest.download_url,
    ) != (config.slug, config.version, config.source_url, config.download_url):
        raise AcquisitionError("acquisition manifest identity does not match dataset configuration")
    expected = config.expected_files
    if tuple(item.path for item in manifest.files) != expected:
        raise AcquisitionError(
            "acquisition manifest file order does not match dataset configuration"
        )
    if any(item.url != _file_url(config, item.path) for item in manifest.files):
        raise AcquisitionError("acquisition manifest file URL does not match dataset configuration")
    expectations = _expected_source_files(config)
    for item in manifest.files:
        _validate_expected_transfer(
            TransferResult(item.size_bytes, item.sha256), expectations[item.path]
        )


def _expected_source_files(config: DatasetConfig) -> dict[str, ExpectedSourceFile]:
    expected_paths = set(config.expected_files)
    by_path = config.expected_source_files_by_path
    missing = sorted(expected_paths - set(by_path))
    unexpected = sorted(set(by_path) - expected_paths)
    if missing or unexpected:
        raise AcquisitionError(
            "committed expected source metadata is incomplete; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return by_path


def _validate_expected_transfer(
    actual: TransferResult, expected: ExpectedSourceFile, *, existing: bool = False
) -> None:
    prefix = "existing source file" if existing else "retrieved source file"
    if actual.size_bytes != expected.size_bytes:
        raise AcquisitionError(
            f"{prefix} size mismatch for {expected.path}: "
            f"expected {expected.size_bytes} bytes, got {actual.size_bytes}"
        )
    if actual.sha256.lower() != expected.sha256:
        raise AcquisitionError(
            f"{prefix} SHA-256 mismatch for {expected.path}: "
            f"expected {expected.sha256}, got {actual.sha256.lower()}"
        )


def _read_manifest(path: Path) -> AcquisitionManifest:
    try:
        return AcquisitionManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AcquisitionError(f"could not read acquisition manifest {path}: {error}") from error


def _write_manifest_once(manifest: AcquisitionManifest, output_path: Path) -> None:
    temporary_path: Path | None = None
    try:
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
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _install_without_overwrite(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise AcquisitionError(
            f"refusing to overwrite acquired file: {destination.name}"
        ) from error
    except OSError as error:
        raise AcquisitionError(
            f"could not install acquired file {destination.name}: {error}"
        ) from error


def _hash_file(path: Path) -> TransferResult:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _file_url(config: DatasetConfig, relative_path: str) -> str:
    return config.download_url + quote(relative_path, safe="")


def _validate_transfer(transfer: TransferResult, relative_path: str) -> None:
    if (
        not isinstance(transfer.size_bytes, int)
        or isinstance(transfer.size_bytes, bool)
        or transfer.size_bytes <= 0
    ):
        raise AcquisitionError(f"retrieved file is empty: {relative_path}")
    if (
        not isinstance(transfer.sha256, str)
        or len(transfer.sha256) != 64
        or any(character not in string.hexdigits for character in transfer.sha256)
    ):
        raise AcquisitionError(f"retrieved file has invalid SHA-256 evidence: {relative_path}")


def _valid_acquired_file(item: AcquiredFile) -> bool:
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
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as error:
        raise AcquisitionError(f"retrieval URL must be an HTTPS file URL: {url}") from error
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
    try:
        requested = urlsplit(requested_url)
        final = urlsplit(final_url)
    except ValueError as error:
        raise AcquisitionError(f"retrieval returned an invalid final URL: {final_url}") from error
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
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as error:
        raise AcquisitionError(f"invalid Content-Length for {url}") from error
    if length < 0:
        raise AcquisitionError(f"invalid Content-Length for {url}")
    return length
