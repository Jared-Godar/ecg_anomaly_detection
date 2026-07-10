"""Versioned configuration and repository path contracts."""

from __future__ import annotations

import string
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """Raised when versioned configuration violates its contract."""


@dataclass(frozen=True, slots=True)
class RepositoryPaths:
    """Canonical paths rooted at a repository checkout."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> RepositoryPaths:
        """Find the nearest parent containing the project metadata file."""
        candidate = (start or Path.cwd()).resolve()
        # A file (rather than a directory) can't itself contain pyproject.toml as a
        # sibling of the search; start the upward walk from its parent directory.
        if candidate.is_file():
            candidate = candidate.parent

        # Walk upward from candidate through every parent directory, stopping at the
        # first one containing pyproject.toml -- this lets discovery work from any
        # subdirectory of the checkout, not just the repository root itself.
        for directory in (candidate, *candidate.parents):
            # Stop at the first match, closest to the starting point.
            if (directory / "pyproject.toml").is_file():
                return cls(root=directory)
        message = f"could not find pyproject.toml from {candidate}"
        raise ConfigurationError(message)

    @property
    def configs(self) -> Path:
        """Return the repository's versioned pipeline-config directory."""

        return self.root / "configs"

    @property
    def raw_data(self) -> Path:
        """Return the repository's gitignored raw-dataset root directory."""

        return self.root / "data" / "raw"

    @property
    def artifacts(self) -> Path:
        """Return the repository's gitignored pipeline-run artifacts directory."""

        return self.root / "artifacts"


@dataclass(frozen=True, slots=True)
class ExpectedSourceFile:
    """Repository-reviewed identity for one versioned source file."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Expected upstream dataset inventory."""

    schema_version: int
    name: str
    slug: str
    version: str
    source_url: str
    download_url: str
    sample_rate_hz: int
    annotation_extension: str
    record_ids: tuple[str, ...]
    required_extensions: tuple[str, ...]
    expected_source_files: tuple[ExpectedSourceFile, ...] = ()

    @property
    def expected_files(self) -> tuple[str, ...]:
        """Return the required record files in deterministic order."""
        return tuple(
            f"{record_id}.{extension}"
            for record_id in self.record_ids
            for extension in self.required_extensions
        )

    @property
    def expected_source_files_by_path(self) -> dict[str, ExpectedSourceFile]:
        """Return committed source expectations keyed by relative path."""
        return {item.path: item for item in self.expected_source_files}


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate a dataset inventory TOML file."""
    # Translate a missing, unreadable, or malformed-TOML file into ConfigurationError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"could not load dataset config {path}: {error}") from error

    schema_version = document.get("schema_version")
    dataset = document.get("dataset")
    # schema_version pins this loader's understanding of the [dataset] table's shape.
    if schema_version != 1 or not isinstance(dataset, dict):
        raise ConfigurationError("dataset config must use schema_version = 1 and a [dataset] table")

    config = DatasetConfig(
        schema_version=schema_version,
        name=_required_string(dataset, "name"),
        slug=_required_string(dataset, "slug"),
        version=_required_string(dataset, "version"),
        source_url=_required_https_url(dataset, "source_url"),
        download_url=_required_https_url(dataset, "download_url", require_trailing_slash=True),
        sample_rate_hz=_required_positive_int(dataset, "sample_rate_hz"),
        annotation_extension=_required_string(dataset, "annotation_extension").lstrip("."),
        record_ids=_required_unique_strings(dataset, "record_ids"),
        required_extensions=_required_unique_strings(dataset, "required_extensions"),
        expected_source_files=_source_files(dataset.get("expected_source_files")),
    )
    # record_ids and required_extensions are joined into filesystem paths elsewhere
    # (see expected_files); a value containing a path separator could otherwise let a
    # config entry construct a path escaping the intended flat directory.
    if any(
        "/" in value or "\\" in value for value in (*config.record_ids, *config.required_extensions)
    ):
        raise ConfigurationError("record IDs and extensions must be path segments, not paths")
    # The annotation file's extension must itself be one of the files acquisition
    # verifies and installs; otherwise records.py's WFDB loader would try to read an
    # annotation file this config never acquired or validated.
    if config.annotation_extension not in config.required_extensions:
        raise ConfigurationError("dataset.annotation_extension must be a required extension")
    configured_paths = set(config.expected_files)
    metadata_paths = {item.path for item in config.expected_source_files}
    # expected_source_files is optional (absent metadata_paths is falsy and skips this
    # check entirely), but when present it must exactly cover the same file set
    # record_ids/required_extensions imply -- a partial or mismatched digest inventory
    # would let acquisition.py silently skip verifying some required files.
    if metadata_paths and metadata_paths != configured_paths:
        missing = sorted(configured_paths - metadata_paths)
        unexpected = sorted(metadata_paths - configured_paths)
        raise ConfigurationError(
            "expected source metadata must exactly match required files; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return config


def _source_files(value: Any) -> tuple[ExpectedSourceFile, ...]:
    """Parse the optional `expected_source_files` array of per-file digest metadata.

    Args:
        value: The raw TOML value for dataset.expected_source_files, or None if the
            config doesn't declare per-file digest metadata.

    Returns:
        The validated entries, or an empty tuple if the field was absent.
    """

    # Absence is a valid, expected state: not every dataset config commits per-file
    # digest metadata (acquisition.py's _expected_source_files enforces that when this
    # IS present, it must exactly cover the required file set -- see load_dataset_config).
    if value is None:
        return ()
    # A present-but-empty or wrong-typed value is a config mistake, not a legitimate
    # "no files" declaration (use omission, i.e. None, for that).
    if not isinstance(value, list) or not value:
        raise ConfigurationError("expected_source_files must be a non-empty array of tables")
    files: list[ExpectedSourceFile] = []
    # Validate each entry independently, including its own index in any error, so a
    # malformed entry anywhere in a long list is easy to locate in the TOML source.
    for index, item in enumerate(value):
        # Every field access below assumes item is a dict.
        if not isinstance(item, dict):
            raise ConfigurationError(f"expected_source_files[{index}] must be a table")
        path = item.get("path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        # path must be a bare relative filename (no directory separators), matching
        # the flat data directory these files are installed into by acquisition.py.
        if not isinstance(path, str) or not path or "/" in path or "\\" in path:
            raise ConfigurationError(
                f"expected_source_files[{index}].path must be a relative file name"
            )
        # bool is an int subclass in Python, so it's excluded explicitly.
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ConfigurationError(
                f"expected_source_files[{index}].size_bytes must be a positive integer"
            )
        # sha256 must be exactly 64 hex characters (the fixed length of a SHA-256
        # digest in hex).
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in string.hexdigits for character in sha256)
        ):
            raise ConfigurationError(
                f"expected_source_files[{index}].sha256 must be a 64-character hex digest"
            )
        files.append(ExpectedSourceFile(path, size_bytes, sha256.lower()))
    paths = [item.path for item in files]
    # A duplicated path would mean two entries claim digest metadata for the same
    # file, making acquisition.py's path-keyed lookup silently pick just one of them.
    if len(paths) != len(set(paths)):
        raise ConfigurationError("expected_source_files must contain unique paths")
    return tuple(files)


def _required_string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested `[dataset]` field.

    Args:
        values: The parsed `[dataset]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value, unstripped (unlike other modules' equivalents, since
        annotation_extension needs to preserve a leading "." for lstrip(".") to strip
        deliberately, and other string fields here have no meaningful surrounding
        whitespace to trim in practice).
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"dataset.{key} must be a non-empty string")
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    """Require and return a strictly positive integer from the requested `[dataset]` field.

    Used for sample_rate_hz; zero or negative would produce a nonsensical dataset rate.

    Args:
        values: The parsed `[dataset]` table to read from.
        key: The field name to extract.

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"dataset.{key} must be a positive integer")
    return value


def _required_https_url(
    values: dict[str, Any], key: str, *, require_trailing_slash: bool = False
) -> str:
    """Require and return a plain HTTPS URL with no embedded credentials or query.

    Shared by source_url and download_url; download_url additionally requires a
    trailing slash since acquisition.py concatenates relative file paths directly onto
    it (see _file_url), and a missing slash would silently merge the last path segment
    with the file name.

    Args:
        values: The parsed `[dataset]` table to read from.
        key: The field name to extract.
        require_trailing_slash: Whether the URL's path must end with "/" (required for
            download_url, since it's used as a base for relative file URLs).

    Returns:
        The validated URL string.
    """

    value = _required_string(values, key).strip()
    # urlsplit can raise ValueError on a malformed URL (e.g. an invalid port).
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"dataset.{key} must be a valid HTTPS URL") from error
    # Reject anything other than a bare https://host/path URL: embedded credentials
    # could leak into logs, and a query string or fragment isn't meaningful for a
    # dataset source/download base URL under this pipeline's trust model.
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"dataset.{key} must be an HTTPS URL without credentials or query")
    # Only enforced for download_url, which acquisition.py concatenates file paths onto.
    if require_trailing_slash and not parsed.path.endswith("/"):
        raise ConfigurationError(f"dataset.{key} must end with a slash")
    return value


def _required_unique_strings(values: dict[str, Any], key: str) -> tuple[str, ...]:
    """Require and return a non-empty array of unique, non-empty, normalized strings.

    Used for record_ids and required_extensions; extensions are additionally
    normalized by stripping a leading "." so both "csv" and ".csv" in a config are
    accepted and treated identically.

    Args:
        values: The parsed `[dataset]` table to read from.
        key: The field name to extract.

    Returns:
        The validated, stripped, and (for extensions) dot-normalized values.
    """

    value = values.get(key)
    # Reject a missing/empty list or any non-string element before normalization below.
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"dataset.{key} must be a non-empty string array")
    normalized = tuple(item.strip().lstrip(".") for item in value)
    # Stripping whitespace/leading dots could reduce a previously-distinct entry to
    # empty or to a duplicate of another entry; re-check both after normalization.
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ConfigurationError(f"dataset.{key} must contain unique, non-empty values")
    return normalized
