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
        if candidate.is_file():
            candidate = candidate.parent

        for directory in (candidate, *candidate.parents):
            if (directory / "pyproject.toml").is_file():
                return cls(root=directory)
        message = f"could not find pyproject.toml from {candidate}"
        raise ConfigurationError(message)

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def raw_data(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def artifacts(self) -> Path:
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
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"could not load dataset config {path}: {error}") from error

    schema_version = document.get("schema_version")
    dataset = document.get("dataset")
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
    if any(
        "/" in value or "\\" in value for value in (*config.record_ids, *config.required_extensions)
    ):
        raise ConfigurationError("record IDs and extensions must be path segments, not paths")
    if config.annotation_extension not in config.required_extensions:
        raise ConfigurationError("dataset.annotation_extension must be a required extension")
    configured_paths = set(config.expected_files)
    metadata_paths = {item.path for item in config.expected_source_files}
    if metadata_paths and metadata_paths != configured_paths:
        missing = sorted(configured_paths - metadata_paths)
        unexpected = sorted(metadata_paths - configured_paths)
        raise ConfigurationError(
            "expected source metadata must exactly match required files; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return config


def _source_files(value: Any) -> tuple[ExpectedSourceFile, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ConfigurationError("expected_source_files must be a non-empty array of tables")
    files: list[ExpectedSourceFile] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ConfigurationError(f"expected_source_files[{index}] must be a table")
        path = item.get("path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not path or "/" in path or "\\" in path:
            raise ConfigurationError(
                f"expected_source_files[{index}].path must be a relative file name"
            )
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ConfigurationError(
                f"expected_source_files[{index}].size_bytes must be a positive integer"
            )
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
    if len(paths) != len(set(paths)):
        raise ConfigurationError("expected_source_files must contain unique paths")
    return tuple(files)


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"dataset.{key} must be a non-empty string")
    return value


def _required_positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"dataset.{key} must be a positive integer")
    return value


def _required_https_url(
    values: dict[str, Any], key: str, *, require_trailing_slash: bool = False
) -> str:
    value = _required_string(values, key).strip()
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"dataset.{key} must be a valid HTTPS URL") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError(f"dataset.{key} must be an HTTPS URL without credentials or query")
    if require_trailing_slash and not parsed.path.endswith("/"):
        raise ConfigurationError(f"dataset.{key} must end with a slash")
    return value


def _required_unique_strings(values: dict[str, Any], key: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"dataset.{key} must be a non-empty string array")
    normalized = tuple(item.strip().lstrip(".") for item in value)
    if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
        raise ConfigurationError(f"dataset.{key} must contain unique, non-empty values")
    return normalized
