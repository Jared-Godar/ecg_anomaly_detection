"""Model-ready partition index over immutable per-record window shards."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.splitting import SplitManifest, read_split_manifest

BUFFER_SIZE = 1024 * 1024


class DatasetIndexError(ValueError):
    """Raised when model-ready shard evidence violates its contract."""


@dataclass(frozen=True, slots=True)
class IndexedFile:
    """Repository-relative file identity and digest."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShardIndex:
    """One immutable record shard assigned to exactly one partition."""

    record_id: str
    window_count: int
    target_value_counts: dict[str, int]
    file: IndexedFile


@dataclass(frozen=True, slots=True)
class PartitionIndex:
    """Ordered shard membership and aggregate counts for one partition."""

    record_count: int
    window_count: int
    target_value_counts: dict[str, int]
    shards: tuple[ShardIndex, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """Machine-readable model input contract without duplicating window arrays."""

    schema_version: int
    split_name: str
    split_version: str
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_index: int
    channel_name: str
    window_samples: int
    total_record_count: int
    total_window_count: int
    split_manifest: IndexedFile
    partitions: dict[str, PartitionIndex]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _InspectedShard:
    index: ShardIndex
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_index: int
    channel_name: str
    window_samples: int


def create_dataset_index(
    repository_root: Path,
    split_manifest_path: Path,
    window_artifact_paths: Sequence[Path],
) -> DatasetIndex:
    """Validate record shards and index them according to grouped split membership."""
    root = repository_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise DatasetIndexError(f"repository root does not contain pyproject.toml: {root}")
    split_path = _resolve_file(root, split_manifest_path, ("artifacts",), "split manifest")
    try:
        split_manifest = read_split_manifest(split_path)
    except ValueError as error:
        raise DatasetIndexError(f"invalid split manifest: {error}") from error
    if not window_artifact_paths:
        raise DatasetIndexError("at least one window artifact is required")

    inspected = [_inspect_shard(root, path) for path in window_artifact_paths]
    by_record: dict[str, _InspectedShard] = {}
    for shard in inspected:
        if shard.index.record_id in by_record:
            raise DatasetIndexError(
                f"record occurs in multiple model-ready shards: {shard.index.record_id}"
            )
        by_record[shard.index.record_id] = shard

    expected_records = {
        record_id
        for partition in split_manifest.partitions.values()
        for record_id in partition.record_ids
    }
    if set(by_record) != expected_records:
        raise DatasetIndexError(
            "window shard records do not match split membership; "
            f"missing={sorted(expected_records - set(by_record))}, "
            f"extra={sorted(set(by_record) - expected_records)}"
        )
    identity = _validate_shard_identity(inspected, split_manifest)
    partitions = {
        name: _build_partition_index(summary.record_ids, summary.target_value_counts, by_record)
        for name, summary in split_manifest.partitions.items()
    }
    for name, partition in partitions.items():
        expected = split_manifest.partitions[name]
        if (
            partition.record_count != expected.record_count
            or partition.window_count != expected.window_count
            or partition.target_value_counts != expected.target_value_counts
        ):
            raise DatasetIndexError(f"indexed partition counts do not match split manifest: {name}")
    return DatasetIndex(
        schema_version=1,
        split_name=split_manifest.split_name,
        split_version=split_manifest.split_version,
        mapping_name=identity.mapping_name,
        mapping_version=identity.mapping_version,
        window_config_name=identity.window_config_name,
        window_config_version=identity.window_config_version,
        sample_rate_hz=identity.sample_rate_hz,
        channel_index=identity.channel_index,
        channel_name=identity.channel_name,
        window_samples=identity.window_samples,
        total_record_count=split_manifest.total_record_count,
        total_window_count=split_manifest.total_window_count,
        split_manifest=_indexed_file(root, split_path),
        partitions=partitions,
    )


def write_dataset_index(index: DatasetIndex, repository_root: Path, output_path: Path) -> None:
    """Write a model-ready dataset index under the ignored processed data zone."""
    root = repository_root.resolve()
    candidate = output_path if output_path.is_absolute() else root / output_path
    if candidate.is_symlink():
        raise DatasetIndexError("dataset index output must not be a symbolic link")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise DatasetIndexError("dataset index output must stay within repository root") from error
    if len(relative.parts) < 2 or relative.parts[:2] != ("data", "processed"):
        raise DatasetIndexError("dataset index output must be written under data/processed/")
    if resolved.suffix != ".json":
        raise DatasetIndexError("dataset index output must use the .json extension")
    if not resolved.parent.is_dir():
        raise DatasetIndexError(f"dataset index parent directory does not exist: {resolved.parent}")
    try:
        with resolved.open("x", encoding="utf-8") as output:
            output.write(index.to_json())
    except FileExistsError as error:
        raise DatasetIndexError(f"dataset index already exists: {resolved}") from error


def _inspect_shard(repository_root: Path, path: Path) -> _InspectedShard:
    resolved = _resolve_file(
        repository_root,
        path,
        ("data", "interim"),
        "window artifact",
    )
    try:
        with np.load(resolved, allow_pickle=False) as artifact:
            required = {
                "schema_version",
                "windows",
                "record_ids",
                "center_sample_indices",
                "source_symbols",
                "target_values",
                "sample_rate_hz",
                "channel_index",
                "channel_name",
                "mapping_name",
                "mapping_version",
                "window_config_name",
                "window_config_version",
            }
            missing = required - set(artifact.files)
            if missing:
                raise DatasetIndexError(f"window artifact {resolved} is missing: {sorted(missing)}")
            if _integer_scalar(artifact["schema_version"], "schema_version") != 1:
                raise DatasetIndexError("window artifact must use schema_version 1")
            windows = np.asarray(artifact["windows"])
            record_ids = _string_vector(artifact["record_ids"], "record_ids")
            center_indices = _integer_vector(
                artifact["center_sample_indices"], "center_sample_indices"
            )
            source_symbols = _string_vector(artifact["source_symbols"], "source_symbols")
            target_values = _integer_vector(artifact["target_values"], "target_values")
            sample_rate_hz = _positive_float_scalar(artifact["sample_rate_hz"], "sample_rate_hz")
            channel_index = _nonnegative_integer_scalar(artifact["channel_index"], "channel_index")
            channel_name = _string_scalar(artifact["channel_name"], "channel_name")
            mapping_name = _string_scalar(artifact["mapping_name"], "mapping_name")
            mapping_version = _string_scalar(artifact["mapping_version"], "mapping_version")
            window_config_name = _string_scalar(
                artifact["window_config_name"], "window_config_name"
            )
            window_config_version = _string_scalar(
                artifact["window_config_version"], "window_config_version"
            )
    except DatasetIndexError:
        raise
    except (BadZipFile, OSError, ValueError) as error:
        raise DatasetIndexError(f"could not inspect window artifact {resolved}: {error}") from error

    row_count = len(record_ids)
    if row_count == 0 or len(set(record_ids)) != 1:
        raise DatasetIndexError("each model-ready shard must contain one non-empty record")
    if (
        windows.ndim != 2
        or windows.shape[0] != row_count
        or windows.shape[1] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise DatasetIndexError("window matrix must be finite floating-point rows")
    if not (len(center_indices) == len(source_symbols) == len(target_values) == row_count):
        raise DatasetIndexError("window shard lineage arrays must have equal row counts")
    if np.any(center_indices < 0) or np.any(np.diff(center_indices) < 0):
        raise DatasetIndexError("window center indices must be nonnegative and ordered")
    counts = Counter(int(value) for value in target_values)
    return _InspectedShard(
        index=ShardIndex(
            record_id=record_ids[0],
            window_count=row_count,
            target_value_counts={str(value): counts[value] for value in sorted(counts)},
            file=_indexed_file(repository_root, resolved),
        ),
        mapping_name=mapping_name,
        mapping_version=mapping_version,
        window_config_name=window_config_name,
        window_config_version=window_config_version,
        sample_rate_hz=sample_rate_hz,
        channel_index=channel_index,
        channel_name=channel_name,
        window_samples=windows.shape[1],
    )


def _validate_shard_identity(
    shards: Sequence[_InspectedShard], split_manifest: SplitManifest
) -> _InspectedShard:
    first = shards[0]
    identity = (
        first.mapping_name,
        first.mapping_version,
        first.window_config_name,
        first.window_config_version,
        first.sample_rate_hz,
        first.channel_index,
        first.channel_name,
        first.window_samples,
    )
    if any(
        (
            shard.mapping_name,
            shard.mapping_version,
            shard.window_config_name,
            shard.window_config_version,
            shard.sample_rate_hz,
            shard.channel_index,
            shard.channel_name,
            shard.window_samples,
        )
        != identity
        for shard in shards[1:]
    ):
        raise DatasetIndexError(
            "window shards do not share one geometry and configuration identity"
        )
    if identity[:4] != (
        split_manifest.mapping_name,
        split_manifest.mapping_version,
        split_manifest.window_config_name,
        split_manifest.window_config_version,
    ):
        raise DatasetIndexError("window shard identity does not match split manifest")
    return first


def _build_partition_index(
    record_ids: tuple[str, ...],
    expected_target_counts: dict[str, int],
    by_record: dict[str, _InspectedShard],
) -> PartitionIndex:
    shards = tuple(by_record[record_id].index for record_id in record_ids)
    target_counts = {
        target: sum(shard.target_value_counts.get(target, 0) for shard in shards)
        for target in expected_target_counts
    }
    return PartitionIndex(
        record_count=len(shards),
        window_count=sum(shard.window_count for shard in shards),
        target_value_counts=target_counts,
        shards=shards,
    )


def _resolve_file(
    repository_root: Path,
    path: Path,
    required_prefix: tuple[str, ...],
    description: str,
) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    if candidate.is_symlink():
        raise DatasetIndexError(f"{description} must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise DatasetIndexError(f"{description} must stay within repository root") from error
    if relative.parts[: len(required_prefix)] != required_prefix:
        required = "/".join(required_prefix)
        raise DatasetIndexError(f"{description} must be stored under {required}/")
    if not resolved.is_file():
        raise DatasetIndexError(f"{description} must be a regular file: {resolved}")
    return resolved


def _indexed_file(repository_root: Path, path: Path) -> IndexedFile:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return IndexedFile(
        path=path.relative_to(repository_root).as_posix(),
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def _string_vector(value: np.ndarray[Any, Any], field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise DatasetIndexError(f"{field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], field: str) -> np.ndarray[Any, Any]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer vector")
    return array


def _string_scalar(value: np.ndarray[Any, Any], field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string scalar")
    result = str(array.item())
    if not result:
        raise DatasetIndexError(f"{field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer scalar")
    return int(array.item())


def _nonnegative_integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    result = _integer_scalar(value, field)
    if result < 0:
        raise DatasetIndexError(f"{field} must be nonnegative")
    return result


def _positive_float_scalar(value: np.ndarray[Any, Any], field: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"f", "i", "u"}:
        raise DatasetIndexError(f"{field} must be a numeric scalar")
    result = float(array.item())
    if not np.isfinite(result) or result <= 0:
        raise DatasetIndexError(f"{field} must be finite and positive")
    return result
