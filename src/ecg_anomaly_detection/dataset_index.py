"""Model-ready partition index over immutable per-record window shards."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.splitting import PartitionSummary, SplitManifest, read_split_manifest

# Centralize BUFFER_SIZE so every caller shares the same documented invariant.
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
    subject_id: str
    window_count: int
    target_value_counts: dict[str, int]
    file: IndexedFile


@dataclass(frozen=True, slots=True)
class PartitionIndex:
    """Ordered shard membership and aggregate counts for one partition."""

    subject_ids: tuple[str, ...]
    subject_count: int
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
    total_subject_count: int
    total_record_count: int
    total_window_count: int
    split_manifest: IndexedFile
    partitions: dict[str, PartitionIndex]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class _InspectedShard:
    """Hold validated identity, counts, and digest evidence for one immutable shard.

    This internal record separates potentially unsafe file inspection from deterministic index
    assembly after every shard has passed its schema and lineage checks.
    """

    index: ShardIndex
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_index: int
    channel_name: str
    channel_selector: str | None
    configured_channel_index: int | None
    configured_channel_name: str | None
    window_samples: int


def create_dataset_index(
    repository_root: Path,
    split_manifest_path: Path,
    window_artifact_paths: Sequence[Path],
) -> DatasetIndex:
    """Validate record shards and index them according to grouped split membership."""
    root = repository_root.resolve()
    # Evaluate `not (root / 'pyproject.toml').is_file()` explicitly so invalid or alternate states
    # follow the documented contract.
    if not (root / "pyproject.toml").is_file():
        raise DatasetIndexError(f"repository root does not contain pyproject.toml: {root}")
    split_path = _resolve_file(root, split_manifest_path, ("artifacts",), "split manifest")
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        split_manifest = read_split_manifest(split_path)
    except ValueError as error:
        raise DatasetIndexError(f"invalid split manifest: {error}") from error
    # Evaluate `not window_artifact_paths` explicitly so invalid or alternate states follow the
    # documented contract.
    if not window_artifact_paths:
        raise DatasetIndexError("at least one window artifact is required")

    inspected = [_inspect_shard(root, path) for path in window_artifact_paths]
    by_record: dict[str, _InspectedShard] = {}
    # Iterate over `inspected` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for shard in inspected:
        # Evaluate `shard.index.record_id in by_record` explicitly so invalid or alternate states
        # follow the documented contract.
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
    # Evaluate `set(by_record) != expected_records` explicitly so invalid or alternate states follow
    # the documented contract.
    if set(by_record) != expected_records:
        raise DatasetIndexError(
            "window shard records do not match split membership; "
            f"missing={sorted(expected_records - set(by_record))}, "
            f"extra={sorted(set(by_record) - expected_records)}"
        )
    identity = _validate_shard_identity(inspected, split_manifest)
    partitions = {
        name: _build_partition_index(summary, by_record)
        for name, summary in split_manifest.partitions.items()
    }
    # Iterate over `partitions.items()` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for name, partition in partitions.items():
        expected = split_manifest.partitions[name]
        # Evaluate `partition.subject_count != expected.subject_count or partition.subject_ids !=
        # expected.subject_ids or partition.recor...` explicitly so invalid or alternate states
        # follow the documented contract.
        if (
            partition.subject_count != expected.subject_count
            or partition.subject_ids != expected.subject_ids
            or partition.record_count != expected.record_count
            or partition.window_count != expected.window_count
            or partition.target_value_counts != expected.target_value_counts
        ):
            raise DatasetIndexError(f"indexed partition counts do not match split manifest: {name}")
    return DatasetIndex(
        schema_version=2,
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
        total_subject_count=split_manifest.total_subject_count,
        total_record_count=split_manifest.total_record_count,
        total_window_count=split_manifest.total_window_count,
        split_manifest=_indexed_file(root, split_path),
        partitions=partitions,
    )


def write_dataset_index(index: DatasetIndex, repository_root: Path, output_path: Path) -> None:
    """Write a model-ready dataset index under the ignored processed data zone."""
    root = repository_root.resolve()
    candidate = output_path if output_path.is_absolute() else root / output_path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise DatasetIndexError("dataset index output must not be a symbolic link")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise DatasetIndexError("dataset index output must stay within repository root") from error
    # Evaluate `len(relative.parts) < 2 or relative.parts[:2] != ('data', 'processed')` explicitly
    # so invalid or alternate states follow the documented contract.
    if len(relative.parts) < 2 or relative.parts[:2] != ("data", "processed"):
        raise DatasetIndexError("dataset index output must be written under data/processed/")
    # Evaluate `resolved.suffix != '.json'` explicitly so invalid or alternate states follow the
    # documented contract.
    if resolved.suffix != ".json":
        raise DatasetIndexError("dataset index output must use the .json extension")
    # Evaluate `not resolved.parent.is_dir()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.parent.is_dir():
        raise DatasetIndexError(f"dataset index parent directory does not exist: {resolved.parent}")
    # Attempt this boundary operation here so FileExistsError can be translated or cleaned up under
    # the repository contract.
    try:
        # Scope `resolved.open('x', encoding='utf-8')` here so resource cleanup occurs on both
        # success and failure paths.
        with resolved.open("x", encoding="utf-8") as output:
            output.write(index.to_json())
    except FileExistsError as error:
        raise DatasetIndexError(f"dataset index already exists: {resolved}") from error


def _inspect_shard(repository_root: Path, path: Path) -> _InspectedShard:
    """Inspect one immutable shard and return its validated identity and counts.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.

    Returns:
        The value produced by the documented operation.
    """

    resolved = _resolve_file(
        repository_root,
        path,
        ("data", "interim"),
        "window artifact",
    )
    # Attempt this boundary operation here so (BadZipFile, OSError, ValueError), DatasetIndexError
    # can be translated or cleaned up under the repository contract.
    try:
        # Scope `np.load(resolved, allow_pickle=False)` here so resource cleanup occurs on both
        # success and failure paths.
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
            # Evaluate `missing` explicitly so invalid or alternate states follow the documented
            # contract.
            if missing:
                raise DatasetIndexError(f"window artifact {resolved} is missing: {sorted(missing)}")
            # Evaluate `_integer_scalar(artifact['schema_version'], 'schema_version') != 1`
            # explicitly so invalid or alternate states follow the documented contract.
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
            channel_selector = _optional_artifact_string_scalar(artifact, "channel_selector")
            # Evaluate `channel_selector is not None and channel_selector not in {'channel_index',
            # 'channel_name'}` explicitly so invalid or alternate states follow the documented
            # contract.
            if channel_selector is not None and channel_selector not in {
                "channel_index",
                "channel_name",
            }:
                raise DatasetIndexError("channel_selector must be channel_index or channel_name")
            configured_channel_index = _optional_artifact_integer_scalar(
                artifact, "configured_channel_index"
            )
            # Evaluate `configured_channel_index == -1` explicitly so invalid or alternate states
            # follow the documented contract.
            if configured_channel_index == -1:
                configured_channel_index = None
            # Evaluate `configured_channel_index is not None and configured_channel_index < 0`
            # explicitly so invalid or alternate states follow the documented contract.
            if configured_channel_index is not None and configured_channel_index < 0:
                raise DatasetIndexError(
                    "configured_channel_index must be nonnegative or -1 when absent"
                )
            configured_channel_name = _optional_artifact_string_scalar(
                artifact, "configured_channel_name", allow_empty=True
            )
            # Evaluate `configured_channel_name == ''` explicitly so invalid or alternate states
            # follow the documented contract.
            if configured_channel_name == "":
                configured_channel_name = None
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
    # Evaluate `row_count == 0 or len(set(record_ids)) != 1` explicitly so invalid or alternate
    # states follow the documented contract.
    if row_count == 0 or len(set(record_ids)) != 1:
        raise DatasetIndexError("each model-ready shard must contain one non-empty record")
    # Evaluate `windows.ndim != 2 or windows.shape[0] != row_count or windows.shape[1] == 0 or
    # (windows.dtype.kind != 'f') or (not np...` explicitly so invalid or alternate states follow
    # the documented contract.
    if (
        windows.ndim != 2
        or windows.shape[0] != row_count
        or windows.shape[1] == 0
        or windows.dtype.kind != "f"
        or not np.isfinite(windows).all()
    ):
        raise DatasetIndexError("window matrix must be finite floating-point rows")
    # Evaluate `not len(center_indices) == len(source_symbols) == len(target_values) == row_count`
    # explicitly so invalid or alternate states follow the documented contract.
    if not (len(center_indices) == len(source_symbols) == len(target_values) == row_count):
        raise DatasetIndexError("window shard lineage arrays must have equal row counts")
    # Evaluate `np.any(center_indices < 0) or np.any(np.diff(center_indices) < 0)` explicitly so
    # invalid or alternate states follow the documented contract.
    if np.any(center_indices < 0) or np.any(np.diff(center_indices) < 0):
        raise DatasetIndexError("window center indices must be nonnegative and ordered")
    counts = Counter(int(value) for value in target_values)
    return _InspectedShard(
        index=ShardIndex(
            record_id=record_ids[0],
            subject_id="",  # populated from the validated split manifest
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
        channel_selector=channel_selector,
        configured_channel_index=configured_channel_index,
        configured_channel_name=configured_channel_name,
        window_samples=windows.shape[1],
    )


def _optional_artifact_string_scalar(
    artifact: Any, name: str, *, allow_empty: bool = False
) -> str | None:
    """Read optional artifact string scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        artifact: The artifact value supplied by the caller or surrounding test fixture.
        name: The name value supplied by the caller or surrounding test fixture.
        allow_empty: The allow empty value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `name not in artifact.files` explicitly so invalid or alternate states follow the
    # documented contract.
    if name not in artifact.files:
        return None
    value = artifact[name]
    # Evaluate `np.asarray(value).shape != ()` explicitly so invalid or alternate states follow the
    # documented contract.
    if np.asarray(value).shape != ():
        raise DatasetIndexError(f"{name} must be a scalar string")
    scalar = value.item()
    # Evaluate `not isinstance(scalar, str)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(scalar, str):
        raise DatasetIndexError(f"{name} must be a scalar string")
    # Evaluate `not allow_empty and (not scalar)` explicitly so invalid or alternate states follow
    # the documented contract.
    if not allow_empty and not scalar:
        raise DatasetIndexError(f"{name} must be a non-empty scalar string")
    return scalar


def _optional_artifact_integer_scalar(artifact: Any, name: str) -> int | None:
    """Read optional artifact integer scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        artifact: The artifact value supplied by the caller or surrounding test fixture.
        name: The name value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `name not in artifact.files` explicitly so invalid or alternate states follow the
    # documented contract.
    if name not in artifact.files:
        return None
    return _integer_scalar(artifact[name], name)


def _validate_shard_identity(
    shards: Sequence[_InspectedShard], split_manifest: SplitManifest
) -> _InspectedShard:
    """Validate shard identity according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        shards: The shards value supplied by the caller or surrounding test fixture.
        split_manifest: The split manifest value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    first = shards[0]
    identity = (
        first.mapping_name,
        first.mapping_version,
        first.window_config_name,
        first.window_config_version,
        first.sample_rate_hz,
        first.channel_name,
        first.window_samples,
    )
    # Evaluate `any(((shard.mapping_name, shard.mapping_version, shard.window_config_name,
    # shard.window_config_version, shard.sample_...` explicitly so invalid or alternate states
    # follow the documented contract.
    if any(
        (
            shard.mapping_name,
            shard.mapping_version,
            shard.window_config_name,
            shard.window_config_version,
            shard.sample_rate_hz,
            shard.channel_name,
            shard.window_samples,
        )
        != identity
        for shard in shards[1:]
    ):
        raise DatasetIndexError(_format_shard_identity_mismatch(shards))
    # Evaluate `identity[:4] != (split_manifest.mapping_name, split_manifest.mapping_version,
    # split_manifest.window_config_name, spli...` explicitly so invalid or alternate states follow
    # the documented contract.
    if identity[:4] != (
        split_manifest.mapping_name,
        split_manifest.mapping_version,
        split_manifest.window_config_name,
        split_manifest.window_config_version,
    ):
        raise DatasetIndexError("window shard identity does not match split manifest")
    return first


def _format_shard_identity_mismatch(shards: Sequence[_InspectedShard]) -> str:
    """Format shard identity mismatch according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        shards: The shards value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    channel_counts = Counter(shard.channel_name for shard in shards)
    # Evaluate `len(channel_counts) <= 1` explicitly so invalid or alternate states follow the
    # documented contract.
    if len(channel_counts) <= 1:
        return "window shards do not share one geometry and configuration identity"

    lines = ["Window extraction produced inconsistent channel identities."]
    configured_selection = _format_configured_channel_selection(shards)
    # Evaluate `configured_selection is not None` explicitly so invalid or alternate states follow
    # the documented contract.
    if configured_selection is not None:
        lines.append(configured_selection)

    lines.append("Observed channel identities:")
    # Iterate over `sorted(channel_counts.items())` one item at a time so ordering, validation, and
    # failure attribution remain explicit.
    for channel_name, count in sorted(channel_counts.items()):
        record_word = "record" if count == 1 else "records"
        lines.append(f"  - {channel_name}: {count} {record_word}")

    first_channel = shards[0].channel_name
    affected = [
        shard
        for shard in sorted(shards, key=lambda item: item.index.record_id)
        if shard.channel_name != first_channel
    ]
    # Evaluate `affected` explicitly so invalid or alternate states follow the documented contract.
    if affected:
        lines.append("Affected records:")
        # Iterate over `affected` one item at a time so ordering, validation, and failure
        # attribution remain explicit.
        for shard in affected:
            lines.append(f"  - {shard.index.record_id}: {shard.channel_name}")

    lines.append(
        "This is not usually fixed by cleaning local artifacts. Configure extraction "
        "by channel name or explicitly exclude/fallback records that cannot satisfy "
        "the channel contract."
    )
    return "\n".join(lines)


def _format_configured_channel_selection(shards: Sequence[_InspectedShard]) -> str | None:
    """Format configured channel selection according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        shards: The shards value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    selections = {
        (
            shard.channel_selector,
            shard.configured_channel_index,
            shard.configured_channel_name,
        )
        for shard in shards
    }
    # Evaluate `len(selections) != 1` explicitly so invalid or alternate states follow the
    # documented contract.
    if len(selections) != 1:
        return None

    selector, configured_index, configured_name = next(iter(selections))
    # Evaluate `selector == 'channel_index' and configured_index is not None` explicitly so invalid
    # or alternate states follow the documented contract.
    if selector == "channel_index" and configured_index is not None:
        return f"Configured channel selection: channel_index = {configured_index}"
    # Evaluate `selector == 'channel_name' and configured_name is not None` explicitly so invalid or
    # alternate states follow the documented contract.
    if selector == "channel_name" and configured_name is not None:
        return f'Configured channel selection: channel_name = "{configured_name}"'
    return None


def _build_partition_index(
    summary: PartitionSummary,
    by_record: dict[str, _InspectedShard],
) -> PartitionIndex:
    """Build partition index according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        summary: The summary value supplied by the caller or surrounding test fixture.
        by_record: The by record value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    shards = tuple(
        ShardIndex(
            record_id=by_record[record_id].index.record_id,
            subject_id=summary.record_subjects[record_id],
            window_count=by_record[record_id].index.window_count,
            target_value_counts=by_record[record_id].index.target_value_counts,
            file=by_record[record_id].index.file,
        )
        for record_id in summary.record_ids
    )
    target_counts = {
        target: sum(shard.target_value_counts.get(target, 0) for shard in shards)
        for target in summary.target_value_counts
    }
    return PartitionIndex(
        subject_ids=summary.subject_ids,
        subject_count=summary.subject_count,
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
    """Resolve file according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
        path: Filesystem path identifying the input or output under review.
        required_prefix: The required prefix value supplied by the caller or surrounding test fixture.
        description: The description value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    candidate = path if path.is_absolute() else repository_root / path
    # Evaluate `candidate.is_symlink()` explicitly so invalid or alternate states follow the
    # documented contract.
    if candidate.is_symlink():
        raise DatasetIndexError(f"{description} must not be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    # Attempt this boundary operation here so ValueError can be translated or cleaned up under the
    # repository contract.
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError as error:
        raise DatasetIndexError(f"{description} must stay within repository root") from error
    # Evaluate `relative.parts[:len(required_prefix)] != required_prefix` explicitly so invalid or
    # alternate states follow the documented contract.
    if relative.parts[: len(required_prefix)] != required_prefix:
        required = "/".join(required_prefix)
        raise DatasetIndexError(f"{description} must be stored under {required}/")
    # Evaluate `not resolved.is_file()` explicitly so invalid or alternate states follow the
    # documented contract.
    if not resolved.is_file():
        raise DatasetIndexError(f"{description} must be a regular file: {resolved}")
    return resolved


def _indexed_file(repository_root: Path, path: Path) -> IndexedFile:
    """Compute and return indexed file for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        repository_root: Repository root used to enforce path and trust boundaries.
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
    return IndexedFile(
        path=path.relative_to(repository_root).as_posix(),
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )


def _string_vector(value: np.ndarray[Any, Any], field: str) -> tuple[str, ...]:
    """Compute and return string vector for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 1 or array.dtype.kind not in {'U', 'S'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    # Evaluate `any((not item for item in result))` explicitly so invalid or alternate states follow
    # the documented contract.
    if any(not item for item in result):
        raise DatasetIndexError(f"{field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], field: str) -> np.ndarray[Any, Any]:
    """Compute and return integer vector for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 1 or array.dtype.kind not in {'i', 'u'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer vector")
    return array


def _string_scalar(value: np.ndarray[Any, Any], field: str) -> str:
    """Compute and return string scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 0 or array.dtype.kind not in {'U', 'S'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise DatasetIndexError(f"{field} must be a string scalar")
    result = str(array.item())
    # Evaluate `not result` explicitly so invalid or alternate states follow the documented
    # contract.
    if not result:
        raise DatasetIndexError(f"{field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    """Compute and return integer scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 0 or array.dtype.kind not in {'i', 'u'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise DatasetIndexError(f"{field} must be an integer scalar")
    return int(array.item())


def _nonnegative_integer_scalar(value: np.ndarray[Any, Any], field: str) -> int:
    """Read and validate nonnegative integer scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    result = _integer_scalar(value, field)
    # Evaluate `result < 0` explicitly so invalid or alternate states follow the documented
    # contract.
    if result < 0:
        raise DatasetIndexError(f"{field} must be nonnegative")
    return result


def _positive_float_scalar(value: np.ndarray[Any, Any], field: str) -> float:
    """Read and validate positive float scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 0 or array.dtype.kind not in {'f', 'i', 'u'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 0 or array.dtype.kind not in {"f", "i", "u"}:
        raise DatasetIndexError(f"{field} must be a numeric scalar")
    result = float(array.item())
    # Evaluate `not np.isfinite(result) or result <= 0` explicitly so invalid or alternate states
    # follow the documented contract.
    if not np.isfinite(result) or result <= 0:
        raise DatasetIndexError(f"{field} must be finite and positive")
    return result
