"""Deterministic record-grouped dataset splitting and audit manifests."""

from __future__ import annotations

import json
import random
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.records import IntegerArray


class SplitError(ValueError):
    """Raised when split configuration or window metadata violates its contract."""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Versioned record-grouped split policy."""

    schema_version: int
    name: str
    version: str
    strategy: str
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float


@dataclass(frozen=True, slots=True)
class WindowMetadata:
    """Minimum window metadata required to assign complete records."""

    record_ids: tuple[str, ...]
    target_values: IntegerArray
    source_artifacts: tuple[str, ...]
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """Record membership and class counts for one partition."""

    record_ids: tuple[str, ...]
    record_count: int
    window_count: int
    target_value_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Machine-readable evidence for a deterministic grouped split."""

    schema_version: int
    split_name: str
    split_version: str
    strategy: str
    seed: int
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    source_artifacts: tuple[str, ...]
    total_record_count: int
    total_window_count: int
    partitions: dict[str, PartitionSummary]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_split_config(path: Path) -> SplitConfig:
    """Load and validate a versioned split configuration."""
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SplitError(f"could not load split config {path}: {error}") from error

    split = document.get("split")
    if document.get("schema_version") != 1 or not isinstance(split, dict):
        raise SplitError("split config must use schema_version = 1 and a [split] table")
    ratios = split.get("ratios")
    if not isinstance(ratios, dict):
        raise SplitError("split config must contain a [split.ratios] table")

    config = SplitConfig(
        schema_version=1,
        name=_required_string(split, "name"),
        version=_required_string(split, "version"),
        strategy=_required_string(split, "strategy"),
        seed=_required_nonnegative_int(split, "seed"),
        train_ratio=_required_ratio(ratios, "train"),
        validation_ratio=_required_ratio(ratios, "validation"),
        test_ratio=_required_ratio(ratios, "test"),
    )
    if config.strategy != "seeded-record-shuffle":
        raise SplitError("split.strategy must be 'seeded-record-shuffle'")
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise SplitError(f"split ratios must sum to 1.0; got {ratio_sum}")
    return config


def load_window_metadata(artifact_paths: Sequence[Path]) -> WindowMetadata:
    """Load lineage fields from non-pickle NPZ window artifacts."""
    if not artifact_paths:
        raise SplitError("at least one window artifact is required")

    record_ids: list[str] = []
    target_arrays: list[IntegerArray] = []
    seen_records: set[str] = set()
    identity: tuple[str, str, str, str] | None = None

    for path in artifact_paths:
        try:
            with np.load(path, allow_pickle=False) as artifact:
                required = {
                    "schema_version",
                    "record_ids",
                    "target_values",
                    "mapping_name",
                    "mapping_version",
                    "window_config_name",
                    "window_config_version",
                }
                missing = required - set(artifact.files)
                if missing:
                    raise SplitError(f"window artifact {path} is missing fields: {sorted(missing)}")
                if _integer_scalar(artifact["schema_version"], path, "schema_version") != 1:
                    raise SplitError(f"window artifact {path} must use schema_version 1")
                artifact_records = _string_vector(artifact["record_ids"], path, "record_ids")
                artifact_targets = _integer_vector(artifact["target_values"], path, "target_values")
                if len(artifact_records) != len(artifact_targets):
                    raise SplitError(f"window artifact {path} has unequal lineage row counts")
                artifact_identity = (
                    _string_scalar(artifact["mapping_name"], path, "mapping_name"),
                    _string_scalar(artifact["mapping_version"], path, "mapping_version"),
                    _string_scalar(artifact["window_config_name"], path, "window_config_name"),
                    _string_scalar(
                        artifact["window_config_version"], path, "window_config_version"
                    ),
                )
        except SplitError:
            raise
        except (BadZipFile, OSError, ValueError) as error:
            raise SplitError(f"could not load window artifact {path}: {error}") from error

        if not artifact_records:
            raise SplitError(f"window artifact {path} contains no windows")
        current_records = set(artifact_records)
        duplicated_records = current_records & seen_records
        if duplicated_records:
            raise SplitError(
                f"records occur in multiple window artifacts: {sorted(duplicated_records)}"
            )
        seen_records.update(current_records)
        if identity is None:
            identity = artifact_identity
        elif artifact_identity != identity:
            raise SplitError("window artifacts must use the same mapping and window configuration")
        record_ids.extend(artifact_records)
        target_arrays.append(artifact_targets)

    if identity is None:  # pragma: no cover - guarded by the non-empty input requirement
        raise SplitError("window artifact identity was not available")
    targets = np.concatenate(target_arrays).astype(np.int64, copy=False)
    targets.setflags(write=False)
    return WindowMetadata(
        record_ids=tuple(record_ids),
        target_values=targets,
        source_artifacts=tuple(str(path) for path in artifact_paths),
        mapping_name=identity[0],
        mapping_version=identity[1],
        window_config_name=identity[2],
        window_config_version=identity[3],
    )


def create_split_manifest(config: SplitConfig, metadata: WindowMetadata) -> SplitManifest:
    """Assign complete records and report record and target counts by partition."""
    if len(metadata.record_ids) != len(metadata.target_values):
        raise SplitError("window metadata record and target row counts must match")
    unique_records = sorted(set(metadata.record_ids))
    if len(unique_records) < 3:
        raise SplitError(
            "record-grouped train/validation/test splitting requires at least 3 records"
        )

    shuffled_records = unique_records.copy()
    random.Random(config.seed).shuffle(shuffled_records)
    sizes = _partition_sizes(
        len(shuffled_records),
        (config.train_ratio, config.validation_ratio, config.test_ratio),
    )
    boundaries = (sizes[0], sizes[0] + sizes[1])
    memberships = {
        "train": tuple(sorted(shuffled_records[: boundaries[0]])),
        "validation": tuple(sorted(shuffled_records[boundaries[0] : boundaries[1]])),
        "test": tuple(sorted(shuffled_records[boundaries[1] :])),
    }
    partitions = {
        name: _summarize_partition(records, metadata) for name, records in memberships.items()
    }
    _validate_partitions(partitions, set(unique_records))
    return SplitManifest(
        schema_version=1,
        split_name=config.name,
        split_version=config.version,
        strategy=config.strategy,
        seed=config.seed,
        mapping_name=metadata.mapping_name,
        mapping_version=metadata.mapping_version,
        window_config_name=metadata.window_config_name,
        window_config_version=metadata.window_config_version,
        source_artifacts=metadata.source_artifacts,
        total_record_count=len(unique_records),
        total_window_count=len(metadata.record_ids),
        partitions=partitions,
    )


def write_split_manifest(manifest: SplitManifest, output_path: Path) -> None:
    """Write a JSON split manifest to an existing directory."""
    if output_path.suffix != ".json":
        raise SplitError("split manifest must use the .json extension")
    if not output_path.parent.is_dir():
        raise SplitError(f"split manifest parent directory does not exist: {output_path.parent}")
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def _partition_sizes(record_count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [record_count * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    remainder = record_count - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(exact[index] - sizes[index]), index))
    for index in order[:remainder]:
        sizes[index] += 1
    for empty_index, size in enumerate(sizes):
        if size == 0:
            donor = max(range(3), key=lambda index: sizes[index])
            if sizes[donor] <= 1:
                raise SplitError("could not create three non-empty record partitions")
            sizes[donor] -= 1
            sizes[empty_index] += 1
    return sizes[0], sizes[1], sizes[2]


def _summarize_partition(records: tuple[str, ...], metadata: WindowMetadata) -> PartitionSummary:
    membership = set(records)
    targets = [
        int(target)
        for record_id, target in zip(metadata.record_ids, metadata.target_values, strict=True)
        if record_id in membership
    ]
    counts = Counter(targets)
    observed_values = sorted({int(value) for value in metadata.target_values})
    return PartitionSummary(
        record_ids=records,
        record_count=len(records),
        window_count=len(targets),
        target_value_counts={str(value): counts[value] for value in observed_values},
    )


def _validate_partitions(
    partitions: dict[str, PartitionSummary], expected_records: set[str]
) -> None:
    record_sets = [set(summary.record_ids) for summary in partitions.values()]
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise SplitError("record leakage detected across partitions")
    if set().union(*record_sets) != expected_records:
        raise SplitError("split partitions do not cover every input record")


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split.{key} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitError(f"split.{key} must be a nonnegative integer")
    return value


def _required_ratio(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
        raise SplitError(f"split.ratios.{key} must be between 0 and 1")
    return float(value)


def _string_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise SplitError(f"window artifact {path} field {field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> IntegerArray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer vector")
    return np.asarray(array, dtype=np.int64)


def _string_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string scalar")
    result = str(array.item())
    if not result:
        raise SplitError(f"window artifact {path} field {field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer scalar")
    return int(array.item())
