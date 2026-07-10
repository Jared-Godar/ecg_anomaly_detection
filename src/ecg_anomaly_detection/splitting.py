"""Deterministic subject-grouped dataset splitting and audit manifests."""

from __future__ import annotations

import json
import random
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from ecg_anomaly_detection.records import IntegerArray


class SplitError(ValueError):
    """Raised when split configuration or window metadata violates its contract."""


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Versioned grouped split policy and record-to-subject metadata."""

    schema_version: int
    name: str
    version: str
    strategy: str
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    record_subjects: dict[str, str]
    quality: SplitQualityConfig = field(default_factory=lambda: SplitQualityConfig())


@dataclass(frozen=True, slots=True)
class SplitQualityConfig:
    """Acceptance thresholds applied after deterministic membership assignment."""

    min_subjects_per_partition: int = 1
    min_records_per_partition: int = 1
    min_windows_per_partition: int = 1
    min_positive_examples_per_partition: int = 0
    required_class_coverage: tuple[str, ...] = ()
    required_classes: tuple[int, ...] = ()
    max_partition_ratio_deviation: float = 1.0
    default_severity: str = "failure"
    warning_checks: tuple[str, ...] = ()


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
    record_shards: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    """Subject and record membership plus class counts for one partition."""

    subject_ids: tuple[str, ...]
    subject_count: int
    record_ids: tuple[str, ...]
    record_subjects: dict[str, str]
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
    total_subject_count: int
    total_record_count: int
    total_window_count: int
    partitions: dict[str, PartitionSummary]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, content: str) -> SplitManifest:
        """Parse and validate a serialized grouped split manifest."""
        # Attempt this boundary operation here so (KeyError, TypeError, json.JSONDecodeError) can be
        # translated or cleaned up under the repository contract.
        try:
            document = json.loads(content)
            schema_version = document["schema_version"]
            partitions_document = document["partitions"]
            # Evaluate `not isinstance(partitions_document, dict) or set(partitions_document) !=
            # {'train', 'validation', 'test'}` explicitly so invalid or alternate states follow the
            # documented contract.
            if not isinstance(partitions_document, dict) or set(partitions_document) != {
                "train",
                "validation",
                "test",
            }:
                raise SplitError("split manifest must define train, validation, and test")
            partitions = {
                name: _parse_partition_summary(name, partitions_document[name], schema_version)
                for name in ("train", "validation", "test")
            }
            source_artifacts_value = document["source_artifacts"]
            # Evaluate `not isinstance(source_artifacts_value, list) or not source_artifacts_value
            # or (not all((isinstance(item, str) and ite...` explicitly so invalid or alternate
            # states follow the documented contract.
            if (
                not isinstance(source_artifacts_value, list)
                or not source_artifacts_value
                or not all(isinstance(item, str) and item for item in source_artifacts_value)
                or len(source_artifacts_value) != len(set(source_artifacts_value))
            ):
                raise SplitError("split manifest source_artifacts must be unique non-empty paths")
            manifest = cls(
                schema_version=schema_version,
                split_name=_manifest_string(document, "split_name"),
                split_version=_manifest_string(document, "split_version"),
                strategy=_manifest_string(document, "strategy"),
                seed=_manifest_nonnegative_int(document, "seed"),
                mapping_name=_manifest_string(document, "mapping_name"),
                mapping_version=_manifest_string(document, "mapping_version"),
                window_config_name=_manifest_string(document, "window_config_name"),
                window_config_version=_manifest_string(document, "window_config_version"),
                source_artifacts=tuple(source_artifacts_value),
                total_subject_count=(
                    _manifest_nonnegative_int(document, "total_subject_count")
                    if schema_version == 2
                    else _manifest_nonnegative_int(document, "total_record_count")
                ),
                total_record_count=_manifest_nonnegative_int(document, "total_record_count"),
                total_window_count=_manifest_nonnegative_int(document, "total_window_count"),
                partitions=partitions,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise SplitError(f"invalid split manifest: {error}") from error
        _validate_serialized_manifest(manifest)
        return manifest


@dataclass(frozen=True, slots=True)
class QualityViolation:
    """One stable, machine-readable acceptance-check result."""

    check: str
    partition: str | None
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class SplitQualitySummary:
    """Deterministic diagnostics and acceptance result for one split."""

    schema_version: int
    split_name: str
    split_version: str
    status: str
    subject_disjoint: bool
    record_disjoint: bool
    configured_ratios: dict[str, float]
    acceptance_checks: dict[str, Any]
    partitions: dict[str, dict[str, Any]]
    violations: tuple[QualityViolation, ...]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_split_config(path: Path) -> SplitConfig:
    """Load and validate a versioned split configuration."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SplitError(f"could not load split config {path}: {error}") from error

    split = document.get("split")
    schema_version = document.get("schema_version")
    # Evaluate `schema_version not in {1, 2} or not isinstance(split, dict)` explicitly so invalid
    # or alternate states follow the documented contract.
    if schema_version not in {1, 2} or not isinstance(split, dict):
        raise SplitError("split config must use schema_version 1 or 2 and a [split] table")
    ratios = split.get("ratios")
    # Evaluate `not isinstance(ratios, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(ratios, dict):
        raise SplitError("split config must contain a [split.ratios] table")

    config = SplitConfig(
        schema_version=schema_version,
        name=_required_string(split, "name"),
        version=_required_string(split, "version"),
        strategy=_required_string(split, "strategy"),
        seed=_required_nonnegative_int(split, "seed"),
        train_ratio=_required_ratio(ratios, "train"),
        validation_ratio=_required_ratio(ratios, "validation"),
        test_ratio=_required_ratio(ratios, "test"),
        record_subjects=_record_subject_mapping(document, schema_version),
        quality=_quality_config(split),
    )
    expected_strategy = "seeded-subject-shuffle" if schema_version == 2 else "seeded-record-shuffle"
    # Evaluate `config.strategy != expected_strategy` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.strategy != expected_strategy:
        raise SplitError(f"split.strategy must be '{expected_strategy}'")
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    # Evaluate `not np.isclose(ratio_sum, 1.0)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not np.isclose(ratio_sum, 1.0):
        raise SplitError(f"split ratios must sum to 1.0; got {ratio_sum}")
    return config


def load_window_metadata(artifact_paths: Sequence[Path]) -> WindowMetadata:
    """Load lineage fields from non-pickle NPZ window artifacts."""
    # Evaluate `not artifact_paths` explicitly so invalid or alternate states follow the documented
    # contract.
    if not artifact_paths:
        raise SplitError("at least one window artifact is required")

    record_ids: list[str] = []
    target_arrays: list[IntegerArray] = []
    seen_records: set[str] = set()
    identity: tuple[str, str, str, str] | None = None
    record_shards: dict[str, str] = {}

    # Iterate over `artifact_paths` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for path in artifact_paths:
        # Attempt this boundary operation here so (BadZipFile, OSError, ValueError), SplitError can
        # be translated or cleaned up under the repository contract.
        try:
            # Scope `np.load(path, allow_pickle=False)` here so resource cleanup occurs on both
            # success and failure paths.
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
                # Evaluate `missing` explicitly so invalid or alternate states follow the documented
                # contract.
                if missing:
                    raise SplitError(f"window artifact {path} is missing fields: {sorted(missing)}")
                # Evaluate `_integer_scalar(artifact['schema_version'], path, 'schema_version') !=
                # 1` explicitly so invalid or alternate states follow the documented contract.
                if _integer_scalar(artifact["schema_version"], path, "schema_version") != 1:
                    raise SplitError(f"window artifact {path} must use schema_version 1")
                artifact_records = _string_vector(artifact["record_ids"], path, "record_ids")
                artifact_targets = _integer_vector(artifact["target_values"], path, "target_values")
                # Evaluate `len(artifact_records) != len(artifact_targets)` explicitly so invalid or
                # alternate states follow the documented contract.
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

        # Evaluate `not artifact_records` explicitly so invalid or alternate states follow the
        # documented contract.
        if not artifact_records:
            raise SplitError(f"window artifact {path} contains no windows")
        current_records = set(artifact_records)
        duplicated_records = current_records & seen_records
        # Evaluate `duplicated_records` explicitly so invalid or alternate states follow the
        # documented contract.
        if duplicated_records:
            raise SplitError(
                f"records occur in multiple window artifacts: {sorted(duplicated_records)}"
            )
        seen_records.update(current_records)
        record_shards.update({record_id: str(path) for record_id in current_records})
        # Evaluate `identity is None` explicitly so invalid or alternate states follow the
        # documented contract.
        if identity is None:
            identity = artifact_identity
        elif artifact_identity != identity:
            raise SplitError("window artifacts must use the same mapping and window configuration")
        record_ids.extend(artifact_records)
        target_arrays.append(artifact_targets)

    # Evaluate `identity is None` explicitly so invalid or alternate states follow the documented
    # contract.
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
        record_shards=dict(sorted(record_shards.items())),
    )


def create_split_quality_summary(
    config: SplitConfig, manifest: SplitManifest, metadata: WindowMetadata
) -> SplitQualitySummary:
    """Build deterministic diagnostics without reading or scoring held-out windows."""
    names = ("train", "validation", "test")
    configured = {
        "train": config.train_ratio,
        "validation": config.validation_ratio,
        "test": config.test_ratio,
    }
    subject_sets = [set(manifest.partitions[name].subject_ids) for name in names]
    record_sets = [set(manifest.partitions[name].record_ids) for name in names]
    subject_disjoint = _sets_are_disjoint(subject_sets)
    record_disjoint = _sets_are_disjoint(record_sets)
    observed_classes = tuple(sorted({int(value) for value in metadata.target_values}))
    binary = observed_classes == (0, 1)
    diagnostics: dict[str, dict[str, Any]] = {}
    violations: list[QualityViolation] = []
    total_unique_shards = len(set(metadata.record_shards.values()))
    # Iterate over `names` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for name in names:
        partition = manifest.partitions[name]
        shards = sorted(
            {
                metadata.record_shards[record_id]
                for record_id in partition.record_ids
                if record_id in metadata.record_shards
            }
        )
        class_counts = partition.target_value_counts
        prevalence = {
            key: (count / partition.window_count if partition.window_count else 0.0)
            for key, count in class_counts.items()
        }
        actual_ratios = {
            "subjects": partition.subject_count / manifest.total_subject_count,
            "records": partition.record_count / manifest.total_record_count,
            "shards": len(shards) / max(1, total_unique_shards),
            "windows": partition.window_count / manifest.total_window_count,
        }
        diagnostics[name] = {
            "subject_count": partition.subject_count,
            "record_count": partition.record_count,
            "shard_count": len(shards),
            "window_count": partition.window_count,
            "class_count": sum(count > 0 for count in class_counts.values()),
            "class_counts": class_counts,
            "class_prevalence": prevalence,
            "binary_counts": (
                {"negative": class_counts["0"], "positive": class_counts["1"]} if binary else None
            ),
            "configured_ratio": configured[name],
            "actual_ratios": actual_ratios,
            "subject_ratio_deviation": abs(actual_ratios["subjects"] - configured[name]),
        }
        checks = (
            (
                "minimum_subjects",
                partition.subject_count,
                config.quality.min_subjects_per_partition,
            ),
            ("minimum_records", partition.record_count, config.quality.min_records_per_partition),
            ("minimum_windows", partition.window_count, config.quality.min_windows_per_partition),
        )
        # Iterate over `checks` one item at a time so ordering, validation, and failure attribution
        # remain explicit.
        for check, actual, minimum in checks:
            # Evaluate `actual < minimum` explicitly so invalid or alternate states follow the
            # documented contract.
            if actual < minimum:
                violations.append(
                    _violation(
                        config, check, name, f"{actual} is below configured minimum {minimum}"
                    )
                )
        # Evaluate `binary and class_counts['1'] <
        # config.quality.min_positive_examples_per_partition` explicitly so invalid or alternate
        # states follow the documented contract.
        if binary and class_counts["1"] < config.quality.min_positive_examples_per_partition:
            violations.append(
                _violation(
                    config,
                    "minimum_positive_examples",
                    name,
                    f"{class_counts['1']} is below configured minimum {config.quality.min_positive_examples_per_partition}",
                )
            )
        # Evaluate `name in config.quality.required_class_coverage` explicitly so invalid or
        # alternate states follow the documented contract.
        if name in config.quality.required_class_coverage:
            missing = [
                value
                for value in config.quality.required_classes
                if class_counts.get(str(value), 0) == 0
            ]
            # Evaluate `missing` explicitly so invalid or alternate states follow the documented
            # contract.
            if missing:
                violations.append(
                    _violation(
                        config,
                        "required_class_coverage",
                        name,
                        f"missing required classes {missing}",
                    )
                )
        # Evaluate `diagnostics[name]['subject_ratio_deviation'] >
        # config.quality.max_partition_ratio_deviation` explicitly so invalid or alternate states
        # follow the documented contract.
        if (
            diagnostics[name]["subject_ratio_deviation"]
            > config.quality.max_partition_ratio_deviation
        ):
            violations.append(
                _violation(
                    config,
                    "partition_ratio_deviation",
                    name,
                    f"subject ratio deviation {diagnostics[name]['subject_ratio_deviation']:.6f} exceeds {config.quality.max_partition_ratio_deviation:.6f}",
                )
            )
    # Evaluate `not subject_disjoint` explicitly so invalid or alternate states follow the
    # documented contract.
    if not subject_disjoint:
        violations.append(
            _violation(
                config, "subject_disjointness", None, "subjects occur in multiple partitions"
            )
        )
    # Evaluate `not record_disjoint` explicitly so invalid or alternate states follow the documented
    # contract.
    if not record_disjoint:
        violations.append(
            _violation(config, "record_disjointness", None, "records occur in multiple partitions")
        )
    violations.sort(key=lambda item: (item.check, item.partition or "", item.message))
    status = (
        "failed"
        if any(item.severity == "failure" for item in violations)
        else ("warning" if violations else "passed")
    )
    return SplitQualitySummary(
        1,
        manifest.split_name,
        manifest.split_version,
        status,
        subject_disjoint,
        record_disjoint,
        configured,
        asdict(config.quality),
        diagnostics,
        tuple(violations),
    )


def write_split_quality_summary(summary: SplitQualitySummary, output_path: Path) -> None:
    """Write split diagnostics to an existing directory."""
    # Evaluate `output_path.suffix != '.json' or not output_path.parent.is_dir()` explicitly so
    # invalid or alternate states follow the documented contract.
    if output_path.suffix != ".json" or not output_path.parent.is_dir():
        raise SplitError("split quality summary must be a JSON file in an existing directory")
    output_path.write_text(summary.to_json(), encoding="utf-8")


def enforce_split_quality(summary: SplitQualitySummary) -> None:
    """Fail closed after the diagnostic artifact has been made available."""
    failures = [item for item in summary.violations if item.severity == "failure"]
    # Evaluate `failures` explicitly so invalid or alternate states follow the documented contract.
    if failures:
        raise SplitError(f"split quality checks failed: {len(failures)} failure(s)")


def create_split_manifest(config: SplitConfig, metadata: WindowMetadata) -> SplitManifest:
    """Assign complete subjects and report subject, record, and target counts."""
    # Evaluate `len(metadata.record_ids) != len(metadata.target_values)` explicitly so invalid or
    # alternate states follow the documented contract.
    if len(metadata.record_ids) != len(metadata.target_values):
        raise SplitError("window metadata record and target row counts must match")
    unique_records = sorted(set(metadata.record_ids))
    record_subjects = (
        {record_id: record_id for record_id in unique_records}
        if config.schema_version == 1
        else config.record_subjects
    )
    # Evaluate `set(record_subjects) != set(unique_records)` explicitly so invalid or alternate
    # states follow the documented contract.
    if set(record_subjects) != set(unique_records):
        raise SplitError(
            "record-to-subject metadata must exactly cover window records; "
            f"missing={sorted(set(unique_records) - set(record_subjects))}, "
            f"extra={sorted(set(record_subjects) - set(unique_records))}"
        )
    unique_subjects = sorted(set(record_subjects.values()))
    # Evaluate `len(unique_subjects) < 3` explicitly so invalid or alternate states follow the
    # documented contract.
    if len(unique_subjects) < 3:
        raise SplitError("subject-grouped splitting requires at least 3 subjects")

    shuffled_subjects = unique_subjects.copy()
    # deterministic seeded shuffle for reproducible train/val/test partitioning,
    # not a cryptographic use.
    random.Random(config.seed).shuffle(shuffled_subjects)  # noqa: S311
    sizes = _partition_sizes(
        len(shuffled_subjects),
        (config.train_ratio, config.validation_ratio, config.test_ratio),
    )
    boundaries = (sizes[0], sizes[0] + sizes[1])
    subject_memberships = {
        "train": tuple(sorted(shuffled_subjects[: boundaries[0]])),
        "validation": tuple(sorted(shuffled_subjects[boundaries[0] : boundaries[1]])),
        "test": tuple(sorted(shuffled_subjects[boundaries[1] :])),
    }
    partitions = {
        name: _summarize_partition(subjects, record_subjects, metadata)
        for name, subjects in subject_memberships.items()
    }
    _validate_partitions(partitions, set(unique_subjects), set(unique_records))
    return SplitManifest(
        schema_version=config.schema_version,
        split_name=config.name,
        split_version=config.version,
        strategy=config.strategy,
        seed=config.seed,
        mapping_name=metadata.mapping_name,
        mapping_version=metadata.mapping_version,
        window_config_name=metadata.window_config_name,
        window_config_version=metadata.window_config_version,
        source_artifacts=metadata.source_artifacts,
        total_subject_count=len(unique_subjects),
        total_record_count=len(unique_records),
        total_window_count=len(metadata.record_ids),
        partitions=partitions,
    )


def write_split_manifest(manifest: SplitManifest, output_path: Path) -> None:
    """Write a JSON split manifest to an existing directory."""
    # Evaluate `output_path.suffix != '.json'` explicitly so invalid or alternate states follow the
    # documented contract.
    if output_path.suffix != ".json":
        raise SplitError("split manifest must use the .json extension")
    # Evaluate `not output_path.parent.is_dir()` explicitly so invalid or alternate states follow
    # the documented contract.
    if not output_path.parent.is_dir():
        raise SplitError(f"split manifest parent directory does not exist: {output_path.parent}")
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def read_split_manifest(path: Path) -> SplitManifest:
    """Read and validate a grouped split manifest from disk."""
    # Attempt this boundary operation here so OSError can be translated or cleaned up under the
    # repository contract.
    try:
        return SplitManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SplitError(f"could not read split manifest {path}: {error}") from error


def _partition_sizes(
    subject_count: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Calculate partition sizes for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        subject_count: The subject count value supplied by the caller or surrounding test fixture.
        ratios: The ratios value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    exact = [subject_count * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    remainder = subject_count - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(exact[index] - sizes[index]), index))
    # Iterate over `order[:remainder]` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for index in order[:remainder]:
        sizes[index] += 1
    # Iterate over `enumerate(sizes)` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for empty_index, size in enumerate(sizes):
        # Evaluate `size == 0` explicitly so invalid or alternate states follow the documented
        # contract.
        if size == 0:
            donor = max(range(3), key=lambda index: sizes[index])
            # Evaluate `sizes[donor] <= 1` explicitly so invalid or alternate states follow the
            # documented contract.
            if sizes[donor] <= 1:
                raise SplitError("could not create three non-empty subject partitions")
            sizes[donor] -= 1
            sizes[empty_index] += 1
    return sizes[0], sizes[1], sizes[2]


def _summarize_partition(
    subjects: tuple[str, ...], record_subjects: dict[str, str], metadata: WindowMetadata
) -> PartitionSummary:
    """Summarize one partition's records, subjects, shards, labels, and ratio diagnostics.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        subjects: The subjects value supplied by the caller or surrounding test fixture.
        record_subjects: The record subjects value supplied by the caller or surrounding test fixture.
        metadata: The metadata value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    subject_membership = set(subjects)
    records = tuple(
        sorted(
            record_id
            for record_id, subject_id in record_subjects.items()
            if subject_id in subject_membership
        )
    )
    membership = set(records)
    targets = [
        int(target)
        for record_id, target in zip(metadata.record_ids, metadata.target_values, strict=True)
        if record_id in membership
    ]
    counts = Counter(targets)
    observed_values = sorted({int(value) for value in metadata.target_values})
    return PartitionSummary(
        subject_ids=subjects,
        subject_count=len(subjects),
        record_ids=records,
        record_subjects={record_id: record_subjects[record_id] for record_id in records},
        record_count=len(records),
        window_count=len(targets),
        target_value_counts={str(value): counts[value] for value in observed_values},
    )


def _validate_partitions(
    partitions: dict[str, PartitionSummary], expected_subjects: set[str], expected_records: set[str]
) -> None:
    """Validate partitions according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        partitions: The partitions value supplied by the caller or surrounding test fixture.
        expected_subjects: The expected subjects value supplied by the caller or surrounding test fixture.
        expected_records: The expected records value supplied by the caller or surrounding test fixture.
    """

    subject_sets = [set(summary.subject_ids) for summary in partitions.values()]
    # Evaluate `any((left & right for index, left in enumerate(subject_sets) for right in
    # subject_sets[index + 1:]))` explicitly so invalid or alternate states follow the documented
    # contract.
    if any(
        left & right
        for index, left in enumerate(subject_sets)
        for right in subject_sets[index + 1 :]
    ):
        raise SplitError("subject leakage detected across partitions")
    # Evaluate `set().union(*subject_sets) != expected_subjects` explicitly so invalid or alternate
    # states follow the documented contract.
    if set().union(*subject_sets) != expected_subjects:
        raise SplitError("split partitions do not cover every input subject")
    record_sets = [set(summary.record_ids) for summary in partitions.values()]
    # Evaluate `any((left & right for index, left in enumerate(record_sets) for right in
    # record_sets[index + 1:]))` explicitly so invalid or alternate states follow the documented
    # contract.
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise SplitError("record leakage detected across partitions")
    # Evaluate `set().union(*record_sets) != expected_records` explicitly so invalid or alternate
    # states follow the documented contract.
    if set().union(*record_sets) != expected_records:
        raise SplitError("split partitions do not cover every input record")


def _parse_partition_summary(name: str, value: Any, schema_version: Any) -> PartitionSummary:
    """Parse partition summary according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        name: The name value supplied by the caller or surrounding test fixture.
        value: Candidate value whose contract is being enforced.
        schema_version: The schema version value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(value, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, dict):
        raise SplitError(f"split partition {name} must be an object")
    record_ids = value.get("record_ids")
    target_counts = value.get("target_value_counts")
    # Evaluate `not isinstance(record_ids, list) or not all((isinstance(item, str) and item for item
    # in record_ids)) or len(record_id...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(record_ids, list)
        or not all(isinstance(item, str) and item for item in record_ids)
        or len(record_ids) != len(set(record_ids))
    ):
        raise SplitError(f"split partition {name} has invalid record IDs")
    # Evaluate `not isinstance(target_counts, dict) or not all((isinstance(key, str) and key and
    # isinstance(count, int) and (not isin...` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(target_counts, dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for key, count in target_counts.items()
    ):
        raise SplitError(f"split partition {name} has invalid target counts")
    subject_ids_value = value.get("subject_ids") if schema_version == 2 else record_ids
    record_subjects_value = (
        value.get("record_subjects") if schema_version == 2 else {item: item for item in record_ids}
    )
    # Evaluate `not isinstance(subject_ids_value, list) or not all((isinstance(item, str) and item
    # for item in subject_ids_value)) or...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(subject_ids_value, list)
        or not all(isinstance(item, str) and item for item in subject_ids_value)
        or len(subject_ids_value) != len(set(subject_ids_value))
        or not isinstance(record_subjects_value, dict)
        or set(record_subjects_value) != set(record_ids)
        or not all(isinstance(item, str) and item for item in record_subjects_value.values())
    ):
        raise SplitError(f"split partition {name} has invalid subject metadata")
    return PartitionSummary(
        subject_ids=tuple(subject_ids_value),
        subject_count=(
            _manifest_nonnegative_int(value, "subject_count")
            if schema_version == 2
            else len(subject_ids_value)
        ),
        record_ids=tuple(record_ids),
        record_subjects=dict(sorted(record_subjects_value.items())),
        record_count=_manifest_nonnegative_int(value, "record_count"),
        window_count=_manifest_nonnegative_int(value, "window_count"),
        target_value_counts=dict(sorted(target_counts.items())),
    )


def _validate_serialized_manifest(manifest: SplitManifest) -> None:
    """Validate serialized manifest according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        manifest: The manifest value supplied by the caller or surrounding test fixture.
    """

    # Evaluate `not isinstance(manifest.schema_version, int) or isinstance(manifest.schema_version,
    # bool) or manifest.schema_version ...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(manifest.schema_version, int)
        or isinstance(manifest.schema_version, bool)
        or manifest.schema_version not in {1, 2}
    ):
        raise SplitError("split manifest must use schema_version 1 or 2")
    record_sets = [set(partition.record_ids) for partition in manifest.partitions.values()]
    # Evaluate `any((left & right for index, left in enumerate(record_sets) for right in
    # record_sets[index + 1:]))` explicitly so invalid or alternate states follow the documented
    # contract.
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise SplitError("split manifest contains record leakage across partitions")
    subject_sets = [set(partition.subject_ids) for partition in manifest.partitions.values()]
    # Evaluate `any((left & right for index, left in enumerate(subject_sets) for right in
    # subject_sets[index + 1:]))` explicitly so invalid or alternate states follow the documented
    # contract.
    if any(
        left & right
        for index, left in enumerate(subject_sets)
        for right in subject_sets[index + 1 :]
    ):
        raise SplitError("split manifest contains subject leakage across partitions")
    # Iterate over `manifest.partitions.items()` one item at a time so ordering, validation, and
    # failure attribution remain explicit.
    for name, partition in manifest.partitions.items():
        # Evaluate `partition.subject_count == 0 or partition.subject_count !=
        # len(partition.subject_ids)` explicitly so invalid or alternate states follow the
        # documented contract.
        if partition.subject_count == 0 or partition.subject_count != len(partition.subject_ids):
            raise SplitError(f"split partition {name} subject count does not match membership")
        # Evaluate `set(partition.record_subjects) != set(partition.record_ids)` explicitly so
        # invalid or alternate states follow the documented contract.
        if set(partition.record_subjects) != set(partition.record_ids):
            raise SplitError(f"split partition {name} record-to-subject metadata is incomplete")
        # Evaluate `set(partition.record_subjects.values()) != set(partition.subject_ids)`
        # explicitly so invalid or alternate states follow the documented contract.
        if set(partition.record_subjects.values()) != set(partition.subject_ids):
            raise SplitError(f"split partition {name} subject membership does not match records")
        # Evaluate `partition.record_count == 0` explicitly so invalid or alternate states follow
        # the documented contract.
        if partition.record_count == 0:
            raise SplitError(f"split partition {name} must contain at least one record")
        # Evaluate `partition.record_count != len(partition.record_ids)` explicitly so invalid or
        # alternate states follow the documented contract.
        if partition.record_count != len(partition.record_ids):
            raise SplitError(f"split partition {name} record count does not match membership")
        # Evaluate `partition.window_count != sum(partition.target_value_counts.values())`
        # explicitly so invalid or alternate states follow the documented contract.
        if partition.window_count != sum(partition.target_value_counts.values()):
            raise SplitError(f"split partition {name} window and target counts do not match")
    # Evaluate `manifest.total_record_count != sum((partition.record_count for partition in
    # manifest.partitions.values()))` explicitly so invalid or alternate states follow the
    # documented contract.
    if manifest.total_record_count != sum(
        partition.record_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total record count does not match partitions")
    # Evaluate `manifest.total_subject_count != sum((partition.subject_count for partition in
    # manifest.partitions.values()))` explicitly so invalid or alternate states follow the
    # documented contract.
    if manifest.total_subject_count != sum(
        partition.subject_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total subject count does not match partitions")
    # Evaluate `manifest.total_window_count != sum((partition.window_count for partition in
    # manifest.partitions.values()))` explicitly so invalid or alternate states follow the
    # documented contract.
    if manifest.total_window_count != sum(
        partition.window_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total window count does not match partitions")
    target_key_sets = {
        frozenset(partition.target_value_counts) for partition in manifest.partitions.values()
    }
    # Evaluate `len(target_key_sets) != 1 or not next(iter(target_key_sets))` explicitly so invalid
    # or alternate states follow the documented contract.
    if len(target_key_sets) != 1 or not next(iter(target_key_sets)):
        raise SplitError("split partitions must report one consistent set of target values")


def _manifest_string(values: dict[str, Any], key: str) -> str:
    """Construct manifest string for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split manifest {key} must be a non-empty string")
    return value.strip()


def _manifest_nonnegative_int(values: dict[str, Any], key: str) -> int:
    """Construct manifest nonnegative int for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, int) or isinstance(value, bool) or value < 0` explicitly so
    # invalid or alternate states follow the documented contract.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitError(f"split manifest {key} must be a nonnegative integer")
    return value


def _required_string(values: dict[str, Any], key: str) -> str:
    """Compute and return required string for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split.{key} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    """Compute and return required nonnegative int for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, int) or isinstance(value, bool) or value < 0` explicitly so
    # invalid or alternate states follow the documented contract.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitError(f"split.{key} must be a nonnegative integer")
    return value


def _record_subject_mapping(document: dict[str, Any], schema_version: int) -> dict[str, str]:
    """Record subject mapping according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        document: Parsed document whose schema and values are being checked.
        schema_version: The schema version value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `schema_version == 1` explicitly so invalid or alternate states follow the documented
    # contract.
    if schema_version == 1:
        return {}
    subjects = document.get("record_subjects")
    # Evaluate `not isinstance(subjects, dict) or not subjects or (not all((isinstance(record_id,
    # str) and record_id and isinstance(s...` explicitly so invalid or alternate states follow the
    # documented contract.
    if (
        not isinstance(subjects, dict)
        or not subjects
        or not all(
            isinstance(record_id, str) and record_id and isinstance(subject_id, str) and subject_id
            for record_id, subject_id in subjects.items()
        )
    ):
        raise SplitError("split config v2 must contain a non-empty [record_subjects] table")
    return dict(sorted(subjects.items()))


def _required_ratio(values: dict[str, Any], key: str) -> float:
    """Compute and return required ratio for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, (int, float)) or isinstance(value, bool) or (not 0 < value <
    # 1)` explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
        raise SplitError(f"split.ratios.{key} must be between 0 and 1")
    return float(value)


def _quality_config(split: dict[str, Any]) -> SplitQualityConfig:
    """Build quality config for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        split: The split value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = split.get("quality", {})
    # Evaluate `not isinstance(value, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, dict):
        raise SplitError("split.quality must be a table")
    severity = value.get("default_severity", "failure")
    warning_checks = value.get("warning_checks", [])
    coverage = value.get("required_class_coverage", [])
    classes = value.get("required_classes", [])
    valid_checks = {
        "minimum_subjects",
        "minimum_records",
        "minimum_windows",
        "minimum_positive_examples",
        "required_class_coverage",
        "partition_ratio_deviation",
        "subject_disjointness",
        "record_disjointness",
    }
    # Evaluate `severity not in {'warning', 'failure'}` explicitly so invalid or alternate states
    # follow the documented contract.
    if severity not in {"warning", "failure"}:
        raise SplitError("split.quality.default_severity must be 'warning' or 'failure'")
    # Evaluate `not isinstance(warning_checks, list) or not all((item in valid_checks for item in
    # warning_checks))` explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(warning_checks, list) or not all(
        item in valid_checks for item in warning_checks
    ):
        raise SplitError("split.quality.warning_checks contains an unknown check")
    # Evaluate `not isinstance(coverage, list) or not all((item in {'train', 'validation', 'test'}
    # for item in coverage))` explicitly so invalid or alternate states follow the documented
    # contract.
    if not isinstance(coverage, list) or not all(
        item in {"train", "validation", "test"} for item in coverage
    ):
        raise SplitError("split.quality.required_class_coverage contains an unknown partition")
    # Evaluate `not isinstance(classes, list) or not all((isinstance(item, int) and (not
    # isinstance(item, bool)) for item in classes))` explicitly so invalid or alternate states
    # follow the documented contract.
    if not isinstance(classes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in classes
    ):
        raise SplitError("split.quality.required_classes must contain integers")

    def nonnegative(name: str, default: int) -> int:
        """Read and validate nonnegative for the documented repository workflow.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            name: The name value supplied by the caller or surrounding test fixture.
            default: The default value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

        item = value.get(name, default)
        # Evaluate `not isinstance(item, int) or isinstance(item, bool) or item < 0` explicitly so
        # invalid or alternate states follow the documented contract.
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise SplitError(f"split.quality.{name} must be a nonnegative integer")
        return item

    deviation = value.get("max_partition_ratio_deviation", 1.0)
    # Evaluate `not isinstance(deviation, (int, float)) or isinstance(deviation, bool) or (not 0 <=
    # deviation <= 1)` explicitly so invalid or alternate states follow the documented contract.
    if (
        not isinstance(deviation, (int, float))
        or isinstance(deviation, bool)
        or not 0 <= deviation <= 1
    ):
        raise SplitError("split.quality.max_partition_ratio_deviation must be between 0 and 1")
    return SplitQualityConfig(
        nonnegative("min_subjects_per_partition", 1),
        nonnegative("min_records_per_partition", 1),
        nonnegative("min_windows_per_partition", 1),
        nonnegative("min_positive_examples_per_partition", 0),
        tuple(coverage),
        tuple(sorted(set(classes))),
        float(deviation),
        severity,
        tuple(sorted(set(warning_checks))),
    )


def _violation(
    config: SplitConfig, check: str, partition: str | None, message: str
) -> QualityViolation:
    """Format violation for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.
        check: The check value supplied by the caller or surrounding test fixture.
        partition: The partition value supplied by the caller or surrounding test fixture.
        message: The message value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    severity = (
        "warning" if check in config.quality.warning_checks else config.quality.default_severity
    )
    return QualityViolation(check, partition, severity, message)


def _sets_are_disjoint(values: list[set[str]]) -> bool:
    """Return whether sets are disjoint under the documented validation contract.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.

    Returns:
        The value produced by the documented operation.
    """

    return not any(
        left & right for index, left in enumerate(values) for right in values[index + 1 :]
    )


def _string_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> tuple[str, ...]:
    """Compute and return string vector for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        path: Filesystem path identifying the input or output under review.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 1 or array.dtype.kind not in {'U', 'S'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    # Evaluate `any((not item for item in result))` explicitly so invalid or alternate states follow
    # the documented contract.
    if any(not item for item in result):
        raise SplitError(f"window artifact {path} field {field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> IntegerArray:
    """Compute and return integer vector for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        path: Filesystem path identifying the input or output under review.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 1 or array.dtype.kind not in {'i', 'u'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer vector")
    return np.asarray(array, dtype=np.int64)


def _string_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> str:
    """Compute and return string scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        path: Filesystem path identifying the input or output under review.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 0 or array.dtype.kind not in {'U', 'S'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string scalar")
    result = str(array.item())
    # Evaluate `not result` explicitly so invalid or alternate states follow the documented
    # contract.
    if not result:
        raise SplitError(f"window artifact {path} field {field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> int:
    """Compute and return integer scalar for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        value: Candidate value whose contract is being enforced.
        path: Filesystem path identifying the input or output under review.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    array = np.asarray(value)
    # Evaluate `array.ndim != 0 or array.dtype.kind not in {'i', 'u'}` explicitly so invalid or
    # alternate states follow the documented contract.
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer scalar")
    return int(array.item())
