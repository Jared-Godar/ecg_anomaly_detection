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
        # Manifests are untrusted JSON read back from disk (or produced by another tool);
        # collapse the ways `json.loads`/dict access can fail into one SplitError so every
        # caller sees a single fail-closed exception type instead of three unrelated ones.
        try:
            document = json.loads(content)
            schema_version = document["schema_version"]
            partitions_document = document["partitions"]
            # A manifest that doesn't name exactly the three fixed partitions was hand-edited
            # or written by an incompatible tool; reject it before any partition is parsed.
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
            # source_artifacts is the lineage back to the window-extraction NPZ files this
            # split was built from; require at least one unique, non-empty path so that
            # lineage can never silently be empty or contain the same artifact twice.
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
    # Wrap the TOML read so a missing config file and a malformed one both surface as one
    # SplitError, rather than leaking tomllib's or the filesystem's own exception types.
    try:
        # Open in binary mode because tomllib.load requires a byte stream; the `with` block
        # guarantees the file handle closes even if parsing raises partway through.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SplitError(f"could not load split config {path}: {error}") from error

    split = document.get("split")
    schema_version = document.get("schema_version")
    # schema_version distinguishes the legacy per-record split (1) from the current
    # subject-grouped split (2); anything else means the config predates or postdates
    # what this loader understands, so fail before touching the [split] table further.
    if schema_version not in {1, 2} or not isinstance(split, dict):
        raise SplitError("split config must use schema_version 1 or 2 and a [split] table")
    ratios = split.get("ratios")
    # The train/validation/test ratios are mandatory and have no sensible default --
    # silently defaulting them would let a config accidentally omit a partition size.
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
    # Pin the declared strategy name to the schema version so a config can't claim
    # record-level shuffling under schema 2 (or vice versa) and silently split the wrong way.
    if config.strategy != expected_strategy:
        raise SplitError(f"split.strategy must be '{expected_strategy}'")
    ratio_sum = config.train_ratio + config.validation_ratio + config.test_ratio
    # Floating-point ratios rarely sum to exactly 1.0; use a tolerance comparison so a
    # config with a tiny rounding remainder isn't rejected, while a true mis-sum still is.
    if not np.isclose(ratio_sum, 1.0):
        raise SplitError(f"split ratios must sum to 1.0; got {ratio_sum}")
    return config


def load_window_metadata(artifact_paths: Sequence[Path]) -> WindowMetadata:
    """Load lineage fields from non-pickle NPZ window artifacts."""
    # A split needs at least one artifact to derive any records/subjects from; an empty
    # list would otherwise fall through to an artificial "no windows" split.
    if not artifact_paths:
        raise SplitError("at least one window artifact is required")

    record_ids: list[str] = []
    target_arrays: list[IntegerArray] = []
    seen_records: set[str] = set()
    identity: tuple[str, str, str, str] | None = None
    record_shards: dict[str, str] = {}

    # Process artifacts one at a time (rather than loading all up front) so the first bad
    # file fails fast with its own path in the error, instead of an aggregated failure.
    for path in artifact_paths:
        # allow_pickle=False is a security/format boundary: these NPZ files may come from a
        # shared or third-party pipeline run, and pickle deserialization of untrusted data is
        # an arbitrary-code-execution risk. Collapse the load/parse failure modes into one
        # SplitError so callers don't need to know numpy's or the filesystem's exception types.
        try:
            # `np.load` returns a lazy NpzFile handle backed by an open zip archive; the
            # `with` block ensures it's closed even if a field is missing or malformed.
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
                # Every downstream field access below assumes these keys exist; check the
                # whole set up front so one missing field is reported clearly instead of a
                # bare KeyError partway through parsing.
                if missing:
                    raise SplitError(f"window artifact {path} is missing fields: {sorted(missing)}")
                # schema_version pins the NPZ layout produced by the window-extraction stage;
                # a mismatch means this artifact was built by an incompatible version of that
                # stage and its field shapes/meanings can't be trusted.
                if _integer_scalar(artifact["schema_version"], path, "schema_version") != 1:
                    raise SplitError(f"window artifact {path} must use schema_version 1")
                artifact_records = _string_vector(artifact["record_ids"], path, "record_ids")
                artifact_targets = _integer_vector(artifact["target_values"], path, "target_values")
                # record_ids and target_values are parallel arrays (one label per window row);
                # an unequal length means the artifact was corrupted or half-written.
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
            # Re-raise our own diagnostics untouched so the specific field-level message
            # from above isn't swallowed by the broader numpy/OSError handler below.
            raise
        except (BadZipFile, OSError, ValueError) as error:
            raise SplitError(f"could not load window artifact {path}: {error}") from error

        # An artifact that parsed but contains zero rows would otherwise contribute nothing
        # while still passing every other check; treat it as a fail-closed configuration error.
        if not artifact_records:
            raise SplitError(f"window artifact {path} contains no windows")
        current_records = set(artifact_records)
        duplicated_records = current_records & seen_records
        # This is the load-bearing leakage guard for splitting: a record must originate from
        # exactly one artifact/shard. If the same record ID appears in two artifacts, later
        # subject-grouped partitioning could not guarantee the record lands in only one split.
        if duplicated_records:
            raise SplitError(
                f"records occur in multiple window artifacts: {sorted(duplicated_records)}"
            )
        seen_records.update(current_records)
        record_shards.update({record_id: str(path) for record_id in current_records})
        # Capture the (mapping, window-config) identity from the first artifact, then require
        # every subsequent artifact to match it -- mixing artifacts built under different
        # annotation mappings or window configs would silently blend incompatible label
        # semantics into one split.
        if identity is None:
            identity = artifact_identity
        elif artifact_identity != identity:
            raise SplitError("window artifacts must use the same mapping and window configuration")
        record_ids.extend(artifact_records)
        target_arrays.append(artifact_targets)

    # Unreachable in practice: the non-empty `artifact_paths` check above guarantees the loop
    # runs at least once, and the "no windows" check above guarantees `identity` gets set on
    # that first iteration. Kept as an explicit guard so a future refactor can't silently
    # return a WindowMetadata built from an unset identity.
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
    # Compute diagnostics and run acceptance checks per partition, in the fixed
    # train/validation/test order, so the resulting JSON summary has stable key ordering.
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
        # A too-small partition (e.g. a validation set with one subject) would make
        # downstream metrics statistically meaningless; run all three size floors together
        # so every violation for this partition is reported in one pass.
        for check, actual, minimum in checks:
            # A strict `<` comparison against the configured minimum: hitting the minimum
            # exactly is acceptable, falling short is not.
            if actual < minimum:
                violations.append(
                    _violation(
                        config, check, name, f"{actual} is below configured minimum {minimum}"
                    )
                )
        # For binary classification only: an anomaly-detection split with too few positive
        # (arrhythmia) examples in a partition can't meaningfully evaluate recall on it.
        if binary and class_counts["1"] < config.quality.min_positive_examples_per_partition:
            violations.append(
                _violation(
                    config,
                    "minimum_positive_examples",
                    name,
                    f"{class_counts['1']} is below configured minimum {config.quality.min_positive_examples_per_partition}",
                )
            )
        # Coverage requirements are opt-in per partition (e.g. "test must contain both
        # classes"); only check the classes list for partitions explicitly named in config.
        if name in config.quality.required_class_coverage:
            missing = [
                value
                for value in config.quality.required_classes
                if class_counts.get(str(value), 0) == 0
            ]
            # A subject-grouped shuffle can, by chance, place all examples of a rare class
            # into one partition; report which specific classes are absent so the config
            # author can widen the partition or adjust required_classes.
            if missing:
                violations.append(
                    _violation(
                        config,
                        "required_class_coverage",
                        name,
                        f"missing required classes {missing}",
                    )
                )
        # A large gap between the configured and actual subject ratio signals the random
        # seed produced a lopsided shuffle for this dataset size; flag it so a reviewer can
        # judge whether to accept the split or pick a different seed.
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
    # This is the diagnostic twin of the leakage guard in load_window_metadata: it should
    # never fire in practice (create_split_manifest already enforces disjointness), but it
    # keeps the quality summary self-verifying against manifests read back from disk, where
    # the guarantee could have been broken by hand-editing or an incompatible writer.
    if not subject_disjoint:
        violations.append(
            _violation(
                config, "subject_disjointness", None, "subjects occur in multiple partitions"
            )
        )
    # Same self-verification, at the record level rather than the subject level.
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
    # Reject a non-JSON extension or a missing parent directory before writing, rather than
    # letting write_text either silently write the wrong file type or raise a bare OSError.
    if output_path.suffix != ".json" or not output_path.parent.is_dir():
        raise SplitError("split quality summary must be a JSON file in an existing directory")
    output_path.write_text(summary.to_json(), encoding="utf-8")


def enforce_split_quality(summary: SplitQualitySummary) -> None:
    """Fail closed after the diagnostic artifact has been made available."""
    failures = [item for item in summary.violations if item.severity == "failure"]
    # Callers are expected to persist the summary (via write_split_quality_summary) before
    # calling this, so a rejected split still leaves a diagnostic artifact on disk for
    # debugging -- this function only decides whether the pipeline should stop.
    if failures:
        raise SplitError(f"split quality checks failed: {len(failures)} failure(s)")


def create_split_manifest(config: SplitConfig, metadata: WindowMetadata) -> SplitManifest:
    """Assign complete subjects and report subject, record, and target counts."""
    # These two arrays are produced together by load_window_metadata and must stay in
    # lockstep; a mismatch here means metadata was constructed some other way.
    if len(metadata.record_ids) != len(metadata.target_values):
        raise SplitError("window metadata record and target row counts must match")
    unique_records = sorted(set(metadata.record_ids))
    record_subjects = (
        {record_id: record_id for record_id in unique_records}
        if config.schema_version == 1
        else config.record_subjects
    )
    # Schema 1 treats each record as its own subject (the legacy, non-grouped behavior);
    # schema 2 requires the config's explicit record-to-subject map to cover every record
    # actually present in the windows -- a mismatch would let some records escape subject
    # grouping entirely, silently reopening the leakage this pipeline exists to prevent.
    if set(record_subjects) != set(unique_records):
        raise SplitError(
            "record-to-subject metadata must exactly cover window records; "
            f"missing={sorted(set(unique_records) - set(record_subjects))}, "
            f"extra={sorted(set(record_subjects) - set(unique_records))}"
        )
    unique_subjects = sorted(set(record_subjects.values()))
    # Three subjects is the practical floor for a non-degenerate train/validation/test
    # split -- fewer would force at least one partition to be empty or share a subject
    # with another, defeating the point of subject-level grouping.
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
    # Enforce the extension so downstream tooling that globs for *.json manifests can rely
    # on finding this file without a separate content sniff.
    if output_path.suffix != ".json":
        raise SplitError("split manifest must use the .json extension")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from write_text.
    if not output_path.parent.is_dir():
        raise SplitError(f"split manifest parent directory does not exist: {output_path.parent}")
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def read_split_manifest(path: Path) -> SplitManifest:
    """Read and validate a grouped split manifest from disk."""
    # Translate a missing or unreadable file into SplitError so callers along the pipeline
    # only need to catch one exception type for every split-related failure.
    try:
        return SplitManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SplitError(f"could not read split manifest {path}: {error}") from error


def _partition_sizes(
    subject_count: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Convert train/validation/test ratios into exact integer subject counts.

    Multiplying a ratio by an integer subject count rarely produces a whole number, and
    naive truncation of all three ratios can under- or over-allocate the total by a few
    subjects. This uses the largest-remainder method: truncate each ratio's exact share,
    then hand out the leftover subjects one at a time to whichever partition's truncated
    share lost the most (the largest fractional remainder), so the three sizes always sum
    to exactly `subject_count`.

    Args:
        subject_count: Total number of unique subjects to distribute.
        ratios: Train, validation, and test ratios (must already sum to ~1.0).

    Returns:
        Exact (train, validation, test) subject counts summing to subject_count.
    """

    exact = [subject_count * ratio for ratio in ratios]
    sizes = [int(value) for value in exact]
    remainder = subject_count - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(exact[index] - sizes[index]), index))
    # Give the leftover subjects (after truncation) to the partitions with the largest
    # fractional remainder first, tie-breaking by index for determinism.
    for index in order[:remainder]:
        sizes[index] += 1
    # With very small subject counts and skewed ratios, largest-remainder allocation can
    # still leave a partition at zero; rebalance by borrowing one subject from whichever
    # partition currently has the most, so every partition ends up non-empty.
    for empty_index, size in enumerate(sizes):
        # Only a zero-sized partition needs rebalancing; anything positive is left alone.
        if size == 0:
            donor = max(range(3), key=lambda index: sizes[index])
            # A donor with only one subject can't give one up without becoming empty
            # itself; at that point there simply aren't enough subjects to split three
            # ways and the caller's minimum-subject requirement (>= 3) has been violated.
            if sizes[donor] <= 1:
                raise SplitError("could not create three non-empty subject partitions")
            sizes[donor] -= 1
            sizes[empty_index] += 1
    return sizes[0], sizes[1], sizes[2]


def _summarize_partition(
    subjects: tuple[str, ...], record_subjects: dict[str, str], metadata: WindowMetadata
) -> PartitionSummary:
    """Summarize one partition's records, subjects, shards, labels, and ratio diagnostics.

    Given the subjects assigned to this partition, derives every record belonging to those
    subjects, then counts target-class occurrences among that partition's windows. This is
    the point where subject-level assignment (from the seeded shuffle) becomes concrete
    record- and window-level membership.

    Args:
        subjects: Subject IDs assigned to this partition by the seeded shuffle.
        record_subjects: Complete record-to-subject mapping for the whole dataset.
        metadata: Window-level lineage (record IDs and target labels) to summarize against.

    Returns:
        Partition membership plus per-class window counts for this partition.
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
    """Confirm freshly assigned partitions have no leakage and cover every input exactly.

    This is the direct enforcement of the repository's core anti-leakage invariant: no
    subject or record may appear in more than one of train/validation/test, and every
    subject/record from the input must end up in exactly one partition. It runs
    immediately after the seeded shuffle in create_split_manifest, before any manifest
    is written to disk, so a bug in the shuffle or boundary arithmetic is caught here
    rather than silently propagating into a leaky split.

    Args:
        partitions: Freshly computed per-partition summaries to check.
        expected_subjects: Every subject ID that should appear across all partitions.
        expected_records: Every record ID that should appear across all partitions.
    """

    subject_sets = [set(summary.subject_ids) for summary in partitions.values()]
    # Pairwise-intersect every partition's subject set against every other partition's;
    # a non-empty intersection means the same subject was assigned to two partitions.
    if any(
        left & right
        for index, left in enumerate(subject_sets)
        for right in subject_sets[index + 1 :]
    ):
        raise SplitError("subject leakage detected across partitions")
    # The union of all partitions' subjects must equal the full input subject set --
    # anything less means a subject was silently dropped by the boundary arithmetic.
    if set().union(*subject_sets) != expected_subjects:
        raise SplitError("split partitions do not cover every input subject")
    record_sets = [set(summary.record_ids) for summary in partitions.values()]
    # Same pairwise leakage check as above, at the record level.
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise SplitError("record leakage detected across partitions")
    # Same full-coverage check as above, at the record level.
    if set().union(*record_sets) != expected_records:
        raise SplitError("split partitions do not cover every input record")


def _parse_partition_summary(name: str, value: Any, schema_version: Any) -> PartitionSummary:
    """Parse and structurally validate one partition of a manifest read back from disk.

    Manifest JSON is untrusted input (it may have been hand-edited, or written by an
    older/incompatible version of this module), so every field is checked before it's
    trusted to build a PartitionSummary. Schema 1 manifests have no explicit subject
    fields and are treated as one subject per record, matching create_split_manifest's
    own schema-1 behavior.

    Args:
        name: Partition name ("train", "validation", or "test"), used only in error text.
        value: The raw JSON object for this partition.
        schema_version: The manifest's declared schema_version (1 or 2).

    Returns:
        A validated PartitionSummary for this partition.
    """

    # Every field access below assumes `value` is a JSON object; reject anything else
    # (e.g. a list or scalar) before attempting `.get()` on it.
    if not isinstance(value, dict):
        raise SplitError(f"split partition {name} must be an object")
    record_ids = value.get("record_ids")
    target_counts = value.get("target_value_counts")
    # record_ids must be a list of non-empty, unique strings -- duplicates would mean the
    # same record was double-counted in this partition's window/record totals.
    if (
        not isinstance(record_ids, list)
        or not all(isinstance(item, str) and item for item in record_ids)
        or len(record_ids) != len(set(record_ids))
    ):
        raise SplitError(f"split partition {name} has invalid record IDs")
    # target_value_counts maps each observed class label to a non-negative window count;
    # a bool is explicitly excluded even though bool is an int subclass, since a boolean
    # count would indicate the field was serialized incorrectly upstream.
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
    # Subject IDs must be unique non-empty strings, and record_subjects must map every
    # record in this partition to exactly one of those subjects -- this is the on-disk
    # form of the same subject/record consistency create_split_manifest enforces at
    # construction time, re-checked here because the manifest may have been hand-edited.
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
    """Re-verify a manifest's global invariants after its partitions parse individually.

    _parse_partition_summary validates each partition in isolation; this pass checks
    cross-partition properties that only make sense once every partition is known --
    leakage-freedom and count consistency -- the same invariants create_split_manifest
    guarantees when building a manifest fresh, re-checked here because from_json is the
    entry point for manifests written by a possibly different or older version of this
    module, or edited by hand.

    Args:
        manifest: The manifest whose partitions have already parsed individually.
    """

    # A bool is technically an int, so it's excluded explicitly; schema_version must be
    # one of the two versions this module understands.
    if (
        not isinstance(manifest.schema_version, int)
        or isinstance(manifest.schema_version, bool)
        or manifest.schema_version not in {1, 2}
    ):
        raise SplitError("split manifest must use schema_version 1 or 2")
    record_sets = [set(partition.record_ids) for partition in manifest.partitions.values()]
    # Pairwise-intersect every partition's record set; a non-empty intersection means the
    # manifest describes a record as belonging to more than one partition.
    if any(
        left & right for index, left in enumerate(record_sets) for right in record_sets[index + 1 :]
    ):
        raise SplitError("split manifest contains record leakage across partitions")
    subject_sets = [set(partition.subject_ids) for partition in manifest.partitions.values()]
    # Same pairwise leakage check as above, at the subject level -- this is the property
    # the whole subject-grouped splitting design exists to guarantee.
    if any(
        left & right
        for index, left in enumerate(subject_sets)
        for right in subject_sets[index + 1 :]
    ):
        raise SplitError("split manifest contains subject leakage across partitions")
    # Cross-check each partition's own internal bookkeeping: its declared counts must
    # match the length of its membership lists, and its record-to-subject map must be
    # exactly consistent with both its record_ids and subject_ids -- catching a manifest
    # where a count field was edited without updating the membership lists, or vice versa.
    for name, partition in manifest.partitions.items():
        # A partition must have at least one subject, and its declared count must match
        # the number of subject IDs actually listed.
        if partition.subject_count == 0 or partition.subject_count != len(partition.subject_ids):
            raise SplitError(f"split partition {name} subject count does not match membership")
        # Every record in this partition must have a subject entry, and no extra ones.
        if set(partition.record_subjects) != set(partition.record_ids):
            raise SplitError(f"split partition {name} record-to-subject metadata is incomplete")
        # The subjects named in record_subjects must be exactly this partition's subjects
        # -- a record pointing at a subject outside the partition would itself be a form
        # of leakage that the earlier subject-set check alone wouldn't catch.
        if set(partition.record_subjects.values()) != set(partition.subject_ids):
            raise SplitError(f"split partition {name} subject membership does not match records")
        # A subject-grouped partition with zero records would be a degenerate split.
        if partition.record_count == 0:
            raise SplitError(f"split partition {name} must contain at least one record")
        # The declared record_count must match the actual length of record_ids.
        if partition.record_count != len(partition.record_ids):
            raise SplitError(f"split partition {name} record count does not match membership")
        # window_count should equal the sum of this partition's own class histogram --
        # a mismatch means the manifest's target_value_counts were edited independently
        # of window_count, or don't actually describe the same windows.
        if partition.window_count != sum(partition.target_value_counts.values()):
            raise SplitError(f"split partition {name} window and target counts do not match")
    # The manifest's top-level totals are redundant with the per-partition sums and exist
    # only for convenient reading; verify they haven't drifted apart from the partitions
    # they're supposed to summarize.
    if manifest.total_record_count != sum(
        partition.record_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total record count does not match partitions")
    # Same drift check as above, at the subject-count total.
    if manifest.total_subject_count != sum(
        partition.subject_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total subject count does not match partitions")
    # Same drift check as above, at the window-count total.
    if manifest.total_window_count != sum(
        partition.window_count for partition in manifest.partitions.values()
    ):
        raise SplitError("split manifest total window count does not match partitions")
    target_key_sets = {
        frozenset(partition.target_value_counts) for partition in manifest.partitions.values()
    }
    # Every partition should report counts for the same set of target classes (even if
    # some counts are zero) so downstream consumers can assume a uniform class vocabulary
    # across train/validation/test without special-casing missing keys.
    if len(target_key_sets) != 1 or not next(iter(target_key_sets)):
        raise SplitError("split partitions must report one consistent set of target values")


def _manifest_string(values: dict[str, Any], key: str) -> str:
    """Extract and validate one required non-empty string field from a parsed manifest.

    A stripped-and-nonempty check catches both a missing/wrong-typed field and a
    whitespace-only placeholder value, either of which would otherwise propagate an
    unusable identifier (e.g. an empty split_name) deep into a SplitManifest.

    Args:
        values: The parsed manifest document (or nested object) to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split manifest {key} must be a non-empty string")
    return value.strip()


def _manifest_nonnegative_int(values: dict[str, Any], key: str) -> int:
    """Extract and validate one required nonnegative integer field from a parsed manifest.

    `bool` is excluded even though it's an `int` subclass in Python, since a JSON boolean
    in a count field (e.g. `"record_count": true`) indicates the manifest was malformed
    upstream rather than a legitimate count of 1.

    Args:
        values: The parsed manifest document (or nested object) to read from.
        key: The field name to extract.

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitError(f"split manifest {key} must be a nonnegative integer")
    return value


def _required_string(values: dict[str, Any], key: str) -> str:
    """Extract and validate one required non-empty string field from a split TOML config.

    Structurally identical to _manifest_string, but raises with a `split.{key}` error
    prefix matching TOML's dotted-path convention, since this validates the *input*
    config rather than a manifest read back from a prior run.

    Args:
        values: The parsed `[split]` table (or nested table) to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split.{key} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    """Extract and validate one required nonnegative integer field from a split TOML config.

    Structurally identical to _manifest_nonnegative_int, but raises with the config's
    `split.{key}` error prefix rather than the manifest's, so a validation failure
    always names which document (input config vs. output manifest) it came from.

    Args:
        values: The parsed `[split]` table (or nested table) to read from.
        key: The field name to extract.

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SplitError(f"split.{key} must be a nonnegative integer")
    return value


def _record_subject_mapping(document: dict[str, Any], schema_version: int) -> dict[str, str]:
    """Load the explicit record-to-subject map required by schema-2 split configs.

    Schema 1 has no concept of subjects distinct from records, so it always returns an
    empty mapping (create_split_manifest falls back to one-subject-per-record for it).
    Schema 2 requires the config author to supply this table explicitly, since inferring
    subject identity from record IDs is exactly the kind of guess that caused the 2022
    pipeline's leakage problem this module was written to fix.

    Args:
        document: The parsed TOML document (the whole config file, not just [split]).
        schema_version: 1 or 2, as already validated by load_split_config.

    Returns:
        A sorted record ID to subject ID mapping (empty for schema 1).
    """

    # Schema 1 has no subject concept at all; nothing to load or validate.
    if schema_version == 1:
        return {}
    subjects = document.get("record_subjects")
    # Every key/value must be a non-empty string; an empty table would silently produce
    # a schema-2 config with no way to assign any record to a subject.
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
    """Extract and validate one train/validation/test ratio from a split TOML config.

    Ratios are constrained to the open interval (0, 1) exclusive: a ratio of exactly 0 or
    1 would produce an empty or all-encompassing partition, which _partition_sizes and the
    minimum-subjects check downstream aren't designed to handle gracefully.

    Args:
        values: The parsed `[split.ratios]` table.
        key: Which ratio to extract ("train", "validation", or "test").

    Returns:
        The ratio as a float.
    """

    value = values.get(key)
    # Exclusive bounds: a ratio of exactly 0 or 1 would produce a degenerate partition.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
        raise SplitError(f"split.ratios.{key} must be between 0 and 1")
    return float(value)


def _quality_config(split: dict[str, Any]) -> SplitQualityConfig:
    """Parse the optional `[split.quality]` table into a validated SplitQualityConfig.

    Every field is optional in the TOML config (SplitQualityConfig's dataclass defaults
    apply when omitted), but any field that *is* present must be well-formed -- this
    guards against, e.g., a typo'd check name in `warning_checks` silently being ignored
    instead of raising, which would let a reviewer believe a check had been downgraded to
    a warning when it was actually still a hard failure.

    Args:
        split: The parsed `[split]` table (the quality sub-table is read from it).

    Returns:
        A validated SplitQualityConfig, using dataclass defaults for omitted fields.
    """

    value = split.get("quality", {})
    # The default {} lets [split.quality] be omitted entirely; anything present that
    # isn't a table means the config was malformed rather than intentionally minimal.
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
    # default_severity governs any check not explicitly listed in warning_checks below.
    if severity not in {"warning", "failure"}:
        raise SplitError("split.quality.default_severity must be 'warning' or 'failure'")
    # Cross-reference against the fixed set of check names create_split_quality_summary
    # actually emits, so a misspelled entry here is caught at config-load time rather
    # than silently never matching any real violation.
    if not isinstance(warning_checks, list) or not all(
        item in valid_checks for item in warning_checks
    ):
        raise SplitError("split.quality.warning_checks contains an unknown check")
    # required_class_coverage names which *partitions* need coverage enforcement, so its
    # values are constrained to the three fixed partition names, not the check names above.
    if not isinstance(coverage, list) or not all(
        item in {"train", "validation", "test"} for item in coverage
    ):
        raise SplitError("split.quality.required_class_coverage contains an unknown partition")
    # required_classes holds raw target-label integers (e.g. 0/1 for binary
    # classification), not check or partition names, so it's validated as plain ints.
    if not isinstance(classes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in classes
    ):
        raise SplitError("split.quality.required_classes must contain integers")

    def nonnegative(name: str, default: int) -> int:
        """Read one nonnegative-integer quality threshold, falling back to its default.

        Local to _quality_config because every one of the four minimum-count thresholds
        needs identical validation; factoring it out avoids repeating the same isinstance
        checks four times with only the field name and default changing.

        Args:
            name: The quality-table key to read (e.g. "min_subjects_per_partition").
            default: The value to use when the key is absent from the config.

        Returns:
            The configured or default threshold.
        """

        item = value.get(name, default)
        # bool is an int subclass in Python, so it's excluded explicitly.
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise SplitError(f"split.quality.{name} must be a nonnegative integer")
        return item

    deviation = value.get("max_partition_ratio_deviation", 1.0)
    # Unlike the count thresholds, this is a ratio-deviation tolerance and must stay
    # within [0, 1] to be a meaningful bound on subject-ratio drift.
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
    """Build one QualityViolation, resolving its severity from the split's quality config.

    Centralizes the severity lookup so every call site in create_split_quality_summary
    treats "warning_checks" consistently: a check named there is always a warning
    regardless of default_severity, and every other check falls back to the configured
    default (failure unless overridden).

    Args:
        config: The split config whose quality settings determine this violation's severity.
        check: Which acceptance check failed (e.g. "minimum_subjects").
        partition: The partition the violation applies to, or None for a split-wide check.
        message: Human-readable detail describing the specific violation.

    Returns:
        A QualityViolation with severity resolved from config.quality.
    """

    severity = (
        "warning" if check in config.quality.warning_checks else config.quality.default_severity
    )
    return QualityViolation(check, partition, severity, message)


def _sets_are_disjoint(values: list[set[str]]) -> bool:
    """Return whether every pair of sets in a list shares no common elements.

    Shared by create_split_quality_summary for both the subject- and record-level
    disjointness diagnostics; O(n^2) pairwise comparison is fine here since it only ever
    runs over the three fixed partitions (train/validation/test), never a large list.

    Args:
        values: The sets to compare pairwise (in practice, per-partition ID sets).

    Returns:
        True if no two sets in the list intersect.
    """

    return not any(
        left & right for index, left in enumerate(values) for right in values[index + 1 :]
    )


def _string_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> tuple[str, ...]:
    """Validate and convert one 1-D string field from a window NPZ artifact.

    NPZ fields round-trip as numpy arrays with an explicit dtype; a field that's the
    wrong shape or an unexpected dtype (e.g. saved as an object array) signals the
    artifact wasn't produced by the expected window-extraction writer.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        path: The artifact's path, included in any error for traceability.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's values as a tuple of Python strings.
    """

    array = np.asarray(value)
    # dtype.kind 'U'/'S' cover numpy's Unicode and byte-string dtypes respectively; a
    # 1-D shape is required because this always represents one value per window row.
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    # record_ids in particular must never contain an empty string, since an empty ID
    # would be indistinguishable from a missing/corrupted lineage entry downstream.
    if any(not item for item in result):
        raise SplitError(f"window artifact {path} field {field} contains an empty string")
    return result


def _integer_vector(value: np.ndarray[Any, Any], path: Path, field: str) -> IntegerArray:
    """Validate and convert one 1-D integer field from a window NPZ artifact.

    Used specifically for target_values, the per-window class labels; the dtype check
    rejects float or object arrays that would otherwise let a fractional or non-numeric
    "label" slip into the labeled dataset undetected.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        path: The artifact's path, included in any error for traceability.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's values as an int64 numpy array.
    """

    array = np.asarray(value)
    # dtype.kind 'i'/'u' cover numpy's signed and unsigned integer dtypes.
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer vector")
    return np.asarray(array, dtype=np.int64)


def _string_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> str:
    """Validate and convert one 0-D (scalar) string field from a window NPZ artifact.

    Used for the artifact-wide identity fields (mapping_name, mapping_version, etc.)
    that apply to the whole file rather than varying per window row, hence 0-D rather
    than the 1-D shape _string_vector expects.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        path: The artifact's path, included in any error for traceability.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python string.
    """

    array = np.asarray(value)
    # 0-D shape distinguishes a scalar identity field from a per-row string vector.
    if array.ndim != 0 or array.dtype.kind not in {"U", "S"}:
        raise SplitError(f"window artifact {path} field {field} must be a string scalar")
    result = str(array.item())
    # An empty identity field (e.g. mapping_name="") would make artifact-identity
    # comparisons in load_window_metadata meaningless -- any artifact would "match".
    if not result:
        raise SplitError(f"window artifact {path} field {field} must not be empty")
    return result


def _integer_scalar(value: np.ndarray[Any, Any], path: Path, field: str) -> int:
    """Validate and convert one 0-D (scalar) integer field from a window NPZ artifact.

    Used for schema_version, which applies to the whole artifact file rather than
    varying per window row.

    Args:
        value: The raw array loaded from the NPZ artifact for this field.
        path: The artifact's path, included in any error for traceability.
        field: The field's name, included in any error for traceability.

    Returns:
        The field's value as a Python int.
    """

    array = np.asarray(value)
    # 0-D shape distinguishes a scalar identity field from a per-row integer vector.
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise SplitError(f"window artifact {path} field {field} must be an integer scalar")
    return int(array.item())
