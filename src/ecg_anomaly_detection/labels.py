"""Versioned mapping from source annotations to project-specific targets."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ecg_anomaly_detection.records import AnnotationSet, IntegerArray


class AnnotationMappingError(ValueError):
    """Raised when mapping configuration or source annotations are invalid."""


@dataclass(frozen=True, slots=True)
class TargetRule:
    """One project target and the source symbols assigned to it."""

    name: str
    value: int
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationMappingConfig:
    """Closed-world annotation mapping contract."""

    schema_version: int
    name: str
    version: str
    unknown_symbol_policy: str
    targets: tuple[TargetRule, ...]
    excluded_symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MappedAnnotationSet:
    """Included source annotations with project-specific integer targets."""

    record_id: str
    sample_indices: IntegerArray
    source_symbols: tuple[str, ...]
    target_values: IntegerArray


@dataclass(frozen=True, slots=True)
class AnnotationMappingReport:
    """Machine-readable counts for included and excluded annotations."""

    schema_version: int
    mapping_name: str
    mapping_version: str
    record_id: str
    input_annotation_count: int
    included_annotation_count: int
    excluded_annotation_count: int
    source_symbol_counts: dict[str, int]
    target_counts: dict[str, int]
    excluded_symbol_counts: dict[str, int]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class AnnotationMappingResult:
    """Mapped annotations and their audit report."""

    annotations: MappedAnnotationSet
    report: AnnotationMappingReport


def load_annotation_mapping(path: Path) -> AnnotationMappingConfig:
    """Load and validate a versioned annotation mapping from TOML."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AnnotationMappingError(
            f"could not load annotation mapping {path}: {error}"
        ) from error

    # Evaluate `document.get('schema_version') != 1` explicitly so invalid or alternate states
    # follow the documented contract.
    if document.get("schema_version") != 1:
        raise AnnotationMappingError("annotation mapping must use schema_version = 1")
    mapping = document.get("mapping")
    targets = document.get("targets")
    exclusions = document.get("exclusions")
    # Evaluate `not isinstance(mapping, dict) or not isinstance(targets, list) or (not
    # isinstance(exclusions, dict))` explicitly so invalid or alternate states follow the documented
    # contract.
    if (
        not isinstance(mapping, dict)
        or not isinstance(targets, list)
        or not isinstance(exclusions, dict)
    ):
        raise AnnotationMappingError(
            "annotation mapping requires [mapping], [[targets]], and [exclusions]"
        )

    rules = tuple(_load_target_rule(item) for item in targets)
    config = AnnotationMappingConfig(
        schema_version=1,
        name=_required_string(mapping, "name"),
        version=_required_string(mapping, "version"),
        unknown_symbol_policy=_required_string(mapping, "unknown_symbol_policy"),
        targets=rules,
        excluded_symbols=_required_symbols(exclusions, "symbols"),
    )
    _validate_mapping_config(config)
    return config


def map_annotations(
    config: AnnotationMappingConfig,
    annotations: AnnotationSet,
) -> AnnotationMappingResult:
    """Apply a closed-world mapping and report every inclusion and exclusion."""
    # Evaluate `len(annotations.sample_indices) != len(annotations.symbols)` explicitly so invalid
    # or alternate states follow the documented contract.
    if len(annotations.sample_indices) != len(annotations.symbols):
        raise AnnotationMappingError("annotation samples and symbols must have equal lengths")
    target_by_symbol = {symbol: rule for rule in config.targets for symbol in rule.symbols}
    excluded = set(config.excluded_symbols)
    unknown = sorted(set(annotations.symbols) - set(target_by_symbol) - excluded)
    # Evaluate `unknown` explicitly so invalid or alternate states follow the documented contract.
    if unknown:
        raise AnnotationMappingError(f"unmapped annotation symbols: {', '.join(unknown)}")

    included_samples: list[int] = []
    included_symbols: list[str] = []
    target_values: list[int] = []
    target_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()

    # Iterate over `zip(annotations.sample_indices, annotations.symbols, strict=True)` one item at a
    # time so ordering, validation, and failure attribution remain explicit.
    for sample_index, symbol in zip(annotations.sample_indices, annotations.symbols, strict=True):
        rule = target_by_symbol.get(symbol)
        # Evaluate `rule is None` explicitly so invalid or alternate states follow the documented
        # contract.
        if rule is None:
            excluded_counts[symbol] += 1
            continue
        included_samples.append(int(sample_index))
        included_symbols.append(symbol)
        target_values.append(rule.value)
        target_counts[rule.name] += 1

    sample_array = np.asarray(included_samples, dtype=np.int64)
    target_array = np.asarray(target_values, dtype=np.int64)
    sample_array.setflags(write=False)
    target_array.setflags(write=False)
    mapped = MappedAnnotationSet(
        record_id=annotations.record_id,
        sample_indices=sample_array,
        source_symbols=tuple(included_symbols),
        target_values=target_array,
    )
    report = AnnotationMappingReport(
        schema_version=1,
        mapping_name=config.name,
        mapping_version=config.version,
        record_id=annotations.record_id,
        input_annotation_count=len(annotations.symbols),
        included_annotation_count=len(included_symbols),
        excluded_annotation_count=sum(excluded_counts.values()),
        source_symbol_counts=dict(sorted(Counter(annotations.symbols).items())),
        target_counts={rule.name: target_counts[rule.name] for rule in config.targets},
        excluded_symbol_counts=dict(sorted(excluded_counts.items())),
    )
    return AnnotationMappingResult(annotations=mapped, report=report)


def write_mapping_report(report: AnnotationMappingReport, output_path: Path) -> None:
    """Write a mapping audit report to an existing directory."""
    # Evaluate `not output_path.parent.is_dir()` explicitly so invalid or alternate states follow
    # the documented contract.
    if not output_path.parent.is_dir():
        raise AnnotationMappingError(
            f"report parent directory does not exist: {output_path.parent}"
        )
    output_path.write_text(report.to_json(), encoding="utf-8")


def _load_target_rule(value: Any) -> TargetRule:
    """Load target rule according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        value: Candidate value whose contract is being enforced.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `not isinstance(value, dict)` explicitly so invalid or alternate states follow the
    # documented contract.
    if not isinstance(value, dict):
        raise AnnotationMappingError("each target must be a TOML table")
    target_value = value.get("value")
    # Evaluate `not isinstance(target_value, int) or isinstance(target_value, bool)` explicitly so
    # invalid or alternate states follow the documented contract.
    if not isinstance(target_value, int) or isinstance(target_value, bool):
        raise AnnotationMappingError("target.value must be an integer")
    return TargetRule(
        name=_required_string(value, "name"),
        value=target_value,
        symbols=_required_symbols(value, "symbols"),
    )


def _validate_mapping_config(config: AnnotationMappingConfig) -> None:
    """Validate mapping config according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
    """

    # Evaluate `config.unknown_symbol_policy != 'error'` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.unknown_symbol_policy != "error":
        raise AnnotationMappingError("unknown_symbol_policy must be 'error'")
    # Evaluate `not config.targets` explicitly so invalid or alternate states follow the documented
    # contract.
    if not config.targets:
        raise AnnotationMappingError("annotation mapping must define at least one target")
    names = [rule.name for rule in config.targets]
    values = [rule.value for rule in config.targets]
    # Evaluate `len(set(names)) != len(names) or len(set(values)) != len(values)` explicitly so
    # invalid or alternate states follow the documented contract.
    if len(set(names)) != len(names) or len(set(values)) != len(values):
        raise AnnotationMappingError("target names and values must be unique")

    classified_symbols = [symbol for rule in config.targets for symbol in rule.symbols]
    all_symbols = classified_symbols + list(config.excluded_symbols)
    # Evaluate `len(set(all_symbols)) != len(all_symbols)` explicitly so invalid or alternate states
    # follow the documented contract.
    if len(set(all_symbols)) != len(all_symbols):
        raise AnnotationMappingError("source symbols must occur in exactly one target or exclusion")


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
        raise AnnotationMappingError(f"{key} must be a non-empty string")
    return value.strip()


def _required_symbols(values: dict[str, Any], key: str) -> tuple[str, ...]:
    """Compute and return required symbols for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, list) or not value or (not all((isinstance(item, str) for item
    # in value)))` explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise AnnotationMappingError(f"{key} must be a non-empty string array")
    symbols = tuple(value)
    # Evaluate `any((not symbol for symbol in symbols)) or len(set(symbols)) != len(symbols)`
    # explicitly so invalid or alternate states follow the documented contract.
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise AnnotationMappingError(f"{key} must contain unique, non-empty symbols")
    return symbols
