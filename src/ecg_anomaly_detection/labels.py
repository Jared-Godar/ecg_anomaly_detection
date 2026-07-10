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
    # Translate a missing, unreadable, or malformed-TOML file into AnnotationMappingError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AnnotationMappingError(
            f"could not load annotation mapping {path}: {error}"
        ) from error

    # schema_version pins this loader's understanding of the document's shape.
    if document.get("schema_version") != 1:
        raise AnnotationMappingError("annotation mapping must use schema_version = 1")
    mapping = document.get("mapping")
    targets = document.get("targets")
    exclusions = document.get("exclusions")
    # All three sections are required: [mapping] carries identity/policy fields,
    # [[targets]] is the array of target rules, and [exclusions] names symbols that
    # are deliberately dropped rather than mapped to any target.
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
    # sample_indices and symbols are parallel arrays (one entry per annotation); an
    # unequal length means the annotation set was corrupted or constructed some other
    # way than the supported record-validation stage.
    if len(annotations.sample_indices) != len(annotations.symbols):
        raise AnnotationMappingError("annotation samples and symbols must have equal lengths")
    target_by_symbol = {symbol: rule for rule in config.targets for symbol in rule.symbols}
    excluded = set(config.excluded_symbols)
    unknown = sorted(set(annotations.symbols) - set(target_by_symbol) - excluded)
    # This is the closed-world enforcement this module is documented for: every source
    # symbol must be explicitly classified as either a target or an exclusion. A symbol
    # this mapping has never seen fails loudly rather than silently falling through to
    # either category, which would misrepresent the mapping's actual coverage.
    if unknown:
        raise AnnotationMappingError(f"unmapped annotation symbols: {', '.join(unknown)}")

    included_samples: list[int] = []
    included_symbols: list[str] = []
    target_values: list[int] = []
    target_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()

    # Classify every annotation in its original order, so the resulting mapped set's
    # row order is deterministic and traceable back to the source annotation sequence.
    for sample_index, symbol in zip(annotations.sample_indices, annotations.symbols, strict=True):
        rule = target_by_symbol.get(symbol)
        # A symbol with no matching target rule was already confirmed to be in the
        # excluded_symbols set by the closed-world check above; count and skip it.
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
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from write_text.
    if not output_path.parent.is_dir():
        raise AnnotationMappingError(
            f"report parent directory does not exist: {output_path.parent}"
        )
    output_path.write_text(report.to_json(), encoding="utf-8")


def _load_target_rule(value: Any) -> TargetRule:
    """Parse and validate one `[[targets]]` table into a TargetRule.

    Args:
        value: One raw TOML target-array entry.

    Returns:
        The validated target rule.
    """

    # Every field access below assumes value is a dict.
    if not isinstance(value, dict):
        raise AnnotationMappingError("each target must be a TOML table")
    target_value = value.get("value")
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(target_value, int) or isinstance(target_value, bool):
        raise AnnotationMappingError("target.value must be an integer")
    return TargetRule(
        name=_required_string(value, "name"),
        value=target_value,
        symbols=_required_symbols(value, "symbols"),
    )


def _validate_mapping_config(config: AnnotationMappingConfig) -> None:
    """Cross-check a mapping config's targets and exclusions as a whole.

    Runs after every individual target rule has already been parsed and validated by
    _load_target_rule; this checks the properties that only make sense once every rule
    is known -- uniqueness across rules, and that no symbol is claimed by more than
    one target or exclusion.

    Args:
        config: The mapping config to validate as a whole.
    """

    # This module only implements the "error" policy (raise on any unmapped symbol,
    # enforced in map_annotations); a config naming any other policy would silently be
    # ignored rather than actually changing behavior.
    if config.unknown_symbol_policy != "error":
        raise AnnotationMappingError("unknown_symbol_policy must be 'error'")
    # A mapping with zero targets would classify every annotation as excluded (or
    # unknown), which is never a useful configuration.
    if not config.targets:
        raise AnnotationMappingError("annotation mapping must define at least one target")
    names = [rule.name for rule in config.targets]
    values = [rule.value for rule in config.targets]
    # Duplicate names would make target_counts (keyed by name) silently merge two
    # distinct rules; duplicate values would make two different target names map to
    # the same encoded integer, losing the distinction downstream.
    if len(set(names)) != len(names) or len(set(values)) != len(values):
        raise AnnotationMappingError("target names and values must be unique")

    classified_symbols = [symbol for rule in config.targets for symbol in rule.symbols]
    all_symbols = classified_symbols + list(config.excluded_symbols)
    # A symbol assigned to two targets (or a target and an exclusion) would make its
    # classification in map_annotations depend on dict-insertion order rather than
    # being a well-defined, closed-world mapping.
    if len(set(all_symbols)) != len(all_symbols):
        raise AnnotationMappingError("source symbols must occur in exactly one target or exclusion")


def _required_string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested structured field.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise AnnotationMappingError(f"{key} must be a non-empty string")
    return value.strip()


def _required_symbols(values: dict[str, Any], key: str) -> tuple[str, ...]:
    """Require and return a non-empty array of unique, non-empty annotation symbols.

    Shared by both target.symbols and exclusions.symbols, since both fields have the
    same shape requirement.

    Args:
        values: The parsed table to read from.
        key: The field name to extract.

    Returns:
        The validated symbols.
    """

    value = values.get(key)
    # Reject a missing/empty list or any non-string element before normalization below.
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise AnnotationMappingError(f"{key} must be a non-empty string array")
    symbols = tuple(value)
    # A duplicated or empty symbol within this one list would already violate the
    # uniqueness this mapping depends on, before even cross-checking against other
    # targets/exclusions in _validate_mapping_config.
    if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
        raise AnnotationMappingError(f"{key} must contain unique, non-empty symbols")
    return symbols
