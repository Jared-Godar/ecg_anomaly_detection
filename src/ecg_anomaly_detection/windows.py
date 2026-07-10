"""Versioned, boundary-safe beat-window extraction."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ecg_anomaly_detection.labels import AnnotationMappingConfig, AnnotationMappingResult
from ecg_anomaly_detection.records import FloatArray, IntegerArray, SignalRecord


class WindowExtractionError(ValueError):
    """Raised when window configuration or extraction violates its contract."""


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Versioned extraction geometry and channel-selection policy."""

    schema_version: int
    name: str
    version: str
    pre_seconds: float
    post_seconds: float
    channel_index: int | None
    channel_name: str | None
    exclude_record_ids: tuple[str, ...]
    boundary_policy: str


@dataclass(frozen=True, slots=True)
class BeatWindowSet:
    """Extracted single-channel windows with row-level source identity."""

    windows: FloatArray
    record_ids: tuple[str, ...]
    center_sample_indices: IntegerArray
    source_symbols: tuple[str, ...]
    target_values: IntegerArray
    sample_rate_hz: float
    channel_selector: str
    configured_channel_index: int | None
    configured_channel_name: str | None
    channel_index: int
    channel_name: str
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str


@dataclass(frozen=True, slots=True)
class WindowExtractionReport:
    """Machine-readable extraction counts and boundary decisions."""

    schema_version: int
    record_id: str
    mapping_name: str
    mapping_version: str
    window_config_name: str
    window_config_version: str
    sample_rate_hz: float
    channel_selector: str
    configured_channel_index: int | None
    configured_channel_name: str | None
    channel_index: int
    channel_name: str
    pre_samples: int
    post_samples: int
    window_samples: int
    input_mapped_annotation_count: int
    emitted_window_count: int
    left_boundary_exclusion_count: int
    right_boundary_exclusion_count: int
    overlapping_adjacent_window_count: int
    emitted_target_counts: dict[str, int]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class WindowExtractionResult:
    """Extracted windows and their audit report."""

    window_set: BeatWindowSet
    report: WindowExtractionReport


def load_window_config(path: Path) -> WindowConfig:
    """Load and validate versioned window configuration from TOML."""
    # Attempt this boundary operation here so (OSError, tomllib.TOMLDecodeError) can be translated
    # or cleaned up under the repository contract.
    try:
        # Scope `path.open('rb')` here so resource cleanup occurs on both success and failure paths.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WindowExtractionError(f"could not load window config {path}: {error}") from error

    # Evaluate `document.get('schema_version') != 1 or not isinstance(document.get('window'), dict)`
    # explicitly so invalid or alternate states follow the documented contract.
    if document.get("schema_version") != 1 or not isinstance(document.get("window"), dict):
        raise WindowExtractionError(
            "window config must use schema_version = 1 and a [window] table"
        )
    window = document["window"]
    has_channel_index = "channel_index" in window
    has_channel_name = "channel_name" in window
    # Evaluate `has_channel_index == has_channel_name` explicitly so invalid or alternate states
    # follow the documented contract.
    if has_channel_index == has_channel_name:
        raise WindowExtractionError(
            "window config must provide exactly one channel selector: channel_name or channel_index"
        )
    config = WindowConfig(
        schema_version=1,
        name=_required_string(window, "name"),
        version=_required_string(window, "version"),
        pre_seconds=_required_positive_number(window, "pre_seconds"),
        post_seconds=_required_positive_number(window, "post_seconds"),
        channel_index=_optional_nonnegative_int(window, "channel_index"),
        channel_name=_optional_string(window, "channel_name"),
        exclude_record_ids=_optional_unique_strings(window, "exclude_record_ids"),
        boundary_policy=_required_string(window, "boundary_policy"),
    )
    # Evaluate `config.boundary_policy != 'exclude'` explicitly so invalid or alternate states
    # follow the documented contract.
    if config.boundary_policy != "exclude":
        raise WindowExtractionError("boundary_policy must be 'exclude'")
    return config


def extract_windows(
    config: WindowConfig,
    mapping_config: AnnotationMappingConfig,
    signal: SignalRecord,
    mapping_result: AnnotationMappingResult,
) -> WindowExtractionResult:
    """Extract complete windows and explicitly report boundary exclusions."""
    mapped = mapping_result.annotations
    # Evaluate `signal.record_id != mapped.record_id or signal.record_id !=
    # mapping_result.report.record_id` explicitly so invalid or alternate states follow the
    # documented contract.
    if signal.record_id != mapped.record_id or signal.record_id != mapping_result.report.record_id:
        raise WindowExtractionError("signal and mapped annotation record IDs do not match")
    # Evaluate `mapping_result.report.mapping_name != mapping_config.name or
    # mapping_result.report.mapping_version != mapping_config....` explicitly so invalid or
    # alternate states follow the documented contract.
    if (
        mapping_result.report.mapping_name != mapping_config.name
        or mapping_result.report.mapping_version != mapping_config.version
    ):
        raise WindowExtractionError("mapping result identity does not match mapping configuration")
    # Evaluate `len(mapped.sample_indices) != len(mapped.source_symbols) or
    # len(mapped.sample_indices) != len(mapped.target_values)` explicitly so invalid or alternate
    # states follow the documented contract.
    if len(mapped.sample_indices) != len(mapped.source_symbols) or len(
        mapped.sample_indices
    ) != len(mapped.target_values):
        raise WindowExtractionError("mapped annotation arrays must have equal lengths")
    # Evaluate `mapped.sample_indices.size and np.any(np.diff(mapped.sample_indices) < 0)`
    # explicitly so invalid or alternate states follow the documented contract.
    if mapped.sample_indices.size and np.any(np.diff(mapped.sample_indices) < 0):
        raise WindowExtractionError("mapped annotation sample indices must be ordered")
    # Evaluate `signal.signals.ndim != 2` explicitly so invalid or alternate states follow the
    # documented contract.
    if signal.signals.ndim != 2:
        raise WindowExtractionError("signals must use a samples-by-channels array")
    channel_selector, resolved_channel_index, resolved_channel_name = _resolve_channel(
        config, signal
    )

    pre_samples = _seconds_to_samples(config.pre_seconds, signal.sample_rate_hz, "pre_seconds")
    post_samples = _seconds_to_samples(config.post_seconds, signal.sample_rate_hz, "post_seconds")
    window_samples = pre_samples + post_samples
    accepted_windows: list[FloatArray] = []
    record_ids: list[str] = []
    center_indices: list[int] = []
    source_symbols: list[str] = []
    target_values: list[int] = []
    accepted_intervals: list[tuple[int, int]] = []
    left_exclusions = 0
    right_exclusions = 0

    # Iterate over `zip(mapped.sample_indices, mapped.source_symbols, mapped.target_values,
    # strict=True)` one item at a time so ordering, validation, and failure attribution remain
    # explicit.
    for center, symbol, target in zip(
        mapped.sample_indices,
        mapped.source_symbols,
        mapped.target_values,
        strict=True,
    ):
        left = int(center) - pre_samples
        right = int(center) + post_samples
        # Evaluate `left < 0` explicitly so invalid or alternate states follow the documented
        # contract.
        if left < 0:
            left_exclusions += 1
            continue
        # Evaluate `right > signal.signals.shape[0]` explicitly so invalid or alternate states
        # follow the documented contract.
        if right > signal.signals.shape[0]:
            right_exclusions += 1
            continue
        window = np.asarray(
            signal.signals[left:right, resolved_channel_index],
            dtype=np.float64,
        ).copy()
        # Evaluate `window.shape != (window_samples,)` explicitly so invalid or alternate states
        # follow the documented contract.
        if window.shape != (window_samples,):
            raise WindowExtractionError("extracted window shape violated the configured contract")
        accepted_windows.append(window)
        record_ids.append(signal.record_id)
        center_indices.append(int(center))
        source_symbols.append(symbol)
        target_values.append(int(target))
        accepted_intervals.append((left, right))

    # Evaluate `accepted_windows` explicitly so invalid or alternate states follow the documented
    # contract.
    if accepted_windows:
        window_array = np.stack(accepted_windows)
    else:
        window_array = np.empty((0, window_samples), dtype=np.float64)
    center_array = np.asarray(center_indices, dtype=np.int64)
    target_array = np.asarray(target_values, dtype=np.int64)
    window_array.setflags(write=False)
    center_array.setflags(write=False)
    target_array.setflags(write=False)

    target_name_by_value = {rule.value: rule.name for rule in mapping_config.targets}
    unknown_target_values = sorted(set(target_values) - set(target_name_by_value))
    # Evaluate `unknown_target_values` explicitly so invalid or alternate states follow the
    # documented contract.
    if unknown_target_values:
        raise WindowExtractionError(f"unconfigured target values: {unknown_target_values}")
    emitted_target_counts = Counter(target_name_by_value[value] for value in target_values)
    overlap_count = sum(
        current[0] < previous[1]
        for previous, current in zip(accepted_intervals, accepted_intervals[1:], strict=False)
    )
    window_set = BeatWindowSet(
        windows=window_array,
        record_ids=tuple(record_ids),
        center_sample_indices=center_array,
        source_symbols=tuple(source_symbols),
        target_values=target_array,
        sample_rate_hz=signal.sample_rate_hz,
        channel_selector=channel_selector,
        configured_channel_index=config.channel_index,
        configured_channel_name=config.channel_name,
        channel_index=resolved_channel_index,
        channel_name=resolved_channel_name,
        mapping_name=mapping_config.name,
        mapping_version=mapping_config.version,
        window_config_name=config.name,
        window_config_version=config.version,
    )
    report = WindowExtractionReport(
        schema_version=1,
        record_id=signal.record_id,
        mapping_name=mapping_config.name,
        mapping_version=mapping_config.version,
        window_config_name=config.name,
        window_config_version=config.version,
        sample_rate_hz=signal.sample_rate_hz,
        channel_selector=channel_selector,
        configured_channel_index=config.channel_index,
        configured_channel_name=config.channel_name,
        channel_index=resolved_channel_index,
        channel_name=resolved_channel_name,
        pre_samples=pre_samples,
        post_samples=post_samples,
        window_samples=window_samples,
        input_mapped_annotation_count=len(mapped.source_symbols),
        emitted_window_count=len(accepted_windows),
        left_boundary_exclusion_count=left_exclusions,
        right_boundary_exclusion_count=right_exclusions,
        overlapping_adjacent_window_count=overlap_count,
        emitted_target_counts={
            rule.name: emitted_target_counts[rule.name] for rule in mapping_config.targets
        },
    )
    return WindowExtractionResult(window_set=window_set, report=report)


def write_window_artifact(window_set: BeatWindowSet, output_path: Path) -> None:
    """Write a non-pickle NPZ artifact with row-level lineage fields."""
    _validate_output_path(output_path, ".npz", "window artifact")
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        windows=window_set.windows,
        record_ids=np.asarray(window_set.record_ids, dtype=np.str_),
        center_sample_indices=window_set.center_sample_indices,
        source_symbols=np.asarray(window_set.source_symbols, dtype=np.str_),
        target_values=window_set.target_values,
        sample_rate_hz=np.asarray(window_set.sample_rate_hz, dtype=np.float64),
        channel_selector=np.asarray(window_set.channel_selector, dtype=np.str_),
        configured_channel_index=np.asarray(
            -1
            if window_set.configured_channel_index is None
            else window_set.configured_channel_index,
            dtype=np.int64,
        ),
        configured_channel_name=np.asarray(
            ""
            if window_set.configured_channel_name is None
            else window_set.configured_channel_name,
            dtype=np.str_,
        ),
        channel_index=np.asarray(window_set.channel_index, dtype=np.int64),
        channel_name=np.asarray(window_set.channel_name, dtype=np.str_),
        resolved_channel_index=np.asarray(window_set.channel_index, dtype=np.int64),
        resolved_channel_name=np.asarray(window_set.channel_name, dtype=np.str_),
        mapping_name=np.asarray(window_set.mapping_name, dtype=np.str_),
        mapping_version=np.asarray(window_set.mapping_version, dtype=np.str_),
        window_config_name=np.asarray(window_set.window_config_name, dtype=np.str_),
        window_config_version=np.asarray(window_set.window_config_version, dtype=np.str_),
    )


def write_window_report(report: WindowExtractionReport, output_path: Path) -> None:
    """Write a JSON extraction report to an existing directory."""
    _validate_output_path(output_path, ".json", "window report")
    output_path.write_text(report.to_json(), encoding="utf-8")


def _seconds_to_samples(seconds: float, sample_rate_hz: float, field: str) -> int:
    """Convert seconds to samples for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        seconds: The seconds value supplied by the caller or surrounding test fixture.
        sample_rate_hz: The sample rate hz value supplied by the caller or surrounding test fixture.
        field: The field value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    exact_samples = seconds * sample_rate_hz
    rounded_samples = round(exact_samples)
    # Evaluate `not np.isclose(exact_samples, rounded_samples) or rounded_samples <= 0` explicitly
    # so invalid or alternate states follow the documented contract.
    if not np.isclose(exact_samples, rounded_samples) or rounded_samples <= 0:
        raise WindowExtractionError(
            f"{field} must resolve to a positive whole sample count; got {exact_samples}"
        )
    return rounded_samples


def _validate_output_path(output_path: Path, suffix: str, description: str) -> None:
    """Validate output path according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        output_path: The output path value supplied by the caller or surrounding test fixture.
        suffix: The suffix value supplied by the caller or surrounding test fixture.
        description: The description value supplied by the caller or surrounding test fixture.
    """

    # Evaluate `output_path.suffix != suffix` explicitly so invalid or alternate states follow the
    # documented contract.
    if output_path.suffix != suffix:
        raise WindowExtractionError(f"{description} must use the {suffix} extension")
    # Evaluate `not output_path.parent.is_dir()` explicitly so invalid or alternate states follow
    # the documented contract.
    if not output_path.parent.is_dir():
        raise WindowExtractionError(
            f"{description} parent directory does not exist: {output_path.parent}"
        )


def _resolve_channel(config: WindowConfig, signal: SignalRecord) -> tuple[str, int, str]:
    """Resolve channel according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
        signal: The signal value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `config.channel_name is not None` explicitly so invalid or alternate states follow
    # the documented contract.
    if config.channel_name is not None:
        # Attempt this boundary operation here so ValueError can be translated or cleaned up under
        # the repository contract.
        try:
            channel_index = signal.channel_names.index(config.channel_name)
        except ValueError as error:
            raise WindowExtractionError(
                f'Configured channel_name = "{config.channel_name}" was not available '
                f"for record {signal.record_id}. available channels: {list(signal.channel_names)}"
            ) from error
        return "channel_name", channel_index, signal.channel_names[channel_index]

    # Evaluate `config.channel_index is None` explicitly so invalid or alternate states follow the
    # documented contract.
    if config.channel_index is None:
        raise WindowExtractionError(
            "window config must provide exactly one channel selector: channel_name or channel_index"
        )
    # Evaluate `config.channel_index >= signal.signals.shape[1]` explicitly so invalid or alternate
    # states follow the documented contract.
    if config.channel_index >= signal.signals.shape[1]:
        raise WindowExtractionError(
            f"channel index {config.channel_index} exceeds signal width {signal.signals.shape[1]}"
        )
    return "channel_index", config.channel_index, signal.channel_names[config.channel_index]


def _optional_string(values: dict[str, Any], key: str) -> str | None:
    """Read optional string for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `key not in values` explicitly so invalid or alternate states follow the documented
    # contract.
    if key not in values:
        return None
    value = values[key]
    # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or alternate
    # states follow the documented contract.
    if not isinstance(value, str) or not value.strip():
        raise WindowExtractionError(f"window.{key} must be a non-empty string")
    return value.strip()


def _optional_nonnegative_int(values: dict[str, Any], key: str) -> int | None:
    """Read optional nonnegative int for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `key not in values` explicitly so invalid or alternate states follow the documented
    # contract.
    if key not in values:
        return None
    return _required_nonnegative_int(values, key)


def _optional_unique_strings(values: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read optional unique strings for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `key not in values` explicitly so invalid or alternate states follow the documented
    # contract.
    if key not in values:
        return ()
    raw_values = values[key]
    # Evaluate `not isinstance(raw_values, list)` explicitly so invalid or alternate states follow
    # the documented contract.
    if not isinstance(raw_values, list):
        raise WindowExtractionError(f"window.{key} must be a list of non-empty strings")
    parsed: list[str] = []
    # Iterate over `raw_values` one item at a time so ordering, validation, and failure attribution
    # remain explicit.
    for value in raw_values:
        # Evaluate `not isinstance(value, str) or not value.strip()` explicitly so invalid or
        # alternate states follow the documented contract.
        if not isinstance(value, str) or not value.strip():
            raise WindowExtractionError(f"window.{key} must be a list of non-empty strings")
        parsed.append(value.strip())
    # Evaluate `len(set(parsed)) != len(parsed)` explicitly so invalid or alternate states follow
    # the documented contract.
    if len(set(parsed)) != len(parsed):
        raise WindowExtractionError(f"window.{key} must not contain duplicates")
    return tuple(parsed)


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
        raise WindowExtractionError(f"window.{key} must be a non-empty string")
    return value.strip()


def _required_positive_number(values: dict[str, Any], key: str) -> float:
    """Compute and return required positive number for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        values: Structured values to validate, transform, or serialize.
        key: The key value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    value = values.get(key)
    # Evaluate `not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0`
    # explicitly so invalid or alternate states follow the documented contract.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise WindowExtractionError(f"window.{key} must be a positive number")
    return float(value)


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
        raise WindowExtractionError(f"window.{key} must be a nonnegative integer")
    return value
