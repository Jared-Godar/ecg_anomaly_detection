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
    channel_index: int
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
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WindowExtractionError(f"could not load window config {path}: {error}") from error

    if document.get("schema_version") != 1 or not isinstance(document.get("window"), dict):
        raise WindowExtractionError(
            "window config must use schema_version = 1 and a [window] table"
        )
    window = document["window"]
    config = WindowConfig(
        schema_version=1,
        name=_required_string(window, "name"),
        version=_required_string(window, "version"),
        pre_seconds=_required_positive_number(window, "pre_seconds"),
        post_seconds=_required_positive_number(window, "post_seconds"),
        channel_index=_required_nonnegative_int(window, "channel_index"),
        boundary_policy=_required_string(window, "boundary_policy"),
    )
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
    if signal.record_id != mapped.record_id or signal.record_id != mapping_result.report.record_id:
        raise WindowExtractionError("signal and mapped annotation record IDs do not match")
    if (
        mapping_result.report.mapping_name != mapping_config.name
        or mapping_result.report.mapping_version != mapping_config.version
    ):
        raise WindowExtractionError("mapping result identity does not match mapping configuration")
    if len(mapped.sample_indices) != len(mapped.source_symbols) or len(
        mapped.sample_indices
    ) != len(mapped.target_values):
        raise WindowExtractionError("mapped annotation arrays must have equal lengths")
    if mapped.sample_indices.size and np.any(np.diff(mapped.sample_indices) < 0):
        raise WindowExtractionError("mapped annotation sample indices must be ordered")
    if signal.signals.ndim != 2:
        raise WindowExtractionError("signals must use a samples-by-channels array")
    if config.channel_index >= signal.signals.shape[1]:
        raise WindowExtractionError(
            f"channel index {config.channel_index} exceeds signal width {signal.signals.shape[1]}"
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

    for center, symbol, target in zip(
        mapped.sample_indices,
        mapped.source_symbols,
        mapped.target_values,
        strict=True,
    ):
        left = int(center) - pre_samples
        right = int(center) + post_samples
        if left < 0:
            left_exclusions += 1
            continue
        if right > signal.signals.shape[0]:
            right_exclusions += 1
            continue
        window = np.asarray(
            signal.signals[left:right, config.channel_index],
            dtype=np.float64,
        ).copy()
        if window.shape != (window_samples,):
            raise WindowExtractionError("extracted window shape violated the configured contract")
        accepted_windows.append(window)
        record_ids.append(signal.record_id)
        center_indices.append(int(center))
        source_symbols.append(symbol)
        target_values.append(int(target))
        accepted_intervals.append((left, right))

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
        channel_index=config.channel_index,
        channel_name=signal.channel_names[config.channel_index],
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
        channel_index=config.channel_index,
        channel_name=signal.channel_names[config.channel_index],
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
        channel_index=np.asarray(window_set.channel_index, dtype=np.int64),
        channel_name=np.asarray(window_set.channel_name, dtype=np.str_),
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
    exact_samples = seconds * sample_rate_hz
    rounded_samples = round(exact_samples)
    if not np.isclose(exact_samples, rounded_samples) or rounded_samples <= 0:
        raise WindowExtractionError(
            f"{field} must resolve to a positive whole sample count; got {exact_samples}"
        )
    return rounded_samples


def _validate_output_path(output_path: Path, suffix: str, description: str) -> None:
    if output_path.suffix != suffix:
        raise WindowExtractionError(f"{description} must use the {suffix} extension")
    if not output_path.parent.is_dir():
        raise WindowExtractionError(
            f"{description} parent directory does not exist: {output_path.parent}"
        )


def _required_string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WindowExtractionError(f"window.{key} must be a non-empty string")
    return value.strip()


def _required_positive_number(values: dict[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise WindowExtractionError(f"window.{key} must be a positive number")
    return float(value)


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WindowExtractionError(f"window.{key} must be a nonnegative integer")
    return value
