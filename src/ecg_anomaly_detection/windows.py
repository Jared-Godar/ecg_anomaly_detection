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
    # Translate a missing, unreadable, or malformed-TOML file into WindowExtractionError.
    try:
        # The `with` block ensures the file handle closes even if tomllib.load raises.
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise WindowExtractionError(f"could not load window config {path}: {error}") from error

    # schema_version pins this loader's understanding of the [window] table's shape.
    if document.get("schema_version") != 1 or not isinstance(document.get("window"), dict):
        raise WindowExtractionError(
            "window config must use schema_version = 1 and a [window] table"
        )
    window = document["window"]
    has_channel_index = "channel_index" in window
    has_channel_name = "channel_name" in window
    # Channel selection must be unambiguous: neither field present, or both present,
    # would leave _resolve_channel with no single source of truth for which channel
    # to extract -- exactly one selector is required, matching this module's
    # documented "channel identity resolved by name or index, never guessed" contract.
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
    # "exclude" is the only implemented boundary policy: windows that would run past a
    # record's edge are dropped rather than padded or clipped (see extract_windows'
    # left/right boundary checks below); a config naming any other policy would
    # otherwise silently be ignored rather than actually changing behavior.
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
    # All three record IDs (the raw signal, its mapped annotations, and the mapping's
    # own report) must agree, or this call is mixing inputs from different records.
    if signal.record_id != mapped.record_id or signal.record_id != mapping_result.report.record_id:
        raise WindowExtractionError("signal and mapped annotation record IDs do not match")
    # The mapping result must have been produced by the exact mapping config passed
    # in here, not merely a same-named one -- a name/version mismatch would mean the
    # target labels below don't correspond to this call's mapping_config.targets.
    if (
        mapping_result.report.mapping_name != mapping_config.name
        or mapping_result.report.mapping_version != mapping_config.version
    ):
        raise WindowExtractionError("mapping result identity does not match mapping configuration")
    # sample_indices, source_symbols, and target_values are parallel arrays (one entry
    # per mapped annotation); an unequal length means mapping_result was corrupted or
    # constructed some other way than the supported annotation-mapping stage.
    if len(mapped.sample_indices) != len(mapped.source_symbols) or len(
        mapped.sample_indices
    ) != len(mapped.target_values):
        raise WindowExtractionError("mapped annotation arrays must have equal lengths")
    # Center indices are expected in ascending sample order (matching the source
    # annotation file's own ordering); the overlap count computed near the end of this
    # function assumes this ordering to detect adjacent overlapping windows correctly.
    if mapped.sample_indices.size and np.any(np.diff(mapped.sample_indices) < 0):
        raise WindowExtractionError("mapped annotation sample indices must be ordered")
    # A samples-by-channels 2-D array is required so channel indexing below
    # (signal.signals[left:right, resolved_channel_index]) selects one channel's
    # values, not an arbitrary slice of a differently-shaped array.
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

    # Attempt to extract one window per mapped annotation, in the annotations' own
    # order, so the emitted rows' order is deterministic and traceable back to the
    # source annotation sequence.
    for center, symbol, target in zip(
        mapped.sample_indices,
        mapped.source_symbols,
        mapped.target_values,
        strict=True,
    ):
        left = int(center) - pre_samples
        right = int(center) + post_samples
        # This is the boundary-safe extraction guarantee this module is named for: a
        # window that would start before sample 0 is dropped (never padded or
        # clipped) rather than silently returning a shorter-than-configured window.
        if left < 0:
            left_exclusions += 1
            continue
        # Same boundary-safety guarantee at the record's right edge.
        if right > signal.signals.shape[0]:
            right_exclusions += 1
            continue
        window = np.asarray(
            signal.signals[left:right, resolved_channel_index],
            dtype=np.float64,
        ).copy()
        # A shape mismatch here would mean the slice above somehow returned a
        # different length than pre_samples + post_samples, which should be
        # unreachable given the boundary checks above -- kept as a fail-closed
        # assertion rather than silently emitting a malformed window.
        if window.shape != (window_samples,):
            raise WindowExtractionError("extracted window shape violated the configured contract")
        accepted_windows.append(window)
        record_ids.append(signal.record_id)
        center_indices.append(int(center))
        source_symbols.append(symbol)
        target_values.append(int(target))
        accepted_intervals.append((left, right))

    # np.stack requires at least one array; a record with zero accepted windows (e.g.
    # every annotation excluded by boundary checks) instead gets an explicit empty
    # array with the correct column width, so downstream concatenation still works.
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
    # Every emitted target value must be nameable by the mapping config; a value this
    # config doesn't know about would mean the mapping stage and this extraction stage
    # were run with mismatched mapping configs, or the closed-world label set drifted.
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
    """Convert a configured window half-width in seconds to a whole sample count.

    Requires the conversion to land (within floating-point tolerance) on a whole
    number, rather than silently rounding an arbitrary fractional value -- a config
    author should choose pre/post_seconds values that align with the record's actual
    sample rate, and a value that doesn't is more likely a mistake than intent.

    Args:
        seconds: The configured half-width in seconds (pre_seconds or post_seconds).
        sample_rate_hz: The record's sample rate.
        field: Which config field this came from, used in the error message.

    Returns:
        The whole sample count.
    """

    exact_samples = seconds * sample_rate_hz
    rounded_samples = round(exact_samples)
    # np.isclose tolerates floating-point rounding noise in the multiplication above,
    # while still rejecting a seconds value that genuinely doesn't align to a whole
    # sample count; rounded_samples <= 0 additionally rejects a zero-width half-window.
    if not np.isclose(exact_samples, rounded_samples) or rounded_samples <= 0:
        raise WindowExtractionError(
            f"{field} must resolve to a positive whole sample count; got {exact_samples}"
        )
    return rounded_samples


def _validate_output_path(output_path: Path, suffix: str, description: str) -> None:
    """Confirm an output path has the expected extension and an existing parent directory.

    Shared by both write_window_artifact (.npz) and write_window_report (.json).

    Args:
        output_path: The candidate output path to validate.
        suffix: The required file extension (including the leading dot).
        description: Human-readable label for this output, used in error messages.
    """

    # Enforce the extension so downstream tooling that globs for *.npz/*.json can rely
    # on finding these files without a separate content sniff.
    if output_path.suffix != suffix:
        raise WindowExtractionError(f"{description} must use the {suffix} extension")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from the underlying writer.
    if not output_path.parent.is_dir():
        raise WindowExtractionError(
            f"{description} parent directory does not exist: {output_path.parent}"
        )


def _resolve_channel(config: WindowConfig, signal: SignalRecord) -> tuple[str, int, str]:
    """Resolve the configured channel selector to a concrete index and name.

    This is the enforcement point for "channel identity is resolved by name, not
    position" (see docs/window-extraction.md): when channel_name is configured, this
    looks the name up in the record's own channel list rather than trusting any
    externally assumed position, so records whose lead order differs from the
    configured index still extract the correct physical channel.

    Args:
        config: Window config supplying exactly one of channel_name/channel_index
            (already enforced as mutually exclusive by load_window_config).
        signal: The record whose channel_names/signal width this selector resolves against.

    Returns:
        The selector method used ("channel_name" or "channel_index"), the resolved
        integer column index, and the resolved channel's name.
    """

    # channel_name takes priority when present (load_window_config already guarantees
    # exactly one of the two is set, so this is not an ambiguous choice).
    if config.channel_name is not None:
        # list.index raises ValueError when the name isn't present; translate that into
        # a message naming the record and listing what channels actually are available,
        # since "not in list" alone wouldn't tell a reviewer what to fix.
        try:
            channel_index = signal.channel_names.index(config.channel_name)
        except ValueError as error:
            raise WindowExtractionError(
                f'Configured channel_name = "{config.channel_name}" was not available '
                f"for record {signal.record_id}. available channels: {list(signal.channel_names)}"
            ) from error
        return "channel_name", channel_index, signal.channel_names[channel_index]

    # Reaching here means channel_name was None, so channel_index must be set --
    # load_window_config's mutual-exclusivity check guarantees this in practice, but
    # this function doesn't assume it without checking, since it may be called from
    # test fixtures that construct a WindowConfig directly.
    if config.channel_index is None:
        raise WindowExtractionError(
            "window config must provide exactly one channel selector: channel_name or channel_index"
        )
    # An out-of-range configured index would otherwise raise a bare numpy IndexError
    # deep inside the window-slicing loop in extract_windows.
    if config.channel_index >= signal.signals.shape[1]:
        raise WindowExtractionError(
            f"channel index {config.channel_index} exceeds signal width {signal.signals.shape[1]}"
        )
    return "channel_index", config.channel_index, signal.channel_names[config.channel_index]


def _optional_string(values: dict[str, Any], key: str) -> str | None:
    """Read an optional non-empty string field, returning None if the key is absent.

    Used for channel_name, which is legitimately absent when channel_index is
    configured instead (load_window_config enforces exactly one is present).

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to look up.

    Returns:
        The field's stripped value, or None if the key is absent.
    """

    # Absence is a valid, expected state for a mutually-exclusive optional field.
    if key not in values:
        return None
    value = values[key]
    # Reject a present-but-wrong-typed or whitespace-only value.
    if not isinstance(value, str) or not value.strip():
        raise WindowExtractionError(f"window.{key} must be a non-empty string")
    return value.strip()


def _optional_nonnegative_int(values: dict[str, Any], key: str) -> int | None:
    """Read an optional nonnegative integer field, returning None if the key is absent.

    Used for channel_index, which is legitimately absent when channel_name is
    configured instead.

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to look up.

    Returns:
        The field's integer value, or None if the key is absent.
    """

    # Absence is a valid, expected state for a mutually-exclusive optional field.
    if key not in values:
        return None
    return _required_nonnegative_int(values, key)


def _optional_unique_strings(values: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read an optional list of unique non-empty strings, defaulting to empty if absent.

    Used for exclude_record_ids, an opt-in denylist of records this config's extraction
    should skip entirely (e.g. records with known data-quality issues).

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to look up.

    Returns:
        The validated, stripped strings as a tuple; empty if the key is absent.
    """

    # Absence means "no exclusions," a valid and common default.
    if key not in values:
        return ()
    raw_values = values[key]
    # Must be a list before individual entries can be validated below.
    if not isinstance(raw_values, list):
        raise WindowExtractionError(f"window.{key} must be a list of non-empty strings")
    parsed: list[str] = []
    # Validate and strip each entry individually so a malformed entry anywhere in the
    # list is caught, not just the first.
    for value in raw_values:
        # Reject a wrong-typed or whitespace-only entry.
        if not isinstance(value, str) or not value.strip():
            raise WindowExtractionError(f"window.{key} must be a list of non-empty strings")
        parsed.append(value.strip())
    # A duplicated record ID would be redundant at best and, if the config author
    # intended two different entries, silently drop one -- reject it explicitly instead.
    if len(set(parsed)) != len(parsed):
        raise WindowExtractionError(f"window.{key} must not contain duplicates")
    return tuple(parsed)


def _required_string(values: dict[str, Any], key: str) -> str:
    """Require and return a non-empty string from the requested `[window]` field.

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value with surrounding whitespace stripped.
    """

    value = values.get(key)
    # Reject a missing/wrong-typed value and a whitespace-only placeholder alike.
    if not isinstance(value, str) or not value.strip():
        raise WindowExtractionError(f"window.{key} must be a non-empty string")
    return value.strip()


def _required_positive_number(values: dict[str, Any], key: str) -> float:
    """Require and return a strictly positive number from the requested `[window]` field.

    Used for pre_seconds/post_seconds; zero or negative would produce a zero-or-negative
    window half-width, which _seconds_to_samples' own positivity check would also catch,
    but rejecting it here gives a more specific, field-scoped error message.

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to extract.

    Returns:
        The field's value as a float.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise WindowExtractionError(f"window.{key} must be a positive number")
    return float(value)


def _required_nonnegative_int(values: dict[str, Any], key: str) -> int:
    """Require and return a nonnegative integer from the requested `[window]` field.

    Args:
        values: The parsed `[window]` table to read from.
        key: The field name to extract.

    Returns:
        The field's integer value.
    """

    value = values.get(key)
    # bool is an int subclass in Python, so it's excluded explicitly.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WindowExtractionError(f"window.{key} must be a nonnegative integer")
    return value
