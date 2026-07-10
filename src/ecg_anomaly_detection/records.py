"""Typed WFDB record loading and validation contracts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import wfdb

from ecg_anomaly_detection.config import DatasetConfig

# Construct FloatArray once so the module exposes one stable shared definition.
FloatArray = npt.NDArray[np.float64]
# Construct IntegerArray once so the module exposes one stable shared definition.
IntegerArray = npt.NDArray[np.int64]


class RecordValidationError(ValueError):
    """Raised when a WFDB record violates the supported data contract."""


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """Physical signal samples and their source metadata."""

    record_id: str
    sample_rate_hz: float
    signals: FloatArray
    channel_names: tuple[str, ...]
    units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationSet:
    """Reference annotation locations and original symbols."""

    record_id: str
    sample_indices: IntegerArray
    symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedRecord:
    """A signal record and annotation set loaded from the same WFDB base path."""

    signal: SignalRecord
    annotations: AnnotationSet


@dataclass(frozen=True, slots=True)
class RecordValidationReport:
    """Machine-readable summary of one validated record."""

    schema_version: int
    dataset_slug: str
    dataset_version: str
    record_id: str
    sample_rate_hz: float
    sample_count: int
    channel_count: int
    channel_names: tuple[str, ...]
    units: tuple[str, ...]
    annotation_count: int
    annotation_symbol_counts: dict[str, int]
    checks: tuple[str, ...]

    def to_json(self) -> str:
        """Serialize with deterministic keys and formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def load_wfdb_record(
    config: DatasetConfig,
    data_dir: Path,
    record_id: str,
) -> LoadedRecord:
    """Load one local WFDB signal and annotation pair for explicit validation."""
    # Evaluate `record_id not in config.record_ids` explicitly so invalid or alternate states follow
    # the documented contract.
    if record_id not in config.record_ids:
        raise RecordValidationError(f"record ID is not configured for {config.slug}: {record_id}")
    _validate_required_record_files(config, data_dir, record_id)

    record_path = data_dir / record_id
    # Attempt this boundary operation here so (OSError, ValueError, IndexError) can be translated or
    # cleaned up under the repository contract.
    try:
        wfdb_record = wfdb.rdrecord(str(record_path))
        wfdb_annotations = wfdb.rdann(str(record_path), config.annotation_extension)
    except (OSError, ValueError, IndexError) as error:
        raise RecordValidationError(f"could not load WFDB record {record_id}: {error}") from error

    signal = _to_signal_record(record_id, wfdb_record)
    annotations = _to_annotation_set(record_id, wfdb_annotations)
    return LoadedRecord(signal=signal, annotations=annotations)


def validate_record(
    config: DatasetConfig,
    signal: SignalRecord,
    annotations: AnnotationSet,
) -> RecordValidationReport:
    """Validate loaded arrays and return a report when every check passes."""
    # Evaluate `signal.record_id != annotations.record_id` explicitly so invalid or alternate states
    # follow the documented contract.
    if signal.record_id != annotations.record_id:
        raise RecordValidationError("signal and annotation record IDs do not match")
    # Evaluate `signal.record_id not in config.record_ids` explicitly so invalid or alternate states
    # follow the documented contract.
    if signal.record_id not in config.record_ids:
        raise RecordValidationError(
            f"record ID is not configured for {config.slug}: {signal.record_id}"
        )
    # Evaluate `not np.isclose(signal.sample_rate_hz, config.sample_rate_hz)` explicitly so invalid
    # or alternate states follow the documented contract.
    if not np.isclose(signal.sample_rate_hz, config.sample_rate_hz):
        raise RecordValidationError(
            f"sample rate mismatch: expected {config.sample_rate_hz}, got {signal.sample_rate_hz}"
        )
    # Evaluate `signal.signals.ndim != 2 or signal.signals.shape[0] == 0 or signal.signals.shape[1]
    # == 0` explicitly so invalid or alternate states follow the documented contract.
    if signal.signals.ndim != 2 or signal.signals.shape[0] == 0 or signal.signals.shape[1] == 0:
        raise RecordValidationError(
            "physical signals must be a non-empty samples-by-channels array"
        )
    # Evaluate `not np.isfinite(signal.signals).all()` explicitly so invalid or alternate states
    # follow the documented contract.
    if not np.isfinite(signal.signals).all():
        raise RecordValidationError("physical signals contain non-finite values")

    channel_count = signal.signals.shape[1]
    # Evaluate `len(signal.channel_names) != channel_count or any((not name for name in
    # signal.channel_names))` explicitly so invalid or alternate states follow the documented
    # contract.
    if len(signal.channel_names) != channel_count or any(not name for name in signal.channel_names):
        raise RecordValidationError("channel names must be non-empty and match the signal width")
    # Evaluate `len(set(signal.channel_names)) != channel_count` explicitly so invalid or alternate
    # states follow the documented contract.
    if len(set(signal.channel_names)) != channel_count:
        raise RecordValidationError("channel names must be unique")
    # Evaluate `len(signal.units) != channel_count or any((not unit for unit in signal.units))`
    # explicitly so invalid or alternate states follow the documented contract.
    if len(signal.units) != channel_count or any(not unit for unit in signal.units):
        raise RecordValidationError("channel units must be non-empty and match the signal width")

    # Evaluate `annotations.sample_indices.ndim != 1` explicitly so invalid or alternate states
    # follow the documented contract.
    if annotations.sample_indices.ndim != 1:
        raise RecordValidationError("annotation sample indices must be one-dimensional")
    # Evaluate `len(annotations.sample_indices) != len(annotations.symbols)` explicitly so invalid
    # or alternate states follow the documented contract.
    if len(annotations.sample_indices) != len(annotations.symbols):
        raise RecordValidationError("annotation samples and symbols must have equal lengths")
    # Evaluate `any((not symbol for symbol in annotations.symbols))` explicitly so invalid or
    # alternate states follow the documented contract.
    if any(not symbol for symbol in annotations.symbols):
        raise RecordValidationError("annotation symbols must be non-empty")
    # Evaluate `annotations.sample_indices.size` explicitly so invalid or alternate states follow
    # the documented contract.
    if annotations.sample_indices.size:
        # Evaluate `annotations.sample_indices.min() < 0` explicitly so invalid or alternate states
        # follow the documented contract.
        if annotations.sample_indices.min() < 0:
            raise RecordValidationError("annotation sample indices cannot be negative")
        # Evaluate `annotations.sample_indices.max() >= signal.signals.shape[0]` explicitly so
        # invalid or alternate states follow the documented contract.
        if annotations.sample_indices.max() >= signal.signals.shape[0]:
            raise RecordValidationError("annotation sample index exceeds the signal boundary")
        # Evaluate `np.any(np.diff(annotations.sample_indices) < 0)` explicitly so invalid or
        # alternate states follow the documented contract.
        if np.any(np.diff(annotations.sample_indices) < 0):
            raise RecordValidationError("annotation sample indices must be ordered")

    return _build_report(config, signal, annotations)


def write_validation_report(report: RecordValidationReport, output_path: Path) -> None:
    """Write a validation report to an existing output directory."""
    # Evaluate `not output_path.parent.is_dir()` explicitly so invalid or alternate states follow
    # the documented contract.
    if not output_path.parent.is_dir():
        raise RecordValidationError(f"report parent directory does not exist: {output_path.parent}")
    output_path.write_text(report.to_json(), encoding="utf-8")


def _validate_required_record_files(
    config: DatasetConfig,
    data_dir: Path,
    record_id: str,
) -> None:
    """Validate required record files according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
        data_dir: The data dir value supplied by the caller or surrounding test fixture.
        record_id: The record id value supplied by the caller or surrounding test fixture.
    """

    invalid = []
    # Iterate over `config.required_extensions` one item at a time so ordering, validation, and
    # failure attribution remain explicit.
    for extension in config.required_extensions:
        file_path = data_dir / f"{record_id}.{extension}"
        # Evaluate `not file_path.is_file() or file_path.is_symlink()` explicitly so invalid or
        # alternate states follow the documented contract.
        if not file_path.is_file() or file_path.is_symlink():
            invalid.append(file_path.name)
    # Evaluate `invalid` explicitly so invalid or alternate states follow the documented contract.
    if invalid:
        raise RecordValidationError(f"missing or invalid required files: {', '.join(invalid)}")


def _to_signal_record(record_id: str, record: Any) -> SignalRecord:
    """Compute and return to signal record for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        record_id: The record id value supplied by the caller or surrounding test fixture.
        record: The record value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    # Evaluate `record.p_signal is None` explicitly so invalid or alternate states follow the
    # documented contract.
    if record.p_signal is None:
        raise RecordValidationError("WFDB record did not provide physical signal values")
    signals = np.asarray(record.p_signal, dtype=np.float64).copy()
    signals.setflags(write=False)
    return SignalRecord(
        record_id=record_id,
        sample_rate_hz=float(record.fs),
        signals=signals,
        channel_names=tuple(record.sig_name or ()),
        units=tuple(record.units or ()),
    )


def _to_annotation_set(record_id: str, annotations: Any) -> AnnotationSet:
    """Compute and return to annotation set for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        record_id: The record id value supplied by the caller or surrounding test fixture.
        annotations: The annotations value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    sample_indices = np.asarray(annotations.sample, dtype=np.int64).copy()
    sample_indices.setflags(write=False)
    return AnnotationSet(
        record_id=record_id,
        sample_indices=sample_indices,
        symbols=tuple(annotations.symbol),
    )


def _build_report(
    config: DatasetConfig,
    signal: SignalRecord,
    annotations: AnnotationSet,
) -> RecordValidationReport:
    """Build report according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        config: Validated configuration controlling the documented operation.
        signal: The signal value supplied by the caller or surrounding test fixture.
        annotations: The annotations value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    symbol_counts = dict(sorted(Counter(annotations.symbols).items()))
    return RecordValidationReport(
        schema_version=1,
        dataset_slug=config.slug,
        dataset_version=config.version,
        record_id=signal.record_id,
        sample_rate_hz=signal.sample_rate_hz,
        sample_count=signal.signals.shape[0],
        channel_count=signal.signals.shape[1],
        channel_names=signal.channel_names,
        units=signal.units,
        annotation_count=len(annotations.symbols),
        annotation_symbol_counts=symbol_counts,
        checks=(
            "configured_record_id",
            "expected_sample_rate",
            "non_empty_signal_shape",
            "finite_physical_signals",
            "channel_metadata",
            "annotation_shape",
            "annotation_bounds",
            "annotation_order",
        ),
    )
