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

# Shared element type for physical signal arrays across the package (float64, matching
# WFDB's own physical-signal precision), so every module agrees on one array alias
# instead of repeating the numpy.typing annotation.
FloatArray = npt.NDArray[np.float64]
# Shared element type for sample-index and label arrays across the package (int64).
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
    # Reject an unconfigured record ID before touching the filesystem, so a typo'd or
    # out-of-scope ID fails with a clear message instead of a generic file-not-found.
    if record_id not in config.record_ids:
        raise RecordValidationError(f"record ID is not configured for {config.slug}: {record_id}")
    _validate_required_record_files(config, data_dir, record_id)

    record_path = data_dir / record_id
    # wfdb.rdrecord/rdann raise a mix of OSError, ValueError, and IndexError depending
    # on what's malformed; collapse them into one RecordValidationError so callers only
    # need to catch this module's own exception type.
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
    # A signal and annotation set from different records would otherwise silently be
    # cross-validated against each other.
    if signal.record_id != annotations.record_id:
        raise RecordValidationError("signal and annotation record IDs do not match")
    # A record could carry local data unrelated to any configured record ID.
    if signal.record_id not in config.record_ids:
        raise RecordValidationError(
            f"record ID is not configured for {config.slug}: {signal.record_id}"
        )
    # np.isclose tolerates floating-point noise in the recorded rate while still
    # catching a genuinely different sample rate than the dataset config declares.
    if not np.isclose(signal.sample_rate_hz, config.sample_rate_hz):
        raise RecordValidationError(
            f"sample rate mismatch: expected {config.sample_rate_hz}, got {signal.sample_rate_hz}"
        )
    # A samples-by-channels 2-D array with both dimensions non-zero is required before
    # any of the shape-dependent checks below (channel_count, annotation bounds) can
    # be trusted.
    if signal.signals.ndim != 2 or signal.signals.shape[0] == 0 or signal.signals.shape[1] == 0:
        raise RecordValidationError(
            "physical signals must be a non-empty samples-by-channels array"
        )
    # A NaN/inf sample would silently corrupt every downstream window extracted from
    # this record.
    if not np.isfinite(signal.signals).all():
        raise RecordValidationError("physical signals contain non-finite values")

    channel_count = signal.signals.shape[1]
    # channel_names must have exactly one non-empty name per physical channel; an empty
    # name would make channel-by-name resolution in windows.py silently unusable.
    if len(signal.channel_names) != channel_count or any(not name for name in signal.channel_names):
        raise RecordValidationError("channel names must be non-empty and match the signal width")
    # Duplicate channel names would make windows.py's name-based channel lookup
    # (`channel_names.index(name)`) ambiguous, always resolving to the first match.
    if len(set(signal.channel_names)) != channel_count:
        raise RecordValidationError("channel names must be unique")
    # Same shape/non-empty requirement as channel_names, for the units array.
    if len(signal.units) != channel_count or any(not unit for unit in signal.units):
        raise RecordValidationError("channel units must be non-empty and match the signal width")

    # The bounds/order checks and downstream window extraction both assume a flat,
    # one-dimensional array of positions.
    if annotations.sample_indices.ndim != 1:
        raise RecordValidationError("annotation sample indices must be one-dimensional")
    # sample_indices and symbols are parallel arrays (one entry per annotation); an
    # unequal length means the WFDB annotation file was corrupted or malformed.
    if len(annotations.sample_indices) != len(annotations.symbols):
        raise RecordValidationError("annotation samples and symbols must have equal lengths")
    # An empty symbol would make target-mapping lookups in labels.py silently unusable.
    if any(not symbol for symbol in annotations.symbols):
        raise RecordValidationError("annotation symbols must be non-empty")
    # The bounds and ordering checks below assume at least one annotation is present;
    # skip them entirely for a record with zero annotations rather than calling
    # .min()/.max() on an empty array, which would raise.
    if annotations.sample_indices.size:
        # A negative sample index can't correspond to any real sample position.
        if annotations.sample_indices.min() < 0:
            raise RecordValidationError("annotation sample indices cannot be negative")
        # An index at or past the signal's own length points outside the record --
        # window extraction later indexes signals using these positions directly.
        if annotations.sample_indices.max() >= signal.signals.shape[0]:
            raise RecordValidationError("annotation sample index exceeds the signal boundary")
        # Annotation ordering is assumed by downstream boundary-overlap detection in
        # windows.py, which relies on ascending sample order to detect adjacency.
        if np.any(np.diff(annotations.sample_indices) < 0):
            raise RecordValidationError("annotation sample indices must be ordered")

    return _build_report(config, signal, annotations)


def write_validation_report(report: RecordValidationReport, output_path: Path) -> None:
    """Write a validation report to an existing output directory."""
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError from write_text.
    if not output_path.parent.is_dir():
        raise RecordValidationError(f"report parent directory does not exist: {output_path.parent}")
    output_path.write_text(report.to_json(), encoding="utf-8")


def _validate_required_record_files(
    config: DatasetConfig,
    data_dir: Path,
    record_id: str,
) -> None:
    """Confirm every WFDB companion file required for one record actually exists.

    A WFDB "record" is really a set of same-stem files with different extensions
    (header, signal, annotation, etc.); this checks the whole configured set exists
    and is regular (not a symlink) before wfdb.rdrecord/rdann attempt to open any of
    them, so a missing companion file fails with a clear file list instead of a
    cryptic WFDB library error partway through parsing.

    Args:
        config: Dataset config supplying the required file extensions for this dataset.
        data_dir: Directory the record's files are expected to live in.
        record_id: The record's base filename (without extension).
    """

    invalid = []
    # Check every required extension so all missing files are reported together,
    # rather than failing on the first one and requiring repeated re-runs to find them all.
    for extension in config.required_extensions:
        file_path = data_dir / f"{record_id}.{extension}"
        # Reject a symlink as well as a missing file: resolving through a link could
        # read a companion file that was swapped out after acquisition/verification.
        if not file_path.is_file() or file_path.is_symlink():
            invalid.append(file_path.name)
    # Report every missing/invalid file together rather than failing on the first.
    if invalid:
        raise RecordValidationError(f"missing or invalid required files: {', '.join(invalid)}")


def _to_signal_record(record_id: str, record: Any) -> SignalRecord:
    """Convert a raw wfdb.Record into this package's typed SignalRecord.

    Args:
        record_id: The record's base filename, attached to the resulting SignalRecord.
        record: The object returned by wfdb.rdrecord.

    Returns:
        A SignalRecord with an immutable, explicitly-typed signal array.
    """

    # p_signal (physical units, as opposed to raw ADC counts) is what every downstream
    # stage assumes; some WFDB records only carry digital samples, which this pipeline
    # doesn't support.
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
    """Convert a raw wfdb.Annotation into this package's typed AnnotationSet.

    Args:
        record_id: The record's base filename, attached to the resulting AnnotationSet.
        annotations: The object returned by wfdb.rdann.

    Returns:
        An AnnotationSet with an immutable sample-index array.
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
    """Assemble the machine-readable validation report after every check has passed.

    The `checks` tuple names each validation step validate_record performed, in order,
    so the report is self-documenting evidence of what was actually verified -- not
    just a pass/fail flag.

    Args:
        config: Dataset config this record was validated against.
        signal: The validated signal record.
        annotations: The validated annotation set.

    Returns:
        A complete validation report for this record.
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
