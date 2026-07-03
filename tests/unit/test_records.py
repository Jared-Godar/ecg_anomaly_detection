"""Tests for signal, annotation, and validation-report contracts."""

from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import DatasetConfig
from ecg_anomaly_detection.records import (
    AnnotationSet,
    RecordValidationError,
    SignalRecord,
    validate_record,
    write_validation_report,
)


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        schema_version=1,
        name="Synthetic fixture",
        slug="synthetic",
        version="1.0.0",
        source_url="https://example.test/synthetic",
        download_url="https://example.test/files/synthetic/",
        sample_rate_hz=360,
        annotation_extension="atr",
        record_ids=("100",),
        required_extensions=("atr", "dat", "hea"),
    )


@pytest.fixture
def signal_record() -> SignalRecord:
    return SignalRecord(
        record_id="100",
        sample_rate_hz=360.0,
        signals=np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
        channel_names=("MLII", "V5"),
        units=("mV", "mV"),
    )


@pytest.fixture
def annotations() -> AnnotationSet:
    return AnnotationSet(
        record_id="100",
        sample_indices=np.array([0, 2], dtype=np.int64),
        symbols=("N", "V"),
    )


def test_valid_record_produces_machine_readable_report(
    tmp_path: Path,
    dataset_config: DatasetConfig,
    signal_record: SignalRecord,
    annotations: AnnotationSet,
) -> None:
    report = validate_record(dataset_config, signal_record, annotations)
    output_path = tmp_path / "report.json"

    write_validation_report(report, output_path)

    assert report.sample_count == 3
    assert report.channel_count == 2
    assert report.annotation_count == 2
    assert report.annotation_symbol_counts == {"N": 1, "V": 1}
    assert '"record_id": "100"' in output_path.read_text(encoding="utf-8")


def test_validation_rejects_wrong_sample_rate(
    dataset_config: DatasetConfig,
    signal_record: SignalRecord,
    annotations: AnnotationSet,
) -> None:
    wrong_rate = SignalRecord(
        record_id=signal_record.record_id,
        sample_rate_hz=250.0,
        signals=signal_record.signals,
        channel_names=signal_record.channel_names,
        units=signal_record.units,
    )

    with pytest.raises(RecordValidationError, match="sample rate mismatch"):
        validate_record(dataset_config, wrong_rate, annotations)


def test_validation_rejects_non_finite_signal(
    dataset_config: DatasetConfig,
    signal_record: SignalRecord,
    annotations: AnnotationSet,
) -> None:
    invalid_signals = signal_record.signals.copy()
    invalid_signals[1, 0] = np.nan
    non_finite = SignalRecord(
        record_id=signal_record.record_id,
        sample_rate_hz=signal_record.sample_rate_hz,
        signals=invalid_signals,
        channel_names=signal_record.channel_names,
        units=signal_record.units,
    )

    with pytest.raises(RecordValidationError, match="non-finite"):
        validate_record(dataset_config, non_finite, annotations)


@pytest.mark.parametrize(
    ("sample_indices", "message"),
    [
        (np.array([-1], dtype=np.int64), "cannot be negative"),
        (np.array([3], dtype=np.int64), "exceeds the signal boundary"),
        (np.array([2, 1], dtype=np.int64), "must be ordered"),
    ],
)
def test_validation_rejects_invalid_annotation_locations(
    dataset_config: DatasetConfig,
    signal_record: SignalRecord,
    sample_indices: np.ndarray,
    message: str,
) -> None:
    invalid = AnnotationSet(
        record_id="100",
        sample_indices=sample_indices,
        symbols=tuple("N" for _ in sample_indices),
    )

    with pytest.raises(RecordValidationError, match=message):
        validate_record(dataset_config, signal_record, invalid)
