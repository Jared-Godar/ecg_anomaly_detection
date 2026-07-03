"""Tests for boundary-safe beat-window extraction and artifacts."""

from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import RepositoryPaths
from ecg_anomaly_detection.labels import (
    AnnotationMappingConfig,
    TargetRule,
    map_annotations,
)
from ecg_anomaly_detection.records import AnnotationSet, SignalRecord
from ecg_anomaly_detection.windows import (
    WindowConfig,
    WindowExtractionError,
    extract_windows,
    load_window_config,
    write_window_artifact,
    write_window_report,
)


@pytest.fixture
def mapping_config() -> AnnotationMappingConfig:
    return AnnotationMappingConfig(
        schema_version=1,
        name="test-mapping",
        version="1.0.0",
        unknown_symbol_policy="error",
        targets=(
            TargetRule(name="reference_normal", value=0, symbols=("N",)),
            TargetRule(name="selected_other", value=1, symbols=("V",)),
        ),
        excluded_symbols=("!",),
    )


@pytest.fixture
def window_config() -> WindowConfig:
    return WindowConfig(
        schema_version=1,
        name="test-window",
        version="1.0.0",
        pre_seconds=2.0,
        post_seconds=2.0,
        channel_index=0,
        boundary_policy="exclude",
    )


@pytest.fixture
def signal_record() -> SignalRecord:
    samples = np.column_stack(
        (np.arange(12, dtype=np.float64), np.arange(100, 112, dtype=np.float64))
    )
    return SignalRecord(
        record_id="100",
        sample_rate_hz=2.0,
        signals=samples,
        channel_names=("MLII", "V5"),
        units=("mV", "mV"),
    )


def test_repository_window_config_preserves_historical_geometry() -> None:
    paths = RepositoryPaths.discover(Path(__file__))
    config = load_window_config(paths.configs / "windowing-v1.toml")

    assert config.name == "historical-six-second"
    assert config.pre_seconds == 3.0
    assert config.post_seconds == 3.0
    assert config.channel_index == 0
    assert config.boundary_policy == "exclude"


def test_extraction_preserves_lineage_and_reports_boundaries(
    tmp_path: Path,
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([2, 4, 6, 10], dtype=np.int64),
        symbols=("N", "V", "N", "V"),
    )
    mapped = map_annotations(mapping_config, source)

    result = extract_windows(window_config, mapping_config, signal_record, mapped)
    artifact_path = tmp_path / "windows.npz"
    report_path = tmp_path / "windows.json"
    write_window_artifact(result.window_set, artifact_path)
    write_window_report(result.report, report_path)

    assert result.window_set.windows.tolist() == [list(range(8)), list(range(2, 10))]
    assert result.window_set.record_ids == ("100", "100")
    assert result.window_set.center_sample_indices.tolist() == [4, 6]
    assert result.window_set.source_symbols == ("V", "N")
    assert result.window_set.target_values.tolist() == [1, 0]
    assert result.window_set.windows.flags.writeable is False
    assert result.report.left_boundary_exclusion_count == 1
    assert result.report.right_boundary_exclusion_count == 1
    assert result.report.overlapping_adjacent_window_count == 1
    assert result.report.emitted_target_counts == {"reference_normal": 1, "selected_other": 1}

    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["windows"].shape == (2, 8)
        assert artifact["record_ids"].tolist() == ["100", "100"]
        assert artifact["mapping_version"].item() == "1.0.0"
    assert '"emitted_window_count": 2' in report_path.read_text(encoding="utf-8")


def test_extraction_returns_typed_empty_matrix_when_all_centers_hit_boundaries(
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([1, 11], dtype=np.int64),
        symbols=("N", "V"),
    )
    mapped = map_annotations(mapping_config, source)

    result = extract_windows(window_config, mapping_config, signal_record, mapped)

    assert result.window_set.windows.shape == (0, 8)
    assert result.report.emitted_window_count == 0
    assert result.report.left_boundary_exclusion_count == 1
    assert result.report.right_boundary_exclusion_count == 1


def test_extraction_rejects_out_of_range_channel(
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    invalid_config = WindowConfig(
        schema_version=window_config.schema_version,
        name=window_config.name,
        version=window_config.version,
        pre_seconds=window_config.pre_seconds,
        post_seconds=window_config.post_seconds,
        channel_index=2,
        boundary_policy=window_config.boundary_policy,
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    with pytest.raises(WindowExtractionError, match="exceeds signal width"):
        extract_windows(
            invalid_config,
            mapping_config,
            signal_record,
            map_annotations(mapping_config, source),
        )


def test_extraction_rejects_fractional_sample_geometry(
    mapping_config: AnnotationMappingConfig,
    signal_record: SignalRecord,
) -> None:
    invalid_config = WindowConfig(
        schema_version=1,
        name="fractional",
        version="1",
        pre_seconds=0.75,
        post_seconds=1.0,
        channel_index=0,
        boundary_policy="exclude",
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    with pytest.raises(WindowExtractionError, match="whole sample count"):
        extract_windows(
            invalid_config,
            mapping_config,
            signal_record,
            map_annotations(mapping_config, source),
        )
