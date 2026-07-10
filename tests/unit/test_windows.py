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
    """Build or exercise the mapping config test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Returns:
        The value produced by the documented operation.
    """

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
    """Build or exercise the window config test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Returns:
        The value produced by the documented operation.
    """

    return WindowConfig(
        schema_version=1,
        name="test-window",
        version="1.0.0",
        pre_seconds=2.0,
        post_seconds=2.0,
        channel_index=0,
        channel_name=None,
        exclude_record_ids=(),
        boundary_policy="exclude",
    )


@pytest.fixture
def signal_record() -> SignalRecord:
    """Build or exercise the signal record test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Returns:
        The value produced by the documented operation.
    """

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
    """Verify that repository window config preserves historical geometry.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    paths = RepositoryPaths.discover(Path(__file__))
    config = load_window_config(paths.configs / "windowing-v1.toml")

    assert config.name == "historical-six-second"
    assert config.pre_seconds == 3.0
    assert config.post_seconds == 3.0
    assert config.channel_index is None
    assert config.channel_name == "MLII"
    assert config.boundary_policy == "exclude"


def test_window_config_rejects_missing_channel_selector(tmp_path: Path) -> None:
    """Verify that window config rejects missing channel selector.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    config_path = tmp_path / "windowing.toml"
    config_path.write_text(
        """
schema_version = 1

[window]
name = "test-window"
version = "1.0.0"
pre_seconds = 2.0
post_seconds = 2.0
boundary_policy = "exclude"
""".lstrip(),
        encoding="utf-8",
    )

    # Scope `pytest.raises(WindowExtractionError, match='exactly one channel selector')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(WindowExtractionError, match="exactly one channel selector"):
        load_window_config(config_path)


def test_window_config_rejects_multiple_channel_selectors(tmp_path: Path) -> None:
    """Verify that window config rejects multiple channel selectors.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    config_path = tmp_path / "windowing.toml"
    config_path.write_text(
        """
schema_version = 1

[window]
name = "test-window"
version = "1.0.0"
pre_seconds = 2.0
post_seconds = 2.0
channel_index = 0
channel_name = "MLII"
boundary_policy = "exclude"
""".lstrip(),
        encoding="utf-8",
    )

    # Scope `pytest.raises(WindowExtractionError, match='exactly one channel selector')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(WindowExtractionError, match="exactly one channel selector"):
        load_window_config(config_path)


def test_extraction_preserves_lineage_and_reports_boundaries(
    tmp_path: Path,
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    """Verify that extraction preserves lineage and reports boundaries.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        window_config: The window config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

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
    assert result.window_set.channel_selector == "channel_index"
    assert result.window_set.configured_channel_index == 0
    assert result.window_set.configured_channel_name is None
    assert result.window_set.channel_index == 0
    assert result.window_set.channel_name == "MLII"
    assert result.report.left_boundary_exclusion_count == 1
    assert result.report.right_boundary_exclusion_count == 1
    assert result.report.overlapping_adjacent_window_count == 1
    assert result.report.emitted_target_counts == {"reference_normal": 1, "selected_other": 1}

    # Scope `np.load(artifact_path, allow_pickle=False)` here so the expected failure and fixture
    # cleanup stay scoped to this assertion.
    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["windows"].shape == (2, 8)
        assert artifact["record_ids"].tolist() == ["100", "100"]
        assert artifact["mapping_version"].item() == "1.0.0"
        assert artifact["channel_selector"].item() == "channel_index"
        assert artifact["configured_channel_index"].item() == 0
        assert artifact["configured_channel_name"].item() == ""
        assert artifact["resolved_channel_index"].item() == 0
        assert artifact["resolved_channel_name"].item() == "MLII"
    assert '"emitted_window_count": 2' in report_path.read_text(encoding="utf-8")


def test_extraction_returns_typed_empty_matrix_when_all_centers_hit_boundaries(
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    """Verify that extraction returns typed empty matrix when all centers hit boundaries.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        window_config: The window config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

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
    """Verify that extraction rejects out of range channel.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        window_config: The window config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

    invalid_config = WindowConfig(
        schema_version=window_config.schema_version,
        name=window_config.name,
        version=window_config.version,
        pre_seconds=window_config.pre_seconds,
        post_seconds=window_config.post_seconds,
        channel_index=2,
        channel_name=None,
        exclude_record_ids=(),
        boundary_policy=window_config.boundary_policy,
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    # Scope `pytest.raises(WindowExtractionError, match='exceeds signal width')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(WindowExtractionError, match="exceeds signal width"):
        extract_windows(
            invalid_config,
            mapping_config,
            signal_record,
            map_annotations(mapping_config, source),
        )


def test_extraction_resolves_named_channel_when_record_order_differs(
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    """Verify that extraction resolves named channel when record order differs.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        window_config: The window config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

    named_config = WindowConfig(
        schema_version=window_config.schema_version,
        name=window_config.name,
        version=window_config.version,
        pre_seconds=window_config.pre_seconds,
        post_seconds=window_config.post_seconds,
        channel_index=None,
        channel_name="MLII",
        exclude_record_ids=(),
        boundary_policy=window_config.boundary_policy,
    )
    reordered_signal = SignalRecord(
        record_id=signal_record.record_id,
        sample_rate_hz=signal_record.sample_rate_hz,
        signals=signal_record.signals[:, [1, 0]],
        channel_names=("V5", "MLII"),
        units=signal_record.units,
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    result = extract_windows(
        named_config,
        mapping_config,
        reordered_signal,
        map_annotations(mapping_config, source),
    )

    assert result.window_set.windows.tolist() == [list(range(2, 10))]
    assert result.window_set.channel_selector == "channel_name"
    assert result.window_set.configured_channel_index is None
    assert result.window_set.configured_channel_name == "MLII"
    assert result.window_set.channel_index == 1
    assert result.window_set.channel_name == "MLII"
    assert result.report.channel_selector == "channel_name"
    assert result.report.configured_channel_name == "MLII"
    assert result.report.channel_index == 1
    assert result.report.channel_name == "MLII"


def test_extraction_rejects_missing_named_channel(
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    """Verify that extraction rejects missing named channel.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        window_config: The window config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

    named_config = WindowConfig(
        schema_version=window_config.schema_version,
        name=window_config.name,
        version=window_config.version,
        pre_seconds=window_config.pre_seconds,
        post_seconds=window_config.post_seconds,
        channel_index=None,
        channel_name="MLII",
        exclude_record_ids=(),
        boundary_policy=window_config.boundary_policy,
    )
    missing_signal = SignalRecord(
        record_id=signal_record.record_id,
        sample_rate_hz=signal_record.sample_rate_hz,
        signals=signal_record.signals,
        channel_names=("V5", "V2"),
        units=signal_record.units,
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    # Scope `pytest.raises(WindowExtractionError, match='Configured channel_name = "MLII" was not
    # available for record 100')` here so the expected failure and fixture cleanup stay scoped to
    # this assertion.
    with pytest.raises(
        WindowExtractionError,
        match='Configured channel_name = "MLII" was not available for record 100',
    ):
        extract_windows(
            named_config,
            mapping_config,
            missing_signal,
            map_annotations(mapping_config, source),
        )


def test_extraction_rejects_fractional_sample_geometry(
    mapping_config: AnnotationMappingConfig,
    signal_record: SignalRecord,
) -> None:
    """Verify that extraction rejects fractional sample geometry.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
        signal_record: The signal record value supplied by the caller or surrounding test fixture.
    """

    invalid_config = WindowConfig(
        schema_version=1,
        name="fractional",
        version="1",
        pre_seconds=0.75,
        post_seconds=1.0,
        channel_index=0,
        channel_name=None,
        exclude_record_ids=(),
        boundary_policy="exclude",
    )
    source = AnnotationSet(
        record_id="100",
        sample_indices=np.array([6], dtype=np.int64),
        symbols=("N",),
    )

    # Scope `pytest.raises(WindowExtractionError, match='whole sample count')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(WindowExtractionError, match="whole sample count"):
        extract_windows(
            invalid_config,
            mapping_config,
            signal_record,
            map_annotations(mapping_config, source),
        )
