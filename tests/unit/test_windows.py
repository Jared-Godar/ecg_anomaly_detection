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
    """A minimal closed-world mapping: N -> reference_normal, V -> selected_other, ! excluded.

    Returns:
        A two-target AnnotationMappingConfig usable across this file's extraction tests.
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
    """A small, deterministic window geometry: 2s pre/post, channel_index=0.

    Returns:
        A WindowConfig sized to fit the short signal_record fixture below.
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
    """A 12-sample, 2-channel synthetic signal at 2 Hz, with distinct per-channel values.

    Each channel uses a different numeric range (0-11 vs 100-111) so tests can verify
    which physical channel was actually selected, not just that some channel was.

    Returns:
        A SignalRecord with channels named "MLII" and "V5", matching real WFDB naming.
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
    """The committed windowing-v1.toml config still matches the original 2022 window geometry.

    Regression test guarding against an accidental edit to the repository's real
    window config: a 3-second pre/post margin selected by channel name ("MLII") is
    the historical geometry the 2022 archived notebook used, and this config is
    expected to preserve it exactly.
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
    """A window config naming neither channel_index nor channel_name is rejected.

    Protects the mutual-exclusivity enforcement in load_window_config: exactly one
    channel selector is required, and omitting both must fail rather than defaulting
    to an implicit channel.
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

    # Neither channel_index nor channel_name is present in the TOML above.
    with pytest.raises(WindowExtractionError, match="exactly one channel selector"):
        load_window_config(config_path)


def test_window_config_rejects_multiple_channel_selectors(tmp_path: Path) -> None:
    """A window config naming both channel_index and channel_name is rejected.

    Same mutual-exclusivity enforcement as the missing-selector test above, exercised
    from the opposite direction: both fields present is equally invalid as neither
    being present, since there would be no single unambiguous channel to select.
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

    # Both channel_index and channel_name are present in the TOML above.
    with pytest.raises(WindowExtractionError, match="exactly one channel selector"):
        load_window_config(config_path)


def test_extraction_preserves_lineage_and_reports_boundaries(
    tmp_path: Path,
    mapping_config: AnnotationMappingConfig,
    window_config: WindowConfig,
    signal_record: SignalRecord,
) -> None:
    """End-to-end extraction preserves row-level lineage and reports every boundary exclusion.

    The most comprehensive test in this file: four annotations at samples 2, 4, 6, 10
    are positioned so exactly one falls too close to the left edge (sample 2), one too
    close to the right edge (sample 10), and the remaining two (4 and 6) are close
    enough together that their extracted windows overlap -- exercising boundary
    exclusion counts, overlap detection, and the full NPZ/JSON artifact write/read
    round-trip in one test.
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

    # Read the NPZ artifact back to confirm every lineage field round-trips correctly,
    # including the writer's -1/"" sentinel-to-int/str encoding of the optional
    # configured_channel_index/name fields.
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
    """Zero surviving windows still yields a correctly shaped (0, window_samples) array.

    Protects extract_windows' empty-array fallback (`np.empty((0, window_samples))`):
    without it, zero accepted windows would make np.stack raise on an empty list
    instead of returning a typed, correctly shaped empty result that downstream
    concatenation can still handle.
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
    """A configured channel_index beyond the signal's actual channel count is rejected.

    Protects _resolve_channel's bounds check: signal_record only has 2 channels
    (indices 0-1), so channel_index=2 must fail with a clear message rather than
    letting a later numpy indexing operation raise a less specific IndexError.
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

    # channel_index=2 exceeds signal_record's actual 2-channel width.
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
    """channel_name resolution finds the right physical channel even when column order differs.

    This is the core regression test for "channel identity resolved by name, not
    position" (docs/window-extraction.md): reordered_signal swaps signal_record's two
    columns so "MLII" is now at index 1 instead of 0. A position-based lookup would
    silently extract the wrong channel's data; resolving by name must still find "MLII"
    correctly regardless of its column position.
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
    """A configured channel_name absent from the record's own channels is rejected.

    Protects _resolve_channel's error path for channel-by-name lookup: missing_signal
    has channels "V5"/"V2" but no "MLII", so the configured name must fail with a
    message naming both the missing channel and what's actually available, rather
    than a bare ValueError from list.index.
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

    # "MLII" is not among missing_signal's channel_names ("V5", "V2").
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
    """A pre_seconds value that doesn't resolve to a whole sample count is rejected.

    Protects _seconds_to_samples' whole-sample-count requirement: at signal_record's
    2 Hz sample rate, 0.75 seconds is 1.5 samples, which can't be a valid window
    half-width -- this must fail at config-validation time rather than silently
    rounding to an unintended window size.
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

    # 0.75s * 2Hz = 1.5 samples, not a whole number.
    with pytest.raises(WindowExtractionError, match="whole sample count"):
        extract_windows(
            invalid_config,
            mapping_config,
            signal_record,
            map_annotations(mapping_config, source),
        )
