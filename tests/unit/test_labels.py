"""Tests for versioned annotation mapping and audit reports."""

from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import RepositoryPaths
from ecg_anomaly_detection.labels import (
    AnnotationMappingConfig,
    AnnotationMappingError,
    TargetRule,
    load_annotation_mapping,
    map_annotations,
    write_mapping_report,
)
from ecg_anomaly_detection.records import AnnotationSet


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
            TargetRule(name="selected_other", value=1, symbols=("V", "A")),
        ),
        excluded_symbols=("!", "Q"),
    )


def test_repository_mapping_preserves_historical_symbol_policy() -> None:
    """Verify that repository mapping preserves historical symbol policy.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    paths = RepositoryPaths.discover(Path(__file__))
    config = load_annotation_mapping(paths.configs / "annotation-map-v1.toml")

    assert config.name == "historical-binary"
    assert config.unknown_symbol_policy == "error"
    assert config.targets[0].symbols == ("N",)
    assert len(config.targets[1].symbols) == 13
    assert len(config.excluded_symbols) == 24


def test_mapping_preserves_source_identity_and_reports_exclusions(
    tmp_path: Path,
    mapping_config: AnnotationMappingConfig,
) -> None:
    """Verify that mapping preserves source identity and reports exclusions.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
    """

    annotations = AnnotationSet(
        record_id="100",
        sample_indices=np.array([2, 4, 6, 8], dtype=np.int64),
        symbols=("N", "V", "!", "A"),
    )

    result = map_annotations(mapping_config, annotations)
    output_path = tmp_path / "mapping-report.json"
    write_mapping_report(result.report, output_path)

    assert result.annotations.record_id == "100"
    assert result.annotations.sample_indices.tolist() == [2, 4, 8]
    assert result.annotations.source_symbols == ("N", "V", "A")
    assert result.annotations.target_values.tolist() == [0, 1, 1]
    assert result.annotations.sample_indices.flags.writeable is False
    assert result.annotations.target_values.flags.writeable is False
    assert result.report.input_annotation_count == 4
    assert result.report.included_annotation_count == 3
    assert result.report.excluded_annotation_count == 1
    assert result.report.target_counts == {"reference_normal": 1, "selected_other": 2}
    assert result.report.excluded_symbol_counts == {"!": 1}
    assert '"mapping_name": "test-mapping"' in output_path.read_text(encoding="utf-8")


def test_mapping_rejects_unknown_source_symbol(
    mapping_config: AnnotationMappingConfig,
) -> None:
    """Verify that mapping rejects unknown source symbol.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        mapping_config: The mapping config value supplied by the caller or surrounding test fixture.
    """

    annotations = AnnotationSet(
        record_id="100",
        sample_indices=np.array([2], dtype=np.int64),
        symbols=("Z",),
    )

    # Scope `pytest.raises(AnnotationMappingError, match='unmapped annotation symbols: Z')` here so
    # the expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AnnotationMappingError, match="unmapped annotation symbols: Z"):
        map_annotations(mapping_config, annotations)


def test_mapping_config_rejects_overlapping_symbols(tmp_path: Path) -> None:
    """Verify that mapping config rejects overlapping symbols.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    config_path = tmp_path / "overlap.toml"
    config_path.write_text(
        """
schema_version = 1
[mapping]
name = "overlap"
version = "1"
unknown_symbol_policy = "error"
[[targets]]
name = "one"
value = 0
symbols = ["N"]
[exclusions]
symbols = ["N"]
""".strip(),
        encoding="utf-8",
    )

    # Scope `pytest.raises(AnnotationMappingError, match='exactly one')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AnnotationMappingError, match="exactly one"):
        load_annotation_mapping(config_path)
