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
    """A two-target binary mapping: "N" -> reference_normal, "V"/"A" -> selected_other, "!"/"Q" excluded.

    Returns:
        An AnnotationMappingConfig small enough to exercise inclusion,
        multi-symbol grouping, and exclusion in a single fixture set.
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
    """The real, committed configs/annotation-map-v1.toml loads with its historical symbol counts intact.

    Confirms the "historical-binary" mapping still classifies "N" alone as
    the reference-normal target, groups 13 symbols into the other target,
    and excludes 24 non-beat symbols -- these counts are a direct check that
    the modernized mapping reproduces the archived 2022 policy exactly.
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
    """Mapping four annotations (one excluded) preserves record/index/symbol lineage and yields
    a correct, immutable, machine-readable audit report.

    Confirms excluded annotations ("!" here) are dropped from the output
    arrays but still counted in the report, that the surviving arrays are
    read-only (guarding against accidental in-place mutation downstream),
    and that the written report file round-trips the mapping's name.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        mapping_config: The two-target binary mapping fixture.
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
    """A symbol ("Z") that appears in neither a target nor the excluded set is rejected under
    the "error" unknown_symbol_policy, naming the offending symbol.

    Args:
        mapping_config: The two-target binary mapping fixture, whose
            unknown_symbol_policy is "error".
    """

    annotations = AnnotationSet(
        record_id="100",
        sample_indices=np.array([2], dtype=np.int64),
        symbols=("Z",),
    )

    # "Z" is not covered by mapping_config's targets or excluded_symbols.
    with pytest.raises(AnnotationMappingError, match="unmapped annotation symbols: Z"):
        map_annotations(mapping_config, annotations)


def test_mapping_config_rejects_overlapping_symbols(tmp_path: Path) -> None:
    """A config assigning symbol "N" to both a target and the excluded set is rejected as ambiguous.

    Every symbol must have exactly one fate (a specific target, or
    exclusion); a symbol in both would make the mapping's classification of
    it ambiguous.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
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

    # "N" above is listed both as a target symbol and an excluded symbol.
    with pytest.raises(AnnotationMappingError, match="exactly one"):
        load_annotation_mapping(config_path)
