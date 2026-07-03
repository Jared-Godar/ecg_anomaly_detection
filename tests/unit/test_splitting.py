"""Tests for deterministic record-grouped splitting."""

from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import RepositoryPaths
from ecg_anomaly_detection.splitting import (
    SplitConfig,
    SplitError,
    WindowMetadata,
    create_split_manifest,
    load_split_config,
    load_window_metadata,
    read_split_manifest,
    write_split_manifest,
)


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig(
        schema_version=1,
        name="test-grouped-split",
        version="1.0.0",
        strategy="seeded-record-shuffle",
        seed=42,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
    )


@pytest.fixture
def metadata() -> WindowMetadata:
    targets = np.asarray([0, 1, 0, 1, 1, 0, 0, 0], dtype=np.int64)
    targets.setflags(write=False)
    return WindowMetadata(
        record_ids=("100", "100", "101", "101", "102", "102", "103", "103"),
        target_values=targets,
        source_artifacts=("windows.npz",),
        mapping_name="binary-map",
        mapping_version="1.0.0",
        window_config_name="six-second",
        window_config_version="1.0.0",
    )


def test_repository_split_config_is_versioned_and_grouped() -> None:
    paths = RepositoryPaths.discover(Path(__file__))
    config = load_split_config(paths.configs / "splitting-v1.toml")

    assert config.name == "record-grouped-holdout"
    assert config.strategy == "seeded-record-shuffle"
    assert config.seed == 2022
    assert config.train_ratio + config.validation_ratio + config.test_ratio == pytest.approx(1.0)


def test_split_is_deterministic_complete_and_record_disjoint(
    split_config: SplitConfig, metadata: WindowMetadata, tmp_path: Path
) -> None:
    first = create_split_manifest(split_config, metadata)
    second = create_split_manifest(split_config, metadata)
    output_path = tmp_path / "split.json"
    write_split_manifest(first, output_path)

    assert first == second
    memberships = [set(summary.record_ids) for summary in first.partitions.values()]
    assert set().union(*memberships) == {"100", "101", "102", "103"}
    assert all(
        not left & right
        for index, left in enumerate(memberships)
        for right in memberships[index + 1 :]
    )
    assert sum(summary.window_count for summary in first.partitions.values()) == 8
    assert (
        sum(sum(summary.target_value_counts.values()) for summary in first.partitions.values()) == 8
    )
    assert all(
        set(summary.target_value_counts) == {"0", "1"} for summary in first.partitions.values()
    )
    assert '"strategy": "seeded-record-shuffle"' in output_path.read_text(encoding="utf-8")
    assert read_split_manifest(output_path) == first


def test_three_records_produce_three_nonempty_partitions(split_config: SplitConfig) -> None:
    metadata = WindowMetadata(
        record_ids=("100", "101", "102"),
        target_values=np.asarray([0, 0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    manifest = create_split_manifest(split_config, metadata)

    assert {summary.record_count for summary in manifest.partitions.values()} == {1}


def test_split_rejects_too_few_records(split_config: SplitConfig) -> None:
    metadata = WindowMetadata(
        record_ids=("100", "101"),
        target_values=np.asarray([0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    with pytest.raises(SplitError, match="at least 3 records"):
        create_split_manifest(split_config, metadata)


def test_window_metadata_loader_rejects_record_reuse_across_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_metadata_artifact(first, ["100"], [0])
    _write_metadata_artifact(second, ["100"], [1])

    with pytest.raises(SplitError, match="multiple window artifacts"):
        load_window_metadata([first, second])


def _write_metadata_artifact(path: Path, record_ids: list[str], target_values: list[int]) -> None:
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        record_ids=np.asarray(record_ids, dtype=np.str_),
        target_values=np.asarray(target_values, dtype=np.int64),
        mapping_name=np.asarray("binary-map", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("six-second", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )
