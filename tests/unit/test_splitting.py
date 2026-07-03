"""Tests for deterministic record-grouped splitting."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.config import RepositoryPaths
from ecg_anomaly_detection.splitting import (
    SplitConfig,
    SplitError,
    SplitManifest,
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
        schema_version=2,
        name="test-subject-split",
        version="2.0.0",
        strategy="seeded-subject-shuffle",
        seed=42,
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        record_subjects={
            "100": "subject-a",
            "101": "subject-a",
            "102": "subject-b",
            "103": "subject-c",
        },
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


def test_repository_split_config_is_versioned_and_subject_grouped() -> None:
    paths = RepositoryPaths.discover(Path(__file__))
    config = load_split_config(paths.configs / "splitting-v2.toml")

    assert config.name == "subject-aware-holdout"
    assert config.strategy == "seeded-subject-shuffle"
    assert config.record_subjects["201"] == config.record_subjects["202"]
    assert config.seed == 2022
    assert config.train_ratio + config.validation_ratio + config.test_ratio == pytest.approx(1.0)


def test_split_is_deterministic_complete_and_subject_disjoint(
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
    subject_memberships = [set(summary.subject_ids) for summary in first.partitions.values()]
    assert all(
        not left & right
        for index, left in enumerate(subject_memberships)
        for right in subject_memberships[index + 1 :]
    )
    assert next(
        name for name, summary in first.partitions.items() if "100" in summary.record_ids
    ) == next(name for name, summary in first.partitions.items() if "101" in summary.record_ids)
    assert sum(summary.window_count for summary in first.partitions.values()) == 8
    assert (
        sum(sum(summary.target_value_counts.values()) for summary in first.partitions.values()) == 8
    )
    assert all(
        set(summary.target_value_counts) == {"0", "1"} for summary in first.partitions.values()
    )
    assert '"strategy": "seeded-subject-shuffle"' in output_path.read_text(encoding="utf-8")
    assert read_split_manifest(output_path) == first


def test_three_subjects_produce_three_nonempty_partitions(split_config: SplitConfig) -> None:
    metadata = WindowMetadata(
        record_ids=("100", "101", "102", "103"),
        target_values=np.asarray([0, 0, 0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    manifest = create_split_manifest(split_config, metadata)

    assert {summary.subject_count for summary in manifest.partitions.values()} == {1}


def test_split_rejects_too_few_subjects(split_config: SplitConfig) -> None:
    metadata = WindowMetadata(
        record_ids=("100", "101", "102"),
        target_values=np.asarray([0, 0, 1], dtype=np.int64),
        source_artifacts=("windows.npz",),
        mapping_name="map",
        mapping_version="1",
        window_config_name="window",
        window_config_version="1",
    )

    with pytest.raises(SplitError, match="at least 3 subjects"):
        create_split_manifest(
            replace(
                split_config,
                record_subjects={
                    "100": "subject-a",
                    "101": "subject-a",
                    "102": "subject-b",
                },
            ),
            metadata,
        )


def test_window_metadata_loader_rejects_record_reuse_across_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_metadata_artifact(first, ["100"], [0])
    _write_metadata_artifact(second, ["100"], [1])

    with pytest.raises(SplitError, match="multiple window artifacts"):
        load_window_metadata([first, second])


def test_manifest_reader_rejects_subject_crossing_partitions(
    split_config: SplitConfig, metadata: WindowMetadata
) -> None:
    document = json.loads(create_split_manifest(split_config, metadata).to_json())
    leaked_subject = document["partitions"]["train"]["subject_ids"][0]
    validation = document["partitions"]["validation"]
    displaced_subject = validation["subject_ids"][0]
    validation["subject_ids"] = [leaked_subject]
    validation["record_subjects"] = {
        record_id: leaked_subject for record_id in validation["record_ids"]
    }
    assert displaced_subject != leaked_subject

    with pytest.raises(SplitError, match="subject leakage"):
        SplitManifest.from_json(json.dumps(document))


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
