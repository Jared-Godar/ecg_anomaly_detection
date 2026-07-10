"""Tests for grouped model-ready dataset indexing."""

from pathlib import Path

import numpy as np
import pytest

from ecg_anomaly_detection.dataset_index import (
    DatasetIndexError,
    create_dataset_index,
    write_dataset_index,
)
from ecg_anomaly_detection.splitting import (
    PartitionSummary,
    SplitManifest,
    write_split_manifest,
)


@pytest.fixture
def indexed_repository(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    """Build or exercise the indexed repository test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.

    Returns:
        The value produced by the documented operation.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    interim = tmp_path / "data" / "interim" / "runs" / "test" / "windows"
    processed = tmp_path / "data" / "processed" / "runs" / "test"
    artifacts = tmp_path / "artifacts" / "runs" / "test"
    interim.mkdir(parents=True)
    processed.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    shard_paths = []
    # Iterate over `('100', '101', '102')` one item at a time so ordering, validation, and failure
    # attribution remain explicit.
    for record_id in ("100", "101", "102"):
        path = interim / f"{record_id}.npz"
        _write_shard(path, record_id)
        shard_paths.append(path)
    split = SplitManifest(
        schema_version=2,
        split_name="grouped",
        split_version="1.0.0",
        strategy="seeded-subject-shuffle",
        seed=7,
        mapping_name="binary",
        mapping_version="1.0.0",
        window_config_name="four-sample",
        window_config_version="1.0.0",
        source_artifacts=tuple(str(path) for path in shard_paths),
        total_subject_count=3,
        total_record_count=3,
        total_window_count=6,
        partitions={
            name: PartitionSummary(
                subject_ids=(f"subject-{record_id}",),
                subject_count=1,
                record_ids=(record_id,),
                record_subjects={record_id: f"subject-{record_id}"},
                record_count=1,
                window_count=2,
                target_value_counts={"0": 1, "1": 1},
            )
            for name, record_id in zip(
                ("train", "validation", "test"),
                ("100", "101", "102"),
                strict=True,
            )
        },
    )
    split_path = artifacts / "split.json"
    write_split_manifest(split, split_path)
    return tmp_path, split_path, shard_paths


def test_dataset_index_preserves_grouped_shards_without_copying_arrays(
    indexed_repository: tuple[Path, Path, list[Path]],
) -> None:
    """Verify that dataset index preserves grouped shards without copying arrays.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        indexed_repository: The indexed repository value supplied by the caller or surrounding test fixture.
    """

    repository, split_path, shard_paths = indexed_repository
    index = create_dataset_index(repository, split_path, shard_paths)
    output = repository / "data" / "processed" / "runs" / "test" / "dataset-index.json"

    write_dataset_index(index, repository, output)

    assert index.total_record_count == 3
    assert index.total_window_count == 6
    assert index.window_samples == 4
    assert index.partitions["train"].shards[0].record_id == "100"
    assert index.partitions["train"].shards[0].subject_id == "subject-100"
    assert index.partitions["test"].target_value_counts == {"0": 1, "1": 1}
    assert index.partitions["validation"].shards[0].file.path.endswith("101.npz")
    assert len(index.partitions["train"].shards[0].file.sha256) == 64
    assert output.is_file()
    assert all(path.is_file() for path in shard_paths)


def test_dataset_index_rejects_nonfinite_window_shard(
    indexed_repository: tuple[Path, Path, list[Path]],
) -> None:
    """Verify that dataset index rejects nonfinite window shard.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        indexed_repository: The indexed repository value supplied by the caller or surrounding test fixture.
    """

    repository, split_path, shard_paths = indexed_repository
    _write_shard(shard_paths[1], "101", nonfinite=True)

    # Scope `pytest.raises(DatasetIndexError, match='finite floating-point')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(DatasetIndexError, match="finite floating-point"):
        create_dataset_index(repository, split_path, shard_paths)


def test_dataset_index_rejects_unordered_window_centers(
    indexed_repository: tuple[Path, Path, list[Path]],
) -> None:
    """Verify that dataset index rejects unordered window centers.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        indexed_repository: The indexed repository value supplied by the caller or surrounding test fixture.
    """

    repository, split_path, shard_paths = indexed_repository
    _write_shard(shard_paths[1], "101", centers=(20, 10))

    # Scope `pytest.raises(DatasetIndexError, match='nonnegative and ordered')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(DatasetIndexError, match="nonnegative and ordered"):
        create_dataset_index(repository, split_path, shard_paths)


def test_dataset_index_rejects_mixed_channel_identity(
    indexed_repository: tuple[Path, Path, list[Path]],
) -> None:
    """Verify that dataset index rejects mixed channel identity.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        indexed_repository: The indexed repository value supplied by the caller or surrounding test fixture.
    """

    repository, split_path, shard_paths = indexed_repository
    _write_shard(shard_paths[2], "102", channel_index=0, channel_name="V5")

    # Scope `pytest.raises(DatasetIndexError)` here so the expected failure and fixture cleanup stay
    # scoped to this assertion.
    with pytest.raises(DatasetIndexError) as error:
        create_dataset_index(repository, split_path, shard_paths)

    message = str(error.value)
    assert "Window extraction produced inconsistent channel identities." in message
    assert 'Configured channel selection: channel_name = "MLII"' in message
    assert "  - MLII: 2 records" in message
    assert "  - V5: 1 record" in message
    assert "  - 102: V5" in message
    assert "This is not usually fixed by cleaning local artifacts." in message


def test_dataset_index_output_must_stay_in_processed_zone(
    indexed_repository: tuple[Path, Path, list[Path]],
) -> None:
    """Verify that dataset index output must stay in processed zone.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        indexed_repository: The indexed repository value supplied by the caller or surrounding test fixture.
    """

    repository, split_path, shard_paths = indexed_repository
    index = create_dataset_index(repository, split_path, shard_paths)

    # Scope `pytest.raises(DatasetIndexError, match='under data/processed')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(DatasetIndexError, match="under data/processed"):
        write_dataset_index(index, repository, Path("artifacts/index.json"))


def _write_shard(
    path: Path,
    record_id: str,
    *,
    nonfinite: bool = False,
    centers: tuple[int, int] = (10, 20),
    channel_index: int = 0,
    channel_name: str = "MLII",
) -> None:
    """Write shard according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        path: Filesystem path identifying the input or output under review.
        record_id: The record id value supplied by the caller or surrounding test fixture.
        nonfinite: The nonfinite value supplied by the caller or surrounding test fixture.
        centers: The centers value supplied by the caller or surrounding test fixture.
        channel_index: The channel index value supplied by the caller or surrounding test fixture.
        channel_name: The channel name value supplied by the caller or surrounding test fixture.
    """

    windows = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    # Exercise the `nonfinite` branch so this regression documents every expected outcome.
    if nonfinite:
        windows[0, 0] = np.nan
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        windows=windows,
        record_ids=np.asarray([record_id, record_id], dtype=np.str_),
        center_sample_indices=np.asarray(centers, dtype=np.int64),
        source_symbols=np.asarray(["N", "V"], dtype=np.str_),
        target_values=np.asarray([0, 1], dtype=np.int64),
        sample_rate_hz=np.asarray(360.0, dtype=np.float64),
        channel_selector=np.asarray("channel_name", dtype=np.str_),
        configured_channel_index=np.asarray(-1, dtype=np.int64),
        configured_channel_name=np.asarray("MLII", dtype=np.str_),
        channel_index=np.asarray(channel_index, dtype=np.int64),
        channel_name=np.asarray(channel_name, dtype=np.str_),
        resolved_channel_index=np.asarray(channel_index, dtype=np.int64),
        resolved_channel_name=np.asarray(channel_name, dtype=np.str_),
        mapping_name=np.asarray("binary", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("four-sample", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )
