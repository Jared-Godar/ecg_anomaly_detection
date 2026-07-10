"""Integration test for the index-dataset CLI, including directory discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecg_anomaly_detection.cli import main


def _write_shard(path: Path, record_id: str) -> None:
    """Write one valid window-shard NPZ artifact for record_id, matching windows.py's real output shape.

    Args:
        path: Where to write the NPZ artifact.
        record_id: The record ID both windows in this shard belong to.
    """

    windows = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        windows=windows,
        record_ids=np.asarray([record_id, record_id], dtype=np.str_),
        center_sample_indices=np.asarray((10, 20), dtype=np.int64),
        source_symbols=np.asarray(["N", "V"], dtype=np.str_),
        target_values=np.asarray([0, 1], dtype=np.int64),
        sample_rate_hz=np.asarray(360.0, dtype=np.float64),
        channel_selector=np.asarray("channel_name", dtype=np.str_),
        configured_channel_index=np.asarray(-1, dtype=np.int64),
        configured_channel_name=np.asarray("MLII", dtype=np.str_),
        channel_index=np.asarray(0, dtype=np.int64),
        channel_name=np.asarray("MLII", dtype=np.str_),
        resolved_channel_index=np.asarray(0, dtype=np.int64),
        resolved_channel_name=np.asarray("MLII", dtype=np.str_),
        mapping_name=np.asarray("binary", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("four-sample", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )


def _write_split_manifest(
    path: Path, record_ids: tuple[str, ...], source_artifacts: tuple[str, ...]
) -> None:
    """Write a split manifest assigning each of record_ids to a distinct partition (train/validation/test).

    Args:
        path: Where to write the split-manifest JSON.
        record_ids: One record ID per partition, assigned in
            (train, validation, test) order; fewer than three leaves the
            remaining partitions unassigned.
        source_artifacts: The shard paths this split manifest claims to
            derive from, recorded for provenance only.
    """

    payload = {
        "schema_version": 2,
        "split_name": "grouped",
        "split_version": "1.0.0",
        "strategy": "seeded-subject-shuffle",
        "seed": 7,
        "mapping_name": "binary",
        "mapping_version": "1.0.0",
        "window_config_name": "four-sample",
        "window_config_version": "1.0.0",
        "source_artifacts": list(source_artifacts),
        "total_subject_count": len(record_ids),
        "total_record_count": len(record_ids),
        "total_window_count": len(record_ids) * 2,
        "partitions": {
            name: {
                "subject_ids": [f"subject-{record_id}"],
                "subject_count": 1,
                "record_ids": [record_id],
                "record_subjects": {record_id: f"subject-{record_id}"},
                "record_count": 1,
                "window_count": 2,
                "target_value_counts": {"0": 1, "1": 1},
            }
            for name, record_id in zip(("train", "validation", "test"), record_ids, strict=False)
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_index_dataset_accepts_a_directory_of_shards(tmp_path: Path) -> None:
    """`ecg-data index-dataset --input <directory>` discovers every shard in that directory and
    builds a dataset index with the correct total record/window counts.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    interim = tmp_path / "data" / "interim" / "runs" / "test" / "windows"
    interim.mkdir(parents=True)
    artifacts = tmp_path / "artifacts" / "runs" / "test"
    artifacts.mkdir(parents=True)
    record_ids = ("100", "101", "102")
    # Write one shard per record, matching the split manifest built below.
    for record_id in record_ids:
        _write_shard(interim / f"{record_id}.npz", record_id)
    split_path = artifacts / "split.json"
    _write_split_manifest(
        split_path,
        record_ids,
        tuple(str(interim / f"{record_id}.npz") for record_id in record_ids),
    )
    output_path = tmp_path / "data" / "processed" / "runs" / "test" / "dataset-index.json"
    output_path.parent.mkdir(parents=True)

    exit_code = main(
        [
            "index-dataset",
            "--repository-root",
            str(tmp_path),
            "--split-manifest",
            str(split_path),
            "--input",
            str(interim),
            "--output",
            str(output_path),
        ]
    )

    index = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert index["total_record_count"] == 3
    assert index["total_window_count"] == 6


def test_index_dataset_reports_a_nonexistent_input_path(tmp_path: Path) -> None:
    """`index-dataset --input <missing directory>` exits 1 rather than crashing with a raw traceback.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts" / "runs" / "test"
    artifacts.mkdir(parents=True)
    split_path = artifacts / "split.json"
    _write_split_manifest(split_path, ("100",), ())

    exit_code = main(
        [
            "index-dataset",
            "--repository-root",
            str(tmp_path),
            "--split-manifest",
            str(split_path),
            "--input",
            str(tmp_path / "data" / "interim" / "runs" / "test" / "windows"),
            "--output",
            str(tmp_path / "dataset-index.json"),
        ]
    )

    assert exit_code == 1
