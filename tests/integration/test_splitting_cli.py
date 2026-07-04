"""Integration test for the record-grouped splitting command."""

import json
from pathlib import Path

import numpy as np

from ecg_anomaly_detection.cli import main


def test_split_windows_writes_auditable_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "split.toml"
    config_path.write_text(
        """
schema_version = 2
[split]
name = "synthetic-subject-split"
version = "2.0.0"
strategy = "seeded-subject-shuffle"
seed = 7
[split.ratios]
train = 0.5
validation = 0.25
test = 0.25
[record_subjects]
100 = "subject-a"
101 = "subject-a"
102 = "subject-b"
103 = "subject-c"
""".strip(),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "windows.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray(1, dtype=np.int64),
        record_ids=np.asarray(["100", "100", "101", "102", "103"], dtype=np.str_),
        target_values=np.asarray([0, 1, 0, 1, 0], dtype=np.int64),
        mapping_name=np.asarray("test-map", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("test-window", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )
    output_path = tmp_path / "split-manifest.json"

    exit_code = main(
        [
            "split-windows",
            "--split-config",
            str(config_path),
            "--input",
            str(artifact_path),
            "--output",
            str(output_path),
        ]
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    quality = json.loads((tmp_path / "split_quality_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert manifest["total_record_count"] == 4
    assert manifest["total_subject_count"] == 3
    assert manifest["total_window_count"] == 5
    assert set(manifest["partitions"]) == {"train", "validation", "test"}
    partitions = manifest["partitions"]
    record_partitions = {
        record_id: name
        for name, partition in partitions.items()
        for record_id in partition["record_ids"]
    }
    assert record_partitions["100"] == record_partitions["101"]
    assert all("subject_ids" in partition for partition in partitions.values())
    assert quality["status"] == "passed"
    assert quality["subject_disjoint"] is True
    assert quality["record_disjoint"] is True
    assert quality["partitions"]["train"]["shard_count"] == 1


def test_split_windows_writes_failed_quality_summary_and_returns_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "split.toml"
    config_path.write_text(
        """
schema_version = 2
[split]
name = "rejected-split"
version = "2.0.0"
strategy = "seeded-subject-shuffle"
seed = 7
[split.ratios]
train = 0.5
validation = 0.25
test = 0.25
[split.quality]
min_windows_per_partition = 10
default_severity = "failure"
[record_subjects]
100 = "subject-a"
101 = "subject-b"
102 = "subject-c"
""".strip(),
        encoding="utf-8",
    )
    artifact_path = tmp_path / "windows.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray(1, dtype=np.int64),
        record_ids=np.asarray(["100", "101", "102"], dtype=np.str_),
        target_values=np.asarray([0, 1, 0], dtype=np.int64),
        mapping_name=np.asarray("test-map", dtype=np.str_),
        mapping_version=np.asarray("1.0.0", dtype=np.str_),
        window_config_name=np.asarray("test-window", dtype=np.str_),
        window_config_version=np.asarray("1.0.0", dtype=np.str_),
    )

    exit_code = main(
        [
            "split-windows",
            "--split-config",
            str(config_path),
            "--input",
            str(artifact_path),
            "--output",
            str(tmp_path / "split.json"),
        ]
    )

    quality = json.loads((tmp_path / "split_quality_summary.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert quality["status"] == "failed"
    assert [item["partition"] for item in quality["violations"]] == [
        "test",
        "train",
        "validation",
    ]
