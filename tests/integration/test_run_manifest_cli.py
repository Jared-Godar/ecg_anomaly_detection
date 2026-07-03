"""Integration test for the run-manifest CLI boundary."""

import json
import subprocess
from pathlib import Path

from ecg_anomaly_detection.cli import main


def test_create_run_manifest_command_records_local_evidence(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    config = tmp_path / "config.toml"
    config.write_text("schema_version=1\n", encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_slug": "synthetic",
                "dataset_version": "1",
                "created_at_utc": "2026-01-01T00:00:00Z",
                "files": [{"path": "100.dat", "size_bytes": 1, "sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_name": "grouped",
                "split_version": "1",
                "strategy": "seeded-record-shuffle",
                "seed": 7,
                "mapping_name": "binary",
                "mapping_version": "1",
                "window_config_name": "window",
                "window_config_version": "1",
                "source_artifacts": ["data/interim/windows.npz"],
                "total_record_count": 3,
                "total_window_count": 3,
                "partitions": {
                    name: {
                        "record_ids": [record_id],
                        "record_count": 1,
                        "window_count": 1,
                        "target_value_counts": {"0": 1},
                    }
                    for name, record_id in zip(
                        ("train", "validation", "test"), ("100", "101", "102"), strict=True
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "untracked.py").write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "artifacts" / "run.json"

    exit_code = main(
        [
            "create-run-manifest",
            "--repository-root",
            str(tmp_path),
            "--inventory-manifest",
            str(inventory),
            "--split-manifest",
            str(split),
            "--config",
            str(config),
            "--output",
            str(output),
        ]
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(manifest["git"]["revision"]) == 40
    assert manifest["git"]["dirty"] is True
    assert manifest["dataset"]["dataset_slug"] == "synthetic"
    assert manifest["dependency_lock"]["path"] == "uv.lock"
    assert manifest["split"]["total_record_count"] == 3
    assert manifest["configuration_files"][0]["path"] == "config.toml"
