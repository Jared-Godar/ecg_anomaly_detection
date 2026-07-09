"""Tests for auditable run-manifest creation."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecg_anomaly_detection.run_manifest import (
    EnvironmentSnapshot,
    GitState,
    RunManifest,
    RunManifestError,
    create_run_manifest,
    read_run_manifest,
    write_run_manifest,
)

RUN_ID = "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "configs" / "dataset.toml").write_text("schema_version=1\n", encoding="utf-8")
    (tmp_path / "artifacts" / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_slug": "synthetic",
                "dataset_version": "1.0.0",
                "created_at_utc": "2026-01-01T00:00:00Z",
                "files": [
                    {"path": "100.dat", "size_bytes": 10, "sha256": "a" * 64},
                    {"path": "100.hea", "size_bytes": 5, "sha256": "b" * 64},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "split.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_name": "grouped",
                "split_version": "1.0.0",
                "strategy": "seeded-record-shuffle",
                "seed": 7,
                "mapping_name": "binary",
                "mapping_version": "1.0.0",
                "window_config_name": "six-second",
                "window_config_version": "1.0.0",
                "source_artifacts": ["data/interim/windows.npz"],
                "total_record_count": 3,
                "total_window_count": 6,
                "partitions": {
                    "train": {
                        "record_ids": ["100"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 1, "1": 1},
                    },
                    "validation": {
                        "record_ids": ["101"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 2, "1": 0},
                    },
                    "test": {
                        "record_ids": ["102"],
                        "record_count": 1,
                        "window_count": 2,
                        "target_value_counts": {"0": 0, "1": 2},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "windows.npz").write_bytes(b"synthetic-artifact")
    return tmp_path


def test_run_manifest_is_deterministic_with_injected_runtime_evidence(
    repository: Path,
) -> None:
    manifest = _create_manifest(repository)
    output = repository / "artifacts" / "run.json"

    write_run_manifest(manifest, repository, output)

    assert manifest.run_id == RUN_ID
    assert manifest.created_at_utc == "2026-01-02T03:04:05Z"
    assert manifest.git.revision == "1" * 40
    assert manifest.dataset.file_count == 2
    assert manifest.dataset.total_size_bytes == 15
    assert manifest.dataset.source_files[0].sha256 == "a" * 64
    assert manifest.dependency_lock.path == "uv.lock"
    assert manifest.split.total_record_count == 3
    assert manifest.split.partitions["test"].record_ids == ("102",)
    assert manifest.configuration_files[0].path == "configs/dataset.toml"
    assert manifest.artifact_files[0].path == "artifacts/windows.npz"
    assert all(not evidence.path.startswith("/") for evidence in manifest.artifact_files)
    assert json.loads(output.read_text(encoding="utf-8"))["environment"]["installed_packages"] == {
        "numpy": "2.4.2"
    }


def test_run_manifest_rejects_evidence_outside_repository(
    repository: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    external = tmp_path_factory.mktemp("external") / "outside.toml"
    external.write_text("secret=false\n", encoding="utf-8")

    with pytest.raises(RunManifestError, match="within repository root"):
        create_run_manifest(
            repository,
            repository / "artifacts" / "inventory.json",
            repository / "artifacts" / "split.json",
            [external],
            git_state_provider=lambda _: GitState("1" * 40, False),
            environment_provider=_environment,
        )


def test_run_manifest_rejects_split_record_leakage(repository: Path) -> None:
    split_path = repository / "artifacts" / "split.json"
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["partitions"]["test"]["record_ids"] = ["100"]
    split_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RunManifestError, match="record leakage"):
        _create_manifest(repository)


def test_run_manifest_rejects_naive_clock(repository: Path) -> None:
    with pytest.raises(RunManifestError, match="timezone-aware"):
        create_run_manifest(
            repository,
            repository / "artifacts" / "inventory.json",
            repository / "artifacts" / "split.json",
            [repository / "configs" / "dataset.toml"],
            clock=lambda: datetime(2026, 1, 2),
            run_id_factory=lambda: RUN_ID,
            git_state_provider=lambda _: GitState("1" * 40, False),
            environment_provider=_environment,
        )


def test_run_manifest_output_must_stay_under_ignored_artifacts(repository: Path) -> None:
    with pytest.raises(RunManifestError, match="under artifacts"):
        write_run_manifest(_create_manifest(repository), repository, Path("run.json"))


def test_read_run_manifest_round_trips_a_written_manifest(repository: Path) -> None:
    manifest = _create_manifest(repository)
    output = repository / "artifacts" / "run.json"
    write_run_manifest(manifest, repository, output)

    reloaded = read_run_manifest(output)

    assert reloaded == manifest


def test_read_run_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(RunManifestError, match="could not read run manifest"):
        read_run_manifest(path)


def test_read_run_manifest_rejects_incomplete_document(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(RunManifestError, match="invalid run manifest"):
        read_run_manifest(path)


def _create_manifest(repository: Path) -> RunManifest:
    return create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        git_state_provider=lambda _: GitState("1" * 40, False),
        environment_provider=_environment,
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python_version="3.12.13",
        python_implementation="CPython",
        platform="darwin",
        machine="arm64",
        installed_packages={"numpy": "2.4.2"},
    )
