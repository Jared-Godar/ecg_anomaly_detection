"""Tests for auditable run-manifest creation."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ecg_anomaly_detection.run_manifest as run_manifest
from ecg_anomaly_detection.run_manifest import (
    UNKNOWN_GIT_REVISION,
    EnvironmentSnapshot,
    GitState,
    RunManifest,
    RunManifestError,
    create_run_manifest,
    read_run_manifest,
    write_run_manifest,
)

# Centralize RUN_ID so every caller shares the same documented invariant.
RUN_ID = "12345678-1234-5678-1234-567812345678"


class _FailingSubprocess:
    """Stand-in for the `subprocess` module when Git is unavailable."""

    CalledProcessError = subprocess.CalledProcessError

    @staticmethod
    def run(*_args: object, **_kwargs: object) -> None:
        """Build or exercise the run test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            _args: The args value supplied by the caller or surrounding test fixture.
            _kwargs: The kwargs value supplied by the caller or surrounding test fixture.
        """

        raise FileNotFoundError("git executable not found")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Build or exercise the repository test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.

    Returns:
        The value produced by the documented operation.
    """

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
    """Verify that run manifest is deterministic with injected runtime evidence.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

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
    """Verify that run manifest rejects evidence outside repository.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        tmp_path_factory: The tmp path factory value supplied by the caller or surrounding test fixture.
    """

    external = tmp_path_factory.mktemp("external") / "outside.toml"
    external.write_text("secret=false\n", encoding="utf-8")

    # Scope `pytest.raises(RunManifestError, match='within repository root')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
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
    """Verify that run manifest rejects split record leakage.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

    split_path = repository / "artifacts" / "split.json"
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["partitions"]["test"]["record_ids"] = ["100"]
    split_path.write_text(json.dumps(document), encoding="utf-8")

    # Scope `pytest.raises(RunManifestError, match='record leakage')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with pytest.raises(RunManifestError, match="record leakage"):
        _create_manifest(repository)


def test_run_manifest_rejects_naive_clock(repository: Path) -> None:
    """Verify that run manifest rejects naive clock.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

    # Scope `pytest.raises(RunManifestError, match='timezone-aware')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
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
    """Verify that run manifest output must stay under ignored artifacts.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

    # Scope `pytest.raises(RunManifestError, match='under artifacts')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with pytest.raises(RunManifestError, match="under artifacts"):
        write_run_manifest(_create_manifest(repository), repository, Path("run.json"))


def test_read_run_manifest_round_trips_a_written_manifest(repository: Path) -> None:
    """Verify that read run manifest round trips a written manifest.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

    manifest = _create_manifest(repository)
    output = repository / "artifacts" / "run.json"
    write_run_manifest(manifest, repository, output)

    reloaded = read_run_manifest(output)

    assert reloaded == manifest


def test_read_run_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    """Verify that read run manifest rejects invalid json.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "run.json"
    path.write_text("not json", encoding="utf-8")

    # Scope `pytest.raises(RunManifestError, match='could not read run manifest')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(RunManifestError, match="could not read run manifest"):
        read_run_manifest(path)


def test_read_run_manifest_rejects_incomplete_document(tmp_path: Path) -> None:
    """Verify that read run manifest rejects incomplete document.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    path = tmp_path / "run.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    # Scope `pytest.raises(RunManifestError, match='invalid run manifest')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(RunManifestError, match="invalid run manifest"):
        read_run_manifest(path)


def test_capture_git_state_degrades_gracefully_when_git_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that capture git state degrades gracefully when git is unavailable.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    monkeypatch.setattr(run_manifest, "subprocess", _FailingSubprocess)

    state = run_manifest._capture_git_state(tmp_path)

    assert state == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)


def test_create_run_manifest_completes_when_git_is_unavailable(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that create run manifest completes when git is unavailable.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    monkeypatch.setattr(run_manifest, "subprocess", _FailingSubprocess)

    manifest = create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        environment_provider=_environment,
    )

    assert manifest.git == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)


def test_run_manifest_round_trips_degraded_git_state_as_json_null(repository: Path) -> None:
    """Verify that run manifest round trips degraded git state as json null.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
    """

    manifest = create_run_manifest(
        repository,
        Path("artifacts/inventory.json"),
        Path("artifacts/split.json"),
        [Path("configs/dataset.toml")],
        artifact_paths=[Path("artifacts/windows.npz")],
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        run_id_factory=lambda: RUN_ID,
        git_state_provider=lambda _: GitState(UNKNOWN_GIT_REVISION, None),
        environment_provider=_environment,
    )
    output = repository / "artifacts" / "run.json"
    write_run_manifest(manifest, repository, output)

    assert json.loads(output.read_text(encoding="utf-8"))["git"] == {
        "revision": UNKNOWN_GIT_REVISION,
        "dirty": None,
    }
    reloaded = read_run_manifest(output)
    assert reloaded.git == GitState(revision=UNKNOWN_GIT_REVISION, dirty=None)
    assert reloaded == manifest


def _create_manifest(repository: Path) -> RunManifest:
    """Create manifest according to the repository contract.

    The helper centralizes validation and failure behavior so every caller follows the same
    documented path.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

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
    """Compute and return environment for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Returns:
        The value produced by the documented operation.
    """

    return EnvironmentSnapshot(
        python_version="3.12.13",
        python_implementation="CPython",
        platform="darwin",
        machine="arm64",
        installed_packages={"numpy": "2.4.2"},
    )
