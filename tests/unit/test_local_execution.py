"""Tests for local run-scoped artifact lifecycle helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ecg_anomaly_detection.local_execution import (
    LocalArtifactLifecycleError,
    list_runs,
    purge_run,
)

RUN_ID = "aaaaaaaa-bbbb-1111-1111-111111111111"
OTHER_RUN_ID = "22222222-2222-2222-2222-222222222222"
MANIFEST_CONTENT = "{}"
MANIFEST_BYTES = len(MANIFEST_CONTENT.encode("utf-8"))


def _init_repository(root: Path) -> None:
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")


def _write_run(root: Path, run_id: str, *, with_manifest: bool = True, blob_bytes: int = 0) -> None:
    artifacts_dir = root / "artifacts" / "runs" / run_id
    interim_dir = root / "data" / "interim" / "runs" / run_id
    processed_dir = root / "data" / "processed" / "runs" / run_id
    for directory in (artifacts_dir, interim_dir, processed_dir):
        directory.mkdir(parents=True)
    if with_manifest:
        (artifacts_dir / "run-manifest.json").write_text(MANIFEST_CONTENT, encoding="utf-8")
    if blob_bytes:
        (interim_dir / "blob.bin").write_bytes(b"\0" * blob_bytes)


def test_list_runs_returns_empty_tuple_when_no_runs_directory(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    assert list_runs(tmp_path) == ()


def test_list_runs_reports_size_manifest_status_and_directories(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID, with_manifest=True, blob_bytes=2048)

    summaries = list_runs(tmp_path)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.run_id == RUN_ID
    assert summary.has_run_manifest is True
    assert summary.total_size_bytes == 2048 + MANIFEST_BYTES
    assert len(summary.directories) == 3


def test_list_runs_orders_newest_first(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID)
    older_directory = tmp_path / "artifacts" / "runs" / RUN_ID
    _write_run(tmp_path, OTHER_RUN_ID)
    newer_directory = tmp_path / "artifacts" / "runs" / OTHER_RUN_ID
    # Force distinguishable, deterministic mtimes regardless of filesystem clock resolution.
    os.utime(older_directory, (1_000_000, 1_000_000))
    os.utime(newer_directory, (2_000_000, 2_000_000))

    summaries = list_runs(tmp_path)

    assert [summary.run_id for summary in summaries] == [OTHER_RUN_ID, RUN_ID]


def test_list_runs_ignores_non_uuid_entries(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    (tmp_path / "artifacts" / "runs" / "not-a-run-id").mkdir(parents=True)

    assert list_runs(tmp_path) == ()


def test_purge_run_removes_all_three_companion_directories(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID, blob_bytes=10)

    result = purge_run(tmp_path, RUN_ID)

    assert result.dry_run is False
    assert result.freed_bytes == 10 + MANIFEST_BYTES
    assert len(result.removed_directories) == 3
    assert not (tmp_path / "artifacts" / "runs" / RUN_ID).exists()
    assert not (tmp_path / "data" / "interim" / "runs" / RUN_ID).exists()
    assert not (tmp_path / "data" / "processed" / "runs" / RUN_ID).exists()


def test_purge_run_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID, blob_bytes=10)

    result = purge_run(tmp_path, RUN_ID, dry_run=True)

    assert result.dry_run is True
    assert result.freed_bytes == 10 + MANIFEST_BYTES
    assert (tmp_path / "artifacts" / "runs" / RUN_ID).is_dir()
    assert (tmp_path / "data" / "interim" / "runs" / RUN_ID).is_dir()
    assert (tmp_path / "data" / "processed" / "runs" / RUN_ID).is_dir()


def test_purge_run_only_touches_the_named_run(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID)
    _write_run(tmp_path, OTHER_RUN_ID)

    purge_run(tmp_path, RUN_ID)

    assert not (tmp_path / "artifacts" / "runs" / RUN_ID).exists()
    assert (tmp_path / "artifacts" / "runs" / OTHER_RUN_ID).is_dir()
    assert (tmp_path / "data" / "interim" / "runs" / OTHER_RUN_ID).is_dir()
    assert (tmp_path / "data" / "processed" / "runs" / OTHER_RUN_ID).is_dir()


def test_purge_run_never_touches_raw_data_or_dataset_acquisition_baseline(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID)
    raw_dir = tmp_path / "data" / "raw" / "mitdb" / "1.0.0"
    raw_dir.mkdir(parents=True)
    (raw_dir / "100.dat").write_bytes(b"\0" * 10)
    dataset_evidence_dir = tmp_path / "artifacts" / "datasets" / "mitdb" / "1.0.0"
    dataset_evidence_dir.mkdir(parents=True)
    (dataset_evidence_dir / "acquisition.json").write_text("{}", encoding="utf-8")

    purge_run(tmp_path, RUN_ID)

    assert (raw_dir / "100.dat").is_file()
    assert (dataset_evidence_dir / "acquisition.json").is_file()


def test_purge_run_rejects_non_canonical_run_id(tmp_path: Path) -> None:
    _init_repository(tmp_path)

    with pytest.raises(LocalArtifactLifecycleError, match="canonical lowercase UUID"):
        purge_run(tmp_path, "not-a-uuid")


def test_purge_run_rejects_uppercase_run_id(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    _write_run(tmp_path, RUN_ID)

    with pytest.raises(LocalArtifactLifecycleError, match="canonical lowercase UUID"):
        purge_run(tmp_path, RUN_ID.upper())


def test_purge_run_rejects_unknown_run_id(tmp_path: Path) -> None:
    _init_repository(tmp_path)

    with pytest.raises(LocalArtifactLifecycleError, match="no local run directories found"):
        purge_run(tmp_path, RUN_ID)


def test_purge_run_refuses_to_delete_through_a_symlink(tmp_path: Path) -> None:
    _init_repository(tmp_path)
    real_target = tmp_path / "elsewhere"
    real_target.mkdir()
    run_parent = tmp_path / "artifacts" / "runs"
    run_parent.mkdir(parents=True)
    (run_parent / RUN_ID).symlink_to(real_target, target_is_directory=True)

    with pytest.raises(LocalArtifactLifecycleError, match="symbolic link"):
        purge_run(tmp_path, RUN_ID)
    assert real_target.is_dir()
