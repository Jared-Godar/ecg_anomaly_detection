"""Tests for deterministic local file inventory and integrity verification."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ecg_anomaly_detection.config import DatasetConfig
from ecg_anomaly_detection.inventory import (
    InventoryError,
    InventoryManifest,
    create_inventory,
    read_manifest,
    verify_inventory,
    write_manifest,
)


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """A two-record synthetic dataset config with three required file extensions per record.

    Returns:
        A DatasetConfig whose expected_files property (six paths: 100/101 x
        atr/dat/hea) drives both fixtures and tests below.
    """

    return DatasetConfig(
        schema_version=1,
        name="Synthetic fixture",
        slug="synthetic",
        version="1.0.0",
        source_url="https://example.test/synthetic",
        download_url="https://example.test/files/synthetic/",
        sample_rate_hz=360,
        annotation_extension="atr",
        record_ids=("100", "101"),
        required_extensions=("atr", "dat", "hea"),
    )


@pytest.fixture
def complete_data_dir(tmp_path: Path, dataset_config: DatasetConfig) -> Path:
    """A directory containing every file dataset_config.expected_files requires.

    Each file's content is a short, distinct marker string (`fixture-<index>`)
    so tests can corrupt a single file and expect exactly one hash mismatch.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        dataset_config: The two-record fixture whose expected_files this
            directory must fully satisfy.

    Returns:
        The populated "raw" data directory.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    # Write one distinct-content file per expected path so a later single-file
    # edit can be attributed to exactly that path in the assertions below.
    for index, relative_path in enumerate(dataset_config.expected_files):
        (data_dir / relative_path).write_bytes(f"fixture-{index}".encode())
    return data_dir


def test_inventory_is_deterministic_and_round_trips(
    tmp_path: Path,
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """A manifest built with a frozen clock has a fixed timestamp and survives a JSON round trip.

    Confirms create_inventory records every expected file with a full 64-character
    SHA-256 hex digest, and that both write_manifest/read_manifest (file-based) and
    to_json/from_json (in-memory) reproduce an identical manifest.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        dataset_config: The two-record fixture defining which files are expected.
        complete_data_dir: A directory already containing every expected file.
    """

    frozen_time = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    manifest = create_inventory(dataset_config, complete_data_dir, clock=lambda: frozen_time)
    output_path = tmp_path / "inventory.json"

    write_manifest(manifest, output_path)

    assert manifest.created_at_utc == "2026-01-02T03:04:05Z"
    assert tuple(item.path for item in manifest.files) == dataset_config.expected_files
    assert all(len(item.sha256) == 64 for item in manifest.files)
    assert read_manifest(output_path) == manifest
    assert InventoryManifest.from_json(manifest.to_json()) == manifest


def test_inventory_reports_every_missing_required_file(
    tmp_path: Path,
    dataset_config: DatasetConfig,
) -> None:
    """Building an inventory over an empty directory names the first missing file in the error.

    tmp_path here has none of dataset_config's six expected files, so
    create_inventory must fail rather than silently produce a partial manifest.

    Args:
        tmp_path: An empty directory standing in for an incomplete download.
        dataset_config: The two-record fixture defining which files are expected.
    """

    # tmp_path is empty, so "100.atr" (the first expected path) is missing.
    with pytest.raises(InventoryError, match=r"missing: 100\.atr"):
        create_inventory(dataset_config, tmp_path)


def test_verification_detects_content_change(
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """A file rewritten after the manifest was created fails verification with its own hash.

    Confirms verify_inventory recomputes hashes rather than trusting file
    presence alone -- the file is neither missing nor renamed, only its bytes
    have changed.

    Args:
        dataset_config: The two-record fixture defining which files are expected.
        complete_data_dir: A directory already containing every expected file.
    """

    manifest = create_inventory(dataset_config, complete_data_dir)
    (complete_data_dir / "101.dat").write_bytes(b"changed")

    # "101.dat" was overwritten above, so its SHA-256 no longer matches the manifest.
    with pytest.raises(InventoryError, match="101.dat"):
        verify_inventory(dataset_config, complete_data_dir, manifest)


def test_verification_accepts_unchanged_required_files(
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """Verifying a manifest against the same, untouched directory it was built from succeeds.

    The positive-path counterpart to the content-change test above: no
    exception means every recorded hash still matches.

    Args:
        dataset_config: The two-record fixture defining which files are expected.
        complete_data_dir: A directory already containing every expected file.
    """

    manifest = create_inventory(dataset_config, complete_data_dir)

    verify_inventory(dataset_config, complete_data_dir, manifest)
