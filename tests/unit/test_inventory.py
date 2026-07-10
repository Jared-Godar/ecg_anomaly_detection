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
    """Build or exercise the dataset config test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Returns:
        The value produced by the documented operation.
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
    """Build or exercise the complete data dir test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    # Iterate over `enumerate(dataset_config.expected_files)` one item at a time so ordering,
    # validation, and failure attribution remain explicit.
    for index, relative_path in enumerate(dataset_config.expected_files):
        (data_dir / relative_path).write_bytes(f"fixture-{index}".encode())
    return data_dir


def test_inventory_is_deterministic_and_round_trips(
    tmp_path: Path,
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """Verify that inventory is deterministic and round trips.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
        complete_data_dir: The complete data dir value supplied by the caller or surrounding test fixture.
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
    """Verify that inventory reports every missing required file.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    # Scope `pytest.raises(InventoryError, match='missing: 100\\.atr')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with pytest.raises(InventoryError, match=r"missing: 100\.atr"):
        create_inventory(dataset_config, tmp_path)


def test_verification_detects_content_change(
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """Verify that verification detects content change.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
        complete_data_dir: The complete data dir value supplied by the caller or surrounding test fixture.
    """

    manifest = create_inventory(dataset_config, complete_data_dir)
    (complete_data_dir / "101.dat").write_bytes(b"changed")

    # Scope `pytest.raises(InventoryError, match='101.dat')` here so the expected failure and
    # fixture cleanup stay scoped to this assertion.
    with pytest.raises(InventoryError, match="101.dat"):
        verify_inventory(dataset_config, complete_data_dir, manifest)


def test_verification_accepts_unchanged_required_files(
    dataset_config: DatasetConfig,
    complete_data_dir: Path,
) -> None:
    """Verify that verification accepts unchanged required files.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
        complete_data_dir: The complete data dir value supplied by the caller or surrounding test fixture.
    """

    manifest = create_inventory(dataset_config, complete_data_dir)

    verify_inventory(dataset_config, complete_data_dir, manifest)
