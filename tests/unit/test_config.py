"""Tests for repository and dataset configuration contracts."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.config import (
    ConfigurationError,
    RepositoryPaths,
    load_dataset_config,
)


def test_repository_paths_discover_project_root() -> None:
    paths = RepositoryPaths.discover(Path(__file__))

    assert (paths.root / "pyproject.toml").is_file()
    assert paths.configs == paths.root / "configs"
    assert paths.raw_data == paths.root / "data" / "raw"
    assert paths.artifacts == paths.root / "artifacts"


def test_mitdb_config_defines_complete_required_inventory() -> None:
    paths = RepositoryPaths.discover(Path(__file__))
    config = load_dataset_config(paths.configs / "mitdb-v1.0.0.toml")

    assert config.slug == "mitdb"
    assert config.version == "1.0.0"
    assert config.sample_rate_hz == 360
    assert len(config.record_ids) == 48
    assert len(config.expected_files) == 144
    assert config.expected_files[:3] == ("100.atr", "100.dat", "100.hea")


def test_config_rejects_duplicate_inventory_values(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
schema_version = 1
[dataset]
name = "Example"
slug = "example"
version = "1"
source_url = "https://example.test"
sample_rate_hz = 1
record_ids = ["100", "100"]
required_extensions = ["dat"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="unique"):
        load_dataset_config(config_path)
