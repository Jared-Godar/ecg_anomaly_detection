"""Tests for repository and dataset configuration contracts."""

from pathlib import Path

import pytest

from ecg_anomaly_detection.config import (
    ConfigurationError,
    RepositoryPaths,
    load_dataset_config,
)


def test_repository_paths_discover_project_root() -> None:
    """RepositoryPaths.discover walks up from this test file to find the repo root (has pyproject.toml)
    and derives configs/, data/raw/, and artifacts/ relative to it."""

    paths = RepositoryPaths.discover(Path(__file__))

    assert (paths.root / "pyproject.toml").is_file()
    assert paths.configs == paths.root / "configs"
    assert paths.raw_data == paths.root / "data" / "raw"
    assert paths.artifacts == paths.root / "artifacts"


def test_mitdb_config_defines_complete_required_inventory() -> None:
    """The real, committed configs/mitdb-v1.0.0.toml loads with the full expected MIT-BIH inventory.

    Confirms the 48-record database resolves to exactly 144 expected files
    (48 records x 3 extensions), and that the first record's known file size
    and SHA-256 digest are captured for later integrity verification.
    """

    paths = RepositoryPaths.discover(Path(__file__))
    config = load_dataset_config(paths.configs / "mitdb-v1.0.0.toml")

    assert config.slug == "mitdb"
    assert config.version == "1.0.0"
    assert config.sample_rate_hz == 360
    assert config.download_url == "https://physionet.org/files/mitdb/1.0.0/"
    assert len(config.record_ids) == 48
    assert len(config.expected_files) == 144
    assert len(config.expected_source_files) == 144
    assert config.expected_files[:3] == ("100.atr", "100.dat", "100.hea")
    assert config.expected_source_files[0].size_bytes == 4558
    assert config.expected_source_files[0].sha256 == (
        "8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471"
    )


def test_config_rejects_duplicate_inventory_values(tmp_path: Path) -> None:
    """A dataset config listing the same record ID twice ("100", "100") is rejected as non-unique.

    A duplicate record ID would make the expected-file inventory ambiguous
    and could silently double-count a record's files.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        """
schema_version = 1
[dataset]
name = "Example"
slug = "example"
version = "1"
source_url = "https://example.test"
download_url = "https://example.test/files/"
sample_rate_hz = 1
annotation_extension = "dat"
record_ids = ["100", "100"]
required_extensions = ["dat"]
""".strip(),
        encoding="utf-8",
    )

    # record_ids = ["100", "100"] above repeats the same ID twice.
    with pytest.raises(ConfigurationError, match="unique"):
        load_dataset_config(config_path)


def test_config_rejects_insecure_download_url(tmp_path: Path) -> None:
    """A dataset config with a plain http:// download_url is rejected; only HTTPS is accepted.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    config_path = tmp_path / "invalid-url.toml"
    config_path.write_text(
        """
schema_version = 1
[dataset]
name = "Example"
slug = "example"
version = "1"
source_url = "https://example.test/content/"
download_url = "http://example.test/files/"
sample_rate_hz = 1
annotation_extension = "dat"
record_ids = ["100"]
required_extensions = ["dat"]
""".strip(),
        encoding="utf-8",
    )

    # download_url above uses "http://", not "https://".
    with pytest.raises(ConfigurationError, match="HTTPS URL"):
        load_dataset_config(config_path)
