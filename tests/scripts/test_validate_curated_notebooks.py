"""Tests for the curated-notebook validation script's pure config builders.

scripts/ holds standalone operational tooling, not the installed package, so
the module under test is loaded directly from its file path rather than
imported as `ecg_anomaly_detection.*`.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

# Locate the script relative to this test file, not the current working
# directory, so the test suite works regardless of where pytest is invoked from.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_curated_notebooks.py"
# Load the script as a module by file path, since it's not installed as part of the
# package (see this file's module docstring for why).
_SPEC = importlib.util.spec_from_file_location("validate_curated_notebooks", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
# The module object every test in this file calls into (e.g. vcn.build_dataset_config_toml).
vcn = importlib.util.module_from_spec(_SPEC)
# Register the loaded module in sys.modules before executing it, matching the
# standard importlib.util pattern.
sys.modules[_SPEC.name] = vcn
_SPEC.loader.exec_module(vcn)

from ecg_anomaly_detection.acquisition import AcquisitionManifest  # noqa: E402
from ecg_anomaly_detection.config import load_dataset_config  # noqa: E402
from ecg_anomaly_detection.splitting import load_split_config  # noqa: E402

# Six synthetic files (atr/dat/hea for two records) shared across every test below,
# standing in for a real curated-notebook download without needing PhysioNet access.
_FILES = (
    vcn.SyntheticFile("900.atr", 128, "a" * 64),
    vcn.SyntheticFile("900.dat", 4096, "b" * 64),
    vcn.SyntheticFile("900.hea", 96, "c" * 64),
    vcn.SyntheticFile("901.atr", 128, "d" * 64),
    vcn.SyntheticFile("901.dat", 4096, "e" * 64),
    vcn.SyntheticFile("901.hea", 96, "f" * 64),
)
# The two record IDs _FILES's six synthetic files belong to.
_RECORD_IDS = ("900", "901")


def test_dataset_config_toml_parses_as_valid_toml() -> None:
    """build_dataset_config_toml's output parses as TOML with the expected slug, record IDs, and file count."""

    document = tomllib.loads(vcn.build_dataset_config_toml(_FILES, _RECORD_IDS))
    assert document["dataset"]["slug"] == "mitdb"
    assert document["dataset"]["record_ids"] == list(_RECORD_IDS)
    assert len(document["dataset"]["expected_source_files"]) == 6


def test_dataset_config_toml_loads_via_the_real_config_loader(tmp_path: Path) -> None:
    """build_dataset_config_toml's output loads successfully through the package's own
    load_dataset_config, not just as generic TOML.

    Confirms the generated file satisfies every schema constraint the real
    config loader enforces (not just TOML syntax), and that the six expected
    files are listed in the same sorted order the loader expects.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    config_path = tmp_path / "mitdb-v1.0.0.toml"
    config_path.write_text(vcn.build_dataset_config_toml(_FILES, _RECORD_IDS), encoding="utf-8")
    config = load_dataset_config(config_path)
    assert config.slug == "mitdb"
    assert config.record_ids == _RECORD_IDS
    assert config.expected_files == (
        "900.atr",
        "900.dat",
        "900.hea",
        "901.atr",
        "901.dat",
        "901.hea",
    )


def test_split_config_toml_loads_via_the_real_split_loader(tmp_path: Path) -> None:
    """build_split_config_toml's output loads successfully through the package's own load_split_config.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    config_path = tmp_path / "splitting-v2.toml"
    config_path.write_text(vcn.build_split_config_toml(_RECORD_IDS), encoding="utf-8")
    config = load_split_config(config_path)
    assert set(config.record_subjects) == set(_RECORD_IDS)
    assert config.quality.min_subjects_per_partition == 1
    assert config.quality.min_records_per_partition == 1


def test_acquisition_manifest_round_trips_via_the_real_manifest_parser() -> None:
    """build_acquisition_manifest's output parses via AcquisitionManifest.from_json with all six files present."""

    content = vcn.build_acquisition_manifest(_FILES, _RECORD_IDS)
    manifest = AcquisitionManifest.from_json(content)
    assert manifest.dataset_slug == "mitdb"
    assert manifest.dataset_version == "1.0.0"
    assert len(manifest.files) == 6
    assert {item.path for item in manifest.files} == {item.name for item in _FILES}


def test_acquisition_manifest_matches_dataset_config_identity(tmp_path: Path) -> None:
    """The manifest and the trimmed dataset config must describe the same dataset.

    acquire_dataset's verify-and-reuse path requires these to agree exactly
    (dataset_slug, dataset_version, source_url, download_url, and each file's
    url/size/sha256) -- this is what lets it skip the network fetcher.
    """
    config_path = tmp_path / "mitdb-v1.0.0.toml"
    config_path.write_text(vcn.build_dataset_config_toml(_FILES, _RECORD_IDS), encoding="utf-8")
    config = load_dataset_config(config_path)
    manifest = AcquisitionManifest.from_json(vcn.build_acquisition_manifest(_FILES, _RECORD_IDS))

    assert manifest.dataset_slug == config.slug
    assert manifest.dataset_version == config.version
    assert manifest.source_url == config.source_url
    assert manifest.download_url == config.download_url
    assert tuple(item.path for item in manifest.files) == config.expected_files
    expectations = config.expected_source_files_by_path
    # Every manifest file's recorded size/digest must match the dataset config's own expectation.
    for item in manifest.files:
        assert item.size_bytes == expectations[item.path].size_bytes
        assert item.sha256 == expectations[item.path].sha256


def test_dataset_config_toml_uses_only_the_given_files_and_record_ids() -> None:
    """Passing only record "900"'s three files produces a config listing just that one record."""

    single_record_files = tuple(item for item in _FILES if item.name.startswith("900"))
    document = tomllib.loads(vcn.build_dataset_config_toml(single_record_files, ("900",)))
    assert document["dataset"]["record_ids"] == ["900"]
    paths = {item["path"] for item in document["dataset"]["expected_source_files"]}
    assert paths == {"900.atr", "900.dat", "900.hea"}
