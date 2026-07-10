"""Integration tests using synthetic files written through the WFDB package."""

from pathlib import Path

import numpy as np
import wfdb

from ecg_anomaly_detection.cli import main
from ecg_anomaly_detection.config import DatasetConfig
from ecg_anomaly_detection.records import load_wfdb_record, validate_record


def test_load_and_validate_synthetic_wfdb_record(tmp_path: Path) -> None:
    """A synthetic record written with the real wfdb package round-trips through load_wfdb_record
    and validate_record with matching shape, immutability, channel names, and symbol counts.

    Writing through wfdb (rather than hand-building a SignalRecord) exercises
    the actual on-disk WFDB format this pipeline must parse, not just the
    in-memory data structures.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_synthetic_record(data_dir)
    config = _synthetic_config()

    loaded = load_wfdb_record(config, data_dir, "100")
    report = validate_record(config, loaded.signal, loaded.annotations)

    assert loaded.signal.signals.shape == (16, 2)
    assert loaded.signal.signals.flags.writeable is False
    assert loaded.annotations.sample_indices.tolist() == [2, 8, 14]
    assert loaded.annotations.sample_indices.flags.writeable is False
    assert report.channel_names == ("MLII", "V5")
    assert report.annotation_symbol_counts == {"N": 2, "V": 1}


def test_validate_record_cli_writes_report(tmp_path: Path) -> None:
    """`ecg-data validate-record` run against a synthetic WFDB record writes a validation report
    with the correct 16-sample count.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_synthetic_record(data_dir)
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        _dataset_config_content(),
        encoding="utf-8",
    )
    output_path = tmp_path / "validation.json"

    exit_code = main(
        [
            "validate-record",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--record-id",
            "100",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert '"sample_count": 16' in output_path.read_text(encoding="utf-8")


def test_map_annotations_cli_writes_audit_report(tmp_path: Path) -> None:
    """`ecg-data map-annotations` run against a synthetic record writes an audit report with the
    correct included-count and per-target tally.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_synthetic_record(data_dir)
    dataset_config_path = tmp_path / "dataset.toml"
    dataset_config_path.write_text(
        _dataset_config_content(),
        encoding="utf-8",
    )
    mapping_config_path = tmp_path / "mapping.toml"
    mapping_config_path.write_text(
        """
schema_version = 1
[mapping]
name = "synthetic"
version = "1.0.0"
unknown_symbol_policy = "error"
[[targets]]
name = "reference_normal"
value = 0
symbols = ["N"]
[[targets]]
name = "selected_other"
value = 1
symbols = ["V"]
[exclusions]
symbols = ["!"]
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "mapping-report.json"

    exit_code = main(
        [
            "map-annotations",
            "--config",
            str(dataset_config_path),
            "--mapping-config",
            str(mapping_config_path),
            "--data-dir",
            str(data_dir),
            "--record-id",
            "100",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    report = output_path.read_text(encoding="utf-8")
    assert '"included_annotation_count": 3' in report
    assert '"selected_other": 1' in report


def test_extract_windows_cli_writes_npz_and_report(tmp_path: Path) -> None:
    """`ecg-data extract-windows` run end-to-end against a synthetic record writes a window NPZ
    shard and a matching report with the correct shape, record IDs, and emitted-window count.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
    """

    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_synthetic_record(data_dir)
    dataset_config_path = tmp_path / "dataset.toml"
    dataset_config_path.write_text(_dataset_config_content(), encoding="utf-8")
    mapping_config_path = tmp_path / "mapping.toml"
    mapping_config_path.write_text(_mapping_config_content(), encoding="utf-8")
    window_config_path = tmp_path / "window.toml"
    window_config_path.write_text(
        """
schema_version = 1
[window]
name = "synthetic-two-sample"
version = "1.0.0"
pre_seconds = 0.002777777777777778
post_seconds = 0.002777777777777778
channel_index = 0
boundary_policy = "exclude"
""".strip(),
        encoding="utf-8",
    )
    output_path = tmp_path / "windows.npz"
    report_path = tmp_path / "windows.json"

    exit_code = main(
        [
            "extract-windows",
            "--config",
            str(dataset_config_path),
            "--mapping-config",
            str(mapping_config_path),
            "--window-config",
            str(window_config_path),
            "--data-dir",
            str(data_dir),
            "--record-id",
            "100",
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    # The window NPZ artifact was just written by the CLI invocation above.
    with np.load(output_path, allow_pickle=False) as artifact:
        assert artifact["windows"].shape == (3, 2)
        assert artifact["record_ids"].tolist() == ["100", "100", "100"]
    assert '"emitted_window_count": 3' in report_path.read_text(encoding="utf-8")


def _write_synthetic_record(data_dir: Path) -> None:
    """Write a 16-sample, 2-channel synthetic WFDB record ("100") with 3 annotations via the real wfdb package.

    Args:
        data_dir: Directory to write the record's .dat/.hea/.atr files into.
    """

    sample_axis = np.linspace(0.0, 1.0, 16, endpoint=False)
    physical_signals = np.column_stack(
        (np.sin(2 * np.pi * sample_axis), np.cos(2 * np.pi * sample_axis))
    )
    wfdb.wrsamp(
        "100",
        fs=360,
        units=["mV", "mV"],
        sig_name=["MLII", "V5"],
        p_signal=physical_signals,
        write_dir=str(data_dir),
    )
    wfdb.wrann(
        "100",
        "atr",
        sample=np.array([2, 8, 14]),
        symbol=["N", "V", "N"],
        write_dir=str(data_dir),
    )


def _synthetic_config() -> DatasetConfig:
    """A one-record DatasetConfig matching the record _write_synthetic_record produces.

    Returns:
        A DatasetConfig for record "100" at 360 Hz with atr/dat/hea files required.
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
        record_ids=("100",),
        required_extensions=("atr", "dat", "hea"),
    )


def _dataset_config_content() -> str:
    """The TOML-file equivalent of _synthetic_config, for tests that exercise config-file loading.

    Returns:
        A dataset config TOML document matching _synthetic_config's fields.
    """

    return """
schema_version = 1
[dataset]
name = "Synthetic fixture"
slug = "synthetic"
version = "1.0.0"
source_url = "https://example.test/synthetic"
download_url = "https://example.test/files/synthetic/"
sample_rate_hz = 360
annotation_extension = "atr"
record_ids = ["100"]
required_extensions = ["atr", "dat", "hea"]
""".strip()


def _mapping_config_content() -> str:
    """A binary annotation-mapping TOML document: "N" -> reference_normal, "V" -> selected_other, "!" excluded.

    Returns:
        A mapping config TOML document matching the symbols
        _write_synthetic_record's annotations use.
    """

    return """
schema_version = 1
[mapping]
name = "synthetic"
version = "1.0.0"
unknown_symbol_policy = "error"
[[targets]]
name = "reference_normal"
value = 0
symbols = ["N"]
[[targets]]
name = "selected_other"
value = 1
symbols = ["V"]
[exclusions]
symbols = ["!"]
""".strip()
