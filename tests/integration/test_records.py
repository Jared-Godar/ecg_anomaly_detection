"""Integration tests using synthetic files written through the WFDB package."""

from pathlib import Path

import numpy as np
import wfdb

from ecg_anomaly_detection.cli import main
from ecg_anomaly_detection.config import DatasetConfig
from ecg_anomaly_detection.records import load_wfdb_record, validate_record


def test_load_and_validate_synthetic_wfdb_record(tmp_path: Path) -> None:
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
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_synthetic_record(data_dir)
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        """
schema_version = 1
[dataset]
name = "Synthetic fixture"
slug = "synthetic"
version = "1.0.0"
source_url = "https://example.test/synthetic"
sample_rate_hz = 360
annotation_extension = "atr"
record_ids = ["100"]
required_extensions = ["atr", "dat", "hea"]
""".strip(),
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


def _write_synthetic_record(data_dir: Path) -> None:
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
    return DatasetConfig(
        schema_version=1,
        name="Synthetic fixture",
        slug="synthetic",
        version="1.0.0",
        source_url="https://example.test/synthetic",
        sample_rate_hz=360,
        annotation_extension="atr",
        record_ids=("100",),
        required_extensions=("atr", "dat", "hea"),
    )
