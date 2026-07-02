"""Integration tests for the data inventory command boundary."""

from pathlib import Path

from ecg_anomaly_detection.cli import main


def test_inventory_then_verify_commands(tmp_path: Path) -> None:
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
record_ids = ["100"]
required_extensions = ["atr", "dat", "hea"]
""".strip(),
        encoding="utf-8",
    )
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    for extension in ("atr", "dat", "hea"):
        (data_dir / f"100.{extension}").write_bytes(extension.encode())
    manifest_path = tmp_path / "inventory.json"

    inventory_exit_code = main(
        [
            "inventory",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--output",
            str(manifest_path),
        ]
    )
    verify_exit_code = main(
        [
            "verify",
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert inventory_exit_code == 0
    assert verify_exit_code == 0
    assert manifest_path.is_file()
