"""Integration tests for command boundaries."""

import json
from pathlib import Path

import nbformat
import pytest

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
download_url = "https://example.test/files/synthetic/"
sample_rate_hz = 360
annotation_extension = "atr"
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


def test_check_local_notebooks_cli_emits_json_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "notebooks/local/example.ipynb"
    path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("raise RuntimeError('must not execute')")]
    )
    nbformat.write(notebook, path)

    exit_code = main(["check-local-notebooks", "--repository-root", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["valid"] is True
    assert payload["notebook_count"] == 1
