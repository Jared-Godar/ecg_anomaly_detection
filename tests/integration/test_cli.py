"""Integration tests for command boundaries."""

import json
from pathlib import Path

import nbformat
import pytest

from ecg_anomaly_detection.cli import main


def test_inventory_then_verify_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`ecg-data inventory` writes a manifest that a subsequent `ecg-data verify` accepts unchanged.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture stdout and confirm each command's own progress
            banners (see #61).
    """

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
    # Write one distinct-content file per required extension, matching dataset.toml above.
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
    inventory_output = capsys.readouterr().out
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
    verify_output = capsys.readouterr().out

    assert inventory_exit_code == 0
    assert verify_exit_code == 0
    assert manifest_path.is_file()
    # Each command's own captured stdout carries its own start/completion
    # progress banners (#61), independent of the other command's output.
    assert "[1/1] inventory: starting" in inventory_output
    assert "[1/1] inventory: complete in" in inventory_output
    assert "[1/1] verify: starting" in verify_output
    assert "[1/1] verify: complete in" in verify_output


def test_check_local_notebooks_cli_emits_json_without_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ecg-data check-local-notebooks --json` reports a notebook as valid without ever running
    its cells, even though the one cell present would raise if executed.

    check_notebooks's "never executes cells" guarantee is what this test
    actually protects; if the CLI ever started executing notebooks, this
    deliberately-raising cell would surface as a crash instead of a clean
    JSON report. It also protects the --json/progress-banner boundary from
    #61: stdout must stay a pure, machine-parseable JSON blob, so this
    command's start/completion banners must land on stderr instead.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.
        capsys: Used to capture and parse main's printed stdout as JSON, and
            to confirm progress banners went to stderr instead.
    """

    path = tmp_path / "notebooks/local/example.ipynb"
    path.parent.mkdir(parents=True)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("raise RuntimeError('must not execute')")]
    )
    nbformat.write(notebook, path)

    exit_code = main(["check-local-notebooks", "--repository-root", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["valid"] is True
    assert payload["notebook_count"] == 1
    # The progress banners must not appear on stdout, or the line above would
    # already have failed to parse as JSON; confirm they landed on stderr.
    assert "[1/1] check-local-notebooks: starting" in captured.err
    assert "[1/1] check-local-notebooks: complete in" in captured.err
