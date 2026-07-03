"""Integration test for the acquisition CLI boundary."""

import hashlib
import json
from pathlib import Path

import pytest

import ecg_anomaly_detection.acquisition as acquisition
from ecg_anomaly_detection.acquisition import TransferResult
from ecg_anomaly_detection.cli import main


def test_acquire_command_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    config = tmp_path / "dataset.toml"
    config.write_text(
        """
schema_version = 1
[dataset]
name = "Synthetic fixture"
slug = "synthetic"
version = "1.0.0"
source_url = "https://example.test/content/synthetic/1.0.0/"
download_url = "https://example.test/files/synthetic/1.0.0/"
sample_rate_hz = 360
annotation_extension = "atr"
record_ids = ["100"]
required_extensions = ["atr", "dat", "hea"]
""".strip(),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_fetch(url: str, output: Path, _: float, __: int) -> TransferResult:
        content = f"fixture-{output.name}".encode()
        output.write_bytes(content)
        calls.append(url)
        return TransferResult(len(content), hashlib.sha256(content).hexdigest())

    monkeypatch.setattr(acquisition, "_fetch_https_file", fake_fetch)
    arguments = [
        "acquire",
        "--repository-root",
        str(tmp_path),
        "--config",
        str(config),
        "--data-dir",
        "data/raw/synthetic/1.0.0",
        "--output",
        "artifacts/acquisition.json",
    ]

    first_exit_code = main(arguments)
    second_exit_code = main(arguments)

    document = json.loads((tmp_path / "artifacts" / "acquisition.json").read_text())
    assert first_exit_code == 0
    assert second_exit_code == 0
    assert len(calls) == 3
    assert document["dataset_slug"] == "synthetic"
    assert len(document["files"]) == 3
