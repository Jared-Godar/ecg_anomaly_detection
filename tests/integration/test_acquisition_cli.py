"""Integration test for the acquisition CLI boundary."""

import hashlib
import json
from pathlib import Path

import pytest

import ecg_anomaly_detection.acquisition as acquisition
from ecg_anomaly_detection.acquisition import TransferResult
from ecg_anomaly_detection.cli import main


def test_acquire_command_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that acquire command is idempotent.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

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
expected_source_files = [
  { path = "100.atr", size_bytes = 15, sha256 = "5ff83a0d5aba6e0450863587c5566dacaa1ee3328f6bfc6c8523f804b295f24f" },
  { path = "100.dat", size_bytes = 15, sha256 = "1c69904e2556c222fa8ba6328c2434416967eab7aa936f10ca5aa13bad96c70a" },
  { path = "100.hea", size_bytes = 15, sha256 = "6ce366847c45118af26db4a50544d7a47bf4e4de0f559a674f16f88f03b54aac" },
]
""".strip(),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_fetch(url: str, output: Path, _: float, __: int) -> TransferResult:
        """Build or exercise the fake fetch test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            url: The url value supplied by the caller or surrounding test fixture.
            output: The output value supplied by the caller or surrounding test fixture.
            _: The operation value supplied by the caller or surrounding test fixture.
            __: The operation value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

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
