"""Tests for repeatable, fail-safe public dataset acquisition."""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import ecg_anomaly_detection.acquisition as acquisition
from ecg_anomaly_detection.acquisition import (
    AcquisitionError,
    Fetcher,
    TransferResult,
    acquire_dataset,
)
from ecg_anomaly_detection.config import DatasetConfig


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig(
        schema_version=1,
        name="Synthetic fixture",
        slug="synthetic",
        version="1.0.0",
        source_url="https://example.test/content/synthetic/1.0.0/",
        download_url="https://example.test/files/synthetic/1.0.0/",
        sample_rate_hz=360,
        annotation_extension="atr",
        record_ids=("100",),
        required_extensions=("atr", "dat", "hea"),
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    return tmp_path


def test_acquisition_downloads_then_reuses_verified_files(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    payloads = _payloads(dataset_config)
    calls: list[str] = []
    fetcher = _fetcher(payloads, calls)

    first = acquire_dataset(
        dataset_config,
        repository,
        Path("data/raw/synthetic/1.0.0"),
        Path("artifacts/acquisition.json"),
        fetcher=fetcher,
        clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    second = acquire_dataset(
        dataset_config,
        repository,
        Path("data/raw/synthetic/1.0.0"),
        Path("artifacts/acquisition.json"),
        fetcher=fetcher,
    )

    assert first.downloaded_file_count == 3
    assert first.reused_file_count == 0
    assert second.downloaded_file_count == 0
    assert second.reused_file_count == 3
    assert len(calls) == 3
    assert first.manifest == second.manifest
    assert first.manifest.created_at_utc == "2026-01-02T03:04:05Z"
    assert all(not item.path.startswith("/") for item in first.manifest.files)


def test_acquisition_restores_only_missing_files_against_baseline(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    payloads = _payloads(dataset_config)
    calls: list[str] = []
    fetcher = _fetcher(payloads, calls)
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    manifest_path = repository / "artifacts" / "acquisition.json"
    acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)
    (data_dir / "100.dat").unlink()

    result = acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)

    assert result.downloaded_file_count == 1
    assert result.reused_file_count == 2
    assert (data_dir / "100.dat").read_bytes() == payloads[dataset_config.download_url + "100.dat"]


def test_acquisition_rejects_changed_existing_file(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    fetcher = _fetcher(_payloads(dataset_config), [])
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    manifest_path = repository / "artifacts" / "acquisition.json"
    acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)
    (data_dir / "100.hea").write_bytes(b"changed")

    with pytest.raises(AcquisitionError, match="differs from acquisition manifest"):
        acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)


def test_acquisition_rejects_changed_source_when_restoring_missing_file(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    payloads = _payloads(dataset_config)
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    manifest_path = repository / "artifacts" / "acquisition.json"
    acquire_dataset(
        dataset_config,
        repository,
        data_dir,
        manifest_path,
        fetcher=_fetcher(payloads, []),
    )
    (data_dir / "100.dat").unlink()
    payloads[dataset_config.download_url + "100.dat"] = b"upstream-changed"

    with pytest.raises(AcquisitionError, match="retrieved file differs"):
        acquire_dataset(
            dataset_config,
            repository,
            data_dir,
            manifest_path,
            fetcher=_fetcher(payloads, []),
        )


def test_acquisition_rejects_existing_required_file_without_baseline(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    data_dir.mkdir(parents=True)
    (data_dir / "100.atr").write_bytes(b"unknown")

    with pytest.raises(AcquisitionError, match="without an acquisition manifest"):
        acquire_dataset(
            dataset_config,
            repository,
            data_dir,
            repository / "artifacts" / "acquisition.json",
            fetcher=_fetcher(_payloads(dataset_config), []),
        )


def test_https_transport_streams_and_hashes_identity_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"synthetic-public-data"
    url = "https://example.test/files/100.dat"
    response = _Response(url, content)

    def fake_open(request: Any, timeout: float) -> _Response:
        assert request.full_url == url
        assert request.get_header("Accept-encoding") == "identity"
        assert timeout == 5.0
        return response

    monkeypatch.setattr(acquisition, "_open_https_request", fake_open)
    output = tmp_path / "100.dat"

    result = acquisition._fetch_https_file(url, output, 5.0, 1024)

    assert output.read_bytes() == content
    assert result == TransferResult(len(content), hashlib.sha256(content).hexdigest())


def test_https_transport_rejects_insecure_url_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(*_: object, **__: object) -> None:
        raise AssertionError("network must not be called")

    monkeypatch.setattr(acquisition, "_open_https_request", fail_if_called)

    with pytest.raises(AcquisitionError, match="must be an HTTPS"):
        acquisition._fetch_https_file("http://example.test/100.dat", tmp_path / "out", 5, 1024)


def test_redirect_handler_rejects_before_following() -> None:
    handler = acquisition._RejectRedirects()
    request = urllib.request.Request("https://example.test/100.dat")

    with pytest.raises(AcquisitionError, match="redirect rejected"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://other.example/100.dat",
        )


def _payloads(config: DatasetConfig) -> dict[str, bytes]:
    return {
        config.download_url + relative_path: f"fixture-{relative_path}".encode()
        for relative_path in config.expected_files
    }


def _fetcher(payloads: dict[str, bytes], calls: list[str]) -> Fetcher:
    def fetch(url: str, output: Path, timeout: float, maximum: int) -> TransferResult:
        content = payloads[url]
        assert timeout > 0
        assert len(content) <= maximum
        output.write_bytes(content)
        calls.append(url)
        return TransferResult(len(content), hashlib.sha256(content).hexdigest())

    return fetch


class _Response:
    def __init__(self, url: str, content: bytes) -> None:
        self._url = url
        self._content = content
        self._read = False
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return self._url

    def read(self, _: int) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._content
