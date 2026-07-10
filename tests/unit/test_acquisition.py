"""Tests for repeatable, fail-safe public dataset acquisition."""

from __future__ import annotations

import errno
import hashlib
import os
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
from ecg_anomaly_detection.config import DatasetConfig, ExpectedSourceFile


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """Build or exercise the dataset config test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Returns:
        The value produced by the documented operation.
    """

    files = tuple(
        ExpectedSourceFile(
            path=f"100.{extension}",
            size_bytes=len(content := f"fixture-100.{extension}".encode()),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for extension in ("atr", "dat", "hea")
    )
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
        expected_source_files=files,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Build or exercise the repository test fixture.

    The helper keeps repeated test setup explicit without hiding the contract under
    examination.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.

    Returns:
        The value produced by the documented operation.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    return tmp_path


def test_acquisition_downloads_then_reuses_verified_files(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition downloads then reuses verified files.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

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
    """Verify that acquisition restores only missing files against baseline.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

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
    """Verify that acquisition rejects changed existing file.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    fetcher = _fetcher(_payloads(dataset_config), [])
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    manifest_path = repository / "artifacts" / "acquisition.json"
    acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)
    (data_dir / "100.hea").write_bytes(b"changed")

    # Scope `pytest.raises(AcquisitionError, match='size mismatch for 100\\.hea')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match=r"size mismatch for 100\.hea"):
        acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)


def test_acquisition_rejects_changed_source_when_restoring_missing_file(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition rejects changed source when restoring missing file.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

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

    # Scope `pytest.raises(AcquisitionError, match='size mismatch for 100\\.dat')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match=r"size mismatch for 100\.dat"):
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
    """Verify that acquisition rejects existing required file without baseline.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    data_dir.mkdir(parents=True)
    (data_dir / "100.atr").write_bytes(b"unknown")

    # Scope `pytest.raises(AcquisitionError, match='without an acquisition manifest')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="without an acquisition manifest"):
        acquire_dataset(
            dataset_config,
            repository,
            data_dir,
            repository / "artifacts" / "acquisition.json",
            fetcher=_fetcher(_payloads(dataset_config), []),
        )


def test_acquisition_rejects_expected_sha256_mismatch(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition rejects expected sha256 mismatch.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    payloads = _payloads(dataset_config)
    payloads[dataset_config.download_url + "100.dat"] = b"fixture-100.dax"

    # Scope `pytest.raises(AcquisitionError, match='SHA-256 mismatch for 100\\.dat')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match=r"SHA-256 mismatch for 100\.dat"):
        acquire_dataset(
            dataset_config,
            repository,
            Path("data/raw/synthetic/1.0.0"),
            Path("artifacts/acquisition.json"),
            fetcher=_fetcher(payloads, []),
        )


def test_acquisition_rejects_expected_size_mismatch(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition rejects expected size mismatch.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    payloads = _payloads(dataset_config)
    payloads[dataset_config.download_url + "100.dat"] = b"wrong-size"

    # Scope `pytest.raises(AcquisitionError, match='size mismatch for 100\\.dat')` here so the
    # expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match=r"size mismatch for 100\.dat"):
        acquire_dataset(
            dataset_config,
            repository,
            Path("data/raw/synthetic/1.0.0"),
            Path("artifacts/acquisition.json"),
            fetcher=_fetcher(payloads, []),
        )


def test_acquisition_rejects_missing_expected_metadata(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition rejects missing expected metadata.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    incomplete = DatasetConfig(
        **{
            field: getattr(dataset_config, field)
            for field in (
                "schema_version",
                "name",
                "slug",
                "version",
                "source_url",
                "download_url",
                "sample_rate_hz",
                "annotation_extension",
                "record_ids",
                "required_extensions",
            )
        }
    )

    # Scope `pytest.raises(AcquisitionError, match='expected source metadata is incomplete')` here
    # so the expected failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="expected source metadata is incomplete"):
        acquire_dataset(
            incomplete,
            repository,
            Path("data/raw/synthetic/1.0.0"),
            Path("artifacts/acquisition.json"),
            fetcher=_fetcher(_payloads(incomplete), []),
        )


def test_acquisition_rejects_unexpected_source_file(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """Verify that acquisition rejects unexpected source file.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        repository: The repository value supplied by the caller or surrounding test fixture.
        dataset_config: The dataset config value supplied by the caller or surrounding test fixture.
    """

    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    data_dir.mkdir(parents=True)
    (data_dir / "README").write_text("unexpected", encoding="utf-8")

    # Scope `pytest.raises(AcquisitionError, match='unexpected source file')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="unexpected source file"):
        acquire_dataset(
            dataset_config,
            repository,
            data_dir,
            Path("artifacts/acquisition.json"),
            fetcher=_fetcher(_payloads(dataset_config), []),
        )


def test_https_transport_streams_and_hashes_identity_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that https transport streams and hashes identity response.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    content = b"synthetic-public-data"
    url = "https://example.test/files/100.dat"
    response = _Response(url, content)

    def fake_open(request: Any, timeout: float) -> _Response:
        """Build or exercise the fake open test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            request: Validated request object crossing the external boundary.
            timeout: The timeout value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

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
    """Verify that https transport rejects insecure url before network.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    def fail_if_called(*_: object, **__: object) -> None:
        """Build or exercise the fail if called test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            _: The operation value supplied by the caller or surrounding test fixture.
            __: The operation value supplied by the caller or surrounding test fixture.
        """

        raise AssertionError("network must not be called")

    monkeypatch.setattr(acquisition, "_open_https_request", fail_if_called)

    # Scope `pytest.raises(AcquisitionError, match='must be an HTTPS')` here so the expected failure
    # and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="must be an HTTPS"):
        acquisition._fetch_https_file("http://example.test/100.dat", tmp_path / "out", 5, 1024)


def test_redirect_handler_rejects_before_following() -> None:
    """Verify that redirect handler rejects before following.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.
    """

    handler = acquisition._RejectRedirects()
    request = urllib.request.Request("https://example.test/100.dat")

    # Scope `pytest.raises(AcquisitionError, match='redirect rejected')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="redirect rejected"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://other.example/100.dat",
        )


def test_install_without_overwrite_hard_links_same_filesystem(tmp_path: Path) -> None:
    """Verify that install without overwrite hard links same filesystem.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
    """

    source = tmp_path / "source.dat"
    source.write_bytes(b"same-filesystem payload")
    destination = tmp_path / "destination.dat"

    acquisition._install_without_overwrite(source, destination)

    assert destination.read_bytes() == b"same-filesystem payload"
    assert destination.stat().st_ino == source.stat().st_ino


def test_install_without_overwrite_falls_back_across_filesystems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that install without overwrite falls back across filesystems.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    source = tmp_path / "source.dat"
    source.write_bytes(b"cross-filesystem payload")
    source.chmod(0o400)
    destination = tmp_path / "destination.dat"

    real_link = os.link
    calls: list[tuple[Path, Path]] = []

    def flaky_link(src: str | Path, dst: str | Path) -> None:
        """Build or exercise the flaky link test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            src: The src value supplied by the caller or surrounding test fixture.
            dst: The dst value supplied by the caller or surrounding test fixture.
        """

        calls.append((Path(src), Path(dst)))
        # Exercise the `len(calls) == 1` branch so this regression documents every expected outcome.
        if len(calls) == 1:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_link(src, dst)

    monkeypatch.setattr(acquisition.os, "link", flaky_link)

    acquisition._install_without_overwrite(source, destination)

    assert destination.read_bytes() == b"cross-filesystem payload"
    assert destination.stat().st_mode == source.stat().st_mode
    assert len(calls) == 2
    assert calls[0] == (source, destination)
    fallback_source, fallback_destination = calls[1]
    assert fallback_destination == destination
    assert fallback_source != source
    assert fallback_source.parent == destination.parent
    assert set(tmp_path.iterdir()) == {source, destination}


def test_install_without_overwrite_refuses_overwrite_after_cross_filesystem_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that install without overwrite refuses overwrite after cross filesystem fallback.

    This regression test makes the named behavior and its failure boundary visible to future
    maintainers.

    Args:
        tmp_path: Temporary filesystem root supplied by pytest for isolated artifacts.
        monkeypatch: Pytest monkeypatch fixture used to isolate external behavior.
    """

    source = tmp_path / "source.dat"
    source.write_bytes(b"new payload")
    destination = tmp_path / "destination.dat"
    destination.write_bytes(b"already acquired")

    real_link = os.link
    calls = 0

    def flaky_link(src: str | Path, dst: str | Path) -> None:
        """Build or exercise the flaky link test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            src: The src value supplied by the caller or surrounding test fixture.
            dst: The dst value supplied by the caller or surrounding test fixture.
        """

        nonlocal calls
        calls += 1
        # Exercise the `calls == 1` branch so this regression documents every expected outcome.
        if calls == 1:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_link(src, dst)

    monkeypatch.setattr(acquisition.os, "link", flaky_link)

    # Scope `pytest.raises(AcquisitionError, match='refusing to overwrite')` here so the expected
    # failure and fixture cleanup stay scoped to this assertion.
    with pytest.raises(AcquisitionError, match="refusing to overwrite"):
        acquisition._install_without_overwrite(source, destination)

    assert destination.read_bytes() == b"already acquired"
    assert set(tmp_path.iterdir()) == {source, destination}


def _payloads(config: DatasetConfig) -> dict[str, bytes]:
    """Compute and return payloads for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        config: Validated configuration controlling the documented operation.

    Returns:
        The value produced by the documented operation.
    """

    return {
        config.download_url + relative_path: f"fixture-{relative_path}".encode()
        for relative_path in config.expected_files
    }


def _fetcher(payloads: dict[str, bytes], calls: list[str]) -> Fetcher:
    """Compute and return fetcher for the documented repository workflow.

    The helper isolates this step so its assumptions, outputs, and failure behavior remain
    reviewable.

    Args:
        payloads: The payloads value supplied by the caller or surrounding test fixture.
        calls: The calls value supplied by the caller or surrounding test fixture.

    Returns:
        The value produced by the documented operation.
    """

    def fetch(url: str, output: Path, timeout: float, maximum: int) -> TransferResult:
        """Build or exercise the fetch test fixture.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            url: The url value supplied by the caller or surrounding test fixture.
            output: The output value supplied by the caller or surrounding test fixture.
            timeout: The timeout value supplied by the caller or surrounding test fixture.
            maximum: The maximum value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

        content = payloads[url]
        assert timeout > 0
        assert len(content) <= maximum
        output.write_bytes(content)
        calls.append(url)
        return TransferResult(len(content), hashlib.sha256(content).hexdigest())

    return fetch


class _Response:
    """Simulate the minimal HTTP response contract used by downloader tests.

    The fixture supports context management, status and URL inspection, content-length headers,
    and one-shot streaming reads without opening a real network connection.
    """

    def __init__(self, url: str, content: bytes) -> None:
        """Initialize this object with the validated state required by its contract.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            url: The url value supplied by the caller or surrounding test fixture.
            content: The content value supplied by the caller or surrounding test fixture.
        """

        self._url = url
        self._content = content
        self._read = False
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> _Response:
        """Return the active fake response when the downloader enters its context.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Returns:
            The value produced by the documented operation.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """Complete fake-response cleanup without suppressing an active exception.

        The helper isolates this step so its assumptions, outputs, and failure behavior remain
        reviewable.

        Args:
            _: The operation value supplied by the caller or surrounding test fixture.

        """

        return None

    def getcode(self) -> int:
        """Return a successful HTTP status code for the simulated transfer.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Returns:
            The value produced by the documented operation.
        """

        return 200

    def geturl(self) -> str:
        """Return the final response URL used for redirect-boundary assertions.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Returns:
            The value produced by the documented operation.
        """

        return self._url

    def read(self, _: int) -> bytes:
        """Return the payload once, then signal end-of-stream with empty bytes.

        The helper keeps repeated test setup explicit without hiding the contract under
        examination.

        Args:
            _: The operation value supplied by the caller or surrounding test fixture.

        Returns:
            The value produced by the documented operation.
        """

        # Exercise the `self._read` branch so this regression documents every expected outcome.
        if self._read:
            return b""
        self._read = True
        return self._content
