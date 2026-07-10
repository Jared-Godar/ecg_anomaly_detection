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
    """A single-record synthetic dataset config with matching expected-file digests.

    Real content/digests (not empty stand-ins) so tests can exercise the actual
    size/SHA-256 verification path acquire_dataset performs, not just its control flow.

    Returns:
        A DatasetConfig for one synthetic record ("100") with three companion files.
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
    """A minimal fake repository root: pyproject.toml plus empty data/raw and artifacts.

    acquire_dataset checks for pyproject.toml before trusting a path as the repository
    root, so tests need this present even though its content is never read.

    Args:
        tmp_path: Pytest's per-test isolated temporary directory.

    Returns:
        The fake repository root path.
    """

    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    return tmp_path


def test_acquisition_downloads_then_reuses_verified_files(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """A first acquisition downloads every file; a second reuses them via the resume path.

    Protects the idempotency guarantee documented on acquire_dataset: re-running
    acquisition against an existing manifest should verify and reuse files rather than
    re-download them, and the resulting manifest (including its timestamp) must be
    identical across both runs.
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
    """A resumed acquisition re-downloads only the specific file that went missing.

    Protects _resume_acquisition's per-file resume logic: deleting one of three
    acquired files and re-running acquisition should download exactly that one file
    and reuse (not re-fetch) the other two, restoring the original content.
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
    """A file modified out-of-band since acquisition fails re-verification.

    Protects the integrity guarantee that a resumed acquisition never silently trusts
    a file that changed after it was first acquired -- overwriting one acquired file's
    bytes must be caught by the size/digest re-check on the next acquire_dataset call.
    """

    fetcher = _fetcher(_payloads(dataset_config), [])
    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    manifest_path = repository / "artifacts" / "acquisition.json"
    acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)
    (data_dir / "100.hea").write_bytes(b"changed")

    # The size changed along with the content, so this specifically exercises the
    # size-mismatch branch of _validate_expected_transfer (existing=True).
    with pytest.raises(AcquisitionError, match=r"size mismatch for 100\.hea"):
        acquire_dataset(dataset_config, repository, data_dir, manifest_path, fetcher=fetcher)


def test_acquisition_rejects_changed_source_when_restoring_missing_file(
    repository: Path, dataset_config: DatasetConfig
) -> None:
    """A re-download that no longer matches the manifest's recorded digest is rejected.

    Protects against a silent version substitution during resume: if the upstream
    source's content for a missing file has changed since the original acquisition
    (simulated here by mutating the fetcher's payload), the re-fetched file must be
    checked against the manifest's original digest and rejected on mismatch, not
    silently accepted as the new baseline.
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

    # The re-fetched content's size no longer matches the manifest's recorded value.
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
    """Files present without a manifest are rejected rather than silently adopted.

    Protects against treating pre-existing, unverified files as a trustworthy starting
    point: acquire_dataset must never assume files found without an accompanying
    manifest are safe to reuse, since they could be from an interrupted or tampered-with
    prior state.
    """

    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    data_dir.mkdir(parents=True)
    (data_dir / "100.atr").write_bytes(b"unknown")

    # A required file exists but no manifest was ever written for it.
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
    """A freshly downloaded file whose digest disagrees with the committed expectation fails.

    Protects the config-level integrity check: even on a first-time download (no prior
    manifest), the fetched content must match the dataset config's committed
    expected_source_files digest, not just whatever the fetcher happens to return.
    """

    payloads = _payloads(dataset_config)
    payloads[dataset_config.download_url + "100.dat"] = b"fixture-100.dax"

    # Same length as the expected content, so this specifically exercises the
    # digest-mismatch branch rather than the size-mismatch one.
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
    """A freshly downloaded file whose size disagrees with the committed expectation fails.

    Same integrity guarantee as the SHA-256 mismatch test above, exercised via the
    size check instead, confirming both fields of ExpectedSourceFile are independently enforced.
    """

    payloads = _payloads(dataset_config)
    payloads[dataset_config.download_url + "100.dat"] = b"wrong-size"

    # A different-length payload exercises the size-mismatch branch specifically.
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
    """A config with expected_files but no matching expected_source_files digests fails.

    Protects _expected_source_files' cross-check between a config's file-name
    inventory and its digest metadata: constructing a DatasetConfig that omits
    expected_source_files entirely (while still declaring record_ids/required_extensions)
    must be rejected before any download is attempted, not silently skip verification.
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

    # expected_source_files is entirely absent from `incomplete`.
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
    """A stray file in the data directory that isn't part of the closed inventory fails.

    Protects _reject_unexpected_source_files: a file the dataset config never declared
    (here, a README) must block acquisition rather than being silently ignored, since
    its presence could indicate a stale or tampered-with data directory.
    """

    data_dir = repository / "data" / "raw" / "synthetic" / "1.0.0"
    data_dir.mkdir(parents=True)
    (data_dir / "README").write_text("unexpected", encoding="utf-8")

    # README is not in dataset_config.expected_files.
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
    """The production HTTPS fetcher streams a response to disk and returns its digest.

    Protects _fetch_https_file's core happy path independent of real network I/O: the
    request must ask for identity encoding (no compression, since size/digest
    verification depends on exact byte counts), and the returned TransferResult must
    match the actual bytes written to output.
    """

    content = b"synthetic-public-data"
    url = "https://example.test/files/100.dat"
    response = _Response(url, content)

    def fake_open(request: Any, timeout: float) -> _Response:
        """Fake urlopen substitute asserting the request shape _fetch_https_file builds.

        Args:
            request: The urllib Request _fetch_https_file constructed.
            timeout: The timeout value passed through from the caller.

        Returns:
            The fake in-memory response to stream from.
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
    """A plain-HTTP URL is rejected before any network call is attempted.

    Protects the fail-fast ordering in _fetch_https_file: _validate_requested_url must
    run and reject before _open_https_request is ever invoked, confirmed here by making
    the network hook raise if called at all.
    """

    def fail_if_called(*_: object, **__: object) -> None:
        """Fail the test immediately if the network transport hook is ever invoked.

        Args:
            _: Unused positional arguments, accepted to match _open_https_request's shape.
            __: Unused keyword arguments, accepted to match _open_https_request's shape.
        """

        raise AssertionError("network must not be called")

    monkeypatch.setattr(acquisition, "_open_https_request", fail_if_called)

    # http:// (not https://) should be rejected by _validate_requested_url.
    with pytest.raises(AcquisitionError, match="must be an HTTPS"):
        acquisition._fetch_https_file("http://example.test/100.dat", tmp_path / "out", 5, 1024)


def test_redirect_handler_rejects_before_following() -> None:
    """_RejectRedirects raises on any redirect instead of following it.

    Protects the redirect-blocking security boundary directly: a 302 response must
    raise AcquisitionError from redirect_request itself, confirming urllib never gets a
    chance to silently follow the link to an unvalidated destination.
    """

    handler = acquisition._RejectRedirects()
    request = urllib.request.Request("https://example.test/100.dat")

    # redirect_request must raise rather than returning a follow-up Request.
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
    """Same-filesystem installs use a hard link, not a data copy.

    Protects the fast-path assumption: source and destination sharing an inode
    (st_ino) proves _install_without_overwrite used os.link rather than falling back
    to the cross-filesystem copy-then-hardlink path.
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
    """An EXDEV (cross-device) link failure falls back to copy-then-hardlink.

    Regression test for the copy-then-hardlink fallback: simulates a real
    cross-filesystem os.link failure (errno.EXDEV) on the first call, then confirms
    the retried install succeeds via the fallback path, preserving both content and
    file mode, and leaves exactly source+destination behind (no leftover temp file).
    """

    source = tmp_path / "source.dat"
    source.write_bytes(b"cross-filesystem payload")
    source.chmod(0o400)
    destination = tmp_path / "destination.dat"

    real_link = os.link
    calls: list[tuple[Path, Path]] = []

    def flaky_link(src: str | Path, dst: str | Path) -> None:
        """Fake os.link that raises EXDEV once, then delegates to the real os.link.

        Simulates exactly the cross-device-link failure _install_without_overwrite's
        EXDEV fallback branch is designed to handle.

        Args:
            src: The link source path.
            dst: The link destination path.
        """

        calls.append((Path(src), Path(dst)))
        # First call simulates the cross-device failure; subsequent calls (the
        # fallback's own hard-link step) succeed via the real os.link.
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
    """The cross-filesystem fallback still refuses to overwrite an existing destination.

    Protects the no-overwrite guarantee specifically along the EXDEV fallback path
    (not just the fast hard-link path): even after falling back to copy-then-hardlink,
    a destination that already exists must still cause a refusal, and the original
    destination content must be left untouched.
    """

    source = tmp_path / "source.dat"
    source.write_bytes(b"new payload")
    destination = tmp_path / "destination.dat"
    destination.write_bytes(b"already acquired")

    real_link = os.link
    calls = 0

    def flaky_link(src: str | Path, dst: str | Path) -> None:
        """Fake os.link that raises EXDEV once, then delegates to the real os.link.

        Args:
            src: The link source path.
            dst: The link destination path.
        """

        nonlocal calls
        calls += 1
        # First call simulates the cross-device failure that triggers the fallback;
        # the fallback's own hard-link attempt (second call) then hits the real
        # FileExistsError from destination already being present.
        if calls == 1:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_link(src, dst)

    monkeypatch.setattr(acquisition.os, "link", flaky_link)

    # destination already exists, so even the EXDEV fallback path must refuse.
    with pytest.raises(AcquisitionError, match="refusing to overwrite"):
        acquisition._install_without_overwrite(source, destination)

    assert destination.read_bytes() == b"already acquired"
    assert set(tmp_path.iterdir()) == {source, destination}


def _payloads(config: DatasetConfig) -> dict[str, bytes]:
    """Build a fake download-URL-to-content mapping matching a config's expected files.

    Args:
        config: The dataset config whose expected_files this builds payloads for.

    Returns:
        A dict from each file's full download URL to deterministic fixture bytes.
    """

    return {
        config.download_url + relative_path: f"fixture-{relative_path}".encode()
        for relative_path in config.expected_files
    }


def _fetcher(payloads: dict[str, bytes], calls: list[str]) -> Fetcher:
    """Build a fake Fetcher serving fixed payloads and recording every call made.

    Args:
        payloads: Map from download URL to the bytes that URL should "download".
        calls: A list this fetcher appends each requested URL to, so tests can
            assert exactly which files were (re-)downloaded.

    Returns:
        A Fetcher-compatible callable usable in place of the real HTTPS transport.
    """

    def fetch(url: str, output: Path, timeout: float, maximum: int) -> TransferResult:
        """Serve one fixed payload for `url`, writing it to `output` and recording the call.

        Args:
            url: The requested download URL; must be a key in the enclosing payloads dict.
            output: Where to write the fake downloaded content.
            timeout: Unused except for the sanity assertion that it's positive.
            maximum: The caller's size cap; asserted against to catch a test fixture
                whose fake payload would have violated the real size-cap contract.

        Returns:
            The digest of the served content.
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
        """Store the fake response's URL and full content for later chunked reads.

        Args:
            url: The URL this fake response reports via geturl().
            content: The full body content, returned in one chunk from the first read().
        """

        self._url = url
        self._content = content
        self._read = False
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self) -> _Response:
        """Return self as the active response, matching the real response's context-manager use.

        Returns:
            This fake response instance.
        """

        return self

    def __exit__(self, *_: object) -> None:
        """No-op exit; nothing needs cleanup for an in-memory fake response.

        Args:
            _: Exception info tuple, ignored (this fake never suppresses exceptions).
        """

        return None

    def getcode(self) -> int:
        """Report a successful (200) HTTP status, matching a real successful download.

        Returns:
            The fixed status code 200.
        """

        return 200

    def geturl(self) -> str:
        """Report the response's final URL, used by _validate_final_url's redirect check.

        Returns:
            This fake response's configured URL.
        """

        return self._url

    def read(self, _: int) -> bytes:
        """Return the full content on the first call, then empty bytes to signal EOF.

        Mirrors the streaming-read contract _fetch_https_file's while-loop depends on:
        a real socket read eventually returns b"" once the response body is exhausted.

        Args:
            _: The requested chunk size, ignored since this fake always returns the
                whole payload in one call.

        Returns:
            The full content on first call; b"" on every call after.
        """

        # The first read serves the whole payload in one chunk; every subsequent read
        # returns b"" to signal end-of-stream, matching how _fetch_https_file's
        # while-loop expects a real socket to eventually behave.
        if self._read:
            return b""
        self._read = True
        return self._content
