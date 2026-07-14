"""Repeatable, fail-safe retrieval of versioned public dataset files."""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import os
import shutil
import string
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ecg_anomaly_detection.config import DatasetConfig, ExpectedSourceFile
from ecg_anomaly_detection.progress import UnitTimingSnapshot, format_unit_timing_suffix

# Chunk size for streaming reads/writes (hashing, HTTPS downloads, file copies). 1 MiB
# balances syscall overhead against peak memory for files that can be tens of megabytes.
BUFFER_SIZE = 1024 * 1024
# Default per-request HTTPS timeout; generous enough for a slow public mirror without
# letting one stalled request hang a pipeline run indefinitely.
DEFAULT_TIMEOUT_SECONDS = 60.0
# Default ceiling on a single downloaded file's size, enforced both from the advertised
# Content-Length header and against the actual bytes streamed, so a misconfigured or
# malicious source can't exhaust local disk during acquisition.
DEFAULT_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
# Total per-file transport attempts (one initial attempt plus bounded retries) for
# transient connectivity failures only, per the repository's defensive-external-calls
# rule (#201): a single brief PhysioNet timeout must not abort a whole acquisition,
# while permanent failures (404, digest/size mismatch) still fail fast on attempt one.
DEFAULT_TRANSIENT_ATTEMPT_COUNT = 3
# First retry backoff wait; each subsequent retry doubles it (2s, then 4s). Bounded and
# short: this absorbs momentary blips, not extended outages, which still fail cleanly.
DEFAULT_RETRY_BACKOFF_BASE_SECONDS = 2.0
# HTTP statuses treated as transient server conditions worth one bounded retry cycle.
# Everything else in the 4xx/5xx range (404 missing file, 403 auth) is permanent:
# retrying could not change the outcome and would only delay the clean failure.
TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AcquisitionError(ValueError):
    """Raised when acquisition cannot preserve its source and integrity contracts."""


class TransientAcquisitionError(AcquisitionError):
    """Raised when one retrieval attempt failed for a plausibly transient network reason.

    The subclass relationship keeps every existing ``except AcquisitionError``
    caller working unchanged, while the retry layer (_fetch_with_transient_retries)
    can catch this specific type to retry only failures that a short wait can
    plausibly fix — timeouts, connection drops (including a response body
    truncated mid-transfer, http.client.IncompleteRead), name-resolution blips,
    and the transient HTTP statuses in TRANSIENT_HTTP_STATUS_CODES. Integrity failures
    (digest/size mismatch, unexpected redirect, size-cap violation) never use this
    type and therefore always fail fast on the first attempt.
    """


@dataclass(frozen=True, slots=True)
class AcquiredFile:
    """Authoritative URL and local digest for one required source file."""

    path: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionManifest:
    """Stable baseline written before downloaded files are committed locally."""

    schema_version: int
    dataset_slug: str
    dataset_version: str
    source_url: str
    download_url: str
    created_at_utc: str
    files: tuple[AcquiredFile, ...]

    def to_json(self) -> str:
        """Serialize with deterministic formatting."""
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, content: str) -> AcquisitionManifest:
        """Parse and validate a serialized acquisition baseline."""
        # Manifests are untrusted JSON read back from disk; collapse the ways parsing or
        # dict/attribute access can fail into one AcquisitionError so callers only need to
        # catch this module's own exception type.
        try:
            document = json.loads(content)
            files = tuple(AcquiredFile(**item) for item in document["files"])
            manifest = cls(
                schema_version=document["schema_version"],
                dataset_slug=document["dataset_slug"],
                dataset_version=document["dataset_version"],
                source_url=document["source_url"],
                download_url=document["download_url"],
                created_at_utc=document["created_at_utc"],
                files=files,
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise AcquisitionError(f"invalid acquisition manifest: {error}") from error
        # This module has only ever written schema_version 1; any other value means the
        # manifest was produced by an incompatible tool or corrupted.
        if manifest.schema_version != 1:
            raise AcquisitionError("acquisition manifest must use schema_version 1")
        # Every identity field must be a genuinely present string -- an empty dataset_slug
        # or source_url would silently break the config-match check in
        # _validate_manifest_for_config, which compares these fields against the live config.
        if not all(
            isinstance(value, str) and value
            for value in (
                manifest.dataset_slug,
                manifest.dataset_version,
                manifest.source_url,
                manifest.download_url,
                manifest.created_at_utc,
            )
        ):
            raise AcquisitionError("acquisition manifest identity fields must be non-empty strings")
        # A duplicated path would mean two entries claim the same on-disk file, which
        # would make _resume_acquisition's per-file resume logic ambiguous.
        if len({item.path for item in manifest.files}) != len(manifest.files):
            raise AcquisitionError("acquisition manifest contains duplicate file paths")
        # Re-validate every file entry's own shape (path safety, URL scheme, digest
        # format) -- see _valid_acquired_file for what "invalid" means here.
        if any(not _valid_acquired_file(item) for item in manifest.files):
            raise AcquisitionError("acquisition manifest contains invalid file evidence")
        return manifest


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Digest returned by a transport after writing a staged file."""

    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Acquisition baseline plus idempotency counts for this invocation."""

    manifest: AcquisitionManifest
    downloaded_file_count: int
    reused_file_count: int


@dataclass(frozen=True, slots=True)
class AcquisitionRecordProgress:
    """Completed transfer counts for one configured dataset record.

    The callback payload is deliberately record-grained rather than file-grained. A
    MIT-BIH record has three required companion files, so one event per record keeps a
    first acquisition visibly active without producing 144 low-value file messages.
    """

    record_index: int
    record_total: int
    record_id: str
    downloaded_file_count: int
    reused_file_count: int


# Fetcher is the transport abstraction acquire_dataset depends on: given a URL,
# destination path, timeout, and size cap, it stages a file and returns its digest. Tests
# substitute a fake Fetcher to exercise acquisition logic without real network access.
Fetcher = Callable[[str, Path, float, int], TransferResult]

# AcquisitionProgressCallback is an observational hook invoked only after all required
# files for one configured record have passed their integrity checks. The callback never
# changes acquisition evidence or determines which files are retrieved.
AcquisitionProgressCallback = Callable[[AcquisitionRecordProgress], None]


def format_acquisition_record_progress(progress: AcquisitionRecordProgress) -> str:
    """Render one concise record-level acquisition progress message.

    Fresh, cached, and partially resumed records use different language so users can
    tell whether network retrieval is active without mistaking a verified cache hit for
    a download. Counts are derived from completed integrity checks, not predictions.

    Args:
        progress: Record identity and completed downloaded/reused file counts.

    Returns:
        One human-readable line suitable for ``ProgressReporter.note``.
    """

    prefix = f"record {progress.record_index}/{progress.record_total} ({progress.record_id}): "
    downloaded = progress.downloaded_file_count
    reused = progress.reused_file_count
    # A fully fresh record reports that every downloaded file also passed the committed
    # size/digest checks; "downloaded" alone would understate the integrity behavior.
    if downloaded and not reused:
        noun = "file" if downloaded == 1 else "files"
        return f"{prefix}downloaded and verified {downloaded} {noun}"
    # A fully cached record is re-hashed and compared with both source expectations and
    # its acquisition manifest, so "verified existing" is more accurate than "skipped".
    if reused and not downloaded:
        noun = "file" if reused == 1 else "files"
        return f"{prefix}verified {reused} existing {noun}"
    # A partially restored record can contain both newly retrieved and reusable files;
    # keep both counts in one record-level line instead of emitting per-file noise.
    downloaded_noun = "file" if downloaded == 1 else "files"
    reused_noun = "file" if reused == 1 else "files"
    return (
        f"{prefix}downloaded and verified {downloaded} missing {downloaded_noun}; "
        f"verified {reused} existing {reused_noun}"
    )


def format_timed_acquisition_record_progress(
    progress: AcquisitionRecordProgress, timing: UnitTimingSnapshot
) -> str:
    """Render one record-level progress message with qualified timing detail (#199).

    Composes the existing record wording with the shared timing suffix so each
    completed record reports its own measured duration, the acquisition phase's
    measured elapsed time, and a clearly qualified approximate remaining duration
    (or an explicit warm-up state) — still exactly one concise line per record.

    Args:
        progress: Record identity and completed downloaded/reused file counts.
        timing: The record's timing snapshot from a per-phase UnitTimingEstimator.

    Returns:
        One human-readable line suitable for ``ProgressReporter.note``.
    """

    return format_acquisition_record_progress(progress) + format_unit_timing_suffix(
        timing, unit_label="record"
    )


def _notify_record_progress(
    callback: AcquisitionProgressCallback | None,
    *,
    record_index: int,
    record_total: int,
    record_id: str,
    downloaded_file_count: int,
    reused_file_count: int,
) -> None:
    """Invoke an optional callback after one record's required files are complete.

    Args:
        callback: Optional record-level observational hook supplied by the caller.
        record_index: One-based configured record position.
        record_total: Total number of configured records.
        record_id: Dataset record identifier just completed.
        downloaded_file_count: Files retrieved and verified for this record.
        reused_file_count: Existing files re-hashed and verified for this record.
    """

    # None is the backwards-compatible silent path for library callers that do not need
    # interactive progress; acquisition results and manifests remain unchanged.
    if callback is None:
        return
    callback(
        AcquisitionRecordProgress(
            record_index=record_index,
            record_total=record_total,
            record_id=record_id,
            downloaded_file_count=downloaded_file_count,
            reused_file_count=reused_file_count,
        )
    )


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Fail before following an HTTP redirect to preserve the configured source boundary."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        """Reject every redirect unconditionally, regardless of destination.

        urllib's HTTPRedirectHandler protocol calls this to build the follow-up request;
        returning None here (after raising) is not an option, so raising is the only way
        to stop urlopen from silently following an HTTP 3xx to a URL this module never
        validated. A redirect could otherwise point the download at an untrusted host
        while looking, from the caller's side, like a normal successful fetch.

        Args:
            request: The original request that received a redirect response.
            file_pointer: Unused; required by HTTPRedirectHandler's method signature.
            code: Unused; required by HTTPRedirectHandler's method signature.
            message: Unused; required by HTTPRedirectHandler's method signature.
            headers: Unused; required by HTTPRedirectHandler's method signature.
            new_url: The redirect target, included in the raised error for diagnosis.
        """

        del request, file_pointer, code, message, headers
        raise AcquisitionError(f"retrieval redirect rejected: {new_url}")


def _is_transient_retrieval_failure(error: BaseException) -> bool:
    """Decide whether one failed retrieval attempt is plausibly transient.

    Transient means a short wait can plausibly change the outcome: a timeout, a
    dropped/refused/reset connection (including one dropped mid-body, surfaced
    by http.client as IncompleteRead), a name-resolution blip, or a momentary
    server-side condition (see TRANSIENT_HTTP_STATUS_CODES). Everything else —
    a definitive HTTP client error such as 404, a malformed protocol exchange
    (http.client's other HTTPException values), or a local OSError like a full
    disk while staging — is permanent and must fail fast, because retrying it
    delays the clean failure without changing the result.

    Args:
        error: The exception one transport attempt raised.

    Returns:
        True when a bounded retry with backoff is justified; False otherwise.
    """

    # HTTPError is checked before its URLError parent class: it carries a
    # definitive server-assigned status code, so only the codes explicitly
    # listed as transient server conditions justify a retry.
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP_STATUS_CODES
    # A non-HTTP URLError means the request never got a definitive answer at all
    # (DNS failure, refused connection, TLS/socket-level timeout) -- the classic
    # transient connectivity class this retry layer exists for.
    if isinstance(error, urllib.error.URLError):
        return True
    # A response that ends before delivering its advertised Content-Length
    # surfaces from response.read() as http.client.IncompleteRead (#206): the
    # connection dropped mid-body, the same transient class as a reset, just
    # reported by http.client's own bookkeeping instead of a socket error.
    # http.client's other HTTPException values (e.g. a malformed status line)
    # are protocol-level defects a wait cannot fix and stay permanent.
    if isinstance(error, http.client.IncompleteRead):
        return True
    # Raw socket-level failures can surface outside urllib's wrapping while the
    # response body is being streamed: timeouts and reset/aborted connections.
    # ConnectionError covers the reset/refused/aborted family; other bare OSError
    # values (e.g. a local disk error) fall through to permanent below.
    return isinstance(error, TimeoutError | ConnectionError)


def _fetch_with_transient_retries(
    fetcher: Fetcher,
    url: str,
    staged_path: Path,
    timeout_seconds: float,
    max_file_size_bytes: int,
    sleep: Callable[[float], None],
) -> TransferResult:
    """Run one file transfer with bounded backoff retries for transient failures only.

    This wrapper sits between acquisition and every transport call (production
    HTTPS or an injected test fetcher), so the retry policy is uniform and
    testable without network access. Only TransientAcquisitionError is retried;
    any other failure — including every integrity failure — propagates
    immediately, unchanged. Between attempts the partially staged file is
    removed, so a retry re-enters the transport with the same clean "must not
    already exist" staging contract as the first attempt; the digest and size
    expectations applied afterward by the caller are identical for every
    attempt, keeping retries integrity-preserving by construction.

    Args:
        fetcher: The transport to invoke for each attempt.
        url: The HTTPS URL to retrieve.
        staged_path: Staging destination the transport writes to.
        timeout_seconds: Per-request timeout passed through to the transport.
        max_file_size_bytes: Per-file size cap passed through to the transport.
        sleep: Wait function for backoff; injectable for deterministic tests.

    Returns:
        The successful attempt's staged size and SHA-256 digest.
    """

    # Attempts are numbered from 1 so the exhaustion message reports the human
    # total ("after 3 attempts"), not a zero-based index.
    for attempt in range(1, DEFAULT_TRANSIENT_ATTEMPT_COUNT + 1):
        # Only the transient subtype is caught below; every other failure —
        # including permanent AcquisitionError — propagates out of the loop
        # unchanged on its first occurrence.
        try:
            return fetcher(url, staged_path, timeout_seconds, max_file_size_bytes)
        except TransientAcquisitionError as error:
            # A failed attempt can leave a partial staged file behind; remove it
            # so the next attempt's exclusive-create ("xb") open cannot collide
            # with debris from this one.
            staged_path.unlink(missing_ok=True)
            # Exhaustion exits gracefully per the defensive-external-calls rule:
            # name what failed, state plainly that it is an external/connectivity
            # condition rather than a code or setup defect, and give remediation.
            if attempt == DEFAULT_TRANSIENT_ATTEMPT_COUNT:
                raise AcquisitionError(
                    f"could not retrieve {url} after {attempt} attempts: {error}. "
                    "This is an external connectivity or service condition (network "
                    "timeout, connection drop, or a transient server error), not a "
                    "defect in this repository or your setup. Check your internet "
                    "connection and whether the source host is reachable, then re-run; "
                    "acquisition is atomic and a re-run restarts cleanly."
                ) from error
            # Exponential backoff (base, 2x base, ...) gives a brief blip time to
            # clear without turning a real outage into a long silent stall.
            sleep(DEFAULT_RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    # Unreachable: every loop iteration either returns or raises, but an explicit
    # error keeps this function total if the constants above are ever misedited.
    raise AcquisitionError(f"retrieval retry loop exited without a result for {url}")


def acquire_dataset(
    config: DatasetConfig,
    repository_root: Path,
    data_dir: Path,
    manifest_path: Path,
    *,
    fetcher: Fetcher | None = None,
    clock: Callable[[], datetime] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    progress_callback: AcquisitionProgressCallback | None = None,
    sleep: Callable[[float], None] | None = None,
) -> AcquisitionResult:
    """Retrieve required files or verify them against an existing acquisition baseline.

    Args:
        config: Versioned dataset identity and closed required-file inventory.
        repository_root: Checkout root containing ``pyproject.toml``.
        data_dir: Canonical raw-data directory for this dataset and version.
        manifest_path: Repository artifact path for acquisition evidence.
        fetcher: Optional transport substitute; defaults to strict HTTPS retrieval.
        clock: Optional acquisition timestamp source used when creating a new manifest.
        timeout_seconds: Positive per-request transport timeout.
        max_file_size_bytes: Positive hard cap for each retrieved file.
        progress_callback: Optional hook called once after each record's required files
            have been downloaded/reused and integrity-verified.
        sleep: Optional retry backoff wait substitute (defaults to ``time.sleep``);
            injectable so retry tests run deterministically without real waiting.

    Returns:
        Stable acquisition evidence and invocation-level transfer/reuse counts.
    """
    expectations = _expected_source_files(config)
    # A non-positive timeout would mean every request fails (or never times out at all),
    # neither of which is a usable configuration.
    if timeout_seconds <= 0:
        raise AcquisitionError("download timeout must be positive")
    # A non-positive size cap would reject every download outright.
    if max_file_size_bytes <= 0:
        raise AcquisitionError("maximum file size must be positive")
    root = repository_root.resolve()
    # A pyproject.toml at the root is the cheapest available signal that this is really
    # the repository root, before any path-boundary checks below trust it as the
    # containment root for the data directory and manifest path.
    if not (root / "pyproject.toml").is_file():
        raise AcquisitionError(f"repository root does not contain pyproject.toml: {root}")
    destination = _canonical_data_directory(config, root, data_dir)
    output = _canonical_manifest_path(root, manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    _reject_unexpected_source_files(config, destination)
    transport = fetcher or _fetch_https_file
    # time.sleep is the production backoff wait; tests inject a recording fake so
    # transient-retry behavior is asserted deterministically without real delays.
    wait = sleep or time.sleep

    # An existing manifest means a prior run already established (or partially
    # established) this acquisition baseline; resume/verify against it rather than
    # re-downloading from scratch, which is what makes acquisition idempotent.
    if output.exists():
        # Reject a symlink or non-regular-file manifest before trusting its contents,
        # since resolving through a link could read/verify against an unrelated file.
        if output.is_symlink() or not output.is_file():
            raise AcquisitionError(f"acquisition manifest must be a regular file: {output}")
        manifest = _read_manifest(output)
        _validate_manifest_for_config(config, manifest)
        return _resume_acquisition(
            config,
            destination,
            manifest,
            transport,
            timeout_seconds,
            max_file_size_bytes,
            progress_callback,
            wait,
        )

    existing = [name for name in config.expected_files if (destination / name).exists()]
    # Files present without a manifest means a previous acquisition was interrupted
    # before the manifest could be written (or the directory was tampered with);
    # neither case is safe to silently treat as a fresh, empty destination.
    if existing:
        raise AcquisitionError(
            "required files already exist without an acquisition manifest: " + ", ".join(existing)
        )
    now = (clock or (lambda: datetime.now(UTC)))()
    # created_at_utc is serialized with an explicit "Z" suffix below; a naive datetime
    # would make that suffix a lie about the actual timezone the timestamp represents.
    if now.tzinfo is None:
        raise AcquisitionError("acquisition clock must return a timezone-aware datetime")

    # Stage every file in a temporary directory alongside the destination (same
    # filesystem, so the later hard-link install is atomic) and let the `with` block
    # clean up the staging directory even if a download fails partway through.
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
        staging = Path(staging_name)
        acquired: list[AcquiredFile] = []
        # The nested config order is the same deterministic order exposed by
        # DatasetConfig.expected_files, while also giving acquisition a natural point to
        # emit exactly one callback after each record's companion files are complete.
        record_total = len(config.record_ids)
        # Complete each record as one user-visible progress unit.
        for record_index, record_id in enumerate(config.record_ids, start=1):
            # Retrieve every required companion extension before reporting the record.
            for extension in config.required_extensions:
                relative_path = f"{record_id}.{extension}"
                url = _file_url(config, relative_path)
                staged_path = staging / relative_path
                # The retry wrapper absorbs bounded transient connectivity blips;
                # every integrity check below still applies to whichever attempt
                # finally succeeded, so retries never weaken the evidence contract.
                transfer = _fetch_with_transient_retries(
                    transport, url, staged_path, timeout_seconds, max_file_size_bytes, wait
                )
                _validate_transfer(transfer, relative_path)
                _validate_expected_transfer(transfer, expectations[relative_path])
                acquired.append(
                    AcquiredFile(
                        path=relative_path,
                        url=url,
                        size_bytes=transfer.size_bytes,
                        sha256=transfer.sha256,
                    )
                )
            _notify_record_progress(
                progress_callback,
                record_index=record_index,
                record_total=record_total,
                record_id=record_id,
                downloaded_file_count=len(config.required_extensions),
                reused_file_count=0,
            )
        manifest = AcquisitionManifest(
            schema_version=1,
            dataset_slug=config.slug,
            dataset_version=config.version,
            source_url=config.source_url,
            download_url=config.download_url,
            created_at_utc=now.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            files=tuple(acquired),
        )
        _write_manifest_once(manifest, output)
        # Install files into the destination only after the manifest is durably written,
        # so a crash between writing files and writing the manifest can never leave files
        # on disk that the manifest doesn't yet account for.
        for item in manifest.files:
            _install_without_overwrite(staging / item.path, destination / item.path)
    return AcquisitionResult(
        manifest=manifest,
        downloaded_file_count=len(manifest.files),
        reused_file_count=0,
    )


def _resume_acquisition(
    config: DatasetConfig,
    destination: Path,
    manifest: AcquisitionManifest,
    fetcher: Fetcher,
    timeout_seconds: float,
    max_file_size_bytes: int,
    progress_callback: AcquisitionProgressCallback | None,
    sleep: Callable[[float], None],
) -> AcquisitionResult:
    """Resume acquisition by verifying existing files and retrieving only missing files.

    This is the idempotency mechanism that lets `run-pipeline` be re-invoked safely: a
    file already on disk is re-hashed and checked against the manifest's recorded digest
    (never blindly trusted, since it could have been modified out of band); a missing
    file is re-downloaded. Both paths converge on the same per-file expectation check
    used during the initial acquisition, so resumed and fresh runs enforce identical
    integrity guarantees.

    Args:
        config: The dataset config this acquisition is running under.
        destination: The validated data directory files are read from/written to.
        manifest: The existing manifest establishing which files are expected.
        fetcher: Transport used to retrieve any file not already present.
        timeout_seconds: Per-request timeout passed through to the fetcher.
        max_file_size_bytes: Per-file size cap passed through to the fetcher.
        progress_callback: Optional hook called once after each record is complete.
        sleep: Retry backoff wait used when a missing file's re-download hits a
            transient connectivity failure.

    Returns:
        Result reporting how many files were reused versus freshly downloaded.
    """

    expectations = _expected_source_files(config)
    downloaded = 0
    reused = 0
    # Manifest/config validation above guarantees these names and their deterministic
    # order match, so indexing by path makes record grouping explicit without weakening
    # any manifest identity check.
    manifest_files = {item.path: item for item in manifest.files}
    record_total = len(config.record_ids)
    # Re-verify or restore each record before emitting its one aggregate callback.
    for record_index, record_id in enumerate(config.record_ids, start=1):
        record_downloaded = 0
        record_reused = 0
        # Count downloaded/reused companion files independently for accurate wording.
        for extension in config.required_extensions:
            item = manifest_files[f"{record_id}.{extension}"]
            destination_path = destination / item.path
            # A file already on disk can potentially be reused instead of re-downloaded --
            # but only after re-verifying it, not merely because a manifest entry exists.
            if destination_path.exists():
                # Reject a symlink or non-regular file rather than hashing through it,
                # since that could reuse the contents of an unrelated file.
                if destination_path.is_symlink() or not destination_path.is_file():
                    raise AcquisitionError(
                        f"acquired path must be a regular file: {destination_path}"
                    )
                current = _hash_file(destination_path)
                _validate_expected_transfer(current, expectations[item.path], existing=True)
                # The manifest recorded a specific digest at acquisition time; if the
                # file no longer matches it, the acquired baseline is no longer trusted.
                if current != TransferResult(item.size_bytes, item.sha256):
                    raise AcquisitionError(
                        f"existing file differs from acquisition manifest: {item.path}"
                    )
                reused += 1
                record_reused += 1
                continue
            # Stage the re-download in a fresh temporary directory so a failed retry
            # never leaves a partial file at the final destination path.
            with tempfile.TemporaryDirectory(prefix=".acquire-", dir=destination) as staging_name:
                staged_path = Path(staging_name) / item.path
                # Restored files get the same bounded transient-retry treatment as a
                # fresh acquisition; the manifest-digest comparison below still binds
                # whichever attempt succeeded to the recorded baseline.
                transfer = _fetch_with_transient_retries(
                    fetcher, item.url, staged_path, timeout_seconds, max_file_size_bytes, sleep
                )
                _validate_transfer(transfer, item.path)
                _validate_expected_transfer(transfer, expectations[item.path])
                # A restored file must match the acquisition manifest, not merely the
                # live config, preventing silent upstream substitution during resume.
                if transfer != TransferResult(item.size_bytes, item.sha256):
                    raise AcquisitionError(
                        f"retrieved file differs from acquisition manifest: {item.path}"
                    )
                _install_without_overwrite(staged_path, destination_path)
            downloaded += 1
            record_downloaded += 1
        _notify_record_progress(
            progress_callback,
            record_index=record_index,
            record_total=record_total,
            record_id=record_id,
            downloaded_file_count=record_downloaded,
            reused_file_count=record_reused,
        )
    return AcquisitionResult(
        manifest=manifest,
        downloaded_file_count=downloaded,
        reused_file_count=reused,
    )


def _fetch_https_file(
    url: str,
    output_path: Path,
    timeout_seconds: float,
    max_file_size_bytes: int,
) -> TransferResult:
    """Fetch one HTTPS resource into a staged file under strict size and digest controls.

    This is the production Fetcher implementation acquire_dataset defaults to; tests
    substitute a fake Fetcher instead of exercising real network I/O. Every safety
    control here (HTTPS-only, no redirects, size-capped, streamed) exists because the
    source URL, while normally trusted, is still remote content this pipeline shouldn't
    let corrupt or exhaust local resources.

    Args:
        url: The HTTPS URL to retrieve; must already be validated by the caller
            (see _validate_requested_url, called below as a second line of defense).
        output_path: Where to write the staged file; must not already exist.
        timeout_seconds: Per-request timeout for the HTTPS connection.
        max_file_size_bytes: Hard cap on both advertised and actual response size.

    Returns:
        The staged file's size and SHA-256 digest.
    """

    _validate_requested_url(url)
    # url's scheme is already validated by _validate_requested_url above.
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "ecg-anomaly-detection-portfolio/0.1",
        },
        method="GET",
    )
    digest = hashlib.sha256()
    size_bytes = 0
    # Collapse every failure mode (network timeout, connection error, our own validation
    # errors) into AcquisitionError, except that AcquisitionError itself passes through
    # unchanged so the specific validation message isn't replaced by a generic one.
    try:
        # The `with` block ensures the HTTPS response is closed even if a size or status
        # check below raises partway through reading it.
        with _open_https_request(request, timeout_seconds) as response:
            # Any non-200 status (redirects are already blocked by _RejectRedirects,
            # so this mainly catches 4xx/5xx) means there's no file content to trust.
            if response.getcode() != 200:
                raise AcquisitionError(f"unexpected HTTP status for {url}: {response.getcode()}")
            _validate_final_url(url, response.geturl())
            content_length = _content_length(response.headers.get("Content-Length"), url)
            # Reject an oversized file based on the advertised length before reading any
            # bytes, so a large file doesn't unnecessarily start streaming to disk.
            if content_length is not None and content_length > max_file_size_bytes:
                raise AcquisitionError(f"remote file exceeds size limit: {url}")
            # Open with mode "xb" (exclusive create, binary) so this can never silently
            # overwrite a pre-existing file at the staged path.
            with output_path.open("xb") as output:
                # The walrus operator lets both the read and the loop's termination
                # condition (an empty final chunk) live in one line without a `break`.
                while chunk := response.read(BUFFER_SIZE):
                    size_bytes += len(chunk)
                    # A server can lie about (or omit) Content-Length; re-check the size
                    # cap against actual bytes streamed so far, not just the advertised
                    # header, so a misbehaving source still can't exhaust local disk.
                    if size_bytes > max_file_size_bytes:
                        raise AcquisitionError(f"remote file exceeds size limit: {url}")
                    output.write(chunk)
                    digest.update(chunk)
            # A Content-Length that doesn't match what was actually streamed indicates a
            # truncated or otherwise unreliable transfer.
            if content_length is not None and content_length != size_bytes:
                raise AcquisitionError(f"Content-Length mismatch for {url}")
    except AcquisitionError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as error:
        # Distinguish plausibly transient connectivity failures (retryable by
        # _fetch_with_transient_retries) from permanent ones (fail fast) while
        # keeping the same single-exception surface for callers: both types are
        # AcquisitionError, so non-retrying callers behave exactly as before.
        # HTTPException is caught since #206: response.read() raises
        # http.client.IncompleteRead (not an OSError or URLError) when the
        # connection drops mid-body, and it previously escaped this collapse.
        if _is_transient_retrieval_failure(error):
            raise TransientAcquisitionError(f"could not retrieve {url}: {error}") from error
        raise AcquisitionError(f"could not retrieve {url}: {error}") from error
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _open_https_request(request: urllib.request.Request, timeout_seconds: float) -> Any:
    """Open one HTTPS request through an opener that rejects every redirect.

    Isolated as its own function so _RejectRedirects (which has meaningful, tested
    behavior of its own) is wired in exactly once, rather than every call site needing
    to remember to build a custom opener.

    Args:
        request: The prepared HTTPS request to send.
        timeout_seconds: Timeout for the underlying socket connection.

    Returns:
        The open response object (a context manager); the caller is responsible for
        closing it, normally via a `with` block.
    """

    opener = urllib.request.build_opener(_RejectRedirects())
    return opener.open(request, timeout=timeout_seconds)


def _canonical_data_directory(config: DatasetConfig, root: Path, data_dir: Path) -> Path:
    """Resolve and enforce the one valid data directory for a dataset/version pair.

    Acquired files always live at a fixed, predictable location
    (data/raw/{slug}/{version}/) derived from the dataset config rather than wherever
    the caller happens to point data_dir -- this prevents a config/CLI-argument mismatch
    from silently acquiring files into (or reading them from) the wrong location.

    Args:
        config: Dataset config supplying the slug/version that fix the expected location.
        root: Repository root used to enforce path and trust boundaries.
        data_dir: The caller-supplied data directory, checked against the expected path.

    Returns:
        The canonical, resolved data directory path.
    """

    raw_root_path = root / "data" / "raw"
    # data/raw must already exist as a real directory (not a symlink) before this
    # function trusts it as the containment root for the dataset-specific subdirectory.
    if raw_root_path.is_symlink() or not raw_root_path.is_dir():
        raise AcquisitionError(
            f"raw data root must be an existing regular directory: {raw_root_path}"
        )
    raw_root = raw_root_path.resolve()
    expected = raw_root / config.slug / config.version
    candidate = data_dir if data_dir.is_absolute() else root / data_dir
    # The caller-supplied data_dir must resolve to exactly the expected canonical path;
    # anything else (including a symlink that happens to point there) is rejected so the
    # actual on-disk location can never silently diverge from what the config implies.
    if candidate.is_symlink() or candidate.resolve() != expected:
        raise AcquisitionError(
            f"data directory must be data/raw/{config.slug}/{config.version} within the repository"
        )
    return expected


def _canonical_manifest_path(root: Path, manifest_path: Path) -> Path:
    """Resolve and enforce that the acquisition manifest lives under artifacts/.

    Manifests are lightweight provenance records (unlike the raw dataset files) and are
    kept in the shared artifacts/ tree rather than alongside the data itself, matching
    this repository's directory contract for pipeline-generated evidence.

    Args:
        root: Repository root used to enforce path and trust boundaries.
        manifest_path: The caller-supplied manifest path, absolute or repo-relative.

    Returns:
        The resolved, validated manifest path.
    """

    candidate = manifest_path if manifest_path.is_absolute() else root / manifest_path
    # Reject a symlink before resolving it, so a link that points outside artifacts/
    # can't be validated against a resolved target it doesn't actually name.
    if candidate.is_symlink():
        raise AcquisitionError("acquisition manifest must not be a symbolic link")
    resolved = candidate.resolve()
    # relative_to raises ValueError when resolved escapes root (e.g. via `..` segments);
    # translate that into this module's own exception type.
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise AcquisitionError("acquisition manifest must stay within repository root") from error
    # Manifests live in artifacts/, matching this repository's directory contract for
    # pipeline-generated provenance evidence rather than raw dataset content.
    if not relative.parts or relative.parts[0] != "artifacts":
        raise AcquisitionError("acquisition manifest must be written under artifacts/")
    # Enforce the extension so downstream tooling that globs for *.json manifests can
    # rely on finding this file without a separate content sniff.
    if resolved.suffix != ".json":
        raise AcquisitionError("acquisition manifest must use the .json extension")
    # Fail before attempting the write rather than letting a missing parent directory
    # surface as a generic OSError later.
    if not resolved.parent.is_dir():
        raise AcquisitionError(f"acquisition manifest parent does not exist: {resolved.parent}")
    return resolved


def _reject_unexpected_source_files(config: DatasetConfig, destination: Path) -> None:
    """Reject source-directory files that are absent from the configured closed inventory.

    The dataset config declares a closed, exact set of expected files; anything else
    present in the destination (a stray download, a manually added file, leftover
    staging debris) would otherwise silently coexist with the tracked files without
    ever being verified or accounted for in the manifest.

    Args:
        config: Dataset config supplying the closed set of expected file names.
        destination: The data directory to check for unexpected entries.
    """

    expected = set(config.expected_files)
    unexpected = sorted(path.name for path in destination.iterdir() if path.name not in expected)
    # Any entry not in the config's declared set is unaccounted-for and untrusted.
    if unexpected:
        raise AcquisitionError(
            "unexpected source file or directory in configured data directory: "
            + ", ".join(unexpected)
        )


def _validate_manifest_for_config(config: DatasetConfig, manifest: AcquisitionManifest) -> None:
    """Confirm an existing on-disk manifest actually matches the current dataset config.

    A manifest from a prior run could describe a different dataset slug/version, a
    different source, or a config that has since been edited; resuming against a
    mismatched manifest would silently verify files against the wrong expectations.

    Args:
        config: The dataset config the caller is currently running acquisition under.
        manifest: The manifest read back from disk, to check against that config.
    """

    # Every identity field the manifest recorded must match the live config exactly.
    if (
        manifest.dataset_slug,
        manifest.dataset_version,
        manifest.source_url,
        manifest.download_url,
    ) != (config.slug, config.version, config.source_url, config.download_url):
        raise AcquisitionError("acquisition manifest identity does not match dataset configuration")
    expected = config.expected_files
    # The manifest's file list must match the config's expected_files in the same order,
    # since order is part of what makes repeated runs deterministic.
    if tuple(item.path for item in manifest.files) != expected:
        raise AcquisitionError(
            "acquisition manifest file order does not match dataset configuration"
        )
    # Each file's recorded URL must still match what the current config's download_url
    # would produce -- if download_url changed, the manifest's URLs are now stale.
    if any(item.url != _file_url(config, item.path) for item in manifest.files):
        raise AcquisitionError("acquisition manifest file URL does not match dataset configuration")
    expectations = _expected_source_files(config)
    # Also re-verify every recorded digest against the committed expected-source
    # metadata, not just against the config's identity fields.
    for item in manifest.files:
        _validate_expected_transfer(
            TransferResult(item.size_bytes, item.sha256), expectations[item.path]
        )


def _expected_source_files(config: DatasetConfig) -> dict[str, ExpectedSourceFile]:
    """Cross-check the config's expected_files list against its committed digest metadata.

    A dataset config declares expected_files (the closed file-name inventory) and
    separately carries per-file digest/size metadata; this function confirms the two
    lists actually agree before any download or verification uses either, since a
    config author could otherwise add a file to one list and forget the other.

    Args:
        config: The dataset config whose two file lists are being cross-checked.

    Returns:
        Digest metadata for every expected file, keyed by relative path.
    """

    expected_paths = set(config.expected_files)
    by_path = config.expected_source_files_by_path
    missing = sorted(expected_paths - set(by_path))
    unexpected = sorted(set(by_path) - expected_paths)
    # Either direction of mismatch means the two committed lists have drifted apart.
    if missing or unexpected:
        raise AcquisitionError(
            "committed expected source metadata is incomplete; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return by_path


def _validate_expected_transfer(
    actual: TransferResult, expected: ExpectedSourceFile, *, existing: bool = False
) -> None:
    """Compare a transfer's actual size/digest against the config's committed expectation.

    Shared by both the fresh-download and resume/verify paths (the `existing` flag only
    changes the error message wording) so every file, however it was obtained, is held
    to the same digest evidence recorded in the dataset config.

    Args:
        actual: The size and SHA-256 actually observed for this file.
        expected: The size and SHA-256 committed in the dataset config for this file.
        existing: Whether `actual` came from a file already on disk (True) or a fresh
            download (False); only affects error message wording.
    """

    prefix = "existing source file" if existing else "retrieved source file"
    # Size and digest are checked separately so a mismatch's specific cause is clear.
    if actual.size_bytes != expected.size_bytes:
        raise AcquisitionError(
            f"{prefix} size mismatch for {expected.path}: "
            f"expected {expected.size_bytes} bytes, got {actual.size_bytes}"
        )
    # A size match alone isn't sufficient evidence of content integrity.
    if actual.sha256.lower() != expected.sha256:
        raise AcquisitionError(
            f"{prefix} SHA-256 mismatch for {expected.path}: "
            f"expected {expected.sha256}, got {actual.sha256.lower()}"
        )


def _read_manifest(path: Path) -> AcquisitionManifest:
    """Read and parse an acquisition manifest from disk.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        The parsed, structurally validated manifest.
    """

    # Translate a missing or unreadable file into AcquisitionError so callers only need
    # to catch one exception type for every acquisition-related failure.
    try:
        return AcquisitionManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AcquisitionError(f"could not read acquisition manifest {path}: {error}") from error


def _write_manifest_once(manifest: AcquisitionManifest, output_path: Path) -> None:
    """Write a manifest exactly once, atomically, never overwriting an existing one.

    Writes to a temporary file in the same directory (so os.link is guaranteed to be an
    atomic same-filesystem operation), fsyncs it, then hard-links it to the final name.
    os.link fails with FileExistsError if the destination already exists, which is what
    makes "exactly once" an enforced guarantee rather than a convention -- a concurrent
    or repeated write can never silently clobber an existing manifest.

    Args:
        manifest: The manifest to serialize and write.
        output_path: The manifest's final path; must not already exist.
    """

    temporary_path: Path | None = None
    # Collapse both the "already exists" and other OS-level failure modes into
    # AcquisitionError, and use `finally` to guarantee the temporary file is cleaned up
    # on every exit path, not just the success path.
    try:
        # delete=False keeps the temp file on disk after the `with` block exits, so it
        # can still be hard-linked afterward; NamedTemporaryFile would otherwise delete
        # it as soon as the block closes the handle.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(manifest.to_json())
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise AcquisitionError(f"acquisition manifest already exists: {output_path}") from error
    except OSError as error:
        raise AcquisitionError(
            f"could not write acquisition manifest {output_path}: {error}"
        ) from error
    finally:
        # Remove the temporary file whether the link succeeded or failed; missing_ok
        # covers the success path, where the link already moved the only reference.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _install_without_overwrite(source: Path, destination: Path) -> None:
    """Install a staged file atomically without overwriting an existing destination.

    Prefers a hard link (instant, no data copy, and atomically fails via FileExistsError
    if destination already exists) and only falls back to a copy when source and
    destination are on different filesystems, where os.link is impossible. The fallback
    still preserves the same no-overwrite atomicity by copying into a temporary file on
    destination's filesystem first, then hard-linking from there.

    Args:
        source: The staged file to install (in a temporary staging directory).
        destination: Where to install it; must not already exist.
    """

    # Collapse both the "already exists" and other OS-level failure modes into
    # AcquisitionError; a bare `return` on success skips the EXDEV fallback below.
    try:
        os.link(source, destination)
        return
    except FileExistsError as error:
        raise AcquisitionError(
            f"refusing to overwrite acquired file: {destination.name}"
        ) from error
    except OSError as error:
        # EXDEV specifically means "cross-device link" -- source and destination are on
        # different filesystems, which os.link cannot bridge. Any other OSError is a
        # genuine failure and should propagate immediately rather than falling through
        # to the copy-then-hardlink path below.
        if error.errno != errno.EXDEV:
            raise AcquisitionError(
                f"could not install acquired file {destination.name}: {error}"
            ) from error

    # source and destination are on different filesystems, so a hard link is
    # not possible. Copy into a temporary file alongside destination (always
    # the same filesystem as destination) and hard-link from there, so the
    # atomic no-overwrite guarantee above still holds instead of being lost to
    # a plain copy-and-replace.
    temporary_path: Path | None = None
    # Same error-collapsing and cleanup pattern as _write_manifest_once above.
    try:
        # delete=False keeps the temp file on disk after the `with` block exits, so it
        # can still be hard-linked afterward.
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            # Stream the copy in fixed-size chunks (via shutil.copyfileobj) rather than
            # reading the whole file into memory, matching this module's other streaming
            # I/O for consistent memory behavior on large files.
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, temporary, BUFFER_SIZE)
            temporary.flush()
            os.fsync(temporary.fileno())
        shutil.copystat(source, temporary_path)
        os.link(temporary_path, destination)
    except FileExistsError as error:
        raise AcquisitionError(
            f"refusing to overwrite acquired file: {destination.name}"
        ) from error
    except OSError as error:
        raise AcquisitionError(
            f"could not install acquired file {destination.name}: {error}"
        ) from error
    finally:
        # Remove the temporary copy whether the hard link succeeded or failed.
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _hash_file(path: Path) -> TransferResult:
    """Calculate stable size and SHA-256 evidence for one local file.

    Used by _resume_acquisition to re-verify a file already present on disk against the
    manifest's recorded digest, without trusting the file's mere existence as proof it's
    unmodified.

    Args:
        path: The local file to hash.

    Returns:
        The file's size and SHA-256 digest.
    """

    digest = hashlib.sha256()
    size_bytes = 0
    # Read in fixed-size chunks rather than the whole file at once, since acquired
    # dataset files can be tens of megabytes.
    with path.open("rb") as source:
        # The walrus operator lets both the read and the loop's termination condition
        # (an empty final chunk) live in one line without a separate `break`.
        while chunk := source.read(BUFFER_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    return TransferResult(size_bytes=size_bytes, sha256=digest.hexdigest())


def _file_url(config: DatasetConfig, relative_path: str) -> str:
    """Build the full download URL for one expected file under the config's download_url.

    Args:
        config: Dataset config supplying the base download_url.
        relative_path: The file's path relative to that base, as declared in
            config.expected_files.

    Returns:
        The full URL to fetch, with the relative path percent-encoded.
    """

    return config.download_url + quote(relative_path, safe="")


def _validate_transfer(transfer: TransferResult, relative_path: str) -> None:
    """Sanity-check a raw transfer result's shape before comparing it to any expectation.

    Runs before _validate_expected_transfer so a malformed TransferResult (e.g. from a
    custom test fetcher with a bug) produces a specific "malformed evidence" error rather
    than a confusing mismatch against the expected digest.

    Args:
        transfer: The size/digest just returned by a fetcher.
        relative_path: The file's path, included in any error for traceability.
    """

    # size_bytes must be a genuine positive int (bool excluded since it's an int
    # subclass in Python); zero or negative would mean nothing was actually transferred.
    if (
        not isinstance(transfer.size_bytes, int)
        or isinstance(transfer.size_bytes, bool)
        or transfer.size_bytes <= 0
    ):
        raise AcquisitionError(f"retrieved file is empty: {relative_path}")
    # sha256 must be exactly 64 hex characters (the fixed length of a SHA-256 digest in
    # hex); anything else means the fetcher returned a malformed or non-hex digest.
    if (
        not isinstance(transfer.sha256, str)
        or len(transfer.sha256) != 64
        or any(character not in string.hexdigits for character in transfer.sha256)
    ):
        raise AcquisitionError(f"retrieved file has invalid SHA-256 evidence: {relative_path}")


def _valid_acquired_file(item: AcquiredFile) -> bool:
    """Structurally validate one AcquiredFile entry read back from a manifest.

    Checks path safety (no path separators, so `path` can never escape the flat data
    directory it's joined against), URL shape, and digest format -- the same shape
    constraints _validate_transfer enforces on a fresh transfer, re-checked here because
    manifest entries are untrusted data read back from disk.

    Args:
        item: One file entry from a parsed AcquisitionManifest.

    Returns:
        True if every field is well-formed; False otherwise (never raises, since this
        is used inside an `any()` check across every file entry in the manifest).
    """

    # path must be a non-empty string with no directory separators (so it can only ever
    # name a file directly inside the data directory, never escape it via `../`);
    # size_bytes and sha256 get the same shape checks as _validate_transfer above.
    if (
        not isinstance(item.path, str)
        or not item.path
        or "/" in item.path
        or "\\" in item.path
        or not isinstance(item.url, str)
        or not isinstance(item.size_bytes, int)
        or isinstance(item.size_bytes, bool)
        or item.size_bytes <= 0
        or not isinstance(item.sha256, str)
        or len(item.sha256) != 64
        or any(character not in string.hexdigits for character in item.sha256)
    ):
        return False
    # urlsplit can raise ValueError on a malformed URL (e.g. an invalid port); treat that
    # as "not valid" rather than letting the exception propagate out of this predicate.
    try:
        parsed = urlsplit(item.url)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _validate_requested_url(url: str) -> None:
    """Confirm a URL is a plain HTTPS file URL before it's used to make a request.

    This is the pre-request half of this module's URL trust boundary (the other half,
    _validate_final_url, checks where the response actually came from); together they
    ensure acquisition only ever talks to plain HTTPS hosts with no embedded
    credentials, query strings, or fragments that could carry unexpected side effects.

    Args:
        url: The candidate URL to validate.
    """

    # urlsplit can raise ValueError on a malformed URL (e.g. an invalid port).
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as error:
        raise AcquisitionError(f"retrieval URL must be an HTTPS file URL: {url}") from error
    # Reject anything other than a bare https://host/path URL: embedded credentials
    # (username/password) could leak into logs, and a query string or fragment isn't
    # meaningful for a static file download under this pipeline's trust model.
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AcquisitionError(f"retrieval URL must be an HTTPS file URL: {url}")


def _validate_final_url(requested_url: str, final_url: str) -> None:
    """Confirm the URL a response actually came from still matches what was requested.

    _RejectRedirects should already prevent urllib from following any redirect, but this
    is a second, independent check against response.geturl() -- defense in depth against
    a future change to the redirect handler (or an edge case it doesn't cover) silently
    reintroducing an unvalidated final destination.

    Args:
        requested_url: The URL that was originally requested.
        final_url: The URL urllib reports the response actually came from.
    """

    # urlsplit can raise ValueError on a malformed URL.
    try:
        requested = urlsplit(requested_url)
        final = urlsplit(final_url)
    except ValueError as error:
        raise AcquisitionError(f"retrieval returned an invalid final URL: {final_url}") from error
    # scheme, host, and path must match exactly (query/fragment/credentials are
    # rejected outright, matching _validate_requested_url's constraints).
    if (
        final.scheme != "https"
        or final.netloc != requested.netloc
        or final.path != requested.path
        or final.username is not None
        or final.password is not None
        or final.query
        or final.fragment
    ):
        raise AcquisitionError(f"retrieval redirected outside the configured file URL: {final_url}")


def _content_length(value: str | None, url: str) -> int | None:
    """Parse an HTTP Content-Length header value, if present.

    Args:
        value: The raw header value, or None if the response didn't send one (some
            servers omit it, particularly for chunked transfer encoding).
        url: The URL this header came from, included in any error for traceability.

    Returns:
        The parsed length, or None if the header was absent.
    """

    # Absence is a valid, expected state -- not every server sends Content-Length.
    if value is None:
        return None
    # A non-numeric header value means the server sent something this parser doesn't
    # understand, which is treated as untrustworthy rather than silently ignored.
    try:
        length = int(value)
    except ValueError as error:
        raise AcquisitionError(f"invalid Content-Length for {url}") from error
    # A negative length is nonsensical and would defeat the size-cap check that uses
    # this value in _fetch_https_file.
    if length < 0:
        raise AcquisitionError(f"invalid Content-Length for {url}")
    return length
